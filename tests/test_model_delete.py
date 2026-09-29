"""Model-level deletion: plan computation and safe execution.

Covers docs/content/models.md "Delete a model": the payload is the model's
own GGUF (all shards of its shard set), never the parent directory by
parenthood; shared mmproj files survive; the directory goes only when it
becomes empty; unsafe paths are refused.
"""
import conftest_paths  # noqa: F401
import os, shutil, tempfile, unittest
from unittest import mock

import model_delete
from model_delete import DeleteError, NotFound, plan, execute


def touch(path, size=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * size)
    return path


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def d(self, *p):
        return os.path.join(self.tmp, *p)

    def test_single_gguf_dedicated_directory(self):
        m = self.d("Qwen-7B", "model-Q4.gguf")
        touch(m)
        p = plan({"model": m})
        self.assertEqual(p["files"], [m])
        self.assertTrue(p["delete_directory"])
        self.assertEqual(p["size_bytes"], 10)

    def test_shared_directory_other_quantization_survives(self):
        q4 = self.d("repo", "model-Q4.gguf")
        q3 = self.d("repo", "model-Q3.gguf")
        touch(q4); touch(q3)
        p = plan({"model": q4}, [{"model": q3}])
        self.assertEqual(p["files"], [q4])
        self.assertFalse(p["delete_directory"])

    def test_shard_set_all_shards_included(self):
        base = self.d("repo", "big")
        shards = [touch(f"{base}-{i:05d}-of-00003.gguf") for i in (1, 2, 3)]
        other = touch(self.d("repo", "small-Q4.gguf"))
        p = plan({"model": shards[0]}, [{"model": other}])
        self.assertEqual(sorted(p["files"]), sorted(shards))
        self.assertFalse(p["delete_directory"])

    def test_shard_set_only_matching_total(self):
        a = [touch(self.d("repo", f"same-{i:05d}-of-00002.gguf")) for i in (1, 2)]
        b = [touch(self.d("repo", f"same-{i:05d}-of-00003.gguf")) for i in (1, 2, 3)]
        p = plan({"model": a[0]})
        self.assertEqual(sorted(p["files"]), sorted(a))

    def test_directory_removed_when_it_becomes_empty(self):
        m = self.d("solo", "model.gguf")
        touch(m)
        p = plan({"model": m})
        ok, err = execute(p)
        self.assertTrue(ok, err)
        self.assertFalse(os.path.exists(m))
        self.assertFalse(os.path.exists(self.d("solo")))

    def test_other_files_keep_the_directory(self):
        m = self.d("repo", "model.gguf")
        readme = touch(self.d("repo", "README.txt"))
        touch(m)
        p = plan({"model": m})
        ok, err = execute(p)
        self.assertTrue(ok, err)
        self.assertFalse(os.path.exists(m))
        self.assertTrue(os.path.exists(readme))
        self.assertTrue(os.path.exists(self.d("repo")))

    def test_exclusive_mmproj_is_deleted(self):
        m = self.d("repo", "model.gguf")
        mm = touch(self.d("repo", "mmproj-model.f16.gguf"))
        touch(m)
        p = plan({"model": m, "mmproj": mm})
        self.assertEqual(p["mmproj"], mm)
        self.assertIn(mm, p["files"])
        self.assertTrue(p["delete_directory"])

    def test_mmproj_referenced_by_another_model_is_preserved(self):
        m = self.d("repo", "model.gguf")
        mm = touch(self.d("repo", "mmproj-shared.gguf"))
        other = touch(self.d("repo", "other.gguf"))
        touch(m)
        p = plan({"model": m, "mmproj": mm},
                 [{"model": other, "mmproj": mm}])
        self.assertIsNone(p["mmproj"])
        self.assertNotIn(mm, p["files"])
        self.assertTrue(os.path.exists(mm))

    def test_mmproj_beside_other_payload_is_preserved(self):
        """A projector in a directory that still holds other model files may
        belong to them; ownership is ambiguous, so it survives."""
        m = self.d("repo", "model.gguf")
        mm = touch(self.d("repo", "mmproj-model.f16.gguf"))
        q3 = touch(self.d("repo", "model-Q3.gguf"))
        touch(m)
        p = plan({"model": m, "mmproj": mm}, [{"model": q3}])
        self.assertIsNone(p["mmproj"])
        self.assertTrue(os.path.exists(mm))

    def test_unreferenced_mmproj_is_never_planned(self):
        m = self.d("repo", "model.gguf")
        mm = touch(self.d("repo", "mmproj-stray.gguf"))
        touch(m)
        p = plan({"model": m})
        self.assertIsNone(p["mmproj"])
        self.assertNotIn(mm, p["files"])

    def test_model_file_shared_by_another_section_is_refused(self):
        m = touch(self.d("repo", "model.gguf"))
        with self.assertRaises(DeleteError):
            plan({"model": m}, [{"model": m}])

    def test_missing_model_file_rejected(self):
        with self.assertRaises(DeleteError):
            plan({"model": self.d("nope", "ghost.gguf")})

    def test_unknown_section_rejected(self):
        with self.assertRaises(DeleteError):
            plan(None)
        with self.assertRaises(DeleteError):
            plan({})

    def test_relative_path_rejected(self):
        with self.assertRaises(DeleteError):
            plan({"model": "relative/model.gguf"})

    def test_directory_as_model_rejected(self):
        d = os.path.join(self.tmp, "adir")
        os.makedirs(d)
        with self.assertRaises(DeleteError):
            plan({"model": d})

    def test_symlinked_model_directory_refused(self):
        real = os.path.join(self.tmp, "real")
        link = os.path.join(self.tmp, "link")
        m = touch(os.path.join(real, "model.gguf"))
        os.symlink(real, link)
        linked_model = os.path.join(link, "model.gguf")
        with self.assertRaises(DeleteError):
            plan({"model": linked_model})

    def test_symlinked_file_is_unlinked_not_followed(self):
        """os.remove on a symlink unlinks the link; the target survives."""
        target = touch(self.d("outside", "target.gguf"))
        m = self.d("repo", "model.gguf")
        os.makedirs(os.path.dirname(m), exist_ok=True)
        os.symlink(target, m)
        p = plan({"model": m})
        ok, err = execute(p)
        self.assertTrue(ok, err)
        self.assertFalse(os.path.exists(m))
        self.assertTrue(os.path.exists(target))

    def test_execute_rmdir_fails_loudly_when_not_empty(self):
        m = self.d("repo", "model.gguf")
        touch(m)
        p = plan({"model": m})
        # Race: something appears in the directory after planning.
        touch(self.d("repo", "surprise.txt"))
        ok, err = execute(p)
        self.assertFalse(ok)
        self.assertIn(self.d("repo"), err)
        self.assertTrue(os.path.exists(self.d("repo")))


class BackendDeleteTest(unittest.TestCase):
    """LlamaCppBackend.delete: loaded guard, section cleanup, router reload."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        import config
        self._saved_cfg = config.CONFIG
        config.CONFIG = os.path.join(self.tmp, "config.json")
        self.ini = os.path.join(self.tmp, "models.ini")
        cfg = config.load(); cfg["models_ini"] = self.ini; config.save(cfg)

        import backends
        self.be = backends.LlamaCppBackend(FakeRoutes())

    def tearDown(self):
        import config
        config.CONFIG = self._saved_cfg

    def _register(self, mid, model_path, **extra):
        import config
        config.set_keys(mid, dict({"model": model_path}, **extra))

    def test_delete_removes_file_section_and_reloads_router(self):
        import config
        m = touch(os.path.join(self.tmp, "repo", "model.gguf"))
        self._register("m", m)
        FakeRoutes.status = "offline"
        ok, err = self.be.delete("m")
        self.assertTrue(ok, err)
        self.assertFalse(os.path.exists(m))
        self.assertNotIn("m", config.read_sections(self.ini))
        self.assertIn("/models?reload=1", [c[0] for c in FakeRoutes.calls])

    def test_loaded_model_is_not_deleted(self):
        m = touch(os.path.join(self.tmp, "repo", "model.gguf"))
        self._register("m", m)
        FakeRoutes.status = "loaded"
        ok, err = self.be.delete("m")
        self.assertFalse(ok)
        self.assertIn("unload", err)
        self.assertTrue(os.path.exists(m))

    def test_unknown_model_raises_not_found(self):
        with self.assertRaises(NotFound):
            self.be.delete("ghost")

    def test_delete_plan_reports_paths_and_size(self):
        m = touch(os.path.join(self.tmp, "repo", "model.gguf"), size=42)
        self._register("m", m)
        p = self.be.delete_plan("m")
        self.assertEqual(p["files"], [m])
        self.assertEqual(p["size_bytes"], 42)
        self.assertTrue(p["delete_directory"])


class FakeRoutes:
    """Minimal routes-deps bundle for LlamaCppBackend."""
    status = "offline"
    calls = []

    def __init__(self):
        FakeRoutes.calls = []

    def cfg(self):
        return {"router_port": 8080}

    def model_state(self):
        return {"models": [{"id": "m", "status": self.status}], "global": {}}

    def router(self, path, method="GET", body=None, timeout=30):
        FakeRoutes.calls.append((path, method))
        return 200, {}

    def schema(self):
        return {"groups": [], "count": 0}

    def _prepare_model_for_load(self, mid):
        pass


class VllmBackendDeleteTest(unittest.TestCase):
    def test_running_vllm_model_refuses_deletion(self):
        import backends
        deps = mock.Mock()
        deps.vllm_mgr.return_value.status.return_value = [
            {"model_id": "org/m", "state": "ready"}]
        be = backends.VllmBackend(deps)
        with mock.patch.object(backends.osplat, "IS_WIN", True):
            ok, err = be.delete("org/m")
        self.assertFalse(ok)
        self.assertIn("stop", err)
        deps.vllm_dl.return_value.delete.assert_not_called()

    def test_idle_vllm_delete_still_goes_through_the_shared_flow(self):
        import backends
        deps = mock.Mock()
        deps.vllm_mgr.return_value.status.return_value = []
        deps.vllm_dl.return_value.delete.return_value = (True, "")
        be = backends.VllmBackend(deps)
        with mock.patch.object(backends.osplat, "IS_WIN", True), \
             mock.patch.object(backends.vllm_registry, "remove") as rm:
            ok, err = be.delete("org/m")
        self.assertTrue(ok)
        rm.assert_called_once_with("org/m")

    def test_vllm_delete_plan_reports_wsl_path(self):
        import backends
        deps = mock.Mock()
        be = backends.VllmBackend(deps)
        with mock.patch.object(backends.vllm_registry, "load",
                              return_value={"org/m": {"wsl_path": "/hf/models--org--m",
                                                     "size_bytes": 123}}):
            p = be.delete_plan("org/m")
        self.assertEqual(p["files"], ["/hf/models--org--m"])
        self.assertEqual(p["size_bytes"], 123)
        with mock.patch.object(backends.vllm_registry, "load", return_value={}):
            with self.assertRaises(NotFound):
                be.delete_plan("org/gone")


class ModelDeleteFrontendTest(unittest.TestCase):
    """The delete confirmation flow in web/js/models.js runs under node."""

    def test_delete_flow(self):
        import shutil, subprocess
        if not shutil.which("node"):
            self.skipTest("Node is required")
        script = os.path.join(os.path.dirname(__file__),
                             "models_delete_frontend_checks.mjs")
        result = subprocess.run(["node", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class RouteDeleteTest(unittest.TestCase):
    """POST /api/models/delete and GET /api/model/delete_plan through the
    engine-agnostic dispatch."""

    def setUp(self):
        import routes
        self.routes = routes

    def test_plan_route_returns_the_backend_plan(self):
        fake = mock.Mock()
        fake.delete_plan.return_value = {"files": ["/m.gguf"], "directory": "/",
                                        "delete_directory": False,
                                        "size_bytes": 7, "mmproj": None}
        with mock.patch.object(self.routes.REGISTRY, "for_model", return_value=fake):
            status, out = self.routes.get_model_delete_plan(
                self.mkreq(q={"model": "m", "backend": "llamacpp"}))
        self.assertEqual(status, 200)
        self.assertEqual(out["files"], ["/m.gguf"])
        self.assertEqual(out["size_bytes"], 7)

    def test_plan_route_404s_for_unknown_model(self):
        fake = mock.Mock()
        fake.delete_plan.side_effect = NotFound("unknown model: ghost")
        with mock.patch.object(self.routes.REGISTRY, "for_model", return_value=fake):
            with self.assertRaises(self.routes.ApiError) as cm:
                self.routes.get_model_delete_plan(self.mkreq(q={"model": "ghost"}))
        self.assertEqual(cm.exception.status, 404)

    def test_delete_route_prunes_binding_on_success(self):
        fake = mock.Mock()
        fake.delete.return_value = (True, "")
        fake.name = "llamacpp"
        with mock.patch.object(self.routes.REGISTRY, "for_model", return_value=fake), \
             mock.patch.object(self.routes.config, "prune_binding") as prune:
            status, out = self.routes.post_model_delete(
                self.mkreq(body={"model": "m", "backend": "llamacpp"}))
        self.assertEqual(status, 200)
        prune.assert_called_once_with("m")

    def test_delete_route_busy_model_is_409(self):
        fake = mock.Mock()
        fake.delete.return_value = (False, "model is loaded or loading - unload it first")
        fake.name = "llamacpp"
        with mock.patch.object(self.routes.REGISTRY, "for_model", return_value=fake), \
             mock.patch.object(self.routes.config, "prune_binding") as prune:
            status, out = self.routes.post_model_delete(
                self.mkreq(body={"model": "m"}))
        self.assertEqual(status, 409)
        prune.assert_not_called()

    def test_delete_route_unknown_model_is_404(self):
        fake = mock.Mock()
        fake.delete.side_effect = NotFound("unknown model: ghost")
        with mock.patch.object(self.routes.REGISTRY, "for_model", return_value=fake):
            with self.assertRaises(self.routes.ApiError) as cm:
                self.routes.post_model_delete(self.mkreq(body={"model": "ghost"}))
        self.assertEqual(cm.exception.status, 404)

    @staticmethod
    def mkreq(q=None, body=None):
        req = mock.Mock()
        req.body = body or {}
        req.q = lambda k, d="": (q or {}).get(k, d)
        req.path = "/api/models/delete"
        return req


if __name__ == "__main__":
    unittest.main()

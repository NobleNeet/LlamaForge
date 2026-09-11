"""The Build card and rebuild execution must agree after backend changes."""
import conftest_paths  # noqa: F401
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import routes
from builder import BuildManager


class BuildBackendSelectionTest(unittest.TestCase):
    def test_preview_and_execution_switch_backends_and_keep_custom_flags(self):
        for target in ("llamacpp", "ikllama"):
            for selected in ("hip", "vulkan", "cuda", "cpu", "metal"):
                with self.subTest(target=target, selected=selected), tempfile.TemporaryDirectory() as tmp:
                    prefix = "ik_llama_" if target == "ikllama" else ""
                    previous = "vulkan" if selected != "vulkan" else "hip"
                    config = {
                        "llama_src": tmp, "build_dir": tmp + "/build",
                        "ik_llama_src": tmp, "ik_llama_build_dir": tmp + "/ik-build",
                        "llama_backend": selected,
                        prefix + "cmake_backend": previous,
                        prefix + "cmake_flags": {
                            "GGML_" + previous.upper(): "ON",
                            "GGML_NATIVE": "OFF", "LLAMA_CURL": "OFF",
                            "GPU_TARGETS": "old-target",
                            "CMAKE_CUDA_ARCHITECTURES": "old-arch",
                        },
                    }
                    rec = {"selected_backend": selected, "cmake_flags": {"GGML_NATIVE": "ON"}}
                    if selected == "hip":
                        rec["cmake_flags"]["GPU_TARGETS"] = "gfx1151"
                    builder = mock.Mock(state={"running": False})
                    builder.start.return_value = True
                    with ExitStack() as stack:
                        for obj, name, value in (
                            (routes, "cfg", config),
                            (routes, "_builder_for", builder),
                            (routes.hardware, "recommend", rec),
                            (routes.prereqs, "status", {}),
                            (routes, "_validate_build_backend", ""),
                            (routes, "_hip_env", {"HIP_PATH": "/test/rocm"}),
                            (routes.router_ctl, "is_running", False),
                            (routes, "_unload_all_models", []),
                        ):
                            stack.enter_context(mock.patch.object(obj, name, return_value=value))
                        update = stack.enter_context(mock.patch.object(routes.config, "update"))
                        _, info = routes.get_build_info(routes.Req(qs={"target": target}))
                        _, result = routes.post_build_start(routes.Req(body={"target": target, "pull": False}))
                    self.assertTrue(result["started"])
                    flags = builder.start.call_args.args[2]
                    self.assertEqual(info["saved_flags"], flags)
                    for backend in ("cuda", "hip", "vulkan", "metal"):
                        self.assertEqual(flags["GGML_" + backend.upper()], "ON" if backend == selected else "OFF")
                    self.assertEqual(flags["GGML_NATIVE"], "OFF")
                    self.assertEqual(flags["LLAMA_CURL"], "OFF")
                    self.assertNotIn("CMAKE_CUDA_ARCHITECTURES", flags)
                    self.assertEqual(flags.get("GPU_TARGETS"), "gfx1151" if selected == "hip" else None)
                    self.assertEqual(update.call_args.args[0][prefix + "cmake_backend"], selected)
                    self.assertEqual(builder.start.call_args.kwargs["env"],
                                     {"HIP_PATH": "/test/rocm"} if selected == "hip" else None)

    def test_same_backend_keeps_custom_architecture_and_repairs_stale_toggle(self):
        config = {"cmake_backend": "hip", "cmake_flags": {
            "GPU_TARGETS": "gfx1100", "GGML_VULKAN": "ON"}}
        flags = routes._resolved_build_flags(config, "llamacpp", {
            "selected_backend": "hip", "cmake_flags": {"GPU_TARGETS": "gfx1151"}})
        self.assertEqual(flags["GPU_TARGETS"], "gfx1100")
        self.assertEqual(flags["GGML_HIP"], "ON")
        self.assertEqual(flags["GGML_VULKAN"], "OFF")

    def test_explicit_api_flags_remain_supported(self):
        builder = mock.Mock(state={"running": False})
        flags = {"GGML_CUDA": "ON", "GGML_VULKAN": "ON", "GGML_NATIVE": "OFF"}
        with mock.patch.object(routes, "cfg", return_value={"ik_llama_src": "/src", "ik_llama_build_dir": "/build"}), \
             mock.patch.object(routes, "_builder_for", return_value=builder), \
             mock.patch.object(routes.hardware, "recommend", return_value={"selected_backend": "cpu", "cmake_flags": {}}), \
             mock.patch.object(routes.prereqs, "status", return_value={}), \
             mock.patch.object(routes.config, "update"), \
             mock.patch.object(BuildManager, "validate_paths", return_value=""):
            routes.post_build_start(routes.Req(body={"target": "ikllama", "flags": flags}))
        self.assertEqual(builder.start.call_args.args[2], flags)

    @unittest.skipUnless(shutil.which("cmake"), "CMake required for cache regression")
    def test_reused_cmake_cache_disables_previous_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, build = Path(tmp) / "src", Path(tmp) / "build"
            src.mkdir()
            (src / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.16)\nproject(backend_test NONE)\n' +
                '\n'.join(f'option(GGML_{be} "backend" OFF)' for be in ("CUDA", "HIP", "VULKAN", "METAL")))
            builder = BuildManager(str(Path(tmp) / "logs"))
            config = {}
            for selected in ("vulkan", "hip", "cpu", "cuda", "metal"):
                flags = routes._resolved_build_flags(config, "llamacpp", {
                    "selected_backend": selected, "cmake_flags": {}})
                builder.run_build(str(src), str(build), flags, pull=False)
                self.assertEqual(builder.state["phase"], "done", builder.tail())
                cache = (build / "CMakeCache.txt").read_text()
                for backend in ("cuda", "hip", "vulkan", "metal"):
                    value = "ON" if backend == selected else "OFF"
                    self.assertRegex(cache, rf'GGML_{backend.upper()}:(?:BOOL|UNINITIALIZED)={value}\b')
                config = {"cmake_backend": selected, "cmake_flags": flags}

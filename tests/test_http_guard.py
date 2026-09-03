"""The trust boundary around the dashboard.

The panel binds 127.0.0.1, which keeps it off the network but leaves it
reachable by every page the user browses - and its routes rebuild llama.cpp,
install packages and rewrite configuration. These tests drive a real
ThreadingHTTPServer the way a hostile page would.
"""
import conftest_paths  # noqa: F401
import json, threading, unittest, urllib.error, urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import routes, server

REAL_CFG = routes.cfg   # the un-patched loader, for the isolation guard below


class GuardUnitTest(unittest.TestCase):
    def test_host_accepts_loopback_names(self):
        for h in ("127.0.0.1:8090", "localhost:8090", "127.0.0.1", "[::1]:8090"):
            self.assertTrue(server._host_ok(h, 8090), h)

    def test_host_rejects_foreign_names_and_ports(self):
        for h in ("evil.com:8090", "192.168.1.5:8090", "127.0.0.1:9999", ""):
            self.assertFalse(server._host_ok(h, 8090), h)

    def test_host_rejects_rebinding_hostname(self):
        """DNS rebinding: attacker's name resolved to 127.0.0.1."""
        self.assertFalse(server._host_ok("attacker.test:8090", 8090))

    def test_origin_absent_is_allowed(self):
        # curl, and agent clients hitting /v1/messages, send no Origin
        self.assertTrue(server._origin_ok("", 8090))

    def test_origin_same_service_allowed(self):
        for o in ("http://127.0.0.1:8090", "http://localhost:8090"):
            self.assertTrue(server._origin_ok(o, 8090), o)

    def test_origin_foreign_rejected(self):
        for o in ("http://evil.com", "https://evil.com", "http://127.0.0.1:9999",
                  "null", "file://"):
            self.assertFalse(server._origin_ok(o, 8090), o)

    def test_allowed_hosts_include_lan_ip_when_panel_is_lan_bound(self):
        allowed = server._allowed_hosts("0.0.0.0", "192.168.1.50")
        self.assertIn("127.0.0.1", allowed)
        self.assertIn("192.168.1.50", allowed)

    def test_host_accepts_lan_ip_when_explicitly_allowed(self):
        allowed = server._allowed_hosts("0.0.0.0", "192.168.1.50")
        self.assertTrue(server._host_ok("192.168.1.50:8090", 8090, allowed))

    def test_origin_accepts_lan_ip_when_explicitly_allowed(self):
        allowed = server._allowed_hosts("0.0.0.0", "192.168.1.50")
        self.assertTrue(server._origin_ok("http://192.168.1.50:8090", 8090, allowed))


class LiveServerTest(unittest.TestCase):
    """Exercises dispatch end to end over a real socket."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # the guard reads panel_port from config; point it at the test socket
        self.cfg_patch = mock.patch.object(
            routes, "cfg", return_value={"panel_port": self.port,
                                         "panel_host": "127.0.0.1",
                                         "anthropic_shim_enabled": False})
        self.cfg_patch.start()
        self.addCleanup(self.cfg_patch.stop)
        self.seen = []

        def probe(req):
            self.seen.append(req)
            return 200, {"ok": True, "body": req.body, "qs": req.qs}

        self.routes_patch = mock.patch.dict(
            routes.GET_ROUTES, {"/api/_probe": probe}, clear=False)
        self.post_patch = mock.patch.dict(
            routes.POST_ROUTES, {"/api/_probe": probe}, clear=False)
        self.routes_patch.start(); self.post_patch.start()
        self.addCleanup(self.routes_patch.stop)
        self.addCleanup(self.post_patch.stop)

    def _req(self, path, method="GET", headers=None, data=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        h = {"Host": f"127.0.0.1:{self.port}"}
        h.update(headers or {})
        body = json.dumps(data).encode() if data is not None else None
        if body is not None:
            h.setdefault("Content-Type", "application/json")
        r = urllib.request.Request(url, data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")

    # ---------------------------------------------------------------- happy
    def test_same_origin_get_is_dispatched(self):
        status, body = self._req(
            "/api/_probe?x=1", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertEqual(body["qs"], {"x": "1"})

    def test_no_origin_get_is_dispatched(self):
        status, body = self._req("/api/_probe")
        self.assertEqual(status, 200)

    def test_json_post_is_dispatched(self):
        status, body = self._req("/api/_probe", "POST", data={"hello": "world"})
        self.assertEqual(status, 200)
        self.assertEqual(body["body"], {"hello": "world"})

    def test_unknown_path_404s(self):
        status, _ = self._req("/api/nope")
        self.assertEqual(status, 404)

    # ------------------------------------------------------------- rejected
    def test_cross_origin_get_is_refused(self):
        status, body = self._req("/api/_probe",
                                 headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.seen, [], "handler ran despite a foreign Origin")

    def test_cross_origin_post_is_refused(self):
        status, _ = self._req("/api/_probe", "POST", data={"a": 1},
                              headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.seen, [])

    def test_rebound_host_is_refused(self):
        status, _ = self._req("/api/_probe", headers={"Host": "attacker.test"})
        self.assertEqual(status, 403)
        self.assertEqual(self.seen, [])

    def test_lan_host_is_accepted_when_panel_is_lan_bound(self):
        self.cfg_patch.stop()
        self.cfg_patch = mock.patch.object(
            routes, "cfg", return_value={"panel_port": self.port,
                                         "panel_host": "0.0.0.0",
                                         "anthropic_shim_enabled": False})
        self.cfg_patch.start()
        # setUp only knows about the patch it made, so this replacement has to
        # register its own teardown: it used to leak, and every later test in
        # the same process then read a stub config with three keys in it.
        self.addCleanup(self.cfg_patch.stop)
        with mock.patch.object(routes.router_ctl, "lan_ip", return_value="192.168.1.50"):
            status, body = self._req("/api/_probe",
                                     headers={"Host": f"192.168.1.50:{self.port}",
                                              "Origin": f"http://192.168.1.50:{self.port}"})
        self.assertEqual(status, 200)
        self.assertEqual(body["ok"], True)

    def test_form_content_type_post_is_refused(self):
        """The CSRF vector: a cross-site <form> can only send these types, and
        the body was previously json.loads()ed regardless of Content-Type."""
        for ctype in ("text/plain", "application/x-www-form-urlencoded",
                      "multipart/form-data"):
            status, _ = self._req("/api/_probe", "POST", data={"a": 1},
                                  headers={"Content-Type": ctype})
            self.assertEqual(status, 415, ctype)
        self.assertEqual(self.seen, [])

    def test_malformed_json_is_a_400_not_a_crash(self):
        url = f"http://127.0.0.1:{self.port}/api/_probe"
        r = urllib.request.Request(
            url, data=b"{not json", method="POST",
            headers={"Host": f"127.0.0.1:{self.port}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        self.assertEqual(status, 400)

    def test_non_object_json_body_is_refused(self):
        status, _ = self._req("/api/_probe", "POST", data=[1, 2, 3])
        self.assertEqual(status, 400)

    def test_handler_exception_becomes_500_not_a_dead_server(self):
        def boom(req):
            raise RuntimeError("kaboom")

        with mock.patch.dict(routes.GET_ROUTES, {"/api/_boom": boom}):
            status, body = self._req("/api/_boom")
        self.assertEqual(status, 500)
        self.assertIn("kaboom", body["error"])
        self.assertEqual(self._req("/api/_probe")[0], 200)   # still serving

    def test_api_error_carries_its_status(self):
        def refuse(req):
            raise routes.ApiError(418, "nope")

        with mock.patch.dict(routes.GET_ROUTES, {"/api/_refuse": refuse}):
            status, body = self._req("/api/_refuse")
        self.assertEqual((status, body["error"]), (418, "nope"))


class PatchIsolationTest(unittest.TestCase):
    """Runs last in this module, on purpose.

    `LiveServerTest` swaps `routes.cfg` for a three-key stub. When one of its
    tests replaced that stub without registering a teardown, the stub survived
    into every test that ran afterwards in the same process - failures reported
    hundreds of tests away, in code that never touched a mock.
    """

    def test_routes_cfg_is_the_real_loader_again(self):
        self.assertIs(routes.cfg, REAL_CFG)


if __name__ == "__main__":
    unittest.main()

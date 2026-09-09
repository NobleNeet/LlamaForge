import conftest_paths  # noqa: F401
import io
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import chat_diagnostics as chat
import routes
import server
import stream_relay


class DiagnosticsTest(unittest.TestCase):
    def test_existing_markdown_renderer_escapes_html_and_omits_images(self):
        html = chat.render_message('**bold** <script>alert(1)</script>\n\n![image](https://example.com/a)')
        self.assertIn('<strong>bold</strong>', html)
        self.assertNotIn('<script>', html)
        self.assertNotIn('<img', html)

    def setUp(self):
        self.args = ['/custom/server', '--model', '/models/a b.gguf', '--alias', 'm',
                     '--ctx-size', '8192', '--port', '60687', '--api-key', 'test-value']
        self.row = {'id':'m', 'backend':'llamacpp', 'status':'loaded', 'settings':{'ctx-size':'99999'}}
        self.runtime = {'id':'m', 'status':{'value':'loaded', 'args':self.args}}
        self.deps = SimpleNamespace(
            cfg=lambda: {'active_engine':'ikllama', 'router_port':8080, 'panel_port':8090},
            REGISTRY=SimpleNamespace(state=lambda: {'models':[self.row]}),
            router=mock.Mock(side_effect=lambda path, **kwargs: (200, {'data':[self.runtime]}) if path == '/models' else
                             (200, {'default_generation_settings':{'n_ctx':8192}})),
            router_ctl=SimpleNamespace(_pid_on_port=mock.Mock(return_value=1234)))

    def test_actual_runtime_not_saved_settings(self):
        result = chat.snapshot(self.deps, 'm')
        self.assertEqual(result['engine'], 'ikllama')
        self.assertEqual(result['context_capacity'], 8192)
        self.assertEqual(result['port'], 60687)
        self.assertEqual(result['pid'], 1234)
        self.assertEqual(result['model_path'], '/models/a b.gguf')
        self.assertNotIn('99999', repr(result))
        self.assertNotIn('test-value', repr(result))
        self.assertEqual(self.args[-1], 'test-value', 'must not mutate router data')

    def test_unloaded_args_are_not_presented_as_live(self):
        self.runtime['status']['value'] = 'unloaded'
        result = chat.snapshot(self.deps, 'm')
        self.assertEqual(result['argv'], [])
        self.assertIsNone(result['context_capacity'])
        self.deps.router_ctl._pid_on_port.assert_not_called()

    def test_missing_model_offline_and_unsupported_backend(self):
        self.assertFalse(chat.snapshot(self.deps, 'absent')['available'])
        self.deps.router.return_value = (599, {})
        self.deps.router.side_effect = None
        result = chat.snapshot(self.deps, 'm')
        self.assertEqual(result['status'], 'offline')
        self.assertFalse(result['router_up'])
        self.row['backend'] = 'vllm'
        self.assertFalse(chat.snapshot(self.deps, 'm')['available'])

    def test_missing_props_does_not_fall_back_to_configured_context(self):
        self.deps.router.side_effect = [(200, {'data':[self.runtime]}), (404, {})]
        self.assertIsNone(chat.snapshot(self.deps, 'm')['context_capacity'])

    def test_redaction_handles_equals_and_preserves_generation_flags(self):
        args = ['--hf-token=example', '--max-tokens', '12', '--api-key', 'value']
        self.assertEqual(chat.runtime_args(args), ['--hf-token=[redacted]', '--max-tokens', '12', '--api-key', '[redacted]'])
        self.assertEqual(chat.runtime_args('not-an-argv'), [])


class RelayTest(unittest.TestCase):
    def test_write_failure_closes_upstream(self):
        response = io.BytesIO(b'data: first\n\ndata: second\n\n')
        with self.assertRaises(BrokenPipeError):
            stream_relay.relay(response, None, mock.Mock(side_effect=BrokenPipeError))
        self.assertTrue(response.closed)

    def test_disconnect_interrupts_blocked_prefill(self):
        client, peer = socket.socketpair()
        upstream, producer = socket.socketpair()
        reader = upstream.makefile('rb')
        response = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=upstream)))
        class Response:
            fp = response.fp
            def __iter__(self): return iter(reader)
            def close(self): reader.close()
        worker = threading.Thread(target=stream_relay.relay, args=(Response(), client, lambda line: None), daemon=True)
        try:
            worker.start()
            peer.close()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive(), 'disconnect must cancel even before a token arrives')
            self.assertTrue(reader.closed)
        finally:
            for sock in (client, peer, upstream, producer): sock.close()

    def test_diagnostic_metadata_is_opt_in_and_stream_is_preserved(self):
        body = {'model':'m', 'messages':[{'role':'system','content':'wiki + system'}], 'stream':True}
        for enabled in (False, True):
            h = object.__new__(server.H)
            h.wfile = io.BytesIO()
            h.headers = {'X-LlamaForge-Diagnostics':'1'} if enabled else {}
            h.connection = None
            h._begin_stream = mock.Mock()
            response = io.BytesIO(b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
            with mock.patch.object(routes, '_router_openai', return_value=(200, response)), \
                 mock.patch.object(server, '_track_api_model_begin'), \
                 mock.patch.object(server, '_track_api_model_end') as end:
                h._openai_proxy_stream(body)
            self.assertIn(b'data: [DONE]', h.wfile.getvalue())
            self.assertEqual(b'llamaforge.diagnostics' in h.wfile.getvalue(), enabled)
            self.assertTrue(response.closed)
            end.assert_called_once_with('m')


if __name__ == '__main__': unittest.main()

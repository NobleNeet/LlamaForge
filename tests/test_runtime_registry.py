"""Fresh runtime schema, isolated registries, and pre-stop failure guarantees."""
import conftest_paths  # noqa: F401
from contextlib import ExitStack
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import argspec
import config
import routes
import runtime_registry
import server
import stats

REAL_HELP_TEXT = argspec._help_text

HELP = '--ctx-size N  context size\n--gpu-layers, --n-gpu-layers N  offload layers\n'
REGISTRY = ('version = 1\n[*]\nctx-size = 4096\nremoved-global = true\n[m]\n'
            'model = /m.gguf\nmmproj = /projector.gguf\nspec-draft-model = /draft.gguf\n'
            'embeddings = true\nload-on-startup = false\ngpu-layers = 5\n'
            'removed-key = true\nfork-only = true\n')


class RuntimeRegistryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(mock.patch.object(config, 'CONFIG', str(self.root / 'config.json')))
        self.binaries = {tid: self.root / tid for tid in ('llamacpp', 'ikllama', 'custom-one', 'custom-two')}
        for binary in self.binaries.values():
            binary.write_text('test fixture')
        targets = {tid: {'name': tid, 'source': str(self.root), 'build': str(self.root),
                         'server_binary': str(self.binaries[tid])} for tid in ('custom-one', 'custom-two')}
        config.update({'models_ini': str(self.root / 'models.ini'),
                       'server_bin': str(self.binaries['llamacpp']),
                       'llama_builtin_server_bin': str(self.binaries['llamacpp']),
                       'ik_llama_server_bin': str(self.binaries['ikllama']),
                       'custom_build_targets': targets})
        self.paths = {}
        for tid in self.binaries:
            c = self.destination(tid)
            path = Path(config.ini_path(c))
            path.write_text(REGISTRY)
            self.paths[tid] = path
        self.help = self.stack.enter_context(mock.patch.object(argspec, '_help_text', side_effect=self.help_for))
        self.stack.enter_context(mock.patch.object(routes.router_ctl, 'supports_router_mode', return_value=True))
        self.stack.enter_context(mock.patch.object(routes.os, 'access', return_value=True))
        self.stop = self.stack.enter_context(mock.patch.object(routes.router_ctl, 'stop', return_value=True))
        self.restart = self.stack.enter_context(mock.patch.object(routes.router_ctl, 'restart', return_value=(True, '')))
        self.stack.enter_context(mock.patch.object(routes.router_ctl, 'is_running', return_value=True))
        self.stack.enter_context(mock.patch.object(routes, '_wait_router_ready', return_value=True))
        self.stack.enter_context(mock.patch.object(routes, 'router', return_value=(200, {'data': []})))
        self.stack.enter_context(mock.patch.object(routes, 'maybe_run_idle_maintenance'))
        for attr in ('_SCHEMA', '_SCHEMA_KEY', '_IK_SCHEMA', '_IK_SCHEMA_KEY'):
            self.stack.enter_context(mock.patch.object(routes, attr, None))

    def destination(self, tid):
        return dict(config.load(), active_engine='ikllama' if tid == 'ikllama' else 'llamacpp',
                    active_llamacpp_build_target=tid,
                    server_bin=str(self.binaries[tid]) if tid != 'ikllama' else str(self.binaries['llamacpp']))

    def help_for(self, binary):
        return HELP + ('--fork-only  custom option\n' if binary == str(self.binaries['custom-one']) else ''), ''

    def assert_clean(self, tid):
        sections = config.read_sections(str(self.paths[tid]))
        self.assertNotIn('removed-global', sections['*'])
        model = sections['m']
        self.assertNotIn('removed-key', model)
        self.assertNotIn('gpu-layers', model)
        self.assertEqual(model['n-gpu-layers'], '5')
        for key in config.EXTRA_MODELS_INI_KEYS:
            self.assertIn(key, model)
        self.assertEqual('fork-only' in model, tid == 'custom-one')

    def test_all_explicit_destinations_sanitize_before_stop_and_share_ui_schema(self):
        for tid in self.binaries:
            with self.subTest(target=tid):
                # Built-in return takes the activation path from a custom runtime.
                config.update(self.destination('custom-two' if tid == 'llamacpp' else 'llamacpp'))
                for path in self.paths.values():
                    path.write_text(REGISTRY)
                before = {name: path.read_bytes() for name, path in self.paths.items()}
                self.stop.side_effect = lambda port: (self.assert_clean(tid), True)[1]
                self.restart.side_effect = lambda binary, path, *a, **kw: (self.assert_clean(tid), (True, ''))[1]
                if tid.startswith('custom'):
                    status, result = routes.post_build_activate(routes.Req(body={'target': tid}))
                else:
                    status, result = routes.post_engine_switch(routes.Req(body={'engine': tid}))
                self.assertEqual(status, 200)
                self.assertTrue(result['ok'])
                self.assertEqual(self.restart.call_args.args[1], str(self.paths[tid]))
                self.assert_clean(tid)
                for other in self.paths:
                    if other != tid:
                        self.assertEqual(self.paths[other].read_bytes(), before[other])
                help_calls = self.help.call_count
                schema = routes.schema()
                self.assertEqual(self.help.call_count, help_calls, 'UI must reuse the sanitized metadata')
                keys = argspec.schema_key_aliases(schema)['keys']
                self.assertEqual('fork-only' in keys, tid == 'custom-one')

    def test_failed_or_unparseable_schema_refuses_switch_without_any_mutation(self):
        for tid in ('llamacpp', 'ikllama', 'custom-one'):
            for response in [('', 'probe timeout'), ('unrecognized output', '')]:
                with self.subTest(target=tid, response=response):
                    config.update(self.destination('custom-two' if tid == 'llamacpp' else 'llamacpp'))
                    before = config.load()
                    files = {k: p.read_bytes() for k, p in self.paths.items()}
                    self.stop.reset_mock()
                    self.restart.reset_mock()
                    self.help.side_effect = None
                    self.help.return_value = response
                    if tid in ('custom-one', 'llamacpp'):
                        with self.assertRaises(routes.ApiError) as caught:
                            if tid == 'llamacpp':
                                routes.post_engine_switch(routes.Req(body={'engine': tid}))
                            else:
                                routes.post_build_activate(routes.Req(body={'target': tid}))
                        self.assertEqual(caught.exception.status, 400)
                    else:
                        status, result = routes.post_engine_switch(routes.Req(body={'engine': tid}))
                        self.assertEqual(status, 400)
                        self.assertFalse(result['ok'])
                    self.assertEqual(config.load(), before)
                    for name, path in self.paths.items():
                        self.assertEqual(path.read_bytes(), files[name])
                    self.stop.assert_not_called()
                    self.restart.assert_not_called()

    def test_discovery_failure_does_not_create_missing_destination(self):
        missing = self.root / 'nested' / 'missing.ini'
        c = self.destination('custom-one')
        c['custom_build_targets']['custom-one']['models_ini'] = str(missing)
        self.help.side_effect = RuntimeError('probe failed')
        with self.assertRaises(runtime_registry.RegistryPreparationError):
            runtime_registry.prepare(c)
        self.assertFalse(missing.exists())
        self.assertFalse(missing.parent.exists())

    def test_successful_first_activation_sanitizes_seed_without_editing_baseline(self):
        c = self.destination('custom-one')
        missing = self.root / 'nested' / 'new.ini'
        c['custom_build_targets']['custom-one']['models_ini'] = str(missing)
        before = self.paths['llamacpp'].read_bytes()
        path, schema, _result = runtime_registry.prepare(c)
        self.assertEqual(path, str(missing))
        sections = config.read_sections(path)
        self.assertNotIn('removed-key', sections['m'])
        self.assertIn('fork-only', sections['m'])
        self.assertEqual(self.paths['llamacpp'].read_bytes(), before)
        content = missing.read_bytes()
        _path, _schema, result = runtime_registry.prepare(c)
        self.assertEqual(result['changed'], [])
        self.assertEqual(missing.read_bytes(), content)

    def test_update_refreshes_same_binary_even_with_unchanged_mtime(self):
        config.update(self.destination('custom-one'))
        routes._prepare_runtime(config.load())
        self.help.side_effect = None
        self.help.return_value = (HELP, '')  # upstream removed the fork option
        routes._PREBUILD_RUNNING, routes._PREBUILD_LOADED = True, []
        self.addCleanup(lambda: (setattr(routes, '_PREBUILD_RUNNING', False), setattr(routes, '_PREBUILD_LOADED', [])))
        routes._bring_router_back(source='automatic update')
        self.assertNotIn('fork-only', config.read_sections()['m'])
        self.restart.assert_called_once()
        self.assertEqual(self.restart.call_args.args[1], str(self.paths['custom-one']))
        self.assertNotIn('fork-only', argspec.schema_key_aliases(routes.schema())['keys'])

    def test_update_failure_preserves_registry_and_does_not_restart(self):
        config.update(self.destination('custom-one'))
        before = self.paths['custom-one'].read_bytes()
        self.help.side_effect = None
        self.help.return_value = ('', 'unavailable schema')
        routes._PREBUILD_RUNNING, routes._PREBUILD_LOADED = True, []
        self.addCleanup(lambda: (setattr(routes, '_PREBUILD_RUNNING', False), setattr(routes, '_PREBUILD_LOADED', [])))
        routes._bring_router_back()
        self.assertEqual(self.paths['custom-one'].read_bytes(), before)
        self.restart.assert_not_called()

    def test_launcher_startup_uses_correct_runtime_and_sanitized_registry(self):
        for tid in self.binaries:
            with self.subTest(target=tid):
                config.update(self.destination(tid))
                with mock.patch.object(routes.router_ctl, 'is_running', return_value=False), \
                     mock.patch.object(routes.router_ctl, 'start', return_value=(True, '')) as start:
                    runtime_registry.start_configured_router()
                self.assert_clean(tid)
                self.assertEqual(start.call_args.args[:2], (str(self.binaries[tid]), str(self.paths[tid])))

    def test_launcher_failure_warns_and_leaves_registry_unchanged(self):
        before = self.paths['llamacpp'].read_bytes()
        self.help.side_effect = None
        self.help.return_value = ('', 'timeout')
        with mock.patch.object(routes.router_ctl, 'is_running', return_value=False), \
             mock.patch.object(routes.router_ctl, 'start') as start, \
             mock.patch('builtins.print') as warning:
            runtime_registry.start_configured_router()
        start.assert_not_called()
        self.assertEqual(self.paths['llamacpp'].read_bytes(), before)
        self.assertIn('WARNING', warning.call_args.args[0])

    def test_panel_startup_failure_warns_and_does_not_apply_ctx_defaults(self):
        class HTTPD:
            def __init__(self, *args): pass
            def serve_forever(self): pass
        with mock.patch.object(config, 'migrate'), \
             mock.patch.object(routes, '_prepare_runtime', side_effect=runtime_registry.RegistryPreparationError('no schema')), \
             mock.patch.object(config, 'apply_ctx_defaults') as defaults, \
             mock.patch.object(server, 'ThreadingHTTPServer', HTTPD), \
             mock.patch.object(server.threading, 'Thread'), \
             mock.patch.object(stats.TRACKER, 'start'), \
             mock.patch('builtins.print') as warning:
            server.main()
        defaults.assert_not_called()
        self.assertTrue(any('WARNING' in str(call) for call in warning.call_args_list))

    def test_sanitizer_requires_schema_preserves_empty_global_and_canonical_value(self):
        path = self.paths['llamacpp']
        path.write_text('[*]\nremoved = true\n[m]\nmodel = /m.gguf\nn-gpu-layers = 3\ngpu-layers = 5\n')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            config.sanitize_models_ini(str(path), valid_keys=set())
        self.assertEqual(path.read_bytes(), before)
        runtime_registry.prepare(config.load())
        sections = config.read_sections(str(path))
        self.assertEqual(sections['*'], {})
        self.assertEqual(sections['m']['n-gpu-layers'], '3')

    def test_startup_defaults_cannot_reintroduce_keys_absent_from_schema(self):
        self.help.side_effect = None
        self.help.return_value = ('--threads N  threads\n', '')
        c = config.load()
        path, _schema, _result = runtime_registry.prepare(c, apply_ctx_defaults=True)
        self.assertNotIn('ctx-size', config.read_sections(path)['*'])
        content = Path(path).read_bytes()
        _path, _schema, result = runtime_registry.prepare(c, apply_ctx_defaults=True)
        self.assertEqual(result['changed'], [])
        self.assertEqual(Path(path).read_bytes(), content)

    def test_failed_activation_rollback_does_not_sanitize_source_registry(self):
        config.update(self.destination('llamacpp'))
        before = self.paths['llamacpp'].read_bytes()
        self.restart.side_effect = [(False, 'new router failed'), (True, '')]
        status, result = routes.post_build_activate(routes.Req(body={'target': 'custom-one'}))
        self.assertEqual(status, 500)
        self.assertTrue(result['rollback_ok'])
        self.assertEqual(self.paths['llamacpp'].read_bytes(), before)
        self.assertEqual(self.restart.call_args.args[1], str(self.paths['llamacpp']))

    def test_builtin_classic_switch_refreshes_its_own_schema(self):
        c = self.destination('ikllama')
        c['active_llamacpp_build_target'] = ''
        config.update(c)
        status, result = routes.post_engine_switch(routes.Req(body={'engine': 'llamacpp'}))
        self.assertEqual(status, 200)
        self.assertTrue(result['ok'])
        self.assert_clean('llamacpp')
        self.assertEqual(self.restart.call_args.args[1], str(self.paths['llamacpp']))

    def test_nonzero_help_exit_rejects_partial_schema(self):
        with mock.patch.object(argspec, '_help_text', REAL_HELP_TEXT), \
             mock.patch.object(argspec.subprocess, 'run', return_value=subprocess.CompletedProcess(
                ['server', '--help'], 1, stdout=HELP, stderr='crashed')):
            schema = argspec.build_schema('server')
        self.assertIn('exit code 1', schema['error'])


if __name__ == '__main__':
    unittest.main()

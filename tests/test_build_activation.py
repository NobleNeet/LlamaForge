"""Runtime identity, transactional activation and recovery with isolated config."""
import conftest_paths  # noqa: F401
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_activation
import config
import routes


class BuildActivationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(mock.patch.object(config, 'CONFIG', str(self.root / 'config.json')))
        self.old = self.root / 'old-server'
        self.new = self.root / 'new-server'
        self.old.write_text('fixture')
        self.new.write_text('fixture')
        self.target = dict(id='custom-one', name='My fork', source=str(self.root), build=str(self.root),
                           server_binary='{source}/new-server')
        config.update({'server_bin': str(self.old), 'custom_build_targets': {'custom-one': self.target}})
        self.stack.enter_context(mock.patch.object(routes.os, 'access', return_value=True))
        self.capable = self.stack.enter_context(mock.patch.object(routes.router_ctl, 'supports_router_mode', return_value=True))
        self.running = self.stack.enter_context(mock.patch.object(routes.router_ctl, 'is_running', return_value=True))
        self.stop = self.stack.enter_context(mock.patch.object(routes.router_ctl, 'stop', return_value=True))
        self.restart = self.stack.enter_context(mock.patch.object(routes.router_ctl, 'restart', return_value=(True, '')))
        self.ready = self.stack.enter_context(mock.patch.object(routes, '_wait_router_ready', return_value=True))
        self.reload = self.stack.enter_context(mock.patch.object(routes, '_reload_loaded_models'))
        self.models = self.stack.enter_context(mock.patch.object(routes, 'router', return_value=(200, {'data': [
            {'id': 'model-a', 'status': {'value': 'loaded'}},
            {'id': 'model-b', 'status': {'value': 'unloaded'}},
            {'id': 'default', 'status': {'value': 'loaded'}}]})))

    def activate(self, tid='custom-one'):
        return routes.post_build_activate(routes.Req(body={'target': tid}))

    def test_activation_and_builtin_restore(self):
        status, result = self.activate()
        self.assertEqual(status, 200)
        self.assertTrue(result['ok'])
        c = config.load()
        self.assertEqual(c['active_engine'], 'llamacpp')
        self.assertEqual(c['server_bin'], str(self.new))
        self.assertEqual(c['llama_builtin_server_bin'], str(self.old))
        self.assertEqual(c['active_llamacpp_build_target'], 'custom-one')
        self.assertEqual(c['custom_build_targets']['custom-one'], self.target)
        self.assertEqual(self.restart.call_args.args[0], str(self.new))
        self.reload.assert_called_with(['model-a'], source='/api/build/activate')
        _, listing = routes.get_build_targets(routes.Req())
        self.assertEqual(listing['active_build']['id'], 'custom-one')
        _, result = routes.post_engine_switch(routes.Req(body={'engine': 'llamacpp'}))
        self.assertTrue(result['ok'])
        self.assertEqual(config.load()['server_bin'], str(self.old))
        self.assertEqual(self.restart.call_args.args[0], str(self.old))
        self.assertEqual(result['active_build']['id'], 'llamacpp')

    def test_invalid_binary_never_stops_or_changes_config(self):
        initial = config.load()
        for patch in (mock.patch.object(routes.os.path, 'isfile', return_value=False),
                      mock.patch.object(routes.os, 'access', return_value=False),
                      mock.patch.object(routes.router_ctl, 'supports_router_mode', return_value=False)):
            with patch, self.assertRaises(routes.ApiError) as caught:
                self.activate()
            self.assertEqual(caught.exception.status, 400)
            self.assertEqual(config.load(), initial)
        self.stop.assert_not_called()
        self.restart.assert_not_called()

    def test_unknown_and_builtin_target_rejection(self):
        for tid in ('missing', 'llamacpp', 'ikllama', [], None):
            with self.subTest(tid=tid), self.assertRaises(routes.ApiError) as caught:
                self.activate(tid)
            self.assertEqual(caught.exception.status, 400)
        self.restart.assert_not_called()

    def test_build_placeholder(self):
        c = config.load()
        c['custom_build_targets']['custom-one']['server_binary'] = '{build}/new-server'
        config.save(c)
        self.activate()
        self.assertEqual(config.load()['server_bin'], str(self.new))

    def test_failed_restart_rolls_back_and_restores_models(self):
        self.restart.side_effect = [(False, 'new process failed'), (True, '')]
        status, result = self.activate()
        self.assertEqual(status, 500)
        self.assertFalse(result['ok'])
        self.assertTrue(result['rollback_ok'])
        self.assertEqual(config.load()['server_bin'], str(self.old))
        self.assertEqual(config.load()['active_llamacpp_build_target'], '')
        self.assertEqual([call.args[0] for call in self.restart.call_args_list], [str(self.new), str(self.old)])
        self.reload.assert_called_with(['model-a'], source='/api/build/activate/rollback')

    def test_ready_timeout_rolls_back(self):
        self.ready.side_effect = [False, True]
        _, result = self.activate()
        self.assertIn('ready', result['error'])
        self.assertTrue(result['rollback_ok'])
        self.assertEqual(config.load()['server_bin'], str(self.old))

    def test_recovery_failure_is_reported_with_old_config(self):
        self.restart.side_effect = [(False, 'new failed'), (False, 'old failed')]
        _, result = self.activate()
        self.assertFalse(result['rollback_ok'])
        self.assertEqual(result['rollback_error'], 'old failed')
        self.assertEqual(config.load()['server_bin'], str(self.old))
        self.reload.assert_not_called()

    def test_stop_exception_during_rollback_still_restores_config(self):
        self.restart.return_value = (False, 'new failed')
        self.stop.side_effect = [True, OSError('stop failed')]
        _, result = self.activate()
        self.assertFalse(result['rollback_ok'])
        self.assertEqual(config.load()['server_bin'], str(self.old))

    def test_stopped_router_is_started_and_failed_attempt_left_stopped(self):
        self.running.return_value = False
        self.restart.return_value = (False, 'failed')
        _, result = self.activate()
        self.assertTrue(result['rollback_ok'])
        self.restart.assert_called_once()
        self.reload.assert_not_called()
        self.assertEqual(config.load()['server_bin'], str(self.old))

    def test_unknown_model_state_and_stop_failure_leave_config_unchanged(self):
        initial = config.load()
        self.models.return_value = (503, {})
        with self.assertRaises(routes.ApiError):
            self.activate()
        self.stop.assert_not_called()
        self.models.return_value = (200, {'data': []})
        self.stop.return_value = False
        with self.assertRaises(routes.ApiError):
            self.activate()
        self.assertEqual(config.load(), initial)
        self.restart.assert_not_called()

    def test_second_custom_does_not_lose_builtin_and_ikllama_stays_a_runtime(self):
        self.activate()
        second = self.root / 'second-server'
        second.write_text('fixture')
        config.mutate(lambda c: c['custom_build_targets'].update({'custom-two': dict(self.target, server_binary=str(second))}))
        self.activate('custom-two')
        self.assertEqual(config.load()['llama_builtin_server_bin'], str(self.old))
        config.update({'ik_llama_server_bin': str(self.old)})
        _, result = routes.post_engine_switch(routes.Req(body={'engine': 'ikllama'}))
        self.assertTrue(result['ok'])
        self.assertEqual(config.load()['active_engine'], 'ikllama')
        self.assertEqual(build_activation.active_build(config.load())['id'], 'ikllama')
        self.activate()
        self.assertEqual(config.load()['active_engine'], 'llamacpp')
        self.assertEqual(config.load()['llama_builtin_server_bin'], str(self.old))

    def test_edit_remove_do_not_lie_about_active_path_or_prevent_builtin_restore(self):
        self.activate()
        config.mutate(lambda c: c['custom_build_targets']['custom-one'].update(server_binary='{build}/changed'))
        self.assertEqual(build_activation.active_build(config.load())['id'], '')
        routes.post_build_target_remove(routes.Req(body={'id': 'custom-one'}))
        self.assertEqual(config.load()['server_bin'], str(self.new))
        routes.post_engine_switch(routes.Req(body={'engine': 'llamacpp'}))
        self.assertEqual(config.load()['server_bin'], str(self.old))

    def test_builtin_callback_and_schedule_preserve_custom_even_if_binary_missing(self):
        self.activate()
        exists = routes.os.path.exists
        with mock.patch.object(routes.os.path, 'exists', side_effect=lambda p: False if p == str(self.new) else exists(p)):
            routes._on_built_llamacpp('server_bin', str(self.old))
        self.assertEqual(config.load()['server_bin'], str(self.new))
        self.assertEqual(config.load()['llama_builtin_server_bin'], str(self.old))
        self.assertEqual(self.restart.call_count, 1)
        self.assertIn('custom', routes._scheduled_build_idle(config.load())[0])

    def test_running_build_blocks_activation(self):
        with mock.patch.dict(routes.BUILDER_LLAMA.state, running=True), self.assertRaises(routes.ApiError) as caught:
            self.activate()
        self.assertEqual(caught.exception.status, 409)
        self.stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()

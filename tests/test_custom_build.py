"""Custom builds use saved recipes, real Git fixtures, and isolated config."""
import conftest_paths  # noqa: F401
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import config
import custom_build
import routes
from builder import BuildManager


class CustomBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / 'source with spaces'
        self.build = self.root / 'build with spaces'
        self.origin = self.root / 'origin'
        self.git('init', '-b', 'main', str(self.origin))
        self.git('-C', str(self.origin), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '--allow-empty', '-m', 'initial')
        self.target = dict(name='My fork', repository='https://example.invalid/fork.git', branch='main',
                           source=str(self.src), build=str(self.build), server_binary='{build}/server',
                           build_command='pwd > cwd.txt\nprintf "built" > server\nprintf "output\\n"\nprintf "error output\\n" >&2')
        self.manager = custom_build.CustomBuildManager(str(self.root / 'logs'))
        real_stream, real_git = self.manager._stream, self.manager._git
        self.commands = []
        def stream(cmd, **kwargs):
            self.commands.append(cmd)
            cmd = [str(self.origin) if x == self.target['repository'] else x for x in cmd]
            return real_stream(cmd, **kwargs)
        def git(src, *args, **kwargs):
            if args == ('remote', 'get-url', 'origin'):
                return subprocess.CompletedProcess([], 0, self.target['repository'] + '\n', '')
            return real_git(src, *args, **kwargs)
        self.manager._stream, self.manager._git = stream, git

    def git(self, *args):
        return subprocess.run(['git', *args], capture_output=True, text=True, check=True)

    def test_new_clone_cwd_multiline_log_binary_and_existing_pull_backup(self):
        self.manager.run_custom(self.target)
        self.assertEqual(self.manager.state['phase'], 'done', self.manager.tail())
        self.assertEqual((self.build / 'cwd.txt').read_text().strip(), str(self.build))
        self.assertIn('error output', self.manager.tail())
        self.assertEqual(self.manager.state['server_bin'], str(self.build / 'server'))
        self.assertTrue(any('clone' in c for c in self.commands))
        self.git('-C', str(self.origin), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                 'commit', '--allow-empty', '-m', 'upstream change')
        update = self.manager.check_updates(str(self.src), 'origin/main')
        self.assertTrue(update['ok'])
        self.assertEqual(update['behind'], 1)
        self.manager.run_custom(self.target)
        self.assertEqual(self.manager.state['phase'], 'done', self.manager.tail())
        self.assertTrue(any('pull' in c and '--ff-only' in c for c in self.commands))
        self.assertTrue(list(self.build.glob('server.backup-*')))
        self.assertEqual(self.manager.check_updates(str(self.src), 'origin/main')['behind'], 0)

    def test_failure_is_not_hidden_by_later_commands_or_old_binary(self):
        self.manager.run_custom(self.target)
        self.target['build_command'] = 'false\nprintf wrong > should-not-exist'
        self.manager.run_custom(self.target)
        self.assertEqual(self.manager.state['phase'], 'failed')
        self.assertFalse((self.build / 'should-not-exist').exists())
        self.assertNotIn('server_bin', self.manager.state)

    def test_missing_binary_and_failed_pipeline(self):
        for command in ('echo no-binary', 'false | cat\ntouch server'):
            self.target['build_command'] = command
            self.manager.run_custom(self.target)
            self.assertEqual(self.manager.state['phase'], 'failed')
        self.assertFalse((self.build / 'server').exists())

    def test_expansion_preserves_shell_syntax(self):
        self.assertEqual(custom_build.expand('{source}|{build}|{jobs}|${source}|{other}|{a,b}', self.target, 3),
                         f'{self.src}|{self.build}|3|${{source}}|{{other}}|{{a,b}}')
        self.target['server_binary'] = 'relative-server'
        self.assertEqual(custom_build.binary_path(self.target), str(self.build / 'relative-server'))

    def test_validation_is_read_only_and_rejects_bad_inputs(self):
        custom_build.validate(self.target)
        self.assertFalse(self.src.exists())
        self.assertFalse(self.build.exists())
        for key, value in [('repository', '--upload-pack=evil'), ('branch', '-bad'),
                           ('build', 'relative'), ('build_command', ''),
                           ('server_binary', '{unknown}/server')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                custom_build.validate(dict(self.target, **{key: value}))
        self.src.mkdir()
        with self.assertRaisesRegex(ValueError, 'Git checkout'):
            custom_build.validate(self.target)

    def test_config_crud_and_remove_preserves_files(self):
        with mock.patch.object(config, 'CONFIG', str(self.root / 'config.json')), \
             mock.patch.object(routes, 'cfg', side_effect=config.load):
            with mock.patch.object(custom_build.CustomBuildManager, 'start_custom') as start:
                routes.post_build_target_validate(routes.Req(body=self.target))
                _, saved = routes.post_build_target_save(routes.Req(body=self.target))
                start.assert_not_called()
            tid = saved['target']['id']
            self.assertEqual(config.load()['custom_build_targets'][tid]['name'], 'My fork')
            edited = dict(saved['target'], name='Renamed')
            routes.post_build_target_save(routes.Req(body=edited))
            self.assertEqual(config.load()['custom_build_targets'][tid]['name'], 'Renamed')
            self.build.mkdir()
            (self.build / 'keep').write_text('keep')
            routes.post_build_target_remove(routes.Req(body={'id': tid}))
            self.assertEqual((self.build / 'keep').read_text(), 'keep')
            self.assertEqual(config.load()['custom_build_targets'], {})
            self.assertNotIn('id', self.target)

    def test_unknown_targets_and_builtin_routing(self):
        with mock.patch.object(routes, 'cfg', return_value={'custom_build_targets': {}}):
            for handler, req in [(routes.get_build_info, routes.Req(qs={'target': 'missing'})),
                                 (routes.get_build_log, routes.Req(qs={'target': 'missing'})),
                                 (routes.post_build_start, routes.Req(body={'target': 'missing'}))]:
                with self.assertRaises(routes.ApiError):
                    handler(req)
            self.assertIs(routes._builder_for('llamacpp'), routes.BUILDER_LLAMA)
            self.assertIs(routes._builder_for('ikllama'), routes.BUILDER_IKLLAMA)

    def test_custom_routes_do_not_generate_flags_or_execute_unsaved_commands(self):
        self.manager.run_custom(self.target)
        c = {'custom_build_targets': {'custom-test': self.target}}
        with mock.patch.object(routes, 'cfg', return_value=c), \
             mock.patch.dict(routes._CUSTOM_BUILDERS, {'custom-test': self.manager}), \
             mock.patch.object(routes.hardware, 'recommend', side_effect=AssertionError('must not inspect hardware')), \
             mock.patch.object(self.manager, 'start_custom', return_value=True) as start:
            _, info = routes.get_build_info(routes.Req(qs={'target': 'custom-test'}))
            self.assertEqual(info['current']['branch'], 'main')
            start.assert_not_called()
            routes.post_build_start(routes.Req(body={'target': 'custom-test', 'build_command': 'evil'}))
            self.assertEqual(start.call_args.args[0]['build_command'], self.target['build_command'])

    def test_update_errors_are_not_reported_as_up_to_date(self):
        self.assertFalse(self.manager.check_updates(str(self.src), 'origin/main')['ok'])
        self.manager.run_custom(self.target)
        self.assertFalse(self.manager.check_updates(str(self.src), 'origin/missing')['ok'])

    def test_running_target_cannot_be_edited_or_removed(self):
        path = str(self.root / 'config.json')
        with mock.patch.object(config, 'CONFIG', path), mock.patch.object(routes, 'cfg', side_effect=config.load):
            _, saved = routes.post_build_target_save(routes.Req(body=self.target))
            tid = saved['target']['id']
            self.manager.state['running'] = True
            with mock.patch.dict(routes._CUSTOM_BUILDERS, {tid: self.manager}):
                for handler, body in [(routes.post_build_target_save, saved['target']),
                                      (routes.post_build_target_remove, {'id': tid})]:
                    with self.assertRaises(routes.ApiError):
                        handler(routes.Req(body=body))
            self.assertIn(tid, config.load()['custom_build_targets'])


if __name__ == '__main__':
    unittest.main()

"""Run history contract: no nearest-load attribution, no historical import."""
import conftest_paths  # noqa: F401
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import routes
import router_ctl
import stats
import stats_history as h


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000.0
        self.ports = {9000: 50}
        self.births = {}
        self.history = h.RunHistory(str(Path(self.tmp.name) / 'history.json'),
            resolve_pid=lambda p: self.ports.get(p), clock=lambda: self.now,
            process_identity=lambda p: self.births.get(p, 'birth1'))

    def load(self, model='alpha', port=9001, pid=101, argv=None):
        self.ports[port] = pid
        self.history.stderr(f'I srv load: spawning server instance with name={model} on port {port}')
        self.history.stderr('I srv load: spawning server instance with args:')
        for arg in argv or ['/external/strix-llama/build/bin/llama-server', '--model', '/models/a b.gguf', '--ctx-size', '65536']:
            self.history.stderr('I srv load:   ' + arg)
        self.history.stderr('I srv load: waiting for child')
        self.history.bind()
        return self.history.active.get(pid)

    def timing(self, pid=101, task=4, mtp=False, tps=True):
        prefix = f'[{pid}] I slot print_timing: id 3 | task {task} | '
        pp = ' ( 57.88 ms per token, 244.34 tokens per second)' if tps else ''
        tg = ' ( 48.33 ms per token, 14.03 tokens per second)' if tps else ''
        lines = [prefix + 'prompt eval time = 1562.75 ms / 27 tokens' + pp,
                 prefix + 'eval time = 2755.01 ms / 58 tokens' + tg,
                 prefix + 'total time = 4317.76 ms / 85 tokens',
                 prefix + 'graphs reused = 6640']
        if mtp:
            lines.append(prefix + 'draft acceptance = 0.51855 ( 699 accepted / 1348 generated), mean len = 2.17')
        return lines

    def run_block(self, **kwargs):
        for line in self.timing(**kwargs):
            self.history.stdout(line)
        self.history.finish(force=True)

    def test_complete_timing_and_optional_mtp(self):
        cid = self.load()
        self.run_block(mtp=True)
        runs = self.history.recent('alpha')['runs']
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual((r['load_config_id'], r['task_id'], r['child_pid']), (cid, 4, 101))
        self.assertEqual((r['prompt_tokens'], r['generated_tokens'], r['total_ms']), (27, 58, 4317.76))
        self.assertEqual((r['prompt_eval_ms'], r['eval_ms'], r['pp_tps'], r['tg_tps']), (1562.75, 2755.01, 244.34, 14.03))
        self.assertEqual((r['mtp_acceptance'], r['mtp_accepted'], r['mtp_generated'], r['mtp_mean_len']), (.51855, 699, 1348, 2.17))
        self.run_block(mtp=True)  # duplicate log block
        self.assertEqual(len(self.history.recent('alpha')['runs']), 1)

    def test_non_mtp_missing_rates_incomplete_and_unknown(self):
        self.load()
        self.history.stdout('[101] unknown engine line')
        self.history.stdout(self.timing(task=1)[0])
        self.history.finish(force=True)
        self.assertEqual(self.history.recent('alpha')['runs'], [])
        self.run_block(task=2, tps=False)
        r = self.history.recent('alpha')['runs'][0]
        for field in ('mtp_acceptance', 'mtp_accepted', 'mtp_generated', 'mtp_mean_len', 'pp_tps', 'tg_tps'):
            self.assertIsNone(r[field])

    def test_interleaved_models_and_tasks(self):
        a = self.load()
        b = self.load('beta', 9002, 202)
        blocks = [self.timing(101, 4, True), self.timing(202, 4), self.timing(101, 5)]
        for i in range(5):
            for block in blocks:
                if i < len(block): self.history.stdout(block[i])
        self.history.finish(force=True)
        ar = self.history.recent('alpha')['runs']
        br = self.history.recent('beta')['runs']
        self.assertEqual(len(ar), 2)
        self.assertTrue(all(r['load_config_id'] == a for r in ar))
        self.assertEqual((br[0]['child_pid'], br[0]['load_config_id']), (202, b))
        self.assertIsNone(br[0]['mtp_acceptance'])

    def test_multiple_pending_argv_blocks_bind_independently(self):
        for model, port in (('alpha', 9001), ('beta', 9002)):
            self.history.stderr(f'I srv load: spawning server instance with name={model} on port {port}')
            self.history.stderr('I srv load: spawning server instance with args:')
            self.history.stderr('I srv load:   /external/llama-server')
            self.history.stderr('I srv load:   --model')
            self.history.stderr('I srv load:   ' + model + '.gguf')
        self.history.stderr('I srv load: child started')
        self.ports.update({9001:101,9002:202})
        self.history.bind()
        self.run_block(pid=101)
        self.run_block(pid=202)
        for model in ('alpha', 'beta'):
            result = self.history.recent(model)
            config = next(iter(result['configs'].values()))
            self.assertEqual(config['main_model'], model + '.gguf')

    def test_slot_only_runs_and_replay(self):
        self.load()
        lines = [line.replace('task 4 | ', '') for line in self.timing()]
        for line in lines: self.history.stdout(line)
        self.history.finish(force=True)
        for line in lines: self.history.stdout(line)
        self.history.finish(force=True)
        self.assertEqual(len(self.history.recent('alpha')['runs']), 1)
        for line in lines: self.history.stdout(line.replace('I slot', '12:10:21 I slot'))
        self.history.finish(force=True)
        self.assertEqual(len(self.history.recent('alpha')['runs']), 2)
        self.assertIsNone(self.history.recent('alpha')['runs'][0]['task_id'])

    def test_reused_pid_unread_logs_cannot_become_new_config(self):
        out = Path(self.tmp.name) / 'out.log'
        out.write_text('')
        follower = h.Follower(str(out))
        self.history.followers['router_stdout'] = follower
        old = self.load()
        with out.open('a') as f: f.write('\n'.join(self.timing())+'\n')
        self.history.close(101)
        new = self.load()
        self.assertGreater(new, old)
        lines = follower.read()
        for line, offset in zip(lines, follower.line_offsets):
            self.history.stdout(line, offset=offset)
        self.history.finish(force=True)
        self.assertEqual(self.history.recent('alpha')['runs'], [])
        with out.open('a') as f: f.write('\n'.join(self.timing(task=5))+'\n')
        lines = follower.read()
        for line, offset in zip(lines, follower.line_offsets):
            self.history.stdout(line, offset=offset)
        self.history.finish(force=True)
        self.assertEqual(self.history.recent('alpha')['runs'][0]['load_config_id'], new)

    def test_invalid_history_never_breaks_tracker(self):
        path = str(Path(self.tmp.name) / 'stats-history.json')
        for raw in ('{broken', '[]', '{"version":1}', '{"version":1,"next_id":1,"configs":{},"runs":{"a":[{}]}}'):
            Path(path).write_text(raw)
            resumed = h.RunHistory(path)
            self.assertEqual(resumed.recent('alpha')['runs'], [])

    def test_redaction_preserves_token_count_options_and_platform_command(self):
        argv = ['/external/llama-server', '--max-tokens', '12', '--hf-token', 'fixture', '--model', "a b'c.gguf"]
        with mock.patch.object(h.router_ctl.osplat, 'IS_WIN', False):
            snapshot = h.snapshot(argv)
        self.assertEqual(snapshot['argv'][2], '12')
        self.assertEqual(snapshot['launch_command'], h.shlex.join(snapshot['argv']))
        self.assertNotIn('fixture', snapshot['launch_command'])
        with mock.patch.object(h.router_ctl.osplat, 'IS_WIN', True):
            snapshot = h.snapshot(argv)
        self.assertEqual(snapshot['launch_command'], subprocess.list2cmdline(snapshot['argv']))

    def test_reload_and_pid_reuse_keep_old_snapshot(self):
        old = self.load()
        self.run_block()
        self.history.close(101)
        new = self.load(argv=['/external/ik_llama/bin/llama-server', '-c', '8192'])
        self.run_block()
        self.assertGreater(new, old)
        result = self.history.recent('alpha')
        self.assertEqual([r['load_config_id'] for r in result['runs']], [new, old])
        self.assertEqual(result['configs'][str(old)]['normalized_options']['performance']['Context (tok)'], '65536')
        self.assertEqual(result['configs'][str(new)]['engine_name'], 'ik_llama')

    def test_pid_reuse_without_observed_port_gap(self):
        self.load()
        self.births[101] = 'birth2'
        self.history.bind()
        self.run_block()
        self.assertEqual(self.history.recent('alpha')['runs'], [])
        self.assertNotIn(101, self.history.active)

    def test_port_replacement_and_model_exit_close_session(self):
        self.load()
        self.ports[9001] = 303
        self.history.bind()
        self.assertNotIn(101, self.history.active)
        self.load('beta', 9002, 202)
        self.history.stderr('I srv load: instance name=beta exited with status 0')
        self.assertNotIn(202, self.history.active)

    def test_pending_retries_timeout_and_ambiguous_pid(self):
        self.load(pid=None)
        self.assertEqual(self.history.active, {})
        self.now += 2
        self.ports[9001] = 101
        self.history.bind()
        self.assertIn(101, self.history.active)
        self.load('beta', 9002, None)
        self.now += h.STARTUP_WINDOW + 1
        with self.assertLogs(h.LOG, level='WARNING'):
            self.history.bind()
        self.assertEqual(self.history.pending, [])
        self.run_block(pid=777)
        self.assertEqual(self.history.recent('beta')['runs'], [])
        with self.assertLogs(h.LOG, level='WARNING'):
            self.load('beta', 9002, 101)
        self.assertNotIn(101, self.history.active)

    def test_unknown_options_and_secret_redaction(self):
        argv = ['/external/llama-server', '--special-fork-mode', 'new value', '--api-key', 'fake-test-key', '--hf-token=fake-test-token', '-ngl', '99', '-m', 'main.gguf', '-md', 'draft.gguf', '--mmproj', 'vision.gguf']
        cid = self.load(argv=argv)
        self.run_block()
        self.history.flush()
        raw = Path(self.history.path).read_text()
        self.assertNotIn('fake-test-key', raw)
        self.assertNotIn('fake-test-token', raw)
        cfg = self.history.config(str(cid), 'alpha')
        self.assertEqual(cfg['other_options'][:2], ['--special-fork-mode', 'new value'])
        self.assertEqual(cfg['argv'][2], 'new value')
        self.assertIn('[REDACTED]', cfg['argv'])
        self.assertEqual((cfg['main_model'], cfg['draft_model'], cfg['mmproj']), ('main.gguf', 'draft.gguf', 'vision.gguf'))
        self.assertEqual(argv[4], 'fake-test-key', 'caller argv must not be modified')

    def test_engine_commit_and_missing_fallback(self):
        checkout = Path(self.tmp.name) / 'strix-llama.cpp'
        (checkout / '.git').mkdir(parents=True)
        exe = str(checkout / 'build/bin/llama-server')
        with mock.patch.object(h.subprocess, 'check_output', return_value='06a64c3\n') as git:
            self.assertEqual(h.engine_identity(exe), ('strix-llama', '06a64c3'))
            self.assertIn('--short=7', git.call_args.args[0])
        with mock.patch.object(h.subprocess, 'check_output', side_effect=OSError):
            self.assertEqual(h.engine_identity(exe), ('strix-llama', None))
        with mock.patch.object(h.subprocess, 'check_output') as git:
            self.assertEqual(h.engine_identity('/external/llama-server'), ('llama.cpp', None))
            git.assert_not_called()

    def test_retention_preserves_referenced_and_active_configs(self):
        old = self.load()
        self.run_block()
        new = self.load(pid=102)
        for task in range(1001):
            self.now += 1
            self.run_block(pid=102, task=task)
        active = self.load('beta', 9002, 202)
        self.history.prune()
        self.assertEqual(len(self.history.data['runs']['alpha']), 1000)
        self.assertNotIn(str(old), self.history.data['configs'])
        self.assertIn(str(new), self.history.data['configs'])
        self.assertIn(str(active), self.history.data['configs'])

    def test_persistence_restart_snapshot_copy(self):
        self.load()
        self.run_block()
        self.history.flush()
        resumed = h.RunHistory(self.history.path)
        result = resumed.recent('alpha')
        result['configs']['1']['argv'].append('edited')
        self.assertNotIn('edited', resumed.recent('alpha')['configs']['1']['argv'])
        self.assertEqual(resumed.active, {})
        self.assertEqual(resumed.data['next_id'], 2)

    def test_api_identifier_limits_and_model_ownership(self):
        cid = self.load()
        self.run_block()
        with mock.patch.object(stats.TRACKER, 'history', self.history):
            for limit in h.LIMITS:
                code, result = routes.get_stats_runs(routes.Req(qs={'model':'alpha','limit':str(limit)}))
                self.assertEqual(code, 200)
                self.assertEqual(len(result['runs']), 1)
            for qs in ({'model':'alpha','limit':'1000'}, {'model':'alpha','limit':'1.0'}, {}, {'model':'a\n'}):
                with self.assertRaises(routes.ApiError): routes.get_stats_runs(routes.Req(qs=qs))
            for config in ('-1', '../1', '1.0', '', '0'):
                with self.assertRaises(routes.ApiError):
                    routes.get_stats_config(routes.Req(qs={'model':'alpha','id':config}))
            with self.assertRaises(routes.ApiError):
                routes.get_stats_config(routes.Req(qs={'model':'beta','id':str(cid)}))
            self.assertEqual(routes.get_stats_config(routes.Req(qs={'model':'alpha','id':str(cid)}))[1]['id'], cid)

    def test_reset_clears_aggregate_history_and_baselines(self):
        logfile = Path(self.tmp.name) / 'router.out.log'
        logfile.write_text('old line\n')
        self.history.followers['router_stdout'] = h.Follower(str(logfile))
        self.load()
        self.run_block()
        with mock.patch.object(stats, 'STATS_FILE', str(Path(self.tmp.name) / 'stats.json')):
            tracker = stats.StatsTracker()
            tracker.history = self.history
            tracker._record_tokens('alpha', 100, 100)
            with logfile.open('a') as f: f.write('\n'.join(self.timing(task=99)) + '\n')
            tracker.reset()
            self.assertEqual(tracker.summary()['per_model'], [])
            self.assertEqual(self.history.data['runs'], {})
            self.assertEqual(self.history.data['configs'], {})
            self.assertEqual(self.history.followers['router_stdout'].read(), [])
            self.assertEqual(self.history.active, {})
            self.assertEqual(self.history.blocks, {})

    def test_old_stats_file_loads_without_migration(self):
        path = str(Path(self.tmp.name) / 'stats.json')
        Path(path).write_text(json.dumps({'models':{'alpha':{'prompt':1,'generated':2,'loaded_secs':3,'runs':4,'last_used':0}}, 'daily':{}}))
        with mock.patch.object(stats, 'STATS_FILE', path):
            tracker = stats.StatsTracker()
        self.assertEqual(tracker.summary()['totals']['tokens'], 3)
        self.assertEqual(tracker.history.recent('alpha')['runs'], [])

    def test_rotation_truncation_partial_lines_and_no_import(self):
        path = Path(self.tmp.name) / 'out.log'
        path.write_text('historical\n')
        follower = h.Follower(str(path))
        self.assertEqual(follower.read(), [])
        with path.open('ab') as f: f.write('新'.encode()[:1])
        self.assertEqual(follower.read(), [])
        with path.open('ab') as f: f.write('新'.encode()[1:] + b' line\npartial')
        self.assertEqual(follower.read(), ['新 line'])
        path.rename(path.with_suffix('.old'))
        path.write_text('new file\n')
        self.assertEqual(follower.read(), ['new file'])
        path.write_text('truncated but longer than previous offset\n')
        self.assertEqual(follower.read(), ['truncated but longer than previous offset'])
        self.assertEqual(follower.read(), [])

    def test_ingestion_actual_blocks_partial_args_reset_restart_rotation(self):
        err = Path(self.tmp.name) / 'router.err.log'
        out = Path(self.tmp.name) / 'router.out.log'
        err.write_text('historical load\n')
        out.write_text('\n'.join(self.timing())+'\n')
        paths = {'router_stderr':str(err), 'router_stdout':str(out)}
        with mock.patch.object(h.log_manager, 'log_path', side_effect=lambda kind: paths[kind]), mock.patch('config.load', return_value={'router_port':9000}):
            self.history.poll_once()
            self.assertEqual(self.history.data['runs'], {})
            self.ports[9001] = 101
            with err.open('a') as f:
                f.write('I srv load: spawning server instance with name=alpha on port 9001\nI srv load: spawning server instance with args:\nI srv load:   /external/llama-server\nI srv load:   --ctx-size\nI srv load:   65')
            self.history.poll_once()
            self.assertEqual(self.history.active, {})
            with err.open('a') as f: f.write('536\nI srv load: child started\n')
            self.history.poll_once()
            self.assertIn(101, self.history.active)
            timing = self.timing(mtp=True)
            with out.open('a') as f: f.write('\n'.join(timing[:-1])+'\n'+timing[-1][:80])
            self.history.poll_once()
            self.now += 2
            self.history.poll_once()
            self.assertEqual(self.history.recent('alpha')['runs'], [], 'partial trailing MTP must not be lost')
            with out.open('a') as f: f.write(timing[-1][80:]+'\n')
            self.history.poll_once()
            self.now += 2
            self.history.poll_once()
            self.assertEqual(len(self.history.recent('alpha')['runs']), 1)
            self.assertEqual(self.history.recent('alpha')['runs'][0]['mtp_mean_len'], 2.17)
            out.rename(out.with_suffix('.old'))
            out.write_text('\n'.join(self.timing(task=5))+'\n')
            self.history.poll_once()
            self.now += 2
            self.history.poll_once()
            self.assertEqual(len(self.history.recent('alpha')['runs']), 2)
            self.history.reset()
            self.history.poll_once()
            self.assertEqual(self.history.recent('alpha')['runs'], [])
            # PID-prefix lines alone after Reset cannot revive the old config.
            with out.open('a') as f: f.write('\n'.join(self.timing(task=6))+'\n')
            self.history.poll_once()
            self.now += 2
            self.history.poll_once()
            self.assertEqual(self.history.recent('alpha')['runs'], [])
            self.load()
            self.ports[9000] = 51  # router restart
            self.history.poll_once()
            self.assertEqual(self.history.active, {})


class PortHelperTests(unittest.TestCase):
    def test_posix_only_unique_listener(self):
        with mock.patch.object(router_ctl.osplat, 'IS_WIN', False), mock.patch.object(router_ctl.osplat, 'run_text') as run:
            for raw, expected in [('101\n101\n', 101), ('101\n202\n', None), ('', None)]:
                run.return_value = raw
                self.assertEqual(router_ctl.listening_pid(9001), expected)
            self.assertIn('-sTCP:LISTEN', run.call_args.args[0])

    def test_process_birth_markers_linux_macos_windows(self):
        with mock.patch.object(router_ctl.osplat, 'IS_WIN', False), mock.patch.object(router_ctl.os.path, 'exists', return_value=True), mock.patch('builtins.open', mock.mock_open(read_data='123 (name with ) spaces) ' + ' '.join(str(i) for i in range(30)))):
            self.assertEqual(router_ctl.process_identity(123), '19')
        with mock.patch.object(router_ctl.osplat, 'IS_WIN', False), mock.patch.object(router_ctl.os.path, 'exists', return_value=False), mock.patch.object(router_ctl.osplat, 'run_text', return_value='Sat Oct 3 05:29:00 2026\n'):
            self.assertEqual(router_ctl.process_identity(123), 'Sat Oct 3 05:29:00 2026')
        with mock.patch.object(router_ctl.osplat, 'IS_WIN', True), mock.patch.object(router_ctl.subprocess, 'check_output', return_value='123456\n'):
            self.assertEqual(router_ctl.process_identity(123), '123456')
        with mock.patch.object(router_ctl.osplat, 'IS_WIN', True), mock.patch.object(router_ctl.subprocess, 'check_output', side_effect=OSError):
            self.assertIsNone(router_ctl.process_identity(123))

    def test_windows_only_unique_listener_and_failure(self):
        with mock.patch.object(router_ctl.osplat, 'IS_WIN', True), mock.patch.object(router_ctl.subprocess, 'check_output') as run:
            for raw, expected in [('101\n', 101), ('101\n202\n', None)]:
                run.return_value = raw
                self.assertEqual(router_ctl.listening_pid(9001), expected)
            run.side_effect = OSError
            self.assertIsNone(router_ctl.listening_pid(9001))


class FrontendTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node required')
    def test_stats_ui(self):
        result = subprocess.run(['node', str(Path(__file__).with_name('stats_frontend_checks.mjs'))], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

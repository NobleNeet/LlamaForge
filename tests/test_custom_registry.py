"""Custom runtime registries seed once, then isolate every model operation."""
import conftest_paths  # noqa: F401
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import config
import routes


class CustomRegistryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(mock.patch.object(config, 'CONFIG', str(self.root / 'config.json')))
        self.base = self.root / 'conf.ini.d' / 'models.ini'
        config.update({'models_ini': str(self.base), 'custom_build_targets': {
            'custom-one': {'name': 'One'}, 'custom-two': {'name': 'Two'}
        }})
        config.ensure_models_ini()
        config.set_keys('shared', {'model': '/missing/shared.gguf', 'ctx-size': '8192'})
        self.baseline = self.base.read_bytes()

    def select(self, tid):
        config.update({'active_engine': 'llamacpp', 'active_llamacpp_build_target': tid})
        return Path(config.ini_path())

    def test_resolution_stable_id_relative_override_and_engine_precedence(self):
        path = self.select('custom-one')
        self.assertEqual(path, self.base.with_name('models-custom-one.ini'))
        self.assertFalse(path.exists(), 'resolving a path must not create it')
        config.mutate(lambda c: c['custom_build_targets']['custom-one'].update(name='Renamed'))
        self.assertEqual(config.ini_path(), str(path))
        config.mutate(lambda c: c['custom_build_targets']['custom-one'].update(models_ini='./registries/fork.ini'))
        self.assertEqual(config.ini_path(), str(Path(config.ROOT) / 'registries/fork.ini'))
        config.update({'active_engine': 'ikllama'})
        self.assertEqual(config.ini_path(), str(self.base.with_name('models-ikllama.ini')))
        self.select('unregistered')
        self.assertEqual(config.ini_path(), str(self.base))

    def test_windows_and_extensionless_paths(self):
        for baseline, expected in [('D:/conf.ini.d/models.ini', 'D:/conf.ini.d/models-custom-one.ini'),
                                   (r'C:\conf\models.ini', r'C:\conf\models-custom-one.ini'),
                                   ('/conf/models', '/conf/models-custom-one.ini')]:
            with self.subTest(baseline=baseline):
                c = dict(config.load(), models_ini=baseline, active_llamacpp_build_target='custom-one')
                self.assertEqual(config.ini_path(c), expected)

    def test_migration_copies_exact_baseline_once_before_first_read_or_edit(self):
        path = self.select('custom-one')
        self.assertEqual(config.read_sections()['shared']['ctx-size'], '8192')
        self.assertEqual(path.read_bytes(), self.baseline)
        config.set_keys('shared', {'spec-draft-adaptive': 'true'})
        fork = path.read_bytes()
        self.select('llamacpp')
        self.assertNotIn('spec-draft-adaptive', config.read_sections()['shared'])
        config.set_keys('shared', {'ctx-size': '4096'})
        self.select('custom-one')
        self.assertFalse(config.ensure_models_ini())
        self.assertEqual(path.read_bytes(), fork)
        self.assertEqual(config.read_sections()['shared']['ctx-size'], '8192')
        self.select('custom-two')
        self.assertEqual(config.read_sections()['shared']['ctx-size'], '4096')
        self.assertNotIn('spec-draft-adaptive', config.read_sections()['shared'])

    def test_existing_override_is_used_unchanged_and_survives_removal(self):
        path = self.root / 'registry' / 'chosen.ini'
        path.parent.mkdir()
        content = b'; user comment\n[*]\n[unique]\nmodel = /unique.gguf\nspec-draft-adaptive = true\n'
        path.write_bytes(content)
        config.mutate(lambda c: c['custom_build_targets']['custom-one'].update(models_ini=str(path)))
        self.select('custom-one')
        self.assertFalse(config.ensure_models_ini())
        self.assertEqual(set(config.read_sections()), {'*', 'unique'})
        routes.post_build_target_remove(routes.Req(body={'id': 'custom-one'}))
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(self.base.read_bytes(), self.baseline)

    def test_missing_baseline_creates_valid_seed_and_nested_override(self):
        missing = self.root / 'fresh' / 'baseline.ini'
        fork = self.root / 'other' / 'fork.ini'
        config.update({'models_ini': str(missing)})
        config.mutate(lambda c: c['custom_build_targets']['custom-one'].update(models_ini=str(fork)))
        self.select('custom-one')
        self.assertTrue(config.ensure_models_ini())
        self.assertEqual(fork.read_bytes(), missing.read_bytes())
        self.assertIn('*', config.read_sections())

    def test_model_routes_presets_scan_registration_ctx_and_prune_are_isolated(self):
        self.select('custom-one')
        with mock.patch.object(routes, 'router', return_value=(200, {'data': []})), \
             mock.patch.object(routes, 'schema', return_value={}), \
             mock.patch.object(config.gguf, 'default_ctx', return_value=2048):
            routes.post_save(routes.Req(body={'model': 'shared', 'settings': {'spec-draft-adaptive': 'true'}}))
            routes._materialize_preset_settings('shared', {'ctx-size': '4096'})
            routes.post_scan_apply(routes.Req(body={'entries': [{'id': 'scanned', 'model': '/missing/scanned.gguf'}]}))
            download = self.root / 'download.gguf'
            download.write_bytes(b'fixture')
            with mock.patch.object(routes.scanner, 'build_entries', return_value=[{'id': 'downloaded', 'model': str(download)}]):
                routes.post_hub_add(routes.Req(body={'path': str(download)}))
            sections = config.read_sections()
            self.assertEqual(sections['shared']['ctx-size'], '2048')
            self.assertEqual(sections['shared']['spec-draft-adaptive'], 'true')
            self.assertIn('scanned', sections)
            self.assertIn('downloaded', sections)
            routes.post_scan_prune(routes.Req(body={'ids': ['scanned']}))
            self.assertNotIn('scanned', config.read_sections())
        self.assertEqual(self.base.read_bytes(), self.baseline)
        self.select('custom-two')
        self.assertEqual(set(config.read_sections()), {'*', 'shared'})


if __name__ == '__main__':
    unittest.main()

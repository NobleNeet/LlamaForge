"""Header consumers and Build use the same backend active-build resolver."""
import conftest_paths  # noqa: F401
import unittest
from unittest import mock

import routes


class RuntimeBadgeStateTests(unittest.TestCase):
    def test_state_and_build_resolve_the_same_effective_identity(self):
        base = {'server_bin': '/builtin/server', 'active_engine': 'llamacpp',
                'active_llamacpp_build_target': 'llamacpp',
                'llama_builtin_server_bin': '/builtin/server',
                'ik_llama_server_bin': '/ik/server',
                'custom_build_targets': {'custom-one': {'name': 'strix-llama.cpp_vulkan',
                    'source': '/src', 'build': '/custom', 'server_binary': '{build}/server'}}}
        cases = [({}, 'llamacpp', 'llama.cpp'),
                 ({'active_engine': 'ikllama'}, 'ikllama', 'ik_llama'),
                 ({'server_bin': '/custom/server', 'active_llamacpp_build_target': 'custom-one'},
                  'custom-one', 'strix-llama.cpp_vulkan'),
                 # Matching an explicitly configured binary wins over a stale ID.
                 ({'server_bin': '/custom/server'}, 'custom-one', 'strix-llama.cpp_vulkan'),
                 ({'server_bin': '/external/server', 'active_llamacpp_build_target': 'custom-one'},
                  '', 'External / unregistered build')]
        for changes, target, name in cases:
            with self.subTest(changes=changes), \
                 mock.patch.object(routes, 'cfg', return_value=dict(base, **changes)), \
                 mock.patch.object(routes.REGISTRY, 'state', return_value={'models': []}), \
                 mock.patch.object(routes.REGISTRY, 'enabled', return_value=[]), \
                 mock.patch.object(routes, '_gpu_telemetry', return_value=[]):
                _, state = routes.get_state(routes.Req())
                _, builds = routes.get_build_targets(routes.Req())
                self.assertEqual(state['active_build'], builds['active_build'])
                self.assertEqual(state['active_build']['id'], target)
                self.assertEqual(state['active_build']['name'], name)

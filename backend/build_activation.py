"""Activate llama.cpp-compatible binaries without adding runtime backends.

The routes module is injected for its existing router/session helpers, avoiding
an additional router controller or shared pre-build session state.
"""
import os

import custom_build

KEYS = ('server_bin', 'active_engine', 'active_llamacpp_build_target', 'llama_builtin_server_bin')


def same_path(a, b):
    return bool(a and b) and os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def active_build(c):
    engine = c.get('active_engine', 'llamacpp')
    if engine == 'ikllama':
        return {'id': 'ikllama', 'name': 'ik_llama', 'server_bin': c.get('ik_llama_server_bin', '')}
    path = c.get('server_bin', '')
    targets = c.get('custom_build_targets', {})
    preferred = c.get('active_llamacpp_build_target', '')
    for tid in sorted(targets, key=lambda tid: tid != preferred):
        try:
            if same_path(path, custom_build.binary_path(targets[tid])):
                return {'id': tid, 'name': targets[tid]['name'], 'server_bin': path}
        except (KeyError, TypeError, ValueError):
            continue
    if not preferred or preferred == 'llamacpp' or same_path(path, c.get('llama_builtin_server_bin')):
        return {'id': 'llamacpp', 'name': 'llama.cpp', 'server_bin': path}
    return {'id': '', 'name': 'External / unregistered build', 'server_bin': path}


def activate(r, binary, target):
    """Called under the target lock. All validation precedes stopping the router."""
    old = r.cfg()
    if not os.path.isfile(binary):
        raise r.ApiError(400, 'Server Binary does not exist. Build this target first.')
    if not os.access(binary, os.X_OK):
        raise r.ApiError(400, 'Server Binary is not executable.')
    if not r.router_ctl.supports_router_mode(binary):
        raise r.ApiError(400, 'Server Binary does not support llama.cpp router mode (--models-preset).')
    if old.get('active_engine', 'llamacpp') == 'llamacpp' and same_path(old.get('server_bin'), binary):
        # Record identity even for a binary previously selected manually in Setup.
        changes = {'active_llamacpp_build_target': target}
        if target != 'llamacpp' and old.get('active_llamacpp_build_target') in (None, '', 'llamacpp'):
            changes['llama_builtin_server_bin'] = old.get('server_bin', '')
        r.config.update(changes)
        return 200, {'ok': True, 'active_engine': 'llamacpp', 'active_build': active_build(r.cfg())}
    port = old.get('router_port', 8080)
    running = r.router_ctl.is_running(port)
    loaded = []
    if running:
        status, data = r.router('/models')
        if status != 200 or not isinstance(data, dict) or not isinstance(data.get('data'), list):
            raise r.ApiError(409, 'Cannot read current router models; nothing changed.')
        loaded = [m['id'] for m in data['data'] if m.get('id') and m['id'] != 'default'
                  and m.get('status', {}).get('value') in ('loaded', 'loading')]
        if not r.router_ctl.stop(port):
            raise r.ApiError(409, 'Could not stop the current router; settings unchanged.')
    previous = {key: old.get(key, 'llamacpp' if key == 'active_engine' else '') for key in KEYS}
    changes = {'server_bin': binary, 'active_engine': 'llamacpp', 'active_llamacpp_build_target': target}
    if target != 'llamacpp' and old.get('active_llamacpp_build_target') in (None, '', 'llamacpp'):
        changes['llama_builtin_server_bin'] = old.get('server_bin', '')

    def restart(c):
        ok, error = r.router_ctl.restart(r._active_server_bin(c), r.config.ini_path(), port,
                                        c.get('router_host', '127.0.0.1'),
                                        c.get('router_api_key', ''), r.LOGDIR,
                                        models_max=r.router_ctl.resolve_models_max(c))
        if not ok:
            raise RuntimeError(error or 'Router restart failed')
        if not r._wait_router_ready(port):
            raise RuntimeError('Router did not become ready')

    try:
        current = r.config.update(changes)
        restart(current)
    except Exception as exc:
        recovery_error = ''
        try:
            # Stop any partially-started new process before restoring the config.
            try:
                stopped = r.router_ctl.stop(port)
            finally:
                r.config.update(previous)
            if not stopped:
                raise RuntimeError('Could not stop the new router')
            if running:
                restart(old)
                r._reload_loaded_models(loaded, source='/api/build/activate/rollback')
        except Exception as recovery:
            recovery_error = str(recovery)
        return 500, {'ok': False, 'error': str(exc), 'rollback_ok': not recovery_error,
                     'rollback_error': recovery_error, 'active_engine': r.cfg().get('active_engine', 'llamacpp'),
                     'active_build': active_build(r.cfg())}
    # Models belong to different INIs across engines. Only restore the previous
    # llama.cpp session on successful activation; rollback restores either one.
    if old.get('active_engine', 'llamacpp') == 'llamacpp':
        r._reload_loaded_models(loaded, source='/api/build/activate')
    return 200, {'ok': True, 'active_engine': 'llamacpp', 'active_build': active_build(r.cfg())}

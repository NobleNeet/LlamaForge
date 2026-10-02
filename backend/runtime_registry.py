"""Prepare one runtime's registry from fresh metadata, before touching its router."""
import argspec
import config


class RegistryPreparationError(ValueError):
    pass


def prepare(c, apply_ctx_defaults=False):
    binary = c.get('ik_llama_server_bin', '') if c.get('active_engine') == 'ikllama' else c.get('server_bin', '')
    # Discovery must precede even first-time initialization: failed discovery
    # must leave an existing registry (or its absence) unchanged.
    try:
        schema = argspec.build_schema(binary)
        metadata = argspec.schema_key_aliases(schema)
        if metadata.get('error'):
            raise RegistryPreparationError(f'Cannot obtain runtime option schema: {metadata["error"]}')
        path = config.ini_path(c)
        config.ensure_models_ini(path, c=c)
        if apply_ctx_defaults and 'ctx-size' in metadata['keys']:
            config.apply_ctx_defaults(path)
        result = config.sanitize_models_ini(path, valid_keys=metadata['keys'],
                                            alias_to_key=metadata['alias_to_key'])
    except RegistryPreparationError:
        raise
    except Exception as exc:
        raise RegistryPreparationError(f'Cannot prepare runtime registry: {exc}') from exc
    return path, schema, result


def start_configured_router():
    """Shared launcher for run.sh/run.ps1; never prevents opening the panel."""
    import os
    import router_ctl
    c = config.load()
    port = c['router_port']
    if router_ctl.is_running(port):
        return
    try:
        path, _schema, _result = prepare(c, apply_ctx_defaults=True)
        binary = c.get('ik_llama_server_bin', '') if c.get('active_engine') == 'ikllama' else c.get('server_bin', '')
        ok, error = router_ctl.start(binary, path, port, c.get('router_host', '127.0.0.1'),
                                     c.get('router_api_key', ''), os.path.join(config.ROOT, 'logs'),
                                     models_max=router_ctl.resolve_models_max(c))
        if not ok:
            print(f'WARNING: router not started: {error}')
    except RegistryPreparationError as exc:
        print(f'WARNING: router not started: {exc}')


if __name__ == '__main__':
    start_configured_router()

---
title: Usage Stats
section: guides
order: 5
---

# Usage Stats

Per-model token counts, run counts, generation speed, daily activity, and detailed run-history design for llama.cpp-family engines.

## What it does

The dashboard itself never sees inference traffic — clients talk to the llama.cpp router directly — and llama.cpp's own Prometheus counters reset when a child process restarts and do not provide persistent per-model history. `backend/stats.py`'s `StatsTracker` works around this with a background poller (`run_forever()`, every `POLL_SECS = 5` seconds) that:

1. Calls the router's `/models` endpoint to learn whether the router is up and which model IDs are currently loaded. More than one model may be resident when `router_models_max > 1`.
2. Scrapes `/metrics?model=<id>` independently for every loaded model. Each model has its own baseline for cumulative `llamacpp:prompt_tokens_total` and `llamacpp:tokens_predicted_total` counters, so token deltas are credited to the model that emitted them rather than to a single globally selected model. Negative deltas mean that child's counters reset and are dropped rather than subtracted.
3. Separately polls vLLM's `/metrics` (`vllm:prompt_tokens_total` / `vllm:generation_tokens_total`) on `vllm_port` the same way, best-effort and silent on failure, so vLLM usage lands in the same aggregate per-model store.
4. Persists aggregate data to `stats.json` at the repo root via an atomic write (write to `.tmp`, then `os.replace`), throttled to at most once every `FLUSH_SECS = 15` seconds while dirty.

Each model's aggregate record in `stats.json` tracks `prompt`, `generated`, `loaded_secs`, `gen_secs`, `runs`, and `last_used`. A "run" in the aggregate counters increments whenever generation transitions from idle to active (`_idle` flag), so it remains an approximation rather than a true request log. Average tokens/sec (`avg_tps`) is `generated / gen_secs`, where `gen_secs` only accumulates during poll windows that had active generation. Daily totals are kept for `DAILY_KEEP = 30` days and trimmed on each write.

Detailed per-run PP/TG/MTP history is specified separately in [`docs/specs/stats-run-history.md`](../specs/stats-run-history.md). That extension is designed to supplement, not replace, the aggregate Prometheus accounting described above.

## Per-model Run History design

The detailed-history extension expands a Per-model Usage row to show recent completed inference runs for supported llama.cpp-family engines.

The Recent Runs table is specified as:

| Column | Meaning |
|---|---|
| `Time` | Completion time shown as `HH:MM`; hover shows `YYYY-MM-DD HH:MM:SS` |
| `Prompt (tok)` | Prompt token count |
| `PP (tok/s)` | Prompt-processing throughput |
| `Generated (tok)` | Generated token count |
| `TG (tok/s)` | Token-generation throughput |
| `Total (s)` | Total run time |
| `MTP Accept (%)` | Draft/MTP acceptance rate when available |
| `Load Config` | Immutable load snapshot used by that run |

The default view shows the newest 10 runs, with 10 / 25 / 50 / 100 selectors. Missing per-run values are shown as `—` instead of changing the table shape.

A Load Config stores the actual execution environment for those runs, including:

- engine name
- engine source commit when available, displayed like `strix-llama (06a64c3)`
- exact executable path
- key performance settings
- speculative-decoding settings
- main/draft/mmproj model paths when present
- fork-specific or otherwise unclassified options
- the exact sanitized launch argv captured at load time

The engine commit is the commit of the engine source repository, not the LlamaForge commit. Commit detection is best-effort; if it cannot be determined, the engine name is shown without a fabricated revision.

### Correct Run -> Load Config association

Detailed history must not associate a run with merely "the most recently parsed load". Several models may be resident and serving at the same time.

The correlation design is:

1. `router.err.log` identifies a model load with `name=<model>` and a unique child server port.
2. The immediately associated `spawning server instance with args:` block is captured as that load session's immutable argv snapshot.
3. LlamaForge resolves the OS process that owns the reported child port and binds its PID to the load session.
4. `router.out.log` inference lines carry a `[PID]` prefix. Timing records from that PID are therefore attached only to the Load Config bound to the same PID/session.
5. When a task ID is present, `(child_pid, task_id)` is used to assemble the prompt timing, generation timing, total timing, and optional MTP lines into one completed Run record.

If the child port cannot be bound unambiguously to a PID, detailed history is skipped for that ambiguous session rather than guessed from timestamps or nearest-log-line heuristics. Aggregate Stats continue working independently.

See [`stats-run-history.md`](../specs/stats-run-history.md) for the complete persistence, retention, rotation, reset, API, and test requirements.

## Network access and API key

The router binds to `127.0.0.1` (local only) by default. The **Network Access** panel — part of the Setup tab's UI, backed by `GET/POST /api/network` — lets you rebind it to `0.0.0.0` so other devices on your network can reach it at `http://<lan-ip>:<port>/`, where the LAN IP comes from `router_ctl.lan_ip()`.

Enforcement is real, not merely a UI toggle: `router_ctl.start()` passes `--api-key <key>` straight to the router process whenever a key is configured. With LAN access on and no key set, the router is reachable by anyone on the network unauthenticated, so the UI warns about this and defaults the "require an API key" checkbox to checked.

The dashboard's own proxied calls send the configured key as `Authorization: Bearer <key>` automatically. `GET /api/network` reports `has_api_key: bool` only; the stored key itself is never sent back to the browser.

## How to use it

1. Open the **Stats** tab. The top row shows total tokens processed, tokens generated, cumulative inference time, distinct models used, approximate run count, and the most-used model.
2. **Live Throughput** shows loaded models, aggregate generation and prompt-eval tok/s, and active request count in real time.
3. **Activity** is a stacked prompt/generated bar chart; toggle **14d** / **30d** to change the window.
4. **Per-model Usage** lists every model with logged usage — total tokens, average tok/s while generating, run count, time loaded, and when it was last used. Click a column chip to sort by it.
5. With the Run History extension implemented, click a supported llama.cpp-family model row to expand Recent Runs, then click a `Load Config` value to inspect the engine, commit, options, model files, and full sanitized launch command.
6. Click **Reset stats** to zero the statistics store. With detailed history implemented, the same reset also clears Run and Load Config history and re-baselines log ingestion.
7. To share the router on your LAN, go to the **Setup** tab's **Network Access** panel, enable network access, optionally generate or enter an API key, then apply/restart the router.

## Screenshot

![Overview](docs/img/overview.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Poll cadence | `stats.py: POLL_SECS` | Router scraped every 5 seconds; `stats.json` flushed at most every 15 seconds (`FLUSH_SECS`) while dirty. |
| Loaded models | `StatsTracker._router_models()` | Returns all currently loaded router model IDs, not only one selected model. |
| Token attribution | `StatsTracker.poll_once()` | `/metrics?model=<id>` is scraped per loaded model; each model keeps its own baseline and counter resets are dropped. |
| Live rates | `StatsTracker.poll_once()` | Per-model live rates are summed for the Live Throughput totals while remaining available by model internally. |
| vLLM usage | `StatsTracker._poll_vllm()` | Separately scrapes vLLM's `/metrics` on `vllm_port`; detailed vLLM Run History is outside the initial history scope. |
| Aggregate run counting | `StatsTracker._poll_model()` | A run increments on an idle-to-active generation transition — an approximation, not a request log. |
| Avg tok/s | `StatsTracker.summary()` | `generated / gen_secs`; `gen_secs` accumulates only during poll windows with active generation. |
| Daily retention | `stats.py: DAILY_KEEP` | 30 days of daily buckets kept; UI toggles between showing the last 14 or 30. |
| Aggregate persistence | `stats.py: STATS_FILE` | `stats.json` at the repo root; atomic write via temp file + `os.replace`. |
| Detailed history | `docs/specs/stats-run-history.md` | 1,000 completed Runs per model; immutable Load Configs; PID/session-safe correlation; UI fetch capped at 100. |
| Router launch | `router_ctl.start()` | Starts the configured router executable with models preset, models-max, host/port, offline mode, metrics, and optional API key. |
| LAN bind | `POST /api/network` | Sets `router_host` to `0.0.0.0` (LAN) or `127.0.0.1` (local only) and restarts the router. |
| Key visibility | `GET /api/network` | Returns `has_api_key: bool` only — the stored key itself is never sent back to the browser. |

## Troubleshooting

If **Live Throughput** shows the router offline but models load fine, confirm the router process is running on `router_port`. The poller re-baselines model counters after router/model process resets rather than treating a counter drop as negative usage.

If one loaded model does not accumulate tokens, inspect that model's `/metrics?model=<id>` endpoint and confirm the router reports it as `loaded`; each resident model is scraped independently.

For detailed Run History, an ambiguous child port -> PID association must produce no guessed Run/Load Config relationship. Check the diagnostic log and router logs for the relevant load session. Log rotation/truncation must be handled by the history ingester without replaying old runs.

If external clients get `401`/`403` after enabling LAN access, verify they send `Authorization: Bearer <key>` with the configured key. A blank key field on save keeps the previously stored key rather than clearing it.

See also [Setup](setup.md) for Network Access, [Models & Tuning](models.md) for per-model configuration, and [`stats-run-history.md`](../specs/stats-run-history.md) for the detailed Run History implementation contract.

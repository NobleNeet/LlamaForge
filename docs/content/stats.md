---
title: Usage Stats
section: guides
order: 5
---

# Usage Stats

Per-model token counts, run counts, and generation speed, a daily activity chart, and the LAN-sharing / API-key toggle for the router.

## What it does

The dashboard itself never sees inference traffic — clients talk to the llama.cpp router directly — and llama.cpp's own Prometheus counters reset on every router restart and keep no per-model history. `backend/stats.py`'s `StatsTracker` works around both limits with a background poller (`run_forever()`, every `POLL_SECS = 5` seconds) that:

1. Calls the router's `/models` endpoint to learn whether it's up and which model is currently loaded (`router_models_max` defaults to `1`, so at most one model is ever loaded at a time — this is what makes attributing token deltas to "the loaded model" safe. Raise that key and the attribution becomes wrong: `stats.py` still credits deltas to one model, and `/models` reports neither load order nor a busy flag, so per-model accounting would have to be built first).
2. Scrapes `/metrics?model=<id>` for that model's cumulative `llamacpp:prompt_tokens_total` and `llamacpp:tokens_predicted_total` counters, diffs them against the previous poll, and adds the delta to that model's running totals — only when the *same* model stayed loaded across both polls, and only when the delta is non-negative (a drop means the router restarted and the counters reset, so it's treated as zero rather than subtracted).
3. Separately polls vLLM's `/metrics` (`vllm:prompt_tokens_total` / `vllm:generation_tokens_total`) on `vllm_port` the same way, best-effort and silent on failure, so vLLM usage lands in the same per-model store.
4. Persists everything to `stats.json` at the repo root via an atomic write (write to `.tmp`, then `os.replace`), throttled to at most once every `FLUSH_SECS = 15` seconds while dirty.

Each model's record in `stats.json` (`{"models": {...}, "daily": {...}, "first_seen": ...}`) tracks `prompt`, `generated`, `loaded_secs`, `gen_secs`, `runs`, and `last_used`. A "run" increments whenever generation transitions from idle to active (`_idle` flag), which approximates a request count without the router exposing one directly. Average tokens/sec (`avg_tps`) is `generated / gen_secs`, where `gen_secs` only accumulates during poll windows that had active generation — so it reflects throughput while generating, not wall-clock time the model was loaded. Daily totals are kept for `DAILY_KEEP = 30` days and trimmed on each write.

**LAN sharing and the API key.** The router binds to `127.0.0.1` (local only) by default. The **Network Access** panel — part of the Setup tab's UI, backed by `GET/POST /api/network` — lets you rebind it to `0.0.0.0` so other devices on your network can reach it at `http://<lan-ip>:<port>/`, where the LAN IP comes from `router_ctl.lan_ip()`. Enforcement is real, not merely a UI toggle: `router_ctl.start()` passes `--api-key <key>` straight to the `llama-server` process whenever a key is configured, so llama.cpp itself rejects unauthenticated requests once LAN access is on and a key is set. With LAN access on and no key set, the router is reachable by anyone on the network unauthenticated — the UI warns about this and defaults the "require an API key" checkbox to checked. The dashboard's own proxied calls (e.g. the OpenAI-compatible shim in `backend/anthropic_shim.py`) send the configured key as `Authorization: Bearer <key>` automatically; external clients must do the same. `GET /api/network` reports `has_api_key` (a boolean, never the key itself) so the UI can show "(unchanged — a key is already set)" without ever re-displaying a saved key.

## How to use it

1. Open the **Stats** tab. The top row shows total tokens processed, tokens generated, cumulative inference time, distinct models used, approximate run count, and the most-used model.
2. **Live Throughput** shows the currently loaded model, generation and prompt-eval tok/s, and active request count in real time (polled every 4 seconds while the tab is open).
3. **Activity** is a stacked prompt/generated bar chart; toggle **14d** / **30d** to change the window.
4. **Per-model Usage** lists every model with logged usage — total tokens, average tok/s while generating, run count, time loaded, and when it was last used. Click a column chip to sort by it.
5. Click **Reset stats** to zero the whole store (`POST /api/stats/reset`) — this is destructive and cannot be undone.
6. To share the router on your LAN: go to the **Setup** tab's **Network Access** panel, check "allow access from other devices on my network," optionally click **Generate Key** (or type your own), then **Apply & Restart Router**.

## Screenshot

![Overview](docs/img/overview.png)

## Reference

| Concept | Source | Behavior |
|---|---|---|
| Poll cadence | `stats.py: POLL_SECS` | Router scraped every 5 seconds; `stats.json` flushed at most every 15 seconds (`FLUSH_SECS`) while dirty. |
| Token attribution | `StatsTracker.poll_once()` | Deltas from `/metrics?model=<id>` credited to a model only if it was also loaded on the prior poll; negative deltas (counter reset) are dropped, not subtracted. |
| vLLM usage | `StatsTracker._poll_vllm()` | Separately scrapes vLLM's `/metrics` on `vllm_port`; same delta/attribution logic, silent on failure. |
| Run counting | `StatsTracker.poll_once()` | A "run" increments on each idle-to-active generation transition — an approximation, not a true request count. |
| Avg tok/s | `StatsTracker.summary()` | `generated / gen_secs`; `gen_secs` accumulates only during poll windows with active generation. |
| Daily retention | `stats.py: DAILY_KEEP` | 30 days of daily buckets kept; UI toggles between showing the last 14 or 30. |
| Persistence | `stats.py: STATS_FILE` | `stats.json` at the repo root; atomic write via temp file + `os.replace`. |
| LAN bind | `POST /api/network` | Sets `router_host` to `0.0.0.0` (LAN) or `127.0.0.1` (local only) and restarts the router. |
| API key enforcement | `router_ctl.start()` | Passed to `llama-server` as `--api-key <key>`; llama.cpp enforces it, not the dashboard. |
| Key visibility | `GET /api/network` | Returns `has_api_key: bool` only — the stored key itself is never sent back to the browser. |

## Troubleshooting

If "Live Throughput" shows the router offline but models load fine, confirm the router process is actually running on `router_port` — `has_api_key`/`router_running` come from `GET /api/network`, and the poller re-baselines (`self._prev = None`) whenever `/models` fails to answer. If per-model totals look stuck at zero for a model you know ran, check that it wasn't reloaded mid-generation: a token delta is only counted when the same model ID was loaded on the previous poll, so a reload during a burst discards that window. If external clients get `401`/`403` after enabling LAN access, verify they're sending `Authorization: Bearer <key>` with the exact key generated in the Network Access panel — a blank key field on save keeps the previously stored key rather than clearing it.

See also [Setup](setup.md) for the Network Access panel this page's LAN/API-key section documents, and [Models & Tuning](models.md) for per-model configuration.

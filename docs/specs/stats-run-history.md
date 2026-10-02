# Stats Run History Specification

## Status

Implementation specification for the Stats per-model run history extension.

This document defines the required behavior. `docs/content/stats.md` remains the user-facing guide and must be kept consistent when implementation changes land.

## Goal

Extend **Stats > Per-model Usage** so a model row can be expanded to inspect recent individual inference runs and the exact load configuration under which each run executed.

The feature is intended for performance comparison and reproducibility. It must make it possible to answer questions such as:

- What PP/TG throughput did this run achieve?
- What was the MTP acceptance rate?
- Which engine was used: upstream `llama.cpp`, `strix-llama`, `ik_llama`, or another llama.cpp-compatible fork?
- Which engine commit was used?
- Which load options and model files were active for that run?
- Did performance change after a reload, engine switch, or option change?

The existing aggregate Stats behavior remains intact.

## Scope

### Initial detailed-history support

Detailed Run History is required for llama.cpp-router-compatible engines whose child `llama-server` output is present in `router.out.log` / `router.err.log`, including at least:

- `llama.cpp`
- `strix-llama`
- `ik_llama` when used in a compatible execution path
- other llama.cpp-derived engines with compatible logging

### vLLM

vLLM continues to participate in the existing aggregate Stats counters.

Detailed Run History is **not** required for vLLM in the initial implementation. A vLLM model row may therefore continue to show aggregate usage without Recent Runs / Load Config detail.

## UI

### Per-model row

Keep the existing compact Per-model Usage row as the default view. Clicking a model row expands its details below the row. Clicking it again collapses the details.

Only the selected model's detail needs to be expanded at a time unless the existing UI makes multiple expansion trivial without visual clutter.

### Recent Runs

The expanded area begins with **RECENT RUNS**.

Default: newest 10 runs.

Selector values:

- 10
- 25
- 50
- 100

Columns, in this order:

| Column | Meaning |
|---|---|
| `Time` | Run completion timestamp displayed as `HH:MM` |
| `Prompt (tok)` | Prompt token count |
| `PP (tok/s)` | Prompt-processing throughput |
| `Generated (tok)` | Generated token count |
| `TG (tok/s)` | Token-generation throughput |
| `Total (s)` | Total run time in seconds |
| `MTP Accept (%)` | Draft/MTP acceptance percentage |
| `Load Config` | Load configuration used by the run, e.g. `#3` |

#### Time tooltip

The visible Time cell stays compact (`HH:MM`). Hovering it shows the full local timestamp including seconds:

`YYYY-MM-DD HH:MM:SS`

Do not spend permanent table width on the date.

#### Missing values

A metric unavailable for a particular engine/run is displayed as `—`. Do not hide/reorder columns on a per-run basis.

#### MTP tooltip

When MTP details are available, hovering or otherwise inspecting the MTP acceptance value exposes:

- acceptance percentage (full available precision)
- accepted tokens
- drafted/generated draft tokens
- mean accepted/draft length when the engine reports it

Example source log:

```text
draft acceptance = 0.51855 ( 699 accepted / 1348 generated), mean len = 2.17
```

Normal table display for this example is `51.9` under `MTP Accept (%)`.

If MTP/speculative decoding was not active, display `—`.

### Load Config boundary

When adjacent rows belong to different load configurations, the UI may show a subtle separator such as `Load Config changed`. The separator is secondary; the `Load Config` column is authoritative.

### Load Config detail

Clicking a `Load Config` value (for example `#3`) displays that immutable load snapshot below Recent Runs.

Suggested structure:

```text
LOAD CONFIG #3                              Loaded 2026-10-03 05:29

ENGINE
Engine             strix-llama (06a64c3)
Executable         /home/.../strix-llama.cpp/build/bin/llama-server

PERFORMANCE
Context (tok)      65,536
Batch (tok)         2,048
UBatch (tok)        1,024
Threads                 4
GPU Layers             99
KV Cache K            q8_0
KV Cache V            q8_0
Flash Attention         on

SPECULATIVE DECODING
Type               draft-mtp
Draft Max (tok)            3
Adaptive                   on

ENGINE-SPECIFIC / OTHER OPTIONS
...

MODELS
Main       gemma-4-31B-it-qat-UD-Q4_K_XL.gguf
Draft      mtp-gemma-4-31B-it.gguf
MMProj     mmproj-F16.gguf

[ Show full launch command ]
```

The exact styling should follow the existing Stats visual language rather than introduce a separate component style.

### Engine label and commit

The Engine line must identify the engine and, when available, the source repository commit used to build/run it.

Display form:

`strix-llama (06a64c3)`

Use a short Git SHA suitable for display (7 characters is preferred).

This commit is the **engine source repository commit**, not the LlamaForge commit.

Minimum persisted engine information:

- normalized/display engine name
- exact executable path
- engine source commit SHA when it can be determined

Commit detection is best-effort and must never block loading or history capture. If the commit cannot be determined, show only the engine name rather than fabricating a revision.

A suitable implementation may locate the source Git worktree from the configured engine/build location and run a read-only equivalent of `git rev-parse --short=7 HEAD`. Do not assume every executable lives inside a Git checkout.

### Option grouping

Known options should be normalized into human-readable groups where possible:

- **ENGINE**
- **PERFORMANCE**
- **SPECULATIVE DECODING**
- **MODELS**

Do not maintain a hard-coded exhaustive list of every fork-specific option.

After extracting known/normalized options, preserve any remaining argv entries under **ENGINE-SPECIFIC / OTHER OPTIONS**. This ensures new fork-specific flags are not silently lost when upstream/forks add options LlamaForge does not yet understand.

### Full Launch Command

`Show full launch command` expands the exact command/argv snapshot used by that load session.

The section must also repeat the engine identity so the command is self-describing:

```text
FULL LAUNCH COMMAND

Engine       strix-llama (06a64c3)
Executable   /home/.../strix-llama.cpp/build/bin/llama-server

/home/.../llama-server \
  --host 127.0.0.1 \
  ...
```

The command is historical data. It must be captured at load time and must not later be reconstructed from current `models.ini`, current Setup values, or current model settings. Editing settings after the run must not mutate old Load Config records.

Secrets must not be persisted/displayed in the historical command. If an argv source can contain secrets such as an API key, redact the value before persistence.

## Data model

Detailed history consists of two related record types.

### LoadConfig

One immutable record per concrete model load session/configuration.

Required logical fields:

```text
id                    monotonically increasing display identity per history store
model_id              router/model alias
loaded_at             timestamp
engine_name           normalized engine display name
engine_executable     exact executable path
engine_commit         source Git SHA, nullable
child_port            child llama-server port
child_pid             child process PID when available; validation only
process_identity      process birth/identity marker when available; validation only
argv                  exact sanitized argv array
normalized_options    parsed known options for UI
other_options         unclassified argv/options for UI
main_model            path/name when known
draft_model           path/name when known
mmproj                 path/name when known
```

`child_port` is the primary runtime correlation identifier exposed by the router logs. `child_pid` and `process_identity` are auxiliary validation/liveness data and must not be confused with the numeric prefix in `router.out.log`.

The stored argv array is authoritative. Human-readable groupings are derived from the same snapshot, not from later configuration state.

### Run

One record per completed inference for which the engine log provides a complete timing record.

Required logical fields:

```text
model_id
load_config_id
timestamp
child_port
task_id                when present in engine log
prompt_tokens
prompt_eval_ms
pp_tps
generated_tokens
eval_ms
tg_tps
total_ms
mtp_acceptance         nullable
mtp_accepted           nullable
mtp_generated          nullable
mtp_mean_len           nullable
```

`child_pid` may also be persisted as diagnostic metadata when known, but Run attribution must not depend on interpreting the `router.out.log` numeric prefix as a PID.

Preserve raw precision sufficient to reproduce the displayed values. Formatting/rounding is a frontend concern.

## Load-session and Run correlation

This is a correctness requirement. **Do not associate a Run with whichever Load Config happened to be parsed most recently.** Multiple models may be resident and generating concurrently.

### Router log identifier semantics

The router load log identifies a child by model name and child port:

```text
load: spawning server instance with name=gemma-... on port 35409
load: spawning server instance with args:
...
```

The corresponding child output in `router.out.log` is prefixed with that **child port**, not the OS PID:

```text
[35409] ... slot print_timing: id 3 | task 0 | prompt eval time = ...
[35409] ... slot print_timing: id 3 | task 0 | eval time = ...
[35409] ... slot print_timing: id 3 | task 0 | total time = ...
[35409] ... slot print_timing: id 3 | task 0 | draft acceptance = ...
```

This correspondence has been observed directly in production logs: `spawning ... on port 35409` and subsequent timing lines beginning with `[35409]` refer to the same child server.

Therefore the numeric `[N]` prefix must be interpreted as `child_port = N`. **It must not be interpreted as an OS PID.**

### Stable load-session identity

A concrete load session is created from the spawn event plus its argv snapshot. Its logical identity is the immutable LoadConfig record; while active, the router child port maps to that record.

Required correlation procedure:

1. When `router.err.log` reports `spawning server instance with name=<model> on port <port>`, open a **pending load session** containing at least `model_id`, `child_port`, and `loaded_at`.
2. Capture the immediately associated `spawning server instance with args:` block as an argv array for that same pending session. Parse the executable from `argv[0]`, derive the engine name, and best-effort resolve the engine Git commit.
3. Persist a new immutable LoadConfig once the spawn metadata and argv snapshot are complete, and maintain an active `child_port -> load_config_id` mapping.
4. The OS process listening on `child_port` may be resolved and stored as `child_pid` / `process_identity` for liveness and reuse validation. PID resolution is supplementary; failure to obtain a PID must not prevent correlation when the router logs themselves provide an unambiguous active child-port session.
5. Parse `router.out.log` timing records by the `[child_port]` prefix. A timing/MTP line beginning with `[P]` may only be attributed to the currently active LoadConfig bound to child port `P`.
6. Use `(load_config_id, task_id)` as the preferred in-memory Run assembly key when `task` is present. Equivalently, `(active session identity, task_id)` is acceptable. Do not use `(child_port, task_id)` alone as a historical identity because a port can be reused by a later load session.
7. Accumulate prompt timing, eval timing, total timing, and MTP acceptance lines for that assembly key until the run is complete, then persist exactly one Run record referencing that LoadConfig.
8. On child exit, unload/reload, router restart, or a new spawn that supersedes an existing session on the same child port, close the old active mapping. A later child reusing the same port creates a new LoadConfig/session; old Runs retain their old `load_config_id`.
9. If PID/process identity is available, use it only to strengthen liveness/session validation and to detect stale/reused processes. Do not require equality between the `router.out.log` prefix and the OS PID.

### Why child-port correlation is required

The router itself emits both sides of the correlation:

```text
router.err.log:  spawning ... on port 35409
router.out.log:  [35409] ... print_timing ...
```

The child port therefore directly bridges the load/config stream and the timing stream. No `port -> PID -> stdout-prefix` translation is required.

This direct mapping is simpler and safer for multiple concurrently loaded models because each active child server has its own router-assigned port. The immutable `load_config_id` supplies the session boundary needed when a port is later reused.

### Failure behavior

Correctness is preferred over guessed history.

If a timing line's child-port prefix has no unambiguous active LoadConfig, do **not** attach it using timestamp/"nearest load" heuristics. Keep aggregate Stats working and skip detailed history for that run/session, with diagnostic logging sufficient to investigate the failure.

Likewise, incomplete timing blocks must not be persisted as complete runs with invented values. Missing optional fields may be null/`—`; the base Run identity/config association must be unambiguous.

### Startup and historical backfill

Detailed history starts from events observed after the feature is installed. **Do not perform a one-time historical import of old completed Runs from router logs.**

Normal `run.sh` startup may start the router before the dashboard/history follower. If a child load session is already active by the time Run History starts, implementations may recover only the currently active LoadConfig metadata needed to correlate future timing lines, provided the recovery is unambiguous. Such recovery must not import completed Runs that predate the history follower baseline.

This startup-recovery path is defensive. Under normal operation, model loads occurring after Run History starts should be captured incrementally from their spawn/argv blocks.

### Platform support

Port/PID process helpers should remain reusable across supported LlamaForge platforms for optional validation/liveness checks. Detailed Run attribution itself is based on the child-port identifiers present in the router logs and must not require platform-specific PID lookup to succeed.

## Log ingestion

Detailed history may be implemented by a dedicated incremental log follower/parser or an equivalent mechanism, but it must satisfy these rules:

- process only newly appended data during normal operation
- survive log rotation/truncation and resume cleanly
- tolerate partial lines while a process is writing
- tolerate unknown log lines
- never make Stats/dashboard startup depend on successful history parsing
- avoid repeatedly rescanning entire large log files
- preserve the child-port/session boundary across interleaved output from multiple loaded models

The existing aggregate Prometheus poller remains the source of aggregate usage accounting unless the implementation deliberately refactors it without changing its behavior.

## Retention

Keep the most recent **1,000 completed Runs per model**.

When pruning Runs:

- remove oldest completed Runs beyond the per-model limit
- retain every LoadConfig still referenced by a retained Run
- unreferenced old LoadConfig records may be pruned
- an active/current LoadConfig must not be pruned merely because no completed Run references it yet

UI exposure remains capped at the selector values `10 / 25 / 50 / 100`.

## Persistence

History must be persisted and survive LlamaForge/dashboard restart.

It may be stored in `stats.json` with a versioned/migratable schema or in a dedicated history file. The implementation should prefer the option that keeps writes atomic and avoids rewriting unnecessarily large payloads on every 5-second aggregate stats poll.

Existing `stats.json` installations must continue to load without manual migration.

Detailed history starts from events observed after the feature is installed. Do not import old completed Runs merely because they are still present in existing log files.

## Reset behavior

The existing **Reset stats** action remains one destructive reset operation.

It must clear:

- aggregate per-model totals
- daily totals
- approximate run counters
- detailed Run history
- LoadConfig history

Any active parser/session state must re-baseline cleanly afterward so pre-reset runs/configs are not re-imported from already-consumed log content.

## Existing aggregate Stats behavior

Do not regress the current multi-model aggregate tracking.

The current implementation discovers all loaded router model IDs and scrapes `/metrics?model=<id>` independently, with a separate baseline per model. Detailed history is additive to that behavior.

vLLM aggregate polling also remains supported.

## API expectations

`GET /api/stats` may continue to serve aggregate data as today. Detailed history may be returned inline or through a separate endpoint, but avoid making the normal Stats refresh transfer all 1,000 runs for every model.

Preferred behavior:

- aggregate Stats request remains lightweight
- model expansion fetches the requested model's recent runs
- requested limit is constrained to supported UI values/max 100
- LoadConfig details can be included with those runs or fetched by ID

The backend must validate model/config identifiers and limits rather than trusting DOM values.

## Formatting examples

### Recent Runs

```text
RECENT RUNS                                                   Latest 10 ▼

Time   Prompt (tok)   PP (tok/s)   Generated (tok)   TG (tok/s)   Total (s)   MTP Accept (%)   Load Config
05:32      22,627        244.34           1,297           14.03       184.99          51.9            #3
05:17       7,806        251.82           1,833           14.21       160.73          53.2            #3
04:58      14,221        239.71             842           13.88       119.90          50.7            #2
```

### Load Config

```text
LOAD CONFIG #3                                       Loaded 2026-10-03 05:29

ENGINE
Engine             strix-llama (06a64c3)
Executable         /home/ubnadmin/LlamaForge-autotune/strix-llama.cpp/build/bin/llama-server
```

## Acceptance criteria

Implementation is complete when all of the following hold:

1. Existing aggregate Stats continue to pass and multi-model metrics remain correct.
2. A supported llama.cpp-family model row can be expanded to show recent completed runs.
3. Run columns and units match this specification.
4. Time shows `HH:MM`; full date/time including seconds is available on hover.
5. MTP acceptance is shown when present and `—` otherwise; detailed MTP values are inspectable.
6. Every displayed Run references one immutable LoadConfig.
7. Reloading the same model with different options creates a new LoadConfig rather than mutating the old one.
8. Switching engine creates a new LoadConfig and the engine label shows e.g. `strix-llama (06a64c3)` when the engine commit is available.
9. Load Config shows executable path, normalized known options, model/draft/mmproj paths when available, and unclassified engine-specific options.
10. Full Launch Command reproduces the sanitized argv captured at load time rather than current settings.
11. Concurrent loaded models cannot cross-attribute runs/configs; child-port/session correlation is tested.
12. The `router.out.log` `[N]` prefix is treated as child port, never as OS PID.
13. Port reuse after a session closes creates a new LoadConfig/session and cannot attach new Runs to the old config.
14. PID/process identity, when available, is auxiliary validation only and is not required for normal timing-line attribution.
15. A timing line whose child port has no active unambiguous LoadConfig is skipped rather than guessed.
16. Retention is 1,000 completed Runs per model; UI can request 10/25/50/100.
17. Reset stats clears both aggregate and detailed history and re-baselines ingestion.
18. Log rotation/truncation and partial lines do not duplicate or corrupt history.
19. Existing installations with old `stats.json` continue to start without manual migration.
20. vLLM aggregate Stats continue working even though vLLM detailed Run History is out of scope.

## Tests to add

At minimum cover:

- parse a complete PP/TG/total timing block into one Run
- parse MTP acceptance and mean length
- non-MTP run yields nullable MTP fields
- regression: `spawning ... on port 35409` followed by `[35409]` timing/MTP lines produces one Run with the correct LoadConfig
- assert explicitly that the `[35409]` prefix is not treated as PID 35409
- two child ports interleaving output are attributed to different LoadConfigs
- same model reload creates a new LoadConfig and old Runs keep the old reference
- child-port reuse after session close creates a new session and does not reuse the old config
- optional PID/process-identity validation can fail/be unavailable without breaking otherwise unambiguous child-port correlation
- engine name + 7-char commit display, plus missing-commit fallback
- unknown engine-specific args survive in `other_options` and Full Launch Command
- secret argv values are redacted before persistence
- retention pruning preserves referenced/active LoadConfigs
- reset clears history and prevents replay from already-consumed log data
- log truncation/rotation resumes without duplicate records
- API limit validation and model/config identifier validation
- frontend model expansion, 10/25/50/100 selector, Time tooltip, MTP tooltip, Load Config detail, and full-command expansion

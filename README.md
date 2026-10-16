# Stash AI

Automatic scene tagging, time-based markers and on-demand facial recognition for [Stash](https://github.com/stashapp/stash).

Stash AI supports four tagging models: JoyTag (`jt`), WD EVA02-Large v3 (`wd`),
DanbooruTagQuery L/16 (`dtq`), and OppaiOracle V1.1 (`op`). Models run
sequentially, with only one model loaded on the GPU at a time.

The default `merged` mode publishes shared `ai:` tags, combines compatible
model predictions, and retains per-model provenance and confidence in a local
SQLite database. Tagging input comes exclusively from Stash VTT files and
sprites. Explicit `recognition run` uses VTT faces and defaults to wider VTT
portraits; optional video portraits decode selected frames. Recognition has its
own face SQLite and configuration.
Use `recognition portraits --performer-id ID` to refresh one existing Stash
performer's portrait from retained evidence, or omit the filter to refresh all
eligible identities, without scanning scenes or rebuilding identities.

## License

Stash AI's original source code is released under the
[GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`).

You may use, modify, and redistribute it under the conditions of GPLv3.
Distributing modified versions or covered derivative works requires
providing the corresponding source code under GPLv3; private use does
not require publishing changes.

Third-party models and other upstream components retain their own
licenses. In particular, DanbooruTagQuery uses the separate DINOv3
License. See [Third-party notices](THIRD_PARTY_NOTICES.md) before
redistributing model assets.

## Requirements

- **Python 3.14.7** — the only supported and tested version;
- **Stash 0.31.1 or newer**, with generated sprites/VTT;
- Dependencies from `requirements.txt`;
- CUDA is recommended. CPU inference works but is substantially slower.

The Stash version is checked before reading scenes or changing server data,
including during `--dry-run`. Local commands such as `--db-update`,
`--db-check`, and `--db-report` do not require a Stash connection.
Failed or interrupted client startup releases acquired HTTP sessions/adapters;
secondary cleanup faults do not replace the original error or Ctrl+C.
Known intervals always publish an end time; occurrences and imported markers
whose end is unknown have `end_seconds: null`.

Both application profiles include tagging and on-demand facial recognition.
Install `requirements-gpu.txt` (also selected by `requirements.txt`) or
`requirements-cpu.txt`, with the matching constraints; keep exactly one ORT package.
Recognition defaults to CUDA. CPU installations use `models.device: cpu` in
`facial_recognition.yml`, or the command's `--device cpu` override.
See [facial recognition](docs/facial-recognition.md) for scope and configuration.

## Quick start

```powershell
pip install -r requirements.txt
# Pinned Python 3.14.7 environments: see constraints/README.md

$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<your Stash API key>"

python .\stash_ai.py --download-models --models all
python .\stash_ai.py --models all --scene-id 123 --dry-run --verbose
python .\stash_ai.py --models all --scene-id 123
```

Set `STASH_API_KEY` explicitly in your environment if your Stash instance
requires authentication. Without this variable, the client does not send an
`ApiKey` header. A historical embedded key has been removed; see
[SECURITY.md](SECURITY.md) for key-rotation guidance.

In `tagger_models.yml`, every selected model must explicitly define
`scene_threshold`, `marker_threshold`, and `batch_size`. Models are selected
using `--models` or the then-required `enabled_models` setting; missing
values do not fall back to legacy JoyTag defaults. Optional
`scene_cache_floor` and `marker_cache_floor` control evidence retention.
Resume reuses matching cached results, but the mere presence of published
tags or markers does not bypass inference.

## Common operations

```powershell
# Normal inference; matching cache entries and checkpoints are reused automatically.
python .\stash_ai.py --models all

# Force fresh inference for a selected scope.
python .\stash_ai.py --models wd --scene-id 123 --restart

# Re-merge stored results without running any models.
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Parallel read-only evidence preparation; Stash publication remains sequential.
python .\stash_ai.py --merge-only --merge-workers 4
# --dry-run and --profile-merge intentionally use the sequential path.

# Apply new thresholds to retained cache without running models.
python .\stash_ai.py --models all --merge-only --rethreshold-markers
python .\stash_ai.py --models all --merge-only --rethreshold-scenes --rethreshold-markers

# Clean up excluded results.
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only

# Repair groups of existing markers.
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run

# Inspect the locally recorded evidence for a tag.
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123

# Remove unused tag definitions.
python .\stash_ai.py --prune-empty-tags-only --dry-run

# Inspect, migrate, validate, and compact the local state database.
python .\stash_ai.py --db-report
python .\stash_ai.py --db-check
python .\stash_ai.py --db-update --dry-run
python .\stash_ai.py --db-update
python .\stash_ai.py --db-compact --dry-run
python .\stash_ai.py --db-compact

# Create a local anonymous diagnostic report for optional manual sharing.
python .\stash_ai.py --anonymous-diagnostics .\stash-ai-diagnostics.json

# Model utility and leave-one-model-out ablation without inference.
# Up to four CPU processes by default; --model-report-workers 1 forces sequential execution.
python .\stash_ai.py --model-report .\model-utility.json
python .\stash_ai.py --model-report .\model-utility.json --model-report-workers 8
python .\stash_ai.py --model-report .\model-utility-scene-123.json --scene-id 123

# Tags worth reviewing: reasons, examples, and per-model counts.
python .\stash_ai.py --tag-review-report .\tag-review.json
python .\stash_ai.py --tag-review-report .\tag-review-scene-123.json --scene-id 123

# Cache completeness against YAML and a plan to backfill missing scene/model pairs.
python .\stash_ai.py --cache-report .\cache-coverage.json
python .\stash_ai.py --cache-report .\cache-coverage-all.json --models all
python .\stash_ai.py --cache-report .\cache-coverage-scene-123.json --models wd --scene-id 123

# Abandon an unfinished merge/cleanup plan.
python .\stash_ai.py --abandon-pending
```

Threshold changes write small rows to the `result_thresholds` table;
compressed inference results and frame-level evidence remain unchanged.
This also applies when changed YAML thresholds trigger automatic cache reuse.
Reading results, merging, cleanup, and tag explanation use the effective
thresholds from that table. Each model/output pair has one active result and
an optional numeric `result_thresholds.threshold`. Databases predating v18
must first use the upgrade bridge described below. Supported schemas use
the single-record/numeric-threshold representation.
`--dry-run` stores thresholds only in a temporary database.

Merge snapshots store shared `extra_support` evidence once and reference it
from contributions, reducing serialization and compression costs.
The current database schema is **v27**, with upgrades supported from **v18**.
New snapshots use format envelope **v7**. `--db-update` converts snapshot
formats v4/v5/v6/v7, including pending and recovery data, in one pass,
preserving interval geometry and publication ownership, normalizing frozen
replay settings, and removing `extra_ids` and redundant
`algorithm: temporal` fields. Full pre-format-5 inference payloads are
retired; dependent publications receive `requires_rebase` and need fresh
inference followed by `--merge-only`. The format-5 evidence archive remains
available for exact reconstruction of published results. Checkpoint-analysis
versioning affects incomplete inference only: obsolete checkpoints are
ignored, and inference restarts that model/scene from the beginning.
Completed format-5 results retain evidence and signatures; older active
results must be recomputed. Dry runs change only a temporary database.

`--explain-tag` outputs scene reports with `format: 2`. Each item in
`recorded_sources` has one `record` object containing `meta`,
`thresholds` and `detections`, rather than the former one-item
`records` array.

To see every option:

```powershell
python .\stash_ai.py --help
```

## State and cache behavior

Before skipping a merge based on cached state, Stash AI refreshes the scene's
current state in Stash. SQL reads are batched; if another automation changed a
marker after the initial scan, a fresh merge is triggered.

Inference identity includes SHA-256 hashes of the model files and JoyTag
normalization (`JOYTAG_CLIP_MEAN` and `JOYTAG_CLIP_STD`).
Weight changes are detected even when file size and modification time are
unchanged. Each model's files are hashed once per run; after downloading
missing assets only the new files are hashed. The first run following this
identity change will refresh older inference results and checkpoints.

The default `tagger_sources.sqlite3` database stores raw model evidence,
checkpoints, published snapshots, and metadata for safe resume.
**Back up this file.** It contains neither images nor the API key.

In `merged` mode, the default resume behavior:

- skips completed inference when the model and input identities match;
- resumes a compatible partial checkpoint;
- can re-run merging without a GPU after publication-policy changes;
- uses retained observations after threshold changes when their coverage
  is sufficient.

`--restart` forces fresh inference for the selected models/scenes. Do not
use it merely to re-run merge, cleanup, or threshold evaluation.

### Starting a new database with existing AI results

`--db-init` can rebuild state after the database is lost, or start a fresh
database while retaining existing Stash results until each scene finishes
inference:

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3
```

A new or empty database and both output types (scene tags and markers) are
required. The command adopts existing `ai:<label>` tag assignments and
application-owned markers. After inference succeeds for all selected models
it removes outdated AI results. Missing VTT/sprites or model errors leave
the scene's previous results intact. Manually assigned `ai:` tags are
also adopted, except category and reserved marker tags.

Resume an interrupted run with the same command **without** `--db-init`,
keeping the model selection, database path, and any `--scene-id` scope.
Repeating the flag also preserves progress. Dry run uses only a temporary
overlay and writes neither to Stash nor to the source database.
See [database initialization](docs/UPGRADE_db_init.md).

### Evidence retention independent of publication thresholds

Optional `scene_cache_floor` and `marker_cache_floor` settings in each
model's `tagger_models.yml` section retain weaker predictions for later
threshold adjustments without a GPU. They **do not** lower the thresholds
for publishing scene tags or markers:

```yaml
wd:
  scene_threshold: 0.80
  marker_threshold: 0.85
  scene_cache_floor: 0.20
  marker_cache_floor: 0.25
```

Values must be in the range 0–1. Omitting a field or setting it to `null`
retains the previous automatic retention behavior. The effective persistence
floor is the minimum of the configured value and the support threshold
required by the current publication policy. A higher configured floor
therefore does not discard observations needed for publication.
The scene cache also stores exact aggregates for weaker labels.

After fresh inference, lower the publication thresholds in YAML and run
`--merge-only --rethreshold-scenes --rethreshold-markers`, provided the
persisted retention floors cover the newly required support thresholds.
Changing retention itself during a normal run triggers fresh inference and
invalidates a partial checkpoint; `--merge-only` cannot recover predictions
that were previously filtered out. Lower retention floors increase database
size and cache-processing cost. These settings affect persistent inference
evidence, not publication thresholds.

An active cache requires explicit retention floors and complete analysis
format 5. `--db-update` retires older inference; run inference again before
merge or cleanup. Overriding publication thresholds cannot restore
observations that were previously discarded.

The rethreshold `Check` and `Apply` phases read compact
`result_metadata` rows in SQLite instead of decoding frames from
`results.data`. New results write this metadata atomically with evidence;
older results populate it on the first check. Each model and result kind
has one active record and an optional numeric publication threshold.
Replacing or removing raw results invalidates their metadata via a trigger,
including writes by older program versions. `Apply` stores thresholds in
`result_thresholds` and updates `result_index` using metadata: result
presence and the current record's most recent inference signature.
When metadata is available, neither phase decodes frame BLOBs.
Dry run only populates a temporary database; `--db-check` verifies
existing metadata against evidence without repairing it.

New inference always replaces the previous result for the given model and
output kind. Upgrading from v18 retires incomplete results stored in older
analysis formats and schedules reinference. Publications retain ownership
and geometry, with required evidence pinned in the archive. Normal merging
and cleanup require native intervals; `OccurrenceTag` points remain points.
Legacy ordinary AI points, YAML v1 hierarchies, and managed `jt:`, `wd:`,
`dtq:` and `op:` prefixes require the
[transition-release migration tools](#migration-tools-from-the-transition-release).
Current code only publishes the `ai:` namespace and does not convert
legacy per-model objects. Manual associations with old tag definitions
are preserved.

### Migration tools from the transition release

Three completed migrations are no longer included in the current package.
If needed, run them from a separate checkout of immutable commit
[`9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
The current package uses schema v27. Databases older than v18 first need
the earlier upgrade bridge described in the database maintenance section.
Back up the database and configuration, and install the checkout's
dependencies in a separate Python 3.14 environment.

```powershell
git clone https://github.com/StormThatComes/stash_ai stash_ai_migration_bridge
git -C stash_ai_migration_bridge checkout --detach 9fcaa52502cd487455457547860b5db7d8710ad8
cd stash_ai_migration_bridge

# Convert custom v1 hierarchies; specify all hierarchy files in use.
python -m stash_ai.commands.hierarchy_migration C:/stash_ai/joytag_hierarchy.yml C:/stash_ai/wd_hierarchy.yml C:/stash_ai/dtq_hierarchy.yml C:/stash_ai/oppai_hierarchy.yml --dry-run

# Legacy AI points: derive end_seconds from published/pending/recovery state.
python -m stash_ai.tools.point_migration --state-db C:/stash_ai/tagger_sources.sqlite3 --dry-run

# Legacy prefixes: rerun inference and publish in the shared ai: namespace.
python -m stash_ai.tools.namespace_migration --models all --state-db C:/stash_ai/tagger_sources.sqlite3 --model-config C:/stash_ai/tagger_models.yml --temporal-policy C:/stash_ai/temporal_policy.yml --marker-core-exclusions C:/stash_ai/marker_core_exclusions.yml --scene-tag-exclusions C:/stash_ai/scene_tag_exclusions.yml --hierarchy C:/stash_ai/joytag_hierarchy.yml --wd-hierarchy C:/stash_ai/wd_hierarchy.yml --dtq-hierarchy C:/stash_ai/dtq_hierarchy.yml --oppai-hierarchy C:/stash_ai/oppai_hierarchy.yml --dry-run
```

Replace `C:/stash_ai` with the absolute path of your existing installation.
Use the same `STASH_URL` and `STASH_API_KEY`, and preserve model selection,
policies, and output scope. If hierarchies are disabled, replace the four
hierarchy options with `--no-hierarchy`. Once the plan is verified, repeat
the relevant commands without `--dry-run`. YAML conversion saves originals
as `*.v1.yml` and retains group assignments and source metadata.
Point migration does not load models, and retains IDs, titles, and manual
tags; occurrences remain points. Prefix migration preserves manually
managed objects and marker extras; matching markers keep their IDs.
Its scope can be limited with `--scene-id`, `--markers-only`, or
`--scene-tags-only`. Finish an interrupted prefix publication with the same
migrator and configuration before returning to the current version.
Pending snapshots remain available for API failure recovery.
Then return to the current installation and perform `--db-check` plus
any required inference.

### Database health, upgrades, and compaction

`--db-update` upgrades the database selected by `--state-db` from
schema **v18 or newer** to the current **v27**, across all scopes.
Normal execution and `--dry-run` require the current schema; new databases
start on v27. Migration adds database-rebuild state but does not adopt
existing Stash results. Adoption is authorized only by an explicit
`--db-init` on a new/empty database.

The updater uses the regular writer lock and runs offline, without Stash or
model configuration. It retains numeric thresholds on current results,
publication ownership, pending publication/cleanup and recovery state.
Older results are marked for reinference and their checkpoints removed.
Snapshot versions v4/v5/v6/v7 are converted in one pass, preserving shared
evidence. Retired algorithm settings are removed while geometry and evidence
references stay intact. Updating cannot reconstruct lost detections.
Pre-format-5 inference requires a rerun.

Result metadata is populated after conversion. Frames in active results
must be chronologically ordered. For databases predating v23, unordered
records are marked for reinference and dependent publications receive
`requires_rebase`. Stored payloads are not sorted or re-signed; ordered
format-5 results remain valid. Reads and writes reject missing provenance,
threshold and retention-floor metadata rather than silently dropping votes.

Normal cleanup accepts only the current fingerprint. An interrupted cleanup
with a legacy fingerprint requires an explicit offline conversion using
**the same YAML files and output scope** (`--cleanup-only`,
`--cleanup-markers-only`, or `--cleanup-scene-tags-only`). The tool validates
every per-model policy, updates only the pending key, and retains the
generation, masks and recovery state:

```bash
python -m stash_ai.tools.cleanup_pending_migration --state-db /path/to/tagger_sources.sqlite3 --cleanup-only --marker-core-exclusions marker_core_exclusions.yml --scene-tag-exclusions scene_tag_exclusions.yml --dry-run
```

Remove `--dry-run` after checking the plan, then resume the original cleanup.
The migration tool neither contacts Stash nor loads models.

For databases **older than v18**, first use compatible commit
[`c5bcd887bafe76b7ee2a9de944666799c9a0fc30`](https://github.com/StormThatComes/stash_ai/commit/c5bcd887bafe76b7ee2a9de944666799c9a0fc30)
in a separate directory with its dependencies. Back up the database and
point the bridge at the same state file:

```bash
git clone https://github.com/StormThatComes/stash_ai stash-ai-upgrade-bridge
git -C stash-ai-upgrade-bridge checkout c5bcd887bafe76b7ee2a9de944666799c9a0fc30
python stash-ai-upgrade-bridge/stash_ai.py --db-update --state-db /path/to/tagger_sources.sqlite3
```

Then run `--db-update` from the current installation. The compatibility
commit converts historical series/imports to v20, which the current
release can upgrade directly.

`--db-update --dry-run` reports source and target schema versions without
migration, file writes or locking. Databases older than v18 receive an
upgrade-path message; foreign or newer schemas are rejected. Payload
conversion, retiring active results, and updating the version execute
within one transaction. Missing-table creation is idempotent, and an
interrupted upgrade can be retried.

`--db-check` performs a deep, read-only consistency check across every
database scope. On older schemas, it only checks the SQLite integrity
and schema preflight, and requires `--db-update` before deep validation.
In addition to a full SQLite `integrity_check`, it validates:
- `results` against `result_index`, including recorded last-inference signatures;
- decoding of persistent BLOBs and valid model/output kinds;
- checkpoint/frame/analysis relationships and checkpoint dimensions;
- cleanup/recovery generations;
- required scene, frame and marker-interval fields, tags, times,
  probabilities and saved temporal statistics;
- checkpoint frames, including `scene_tags`, and float32 values
  (rejecting `NaN`, infinities and values outside `[0, 1]`).

Active results must satisfy the current inference contract. Legacy
intervals and incomplete statistics are only accepted in archived evidence.
Imports and `score: null` assessments are rejected. Retired
`legacy_imports` and `refinement_progress` tables are flagged without
reading their payloads. Some missing snapshot/recovery associations are
warnings. Pending publication/cleanup and refresh requests are treated as
in-progress work, not automatically as errors.

The check does not connect to Stash, load models, repair data, or modify
anything. A progress bar displays the current phase (e.g. SQLite integrity,
results/index, state payloads, checkpoints). Exit code `0` indicates no
logical errors, `1` indicates inconsistencies; warnings alone do not fail it.

`--db-report` needs neither Stash nor models. It reports schema version,
SQLite `quick_check`, DB/WAL/SHM size, free pages, result counts grouped by
model/output kind, and checkpoint, refresh, pending/recovery and merge-check
counts for the current scope. It does not print tag names, scene IDs or scores.

`--db-compact` runs SQLite `VACUUM` under the normal writer lock. It first
checks database integrity, free disk space, and the absence of unfinished
merge/cleanup operations. `--dry-run` runs only the preflight and estimates
space reclaimable from unused pages. Recovery evidence and checkpoints
are preserved.

### Anonymous developer diagnostics

`--anonymous-diagnostics NEW.json` creates a local report for manual
inspection and, optionally, sharing with a developer. The application
**never uploads it automatically** and refuses to overwrite an existing file.

The report uses `format: 2`. For v18–v20 databases, it reports basic
database information and `database.upgrade_required: true` without decoding
evidence or computing detailed statistics. Generate a full report after
`--db-update`. Legacy modes, provenance and analysis formats are identified
as `other`, without dedicated fields for historical intervals or
`scene_frames_format`.

The report covers the entire selected state database but exports no scene
IDs, `scope` values, URLs, file paths, tag names, scores, marker timestamps,
signatures or input hashes. Unrecognized model/output names become `other`;
custom tables are reported only by count.

For schema optimization, the report includes:
- Python/SQLite versions and operating-system family;
- page size/count, freelist, DB/WAL size, number of scopes and scenes;
- record counts for known tables and BLOB-size distributions;
- per model/output kind: compressed JSON size percentiles, compression
  ratio, items per record, detections per frame and scene observations;
- aggregate checkpoint, result-index and refresh-request state.

Although it removes direct identifiers and content, exact counts and size
aggregates can still characterize a particular installation. Review the
generated JSON before sharing it.

### Run history and profiling report

Inference, merge, cleanup and state-writing commands automatically record
run history in a separate `<state-db>.runs.sqlite3` file. Set
`--run-history-db PATH` to choose another file or `--no-run-history` to
disable recording. Run history is independent of cache and does not alter
the cache schema, signatures, or resume rules. Back it up together with
state if you need historical comparisons.

Inspection and reporting commands do not record history by default. Use
`--record-run` to profile, for example, `--model-report`, `--cache-report`
or `--db-check`. `--dry-run` and programmatic `DRY_RUN` never write history,
even with `--record-run`. History recording does not force sequential
merges or model reports: CPU workers return their aggregates to the parent.

```powershell
python .\stash_ai.py --models all
python .\stash_ai.py --merge-only
python .\stash_ai.py --model-report .\models.json --record-run
python .\stash_ai.py --run-report .\runs.json
python .\stash_ai.py --run-report .\wd-runs.json --run-model wd --run-command inference --run-status finished --run-limit 100
```

`--run-report NEW.json` reads only the history database, prints a summary,
and creates a new JSON file. It does not need the cache database, models,
or Stash. Filters cover command, status and model; a limit of 1–10000
selects the newest matching runs (50 by default). Existing report files
are not overwritten.
Malformed counters, outcomes and elapsed values are excluded with bounded
structured warnings. Affected records stay in separate unverified comparison
groups; usable runs and weighted metrics remain available, and unknown elapsed
values are reported explicitly. Reading a report never repairs its source DB.
History initialization/upgrades serialize with an explicit SQLite transaction;
tables, indexes and version stamps commit together. Known missing history
features can be restored by a writable open while existing rows/proofs remain;
read-only access rejects an incomplete current schema with a maintenance error.

Legacy runs without a configuration fingerprint remain visible in isolated,
explicitly incomparable groups. Invalid metric entries are excluded from the
export and aggregates, with warning counts and at most 20 diagnostic examples;
the stored history is unchanged. Reconciliation leaves non-object or malformed
owner configurations unverified while continuing other selected run IDs.

Each run records status, wall time, parent CPU time, configuration hashes,
worker settings, models and revisions, available weight/vocabulary hashes,
change and error counts, and stage/API timing. API keys, media URLs,
payloads, and error contents are not recorded. This is **not** an
anonymous report: run IDs, dates, configuration and aggregates describe
your installation. Run start is saved before execution; Ctrl+C and errors
finalize the record. Partial aggregates are flushed at stage boundaries,
every 30 seconds, or after 32 completed model/scene/merge work units.
Throttled checks read only those unit counters; a full snapshot is built when
an update is due, explicitly forced, or the run finishes (including Ctrl+C).
`last_progress_at` and `progress_seq` identify the most recent update;
`partial=true` means a run did not finish and is not counted as a
successful complete run. `running` means no end was recorded: the process
may still be running or may have been killed. History-write failures
neither stop the real command nor mask its errors.

The report includes available timings for scenes, inference batches,
frame preparation, checkpoints, cache, rethreshold, merge, publication
and API calls. Fresh inference, resume and cache reuse are distinguished.
Metrics include call counts, total/mean/min/max durations, approximate
p50/p95 from bounded histograms, and throughput where work-unit counts
are available. Mean is total time divided by calls, including when
aggregating runs. Comparisons group compatible configurations, environments,
and model identities.

Configuration includes a hash of the sorted set of selected scene IDs
(order and duplicates do not change the hash), code hash, available Git
revision, and installed library versions. `execution_observed=false`
means the model's execution device was not observed, for example when
cache was reused. Verified inputs have a separate workload identity:
pair counts, cue counts and input fingerprints, used for grouping.
These diagnostics do not authorize cache reuse or change stored prediction
identities. The report separates completion statuses and successful/failed
operations.

Stage timings include inclusive time and self time. Threads/processes
may overlap, so summed stage times are not elapsed run time.
`cpu_seconds` covers the parent; `worker_cpu` covers measured worker tasks.
A failed worker task has `outcome=error`, increments `errors` and
does not count as a completed scene (`units=0`), even when delivered
to the parent as a result. Merge continues with other scenes after
such errors; the model report indicates failure.

Histograms have bounded size; history grows by one record per run without
storing every timing sample. Filtered/limited reports summarize only
selected runs, not necessarily all of history.

### Why was a scene recomputed?

Schema v27 adds verified inference identity to the state database.
Run `python .\stash_ai.py --db-update` before opening an older database.
Migration preserves original result BLOBs, checkpoints and frozen proofs;
it does not run any models. On the first compatible read, the CLI can
persist missing identity data without rerunning inference.

Subsequent changes to `batch_size`, relocation of identical model weights,
or less restrictive retention can reuse the same scores when input
verification and cache-coverage checks succeed. A new inference with a
different batch size may have minor numerical differences; reuse preserves
the existing predictions, not bitwise equivalence with a fresh run.

For old cache without recorded inference identity, make one compatible run
with the original configuration **before** relocating model files or
changing batch size/retention. When the old identity cannot be proven,
the CLI will not infer it from diagnostic history. Weight/input changes
or missing required scores can still trigger inference for specific
scene/model pairs. Compatible checkpoints can be resumed after changing
batch size as long as retention policy remains compatible.

Normal inference logs a decision for each scene/model pair:
`infer`, `cache_reused`, `rethreshold`, `checkpoint_resumed` or
`blocked`. Use `--verbose` to display decisions and reasons while running.
A separate offline report reads those decisions, prints per-model totals
and creates a JSON file:

```powershell
python .\stash_ai.py --reuse-report .\reuse.json
python .\stash_ai.py --reuse-report .\wd-scene.json --models wd --scene-id 123
python .\stash_ai.py --reuse-report .\recent-reuse.json --run-limit 20 --run-status finished
```

By default, the report uses the latest matching run with inference decisions.
`--run-limit` selects 1–10000 runs. Model (`--models` or `--run-model`),
scene (repeatable `--scene-id`), command and status filters apply before
the limit. A later merge or maintenance command without inference
decisions does not hide the last relevant report. This report describes
**recorded executions**, not the current files or current Stash media.

Reasons include missing evidence, explicit restart, disabled resume,
pending refresh, incomplete index, and changes to model/revision/weights,
vocabulary, batch size, aggregation, retention, thresholds, VTT or sprite.
When the previous identity is available, JSON includes before/after
values, cue counts and the median spacing between cue start times.
Insufficient cache coverage includes the stored floor and required
retention; checkpoint decisions include the resumed frame count and
acceptance/rejection reason. Multiple reasons may apply to one decision,
so their counts need not sum to the scene count.

Older cache without history still works. Compatible reuse can add its
diagnostic identity without inference. Missing historical data is
identified by `historical_identity_unavailable` and generic
`input_changed` or `model_or_configuration_changed` reasons.
Diagnostics do not change cache signatures, schema or reuse rules.
Format-1 history can be read unchanged and upgrades to format 2 on
the next write.

History records scene/model IDs, reasons, settings and input hashes,
but not media URLs or API keys. Decision storage grows with
scene/model/run pairs; prior identity uses one record per
scene/model/output kind. Writes are buffered into short transactions
of up to 64 operations. Normal completion and Ctrl+C flush the buffer;
a hard process kill may lose the latest unflushed history operations.
`--no-run-history` disables recording; reports require existing data.
Without `--verbose`, diagnostic proof construction is skipped too,
while input verification and cache identity checks remain active.
Failures of optional GPU metadata or run history do not change
command outcomes or inference-reuse decisions.

### Inference progress

In an interactive terminal, inference shows two progress bars: scenes
for the current model, and frames within the current scene, e.g.
`wd scene 6306: 128/729 frame`. Frame progress counts VTT cues
selected for inference, **not** every video frame. Resumed processing
starts after the restored checkpoint and advances after CPU handling
of each batch, even if no tags were retained. Prefetch or inference for
the next scene does not advance the current scene's frame bar.

After the frame bar reaches 100%, result finalization and saving may
still be in progress. The scene bar advances only after scene handling
completes. Cache-only scenes do not show a frame bar. Non-interactive
terminals hide the per-frame bar to avoid cursor-motion log artifacts.

Two buffer-occupancy lines appear below the progress bars, for example:

```text
Buffers: in=ready=12 in-flight=2 occupied=14/17 out=ready=5 in-flight=2 occupied=7/34 | next=off
Cache: 1.2GiB/24.0GiB | checkpoint interval=7/32 batch
```

`ready` counts complete batches waiting in the queue. `in-flight` counts
scheduled/active preparation or batches currently owned by the producer and
consumer, including a producer waiting for a full queue. `occupied` is their
sum. Input futures have `prefetch_batches` slots plus one consumer-owned batch;
streamed next-scene inputs and GPU results have their configured queue slots
plus one producer and one consumer slot. These are batch capacities, not byte
budgets. A batch transferring into or out of a ready queue is counted once.
`next[ID]` shows separate input and result buffers for the next scene. End
markers, failed/empty preparation and errors are excluded. Counts reset when
streams finish or close. With `prefetch_batches: 0`, queues show `off`; without
a next scene, they show `next=off`.

`Cache` is estimated retained memory in the shared RAM cache relative
to its effective configured limit, after applying `ram_budget_gb`.
It covers shared media, input, result and index caches, **not**
the process's total RAM consumption or VRAM usage.
`checkpoint interval` is processed batches since the last successful
checkpoint divided by `checkpoint_batches`; it is not queue capacity.
The counter resets after a checkpoint is written.

Occupancy is sampled whenever frame progress refreshes (by default
no faster than every 0.25 seconds), and around finalization/scene
persistence. Samples do not wait for producers, consume queue entries,
query SQLite or update LRU state. Extra lines are hidden outside an
interactive terminal and cleared on errors or Ctrl+C.

Read-only merge and model-report workers return ordered chunks of at most eight
scenes. Compatible profiling histograms and model-report counters are combined
once per chunk; merge evidence, diagnostics and errors retain their scene IDs.
The parent rejects missing, extra, duplicate or mismatched scene results before
using an affected chunk. Only the parent publishes or writes run history.

### Model utility / ablation report

`--model-report NEW.json` works entirely from saved evidence, without
Stash connectivity or model inference. It first reconstructs the normal
result across all recorded models for each scene, then computes four
counterfactual `all minus one model` results. Omitting a model
recalculates consensus seeds, temporal rules, shared scene tags,
marker extras and Occurrence from scratch; it does not merely filter
the resulting markers.

Per-model metrics include:
- scenes with retained evidence and scenes whose result changes when
  that model is omitted;
- final scene tags and marker components supported by the model;
- scene tags, markers, extras and Occurrence points lost without it;
- surviving markers with changed intervals or extras;
- the share of evidence-bearing scenes with a measurable marginal impact.

A change only in the marker's provenance badge, e.g. `[jt+wd]` to
`[wd]`, is not counted as a semantic change. The report measures
the effect on tagging, geometry and extras rather than merely
model attribution. Use `--scene-id` to restrict the sample.

### Tag review report

`--tag-review-report NEW.json` examines stored scene tags and primary
marker labels by replaying merge with effective database thresholds.
It uses the selected temporal policy and exclusion rules. It does not
load models, retrieve media, connect to Stash, modify source state,
or publish changes.

Scenes whose `input_checks` state is `missing_media`, `read_error`
or `deleted` are skipped with `unverified_inputs`, even when an
unselected model caused the problem on that scene. The report keeps
the stored evidence and includes the reason in its examples.

Review priorities are determined by the following heuristics:

- `weak_or_borderline`: best score is no more than
  `--tag-review-margin` (default `0.05`) above the relevant threshold.
  For scene tags qualifying via `strong_peak`, the strong-peak
  threshold applies; otherwise use that scene/marker's own threshold.
  Weak results accepted through temporal or consensus rules can
  also be flagged.
- `isolated_peak`: a scene tag qualified through a strong peak
  but only one sample exceeds the scene threshold, with at least
  two samples present.
- `model_disagreement`: another model retained the same label for
  identical input but did not qualify it under its own scene rules.
  Missing models/labels are **not** negative votes, and excluded
  labels are ignored for comparison.
- `long_marker`: a merged interval is at least
  `--tag-review-long-marker-seconds` (default `300`) seconds long.

The terminal table and JSON `models` section show JoyTag, WD, DTQ,
and Oppai counts for distinct flagged labels, scene/tag pairs,
and scenes. Each model's scene/tag pair counts once, even when
several intervals exist, the tag is also a scene tag or marker core,
or it came from multiple recorded runs. Global counts deduplicate
across models; summing model rows does not give total distinct tags.
Reason counts may overlap. A model contributing to marker-core
consensus also counts even if it created no independent interval:
the supporting score is derived from compatible retained observations
within the contribution, and its example has `support_only: true`.

The JSON `tags` section contains model/label rows, reason counts,
and up to three example scenes with scores, thresholds and times.
Configure 1–20 examples using `--tag-review-examples`; counts always
include all analyzed scenes. Rows sort by affected-scene count,
then reason count. `--scene-id` narrows the analysis. Existing report
files are never overwritten.

Scenes awaiting refresh/publication or with unusable evidence are
skipped with explicit reasons instead of participating in comparisons.
The report shows **replayed cache output**, not live Stash state.
Heuristics are not proof of an error: a valid tag can have one peak,
a long interval or support from a single model. Model scores are
not averaged.

### Cache coverage report and backfill plan

`--cache-report NEW.json` checks scene and marker evidence for models
enabled in YAML or explicitly selected with `--models`. It runs
offline without loading models, vocabularies or media and without
connecting to Stash. It changes neither source state nor published
results and refuses to overwrite an existing report.

By default, it covers scenes known to the **current local database scope**,
including scenes only referenced by checkpoints, refresh requests,
snapshots, recovery or `input_checks`. Stash scenes never encountered
by this database are not counted. Repeatable `--scene-id` arguments
set an exact scope; IDs without any local state then count as missing.
The report does not validate current weights, vocabulary, VTT or sprites;
a normal inference run performs those checks.

The terminal table and JSON `models` list per-model scene/model pairs
that are ready, rethresholdable without GPU, requiring inference,
requiring rethreshold only, or blocked. `outputs` splits the counts
by scene/marker outputs and mutually exclusive statuses:

| Status | Meaning |
|---|---|
| `ready` | Completed inference, consistent index, effective thresholds, sufficient retention |
| `rethreshold` | Cache covers the YAML thresholds; only GPU-free rethreshold is required |
| `missing` | No result for this output kind |
| `partial` | No result, but a checkpoint exists for resume-time validation |
| `refresh_required` | Pending fresh-inference request invalidates old results |
| `insufficient` | No frame evidence, retention floor too high, or required labels previously filtered out |
| `inconsistent` | Result/index mismatch or unreadable evidence; investigate with `--db-check` |

Completed inference with zero detections is valid cache, not missing
evidence. `evidence_ready_pairs` describes coverage of persisted
results. Known deferred input checks block `fully_ready_scenes` and the
whole plan for that scene, even if they concern an unselected model.
`issues.input_checks` identifies the model and reason. Restore the media
or resolve scene deletion, then verify input during a normal run;
the blockage alone does not require `--restart`.

Retention is compared against the lower of the YAML floor and the
support threshold needed for publication. Exactly one active result
per model/output is checked. With populated `result_metadata`, the
report avoids decoding prediction BLOBs; missing legacy metadata is
filled only in a temporary database. Use `--db-check` for deep
payload validation.

`issues` identifies scene/model pairs, reasons, required thresholds,
blockers and checkpoint progress. `plan` provides ordered, grouped
**argument arrays (`argv`)** with at most 50 scenes per command:

- `resume_or_infer`: fill selected models/scenes; normal resume
  validates checkpoints and reuses matching completed results;
- `restart_inference`: forcibly replace unusable completed cache.
  Active missing/refresh states with checkpoints use normal resume
  to retain progress;
- `rethreshold`: apply both YAML thresholds without inference;
- `merge_after_backfill`: refresh model consensus and Occurrence last.

Commands preserve `--state-db` and YAML paths for models, temporal
policy and exclusions. The report **does not execute** its plan.
Manually running these commands can modify Stash; keep the same
`STASH_URL` and environment. Publication may remain blocked until
all models with pending refreshes have finished.

Pending publication, global cleanup, refresh for an unselected model,
unreadable evidence, or disabled replacement of unusable history
blocks all planned commands for an affected scene, including
backfilling other models and the final merge. Every command might
attempt publication. Finish the original operation or resolve the
inconsistency, then generate the report again. Evidence readiness
and publication blockage can overlap in aggregate counts.

Merge also uses retained evidence from **unselected models**.
The report checks their publication eligibility using stored effective
thresholds and the selected policy, without requiring those models'
current YAML retention settings. Incomplete or unreadable evidence
from such a model blocks the scene. `issues[].publication_dependencies`
identifies its model, output kind, reason and remedy. Include it in
`--models` and regenerate the report to plan repairs. Completeness
counts and inference targets still concern only selected models;
the report does not silently expand the inference scope.
`issues[].blocking_models` identifies selected models whose blockage
prevents planning other pairs for the same scene.

Exit code `0` means the report was successfully generated, even
when gaps remain. Archived evidence after `--abandon-pending` does
not block cache or inference. `publication_recovery_scene_ids`
lists affected scenes; the plan's final merge reconciles their
publication even with complete cache, provided no other
publication blocker exists.

## Results in Stash

In `merged` mode:

| Item | Example |
|---|---|
| Scene tag / marker core | `ai:blue_hair` |
| Marker title | `[AI] blue_hair [jt+wd+op]` |
| Ownership tag | `ai:marker:stash-taggers` |
| Occurrence point | Also has `ai:marker:occurrence` |
| Regular marker group | `m:blue_hair` |
| Occurrence group | `m:blue_hair:occurrence` |
| Marker group root | `ai_markers` |
| Hierarchy categories | `ai:category:...` |

Occurrence groups use `:occurrence`. The one-time spelling migration is
complete; current versions manage only the canonical names.
See [marker groups](docs/UPGRADE_marker_groups.md) for details.

New results always use shared `ai:` publication and temporal rules.
`--output-mode`, `--no-temporal`, and the YAML `enabled` key have been
removed; remove them from scripts and `temporal_policy.yml`. Individual
rules, including `occurrence_markers`, are still configurable via the
temporal-policy YAML. Historical per-model namespaces require the
separate migrator. Normal publication, cleanup and default pruning
manage current `ai:` results and `m:` groups. Legacy tag definitions
can be explicitly pruned with, for example,
`--prune-empty-tags-only --prune-prefixes jt: wd: dtq: op:`.
Manual extras from other namespaces are preserved.
Before deleting an unused tag, pruning refreshes its usage and requires the
response to identify exactly that tag. A malformed/mismatched ID stops pruning
before deleting that candidate; a tag already removed by another client is skipped.

Fresh inference writes complete analysis format 5: input and inference
signatures, explicit retention floors, and statistics or explicit
`null` for non-qualifying aggregates. `--db-update` retires older
active results requiring reinference. The pinned archive retains
exact publication evidence, and historical snapshot geometry and
ownership remain available for recovery and point migration. Cleanup
requires rebuilding retired publications via fresh inference and
`--merge-only`.

The new cleanup fingerprint always includes output scope and no
constant empty blocklist. Interrupted cleanup retains its generation
and masks. Old fingerprints require the explicit
`stash_ai.tools.cleanup_pending_migration` tool, which validates
the same exclusion policy and output scope. New operations store only
the current fingerprint.

## Configuration

All models share one `tag_hierarchy.yml` file: the union of the former
four hierarchies, containing 21,154 distinct tags across 92 groups.
`--hierarchy YAML` selects this shared file even if only one model is
enabled. `--wd-hierarchy`, `--dtq-hierarchy` and `--oppai-hierarchy`
have been removed.

The hierarchy uses `version: 2` and the `source` and `groups` fields.
Group paths directly form `ai:category:` categories under
`ai:category:AI`. Validation requires all tags of selected models to
be assigned; including tags from other models is fine. Optional
`source.tag_count` and `source.vocabulary_sha256` describe the
complete shared tag set independently of the selected models.
Reorganizing groups needs neither fresh inference nor a new SQLite DB;
use `--sync-hierarchy-only` to synchronize assignments.

If you customized the old hierarchy files, transfer those changes
into the shared YAML. The updater preserves legacy local configurations,
but current code does not read them. Convert custom YAML v1 files
using the [transition release](#migration-tools-from-the-transition-release).
The current validator requires v2 and directs users to the migration
commit without changing legacy files.

| File / option | Purpose |
|---|---|
| `tagger_models.yml` | Models, directories, thresholds and batch sizes |
| `temporal_policy.yml` | Temporal rules, consensus, extras and Occurrence |
| `scene_tag_exclusions.yml` | Global scene-tag exclusions |
| `marker_core_exclusions.yml` | Marker primary-tag exclusions |
| `tag_hierarchy.yml` | Shared category hierarchy for all models |
| `--state-db` | Persistent state database path |
| `--db-init` | Initialize a new/empty database and adopt existing AI results |

Publication has **no maximum number of scene tags or markers**.
All results qualifying under thresholds, temporal policies and
exclusions may be published, for individual models, combined results,
and Occurrence points. `MAX_TAGS` and `MAX_MARKERS_PER_SCENE` were
removed. The independent per-frame tag limit `MARKER_MAX_TAGS_PER_FRAME`
remains (default `0`).

Databases on schema v18–v26 require `--db-update`. Upgrading to v27
writes snapshot format v7 without the legacy output-count limits,
while retaining geometry, ownership and pending/recovery plans.
It neither modifies inference cache nor writes to Stash.
To regenerate previously truncated published results from cached evidence:

```powershell
python .\stash_ai.py --db-update
python .\stash_ai.py --merge-only
```

`--merge-only` applies the current publication rules without a GPU;
cache-completeness requirements still apply. Use `--dry-run` to preview
publication.

Current memory and parallelism defaults:

- `--ram-budget-gb 32`;
- `--ram-cache-gb 24` — lazy allocation, capped at 75% of the RAM budget;
- `--prepare-workers 4`;
- `--merge-workers 4` — parallel CPU evidence replay, with sequential
  Stash/SQLite writes;
- Global `checkpoint_batches: 8` in `tagger_models.yml`, or
  `--checkpoint-batches N`, saves progress every N processed batches.
  It must be an integer >= 1. CLI takes precedence, and the default
  remains 8 when omitted. Final checkpoints and saves on supported
  interruption/error paths are independent of this interval.
- Global `prefetch_batches: 4` in `tagger_models.yml` (0–128; zero
  disables prefetch). Explicit `--prefetch-batches N` takes precedence;
  omission retains the default 4. This controls preparation of future
  batches independently of inference `batch_size` and does not
  invalidate stored predictions.
- Global `result_prefetch_batches: 32` (1–128), or
  `--result-prefetch-batches N`, separately controls the CPU result
  queue. It applies when `prefetch_batches > 0`. CLI overrides YAML,
  and the default remains 32 when omitted. Changing it does not
  invalidate results or checkpoints.

With `prefetch_batches > 0`, one persistent thread loads a model,
runs inference for all its scenes, then closes it. A separate
backend thread is **not** created for every scene, so CUDA resources
and stream pools can retain a single owner. Meanwhile, the main
thread processes previous results and persists checkpoints/SQLite.
The result queue holds at most `result_prefetch_batches` batches;
one more result may be pending insertion and one inference call may
be in flight. Results are consumed in order and checkpoints cover
only fully processed prefixes. Failure in a later batch preserves
earlier progress. Interruption cancels scene preparation and waits
for inference, after which the model is closed on its owning thread.
Prefetched inputs and queued results occupy **RAM, not VRAM**.

A larger result queue allows inference to continue while previous
results undergo slower CPU post-processing, compression and saving.
This queue occupies RAM **outside the cache limit**. Float32 result
arrays alone use approximately
`result_prefetch_batches × batch_size × label_count × 4` bytes,
plus any in-flight result and frame metadata. The queue buffers
temporary stalls; if the consumer is consistently slower than GPU
inference, it will still fill up.

Prefetch also handles input for **one next scene**. Downloading
VTT/sprites, verifying their hashes, decoding images and building
CPU batches can overlap with inference of the current scene.
It uses a separate HTTP session and the shared RAM cache.
The media buffer retains at most two compressed sprites within
16 MiB; the shared cache still respects its memory budget.
The next scene's input queue holds `prefetch_batches` prepared
batches, potentially with one batch waiting to enter it and
a transformation window of up to
`max(batch_size, prepare_workers)` frames.

Both scenes share one `prepare_workers` pool, so the thread count
does not double. Input verification uses its own executor, and at
most two CPU batch producers have isolated HTTP sessions.
A full current-scene queue does not block fetching or preparing
the next scene. Each active input queue has a separate capacity:
current and next scenes can both retain batches simultaneously.
These are live CPU inputs and lie outside `--ram-cache-gb`.
Larger prefetch depth raises memory use proportionally to
`prefetch_batches` and `batch_size`; it does not increase the
number of look-ahead scenes or the GPU result-queue depth.
Scenes already eligible for reuse are not prefetched.

Read failures are surfaced only when their scene is reached,
so they do not interrupt saving previous results. The same
holds for next-scene decoding or inference errors.

After the final batch of the current scene, the main thread may
consume ready next-scene input, validate reuse, and restore its
checkpoint. Only then can it start that scene on the **same GPU
owner**, before finalizing, assembling and committing the previous
scene. Its output waits in the bounded RAM queue; checkpoints and
completed results are still written in scene order on the main
thread. If next-scene input is still downloading or verification
failed, the previous scene finalizes without waiting. Handoff
is attempted again before result assembly and commit if input
became ready in the meantime.

During resume, speculative CPU batches from the beginning of the
next scene are discarded; the GPU processes only the remaining
frames after the verified checkpoint. Ctrl+C also cancels
next-scene download and inference; unprocessed GPU results do
not extend the saved checkpoint.
`--prefetch-batches 0` disables look-ahead scene preparation too.

CPU preparation constructs complete inputs (stacking, masks and
fixed-size ONNX batch padding) before passing them to inference.
Transformation workers can process frames from multiple batches:
a batch size of 4 no longer restricts 8 workers to four simultaneous
tasks. Media/PIL reading continues to have one owner. Besides the
batch queue, the transformation window is limited to
`max(batch_size, prepare_workers)` frames. With prefetch disabled,
inference and result processing stay on the main thread with
the existing transformation pool.

Run history separates `sprite_frame_read`, `image_prepare`,
`batch_collate`, `inference_input_wait`, `inference_batch`,
`inference_result_wait`, `inference_result_queue_wait` and
`inference_postprocess`. A full result queue suggests that the
consumer cannot keep up; a long `inference_input_wait` suggests
input preparation is the bottleneck. `scene_input_prefetch`
measures background verification, while `scene_input_wait`
measures waiting for prepared next-scene input.
`scene_batch_prefetch` and `scene_batch_queue_wait` cover
CPU preparation and waiting for space in the next-scene queue.
`scene_handoff` measures authorizing and starting the next
scene before finalization of the previous one.
`inference_scene_finalize`, `inference_output_capture` and
`inference_commit` distinguish CPU finalization, result assembly
and persistence, to help locate pauses at scene boundaries.
On the synchronous path `input_verify` still includes reading
and hashing; with prefetch it covers receiving input, with
waiting tracked separately by `scene_input_wait`.

`inference_batch` includes transfers and validation, not just
CUDA kernel execution. Parallel stage timings may overlap and
must not be summed as wall-clock runtime. Pipeline changes
require neither database migration nor rejection of compatible
results/checkpoints.

Results and checkpoints use faster zlib level-1 compression.
Older BLOBs remain readable; faster writes may increase database
size. `state_json_encode` and `state_compress` track result
serialization; `checkpoint_json_encode` and
`checkpoint_compress` cover checkpoint preparation, including
top-score array compression. `checkpoint_prepare` includes
BLOB preparation before the transaction, `checkpoint_sqlite_write`
covers SQL and commit, and `checkpoint_write` includes both.

`--db-compact` retains the write lock until final database validation
and post-VACUUM counter comparison.

Parallel merge divides the RAM and cache budget between helpers
and the parent process responsible for publication. With four
helpers, each process gets one-fifth of the budget; once they
exit, the parent regains the entire budget. Model-report workers
use the same eligibility settings and limits as the sequential
report.

## Updating the application

`update_stash_ai.py` updates managed application files without Git
or `pip`. For this private repository it requires a GitHub token
with `Contents: read` access.

The current updater supports the package layout (`stash_ai/`).
Older installations using flat `stash_*.py` modules require a
one-time migration via the updater from
[commit `9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
Stop the application and download that version as a separate file
inside the installation directory:

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"
$bridge = "9fcaa52502cd487455457547860b5db7d8710ad8"
$headers = @{ Authorization = "Bearer $env:GH_TOKEN"; Accept = "application/vnd.github.raw+json" }
Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/StormThatComes/stash_ai/contents/update_stash_ai.py?ref=$bridge" -OutFile .\update_stash_ai_bridge.py
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout --dry-run
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout
```

If a legacy update was interrupted, first run
`python .\update_stash_ai_bridge.py --install-dir . --recover`.
The bridge installs the current package while preserving backups,
local YAML files, database and models; use `--force-code` for
locally modified code. Continue with the regular updater afterwards.
The current updater does not scan for or delete untracked legacy
modules. A `git pull` directly into the package layout does not
require the updater bridge.

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"

python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run
python .\update_stash_ai.py
```

The updater:
- downloads application files from a single `main` commit SHA;
- installs launchers, the `stash_ai/` package, YAML files, all four
  requirements files and CPU/GPU constraints (including their README);
- preserves locally modified YAML and writes `.incoming` proposals;
- refuses to overwrite locally changed code or dependency files without `--force-code`;
- does not update models, `.env`, SQLite or Python dependencies;
- uses an installation lock and recovery journal.

After updating, prepare or refresh the local environment manually using the
downloaded dependency files. Choose one application profile:

```powershell
python -m pip install -r requirements-gpu.txt -c constraints/gpu-py314.txt
# Or, for a CPU environment:
python -m pip install -r requirements-cpu.txt -c constraints/cpu-py314.txt
```

`requirements.txt` selects the GPU profile. `requirements-dev.txt` is available
for development and tests. The updater downloads these manifests but does not
run `pip` or modify the installed Python packages.

If the installed updater predates dependency-file downloads, run it once to
update itself, then run it again to fetch requirements and constraints.

After an interrupted update:

```powershell
python .\update_stash_ai.py --recover
```

## Security

- Use a newly generated/rotated `STASH_API_KEY` from the environment.
- Never commit GitHub tokens or Stash API keys.
- Read-only GraphQL requests and media GETs may be retried on transient errors.
- GraphQL **mutations are not retried automatically**, because the server
  may have applied a write despite a lost response.
- `--dry-run` writes neither to Stash nor to the persistent source database.

## Tests and CI

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

CI tests Python 3.14.7. The full suite runs on Linux, with extra
Windows coverage for the updater. Tests use mocked Stash and
models; a production server is not needed.

## Documentation

- [Current user guide](docs/README_multimodel.md)
- [Documentation index](docs/README.md)
- [SQL cleanup](docs/UPGRADE_cleanup_sql.md)
- [Performance and memory](docs/UPGRADE_performance.md)
- [Merge and fast path](docs/UPGRADE_merge.md)
- [Exclusions and cleanup](docs/UPGRADE_exclusions.md)
- [Occurrence](docs/UPGRADE_occurrences.md)
- [Marker groups](docs/UPGRADE_marker_groups.md)
- [Abandoning interrupted operations](docs/UPGRADE_abandon_pending.md)
- [Initializing a fresh database](docs/UPGRADE_db_init.md)
- [Merge diagnostics](docs/UPGRADE_diagnostics.md)
- [Benchmarks](benchmarks/README.md)

Files under `docs/validation/` are historical results from specific
validation and benchmark runs. They are not authoritative descriptions
of the current test count or application contract.

Configuration and CLI naming changes are documented in the
[exclusion migration guide](docs/UPGRADE_exclusions.md#renamed-configuration-and-cli-options).
Use `excluded_scene_tags`, `--models`,
`--scene-tag-exclusions` and `--cleanup-markers-only`;
the default marker-exclusions file is `marker_core_exclusions.yml`.

An active result is one analysis-format-5 record with an optional
numeric publication threshold. New inference replaces the record and
resets that threshold. The archive retains only specific proofs pinned
by published/pending/recovery state; ordinary merge, rethreshold, reports
and completion indexing do not read historical segments.
`--db-update` preserves geometry, ownership and recovery, and schedules
incomplete legacy results for reinference. `--merge-only` requires
fresh evidence before cleanup. A missing `end_seconds` in recovery is
an exact point, not an interval.

### Deferred input verification (schema v26)

Run `python stash_ai.py --db-update` once for existing databases.
This adds a small input-verification table without changing raw inference
records or publication snapshots. Missing VTT/sprite paths defer
publication and retain previous predictions. After restoring identical
inputs, `--resume` verifies and reuses compatible results instead of
forcing inference. Model-utility reports exclude scenes awaiting input
verification or fresh inference and include skipped counts.

Explicit missing scene IDs are reported separately. During resumed
`--db-init`, a successful Stash read confirming scene deletion retires
that scene so remaining scenes can complete. Network/read errors
cannot authorize retirement.

For a detailed cache-based publication preview, use
`--merge-only --dry-run --dry-run-plan review.jsonl`.
Event details and symbolic IDs are described in
[the dry-run plan](docs/dry_run_plan.md).

### Run history retention

`--prune-run-history` affects only the separate run-history database,
within the current `STASH_URL` scope. Provide
`--history-keep-last N` and/or `--history-max-age-days DAYS`.
Combined limits delete a finished run exceeding either limit;
`N=0` removes all finished runs in the scope. Preview affected
run/decision counts and example IDs using `--dry-run`:

```powershell
python .\stash_ai.py --prune-run-history --history-keep-last 100 --history-max-age-days 90 --dry-run
python .\stash_ai.py --prune-run-history --history-keep-last 100 --history-max-age-days 90
```

Run and decision deletion is one transaction. `running` records,
records without completion, other scopes and `reuse_proofs` remain.
The source database's results, checkpoints and snapshots are never
opened. History formats v1/v2 can still be read without mutation;
the next write upgrades only run history to v3. Pruning is
opt-in and does not run `VACUUM`.

### Republishing dependencies

The merge fingerprint covers modules responsible for replaying
retained evidence and publishing tags/markers, along with the
effective policy, publication options, exclusions, hierarchy
and observed catalog/scene. Changes limited to backend, checkpoints,
input preparation, prefetch, worker settings or history do not by
themselves invalidate a completed publication because a source
file changed. New results and actual scene/policy changes still do.

The dependency manifest is conservative at the module level:
changing a combined storage or domain module may still require
rechecking merge. A one-time manifest change rechecks existing
merge checks against stored results, without altering inference
identity or result/checkpoint/snapshot formats.

Pinned published/pending/recovery evidence is lazily read once
during a scene+marker result write. The read is not reused
across subsequent writes, rollbacks or publication changes.
The complete archive of the specific pinned proofs is retained.

### Worker counts in tagger_models.yml

Global settings independently control worker counts for each stage:

```yaml
checkpoint_batches: 8
prefetch_batches: 4
result_prefetch_batches: 32
prepare_workers: 4
merge_workers: 4
model_report_workers: 4
```

`prepare_workers` controls CPU image transformation,
`merge_workers` controls evidence reading/replay, and
`model_report_workers` controls model-report processes.
Worker counts are integers from 1 to 16; `1` means sequential
execution. Corresponding CLI options (`--prepare-workers`,
`--merge-workers`, `--model-report-workers`) override YAML.
Without these keys, defaults remain 4/4/up to 4 available CPUs.

Merge and model reports read worker settings from an existing
YAML file without loading models. Missing YAML does not prevent
replaying saved evidence: defaults or CLI settings apply.
Other offline commands do not read this file for worker counts.
Changes to worker counts do not invalidate inference identity
or persisted results. Model inference and Stash writes remain
sequential.

### Rerunning without refetching verified inputs

By default, normal execution/resume trusts already verified input
for each completed model/scene. It skips fetching VTT and sprites
from Stash and skips model loading, even if another selected model
still needs to process that scene. It continues to check model
identity, model-file contents, vocabulary, result completion and
retained evidence's threshold coverage. Compatible changes to
batch size, workers or publication thresholds do not force inference.

After regenerating VTT/sprites **at the same URLs**, force a fresh
input verification:

```powershell
python .\stash_ai.py --force-fetch-vtt
```

This option fetches and verifies VTT and relevant sprites even
for completed scenes. It does **not** force inference:
unchanged inputs with compatible results are reused. Only
results with modified inputs/models or insufficient evidence
coverage are recomputed. This applies to normal inference/resume,
not reports, `--merge-only` or standalone rethreshold commands.

New/incomplete results and checkpoints still fetch their inputs
during the corresponding model pass. For example, if JoyTag
finished but Oppai was interrupted, JoyTag skips fetching
while Oppai fetches input and resumes. If that later model
detects a changed VTT/sprite, previously saved results based
on the old inputs are immediately blocked and queued for
refresh before the shared merge. The block persists through
interruption or failure. Results saved with different inputs,
pending verification, changed tracked paths, or sampling limits
also require verification.

No database migration or inference/checkpoint format change is
needed. Existing v27 identity records work immediately with
the default full sampling. Older entries did not record paths;
run `--force-fetch-vtt` after changing those paths.
Missing identity or legacy identity with reduced sampling
requires input verification, after which compatible results
may still be preserved. New/verified records store a small
checksum and input context without rewriting raw results.
Run history marks skipped fetches using `input_fetch_skipped`;
it neither pretends that verification occurred nor authorizes reuse.

### Offline schema updates and repairs

`--db-update` checks schema shape and completed data revisions first.
An already current, complete database returns without decoding
inference BLOBs or running SQLite `quick_check`. The small metadata/key
scan can grow with result count. Required historical conversions
(minimum schema v18) run automatically before recording completion.
A previously unrecorded data revision may require a one-time full pass.
A DDL-only upgrade with completed data revisions does not require
a global repair.

Use `--db-repair` for an explicit full scan on the current schema:
it repairs derived metadata and identical scene aggregate duplicates
while preserving archived/publication proofs. Conflicting duplicates
stop the operation and roll it back. Investigate semantic corruption
with `--db-check`; a fast preflight cannot prove arbitrary BLOB contents.
Update and repair both work offline and support `--dry-run`; the repair
preview uses a temporary SQLite copy. Neither command runs models,
contacts Stash, or globally requests reinference.

Repair spools canonical BLOB rewrites and signature replacements into
a disk-backed temporary SQLite database with a bounded page cache.
It scans each original archive record once and applies evidence,
metadata and publication-reference changes in one transaction.
Interruption before commit rolls back the entire repair.
Available memory must still fit one decoded evidence/publication record,
and temporary disk space must fit rewritten BLOBs.

CLI maintenance shows the active phase and scanned/written row counts.
A current-schema update that restores metadata or completes a data
revision reports that work rather than claiming no changes were
needed. A complete no-op creates no progress bar and performs
no full scan.

Replay proof caches reuse exact scores only while immutable representative
versions remain live. Directional constituent compatibility (including rejected
pairs) and earliest distinct support-cue prefixes share the byte-bounded RAM
owner. Every changed representative set, threshold or support minimum causes
revalidation; evicted entries are recomputed. Co-presence still includes all
eligible selected scenes, and membership checks remain current. Weak input
owners prevent caches from keeping obsolete reference sets across reloads.

Cold replay batches historical ownership reads and prepares a single ordered
label-vote stream with frozen alias/performer metadata. A private, transactionally
maintained candidate-address relation avoids repeated payload decoding in hot
joins; durable payloads and their integrity checks keep the existing format.
Physical sample keys use a bounded RAM cache keyed by every canonical dependency,
so changed geometry, source or scan input cannot reuse an old key. Final candidate
payloads receive their reference context during insertion rather than a second
JSON update pass.

The disposable replay database now resides in RAM, including SQLite temporary
work. Half of `execution.rebuild_memory_bytes` caps its main database pages;
reference storage, model memory, transient SQLite sorts and separately bounded
Python caches are additional allocations. Exceeding the page ceiling aborts the
private replay before durable apply; increase the setting rather than spilling
to disk. Durable FaceDatabase transactions retain their existing guarantees.

`execution.rebuild_prefetch_scenes` defaults to false. This experimental option
uses one isolated read-only reader to validate at most one upcoming scene
(256 observations, at most 8 MiB and further limited by the working budget).
Only immutable vectors are prepared; assignments and SQLite writes stay ordered
on the coordinator. Large scenes use the serial path. Invalid vectors are
reported when originally consumed, and cancellation closes the reader/worker.

Run `python -m benchmarks.benchmark_recognition_replay --scenes 960 --repeats 2`
for an offline comparison with the former temporary-file path. One Linux run
(Python 3.14.7, NumPy 2.5.3, SQLite 3.53.1; 2,880 observations, six orthogonal
identities, 128 dimensions, four representatives) measured medians 4.55 s file,
4.19 s RAM serial, 4.72 s RAM lookahead, with identical partitions. The replay
image was 5.5 MiB. These synthetic results support keeping serial preparation
default; they do not predict a production rebuild's completion time. The
benchmark reports process RSS after each trial, not isolated peak process RAM.

Matching policy v8 separates representative/decision semantics from the
`max_reference_bytes` and `max_people_per_lookup` admission limits. Existing v7
candidates receive one retained-evidence replay; no new inference is requested.
Completed rebuilds can reuse results after monotonic capacity increases only
when all current input, reference, label-origin, alias and exact candidate
integrity checks pass. Decreases replay conservatively, as does an increase
when historical label matching previously skipped an over-budget reference set.
Standalone scene caches also record and check their original admission limits.

Keep at most eight completed selection descriptors per scope/namespace. A subset
run does not discard a full selection's descriptor, but it can invalidate that
selection's dependencies or assignments. Descriptors contain bounded hashes and
limits, not copied vectors or historical assignments. Current memberships,
accepted/withheld coverage, exact candidate UUID addresses, raw input repairs and
performer uniqueness are revalidated. Broken derived descriptors/candidates fall
back to retained-evidence replay. Warm checks read 256-scan chunks and validate
homogeneous vector blocks of at most 128 rows/1 MiB; norm-boundary and malformed
records use the original scalar validator for exact error behavior.

VTT portrait refresh can reuse a completed full ranking after verifying the
exact selected memberships/inputs, reference context, assignment proofs,
measurement contracts/bases, target and portrait metadata. Cached ranks do not
skip live ownership/current-image checks before publication or JPEG digest and
framing checks. Missing measurements, recognition-crop approximations, quality
errors and source failures keep the full evaluation/retry path. Corrupt or
changed descriptors also recompute. The catalogue retains at most 64 descriptors
of at most 1 MiB each; oversized results are not cached or truncated. Ranking
continues to score every eligible VTT observation in RAM.

Changed VTT quality/candidate metadata writes are committed in batches bounded
by 128 writes/4 MiB, with a final flush before winner preparation/return.
Unchanged metadata does not start a write transaction. Failed tails roll back;
previous complete batches remain reusable. Publication intent, dispatch and
confirmation retain their separate durable transactions. A partial provenance
index covers scope/person/observation lookups without indexing malformed JSON.
Explicit scene selections drive portrait-target discovery from selected scans
in 256-scene chunks; all-scene discovery preserves the retained-scan contract.

`execution.prefetch_portraits` enables bounded VTT preparation (default true;
false keeps the serial path). The metadata phase shares two verified scene
sources through an isolated read owner, prefetches at most two upcoming
thumbnails, and retains at most eight decoded thumbnails. Queue allowances
scale from `execution.rebuild_memory_bytes`, capped at 256 MiB: thumbnail pixels
use at most one quarter, and detached next-target descriptors use one eighth
and at most 128 observations. Oversized sources/frames use original serial
acquisition; no working data spills to disk. Source caches and decoded sheets
have separate bounded item/pixel allowances; model/runtime memory is additional.

The coordinator snapshots descriptors, identity proofs and policy; a source
worker owns its read sessions, a preparation worker handles detached quality
results, and one model consumer serializes visibility/expression initialization,
inference and recovery. Model assets and warning rendering stay on the
coordinator. The next new target can prepare while the current verified image
is published; existing portrait targets keep demand-driven evaluation without
a second history prepass. All eligible observations still reach ordered ranking.
The queues do not predict assignments or mutate memberships.

Preflight performer/scene reads support validated GraphQL aliases in chunks of
32. Null or invalid IDs stay isolated, and an envelope error falls back to
separate proven reads. Initial ownership/image snapshots can be prefetched;
pending/unknown image reconciliation starts from fresh reads, and every
publication still refreshes live ownership and image bytes before dispatch.
Workers never reconcile publication rows or use the coordinator's SQLite.
Image copies remain owned until inference finishes even if their consumer is
interrupted. Shutdown cancels reads, joins owners and releases queued images.

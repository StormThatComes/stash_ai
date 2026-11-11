# Tagging run diagnostics

[Documentation index](../README.md) · [Tagging workflow](README.md)

- [Anonymous developer diagnostics](#anonymous-developer-diagnostics)
- [Explain one tag](#explain-one-tag)

## Anonymous developer diagnostics

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
to its effective configured limit, after applying `ram_budget_mb`.
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

## Explain one tag

```powershell
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123
```

Scene reports use `format: 2`. Each `recorded_sources` item has one `record`
containing `meta`, `thresholds` and `detections`. Retained observations and
effective thresholds explain the result without running models.

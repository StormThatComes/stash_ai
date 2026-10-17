# Measuring recognition identity rebuild

Rebuild consumes retained embeddings. Its benchmark does not load recognition
models, fetch media, contact Stash or publish assignments. GPU inactivity during
this stage is expected. The serial default remains `rebuild_workers: 1`.

Start with a representative selection and the same facial configuration used by
the run. The local benchmark opens the source SQLite read-only, takes a consistent
snapshot, and closes the source before measuring. It copies recognition evidence
and historical labels in the selected scope/namespace into RAM; it does not read
JPEG crops, portrait images, quality BLOBs or run history. Every worker-budget
trial starts from the same private snapshot. Disk replay checkpoints are disabled
for this command, regardless of the configuration.

```powershell
python -m benchmarks.benchmark_recognition_retained --config facial_recognition.yml --scene-id 100 101 102 --workers 1 8 16 20 --repeats 3 > rebuild-baseline.json
```

Use `--all-scenes` instead of `--scene-id` for the full retained library. Choose
`--database` to override the configured source, and `--scope` if the namespace is
present in multiple Stash scopes. `--namespace` is an explicit override for older
retained evidence; normally the recognizer namespace comes from the configuration.
An explicitly selected scene needs its latest scan to be complete. All-scenes
mode excludes latest incomplete scans and reports their count; it never silently
substitutes an older complete scan.

The snapshot retains historical references needed for stable identity binding,
including observations outside the selected scenes. This preserves the real
rebuild's input semantics. RAM can therefore exceed just the selected embeddings,
and several RAM database copies coexist during a trial. Snapshot time is reported
separately from cold/cached rebuild wall time. No disk-backed working cache is
created. The JSON output itself is a report that the shell can save to a file.

Output contains input/configuration/selection hashes, counts, dimensions,
environment, effective worker/native limits, cold and cached timings, process
CPU seconds and semantic-output hashes. It omits scene/person IDs, URLs,
installation IDs, embeddings and SQL. Nothing is uploaded. Treat hashes and
workload counts as local diagnostics and review any report before sharing it.

Memory uses a 50 ms process-RSS sampler. `sampled_peak_bytes` can miss shorter
peaks and includes the interpreter, source snapshot and private trial/replay.
`rss_after_bytes` is current end-of-trial RSS, not a peak or retained-buffer cap.
Compare trials in the same process/interpreter and report the sampling method.

## Interpreting CPU diagnostics

The console's CPU section shows active/configured workers, computed/retired
blocks, admitted queue bytes, coordinator wait reason and strict scenes applied.
Workers update only lock-protected execution counters. The coordinator owns all
console updates. Computed blocks never advance the scene bar or its ETA.

With `profile_rebuild: true`, `rebuild_profile.cpu` includes:

- `completed_blocks`, `failed_blocks`, `active_workers`, `pending_blocks` and
  `retired_blocks`. Legacy `applied_blocks` also means retired numerical blocks;
  it does not mean identity decisions or a committed durable rebuild.
- `strict_scenes_applied`, counting completed strict replay scenes. Checkpoint
  reuse and final durable publication are separate stages.
- Bounded `fallbacks` for disabled parallelism, small catalogues/single shards,
  unsupported layouts, mixed dimensions and admission limits. A fallback does
  not suppress the original retained-evidence validation error.
- `preparation` time for vectors and packed matrices; profile counters distinguish
  `repacked_bytes` and `reused_packed_bytes`.
- `records`, separating computed, consumed and discarded score rows, evidence
  rows and constituent proofs. Consumed means delivered to the canonical
  numerical consumer, not an accepted identity. An early exit can leave valid
  speculative answers unused.

Worker wall seconds are summed task durations. Worker-thread CPU excludes native
helper threads. Neither metric is elapsed run time, CPU utilization or ETA.
Preparation and other stage timers can nest; do not sum them as disjoint shares.

Measure a baseline now and repeat representative small, merge-heavy and full
selections after major CPU changes. For AuraFace, prioritize 512-dimensional
workloads. Compare 1/8/16/20 workers with native limits recorded in the report;
the largest count is not presumed fastest. Different hashes between budgets are
a correctness failure: the command exits unsuccessfully.

The synthetic `benchmark_recognition_parallel` remains useful for offline smoke
tests, but its orthogonal repeated identities do not predict a real 9,000-scene
library. Representative 24-core measurements and a measured tuning/default
recommendation remain tracked in #856.

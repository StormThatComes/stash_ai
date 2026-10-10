# Offline benchmarks

Run these scripts from the repository root after installing
`requirements-dev.txt`. In CI, all twelve scripts pass smoke tests
against small datasets. Fixtures use the current analysis format
and complete metadata. Benchmarks operate on synthetic data,
temporary local databases, or a Stash mock: they do **not**
connect to Stash or run GPU models.

```powershell
python -m benchmarks.benchmark_performance --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_marker_matching --markers 1000 --labels 1 --repeats 5
python -m benchmarks.benchmark_retained_merge --frames 400 --labels 12 --weak-labels 108 --repeats 5
python -m benchmarks.benchmark_occurrences --frames 500 --labels 100 --repeats 5
python -m benchmarks.benchmark_cleanup_sql --markers 2500000 --legacy-time-budget 5
python -m benchmarks.benchmark_rethreshold --scenes 6 --frames 600 --labels 12 --repeats 5
python -m benchmarks.benchmark_tag_cascade --markers 100000 --batch-size 200 --repeats 3
python -m benchmarks.benchmark_scoped_reports --scenes 9000 --selected 32 --repeats 3
python -m benchmarks.benchmark_run_history --frames 128 --labels 12 --repeats 3
python -m benchmarks.benchmark_pinned_proofs --frames 256 --labels 12 --repeats 3
python -m benchmarks.benchmark_state_compression --frames 729 --labels 128 --repeats 5
```

| Script | What it measures |
|---|---|
| `benchmark_performance` | Marker construction and merging, including output hashes |
| `benchmark_merge_pipeline` | SQLite-backed merge, its stages and a mocked API |
| `benchmark_marker_matching` | Re-merge against existing markers; full synchronization and matching time, logical snapshot and Stash hashes |
| `benchmark_retained_merge` | Shared observations, single-model temporal markers and CPU merge with broad prediction retention, hashing every stage |
| `benchmark_occurrences` | Selection of representative Occurrence points |
| `benchmark_cleanup_sql` | SQL-based candidate discovery, excluding API and deletion cost |
| `benchmark_rethreshold` | Threshold updates, Check/Apply time and decoding, evidence writes, result and `result_index` hashes, excluding merge/API |
| `benchmark_tag_cascade` | Popular-tag cascading versus prior batched relationship detachment; total time, longest transaction, mutation counts and retained-state hashes |
| `benchmark_scoped_reports` | Cache report and complete Check/Apply for a small selection in a larger database, writer versus preview: duration, SQL statements, writes, decoding, output hashes |
| `benchmark_run_history` | History enabled/disabled during fresh inference, cache reuse and checkpoint resume: duration, RSS, SQL, retention and raw-result equivalence |
| `benchmark_pinned_proofs` | Full result writes with published/pending/recovery evidence, writer and preview: duration, RSS, decoding and final-state equivalence |
| `benchmark_state_compression` | zlib level 6 vs 1 using identical scene/marker evidence: compression time, BLOB size and decoding equivalence, excluding JSON encoding, SQLite and GPU |

`results/scoped_reports_2026-10-07.json` compares PR 4 to PR 5 for
32 selected scenes and four models with either 900 or 9,000 scenes
in the database. With 9,000 scenes, median preview report time
fell from 437 to 7 ms, Check/Apply writer time from 201 to 12 ms,
and Check/Apply preview time from 420 to 15 ms. Output hashes
remained identical, raw evidence was unchanged, and no variant
decoded prediction BLOBs. Post-fix SQL statement count stayed
constant at both database sizes. These are local synthetic
workflow measurements without models or API.

Every script also supports direct execution by path, e.g.
`python benchmarks/benchmark_performance.py`. Benchmarks comparing
application versions without `--project` target the application
directory independently of the terminal's current directory.

For comparison against an earlier source-tree copy:

```powershell
python -m benchmarks.benchmark_performance --project C:\previous_build\stash_ai
python -m benchmarks.benchmark_merge_pipeline --project C:\previous_build\stash_ai
python -m benchmarks.benchmark_rethreshold --project C:\previous_build\stash_ai
```

The selected directory must contain a complete installation:
`stash_ai.py`, `update_stash_ai.py`, and the `stash_ai/` package.
Compared revisions must expose the current APIs required by the
scripts: `union_intervals(..., borrow=True)`, NumPy-array filtering
through `select_scores`, `MergeDiagnostics`, and
`StateStore.result_metadata`. Benchmarks do not adapt their calls
to older interfaces.

Benchmark tools and the Stash mock come from the current
repository, so the target installation's tests do not need
to be run. Use identical data and parameters; output hashes
may also differ when application behavior changes.

`benchmark_tag_cascade` creates only its own temporary SQLite
database and does not accept `--project`. It models indexed
relationships between tags, markers, scenes and images while
omitting network operations, hooks and Stash object loading,
giving pre-detachment an optimistic cost. Fixture cloning and
hash verification are excluded from the timed portion.
The database uses WAL/NORMAL and immediate write transactions,
matching Stash v0.31.1. See
[cleanup performance](../docs/cleanup_performance.md) for the
results and the selected cleanup strategy.

`benchmark_merge_pipeline` reports the JSON and compressed
BLOB sizes of each snapshot. When comparing storage formats,
use `input_sha256`, `logical_snapshot_sha256`, `stash_sha256`
and operation counts. `snapshot_sha256` covers raw BLOB bytes,
which can change even if logical results remain identical.
Decoding and hashing for verification occur outside the
measured interval.

`benchmark_marker_matching` first publishes disjoint intervals,
then measures re-synchronization of those same markers.
`--labels` distributes them across multiple primary tags,
`--point-only` tests point-marker publication, and `--cleanup`
also includes checking marker existence before cleanup.
The script supports `--project` and `--output`. When comparing
revisions, use the same script and settings; hashes and counts
must match. Verification is excluded from timing; results include
local snapshot writes without inference or API latency.

`benchmark_rethreshold` measures results written by the selected
application version with populated validation metadata by default.
Use `--mode scene` or `--mode markers` to change only scene
or marker thresholds; `--mode combined` (default) changes both.
Use `--legacy-metadata` to measure initial metadata population
for older cached results. This option deletes only derived
metadata inside the temporary benchmark database. The report
separately reports evidence decoding and planning time for
`Check` and `Apply`, while total time includes transactions and
threshold writes. Hashes cover input BLOBs, effective results
and completion indexes.

Scripts print JSON to stdout. `benchmark_merge_pipeline` and
`benchmark_occurrences` also support `--output PATH`. Use a new
filename for each measurement:

```powershell
python -m benchmarks.benchmark_occurrences --output benchmarks/results/local_occurrences.json
```

Local `local_*` result files are gitignored. Historical measurements
in [results](results/README.md) are reference data: do not overwrite
them during routine runs. Synthetic benchmark timings are not
forecasts for processing an entire library.

`benchmark_retained_merge` retains 12 strong/supporting and 108 weak
observations per frame at a publication threshold of 0.8 and
retention floor of 0.1. Use `--weak-labels 0` to test a case
without extra weak observations. For version comparisons, run
the same script and settings with a different `--project` path.
Identical hashes for the input and three stages verify evidence,
marker and data consistency. The complete `merge_cpu` measurement
includes building and merging markers before SQLite writes or
Stash API calls.

## Run history overhead

`python -m benchmarks.benchmark_run_history --frames 128 --labels 12 --repeats 3`
compares run history on/off for fresh inference, cache reuse
and checkpoint resume. Every sample runs in a separate process.
The report records wall/CPU time, changed SQL rows in state and
history databases, process peak RSS, database sizes, and
separate retention cost. Completed raw-result hashes must
be identical; the cache variant must not perform prediction.
Retention removes 40 completed fixture runs while protecting
proofs. The fixture exercises actual orchestration, VTT/PNG
and checkpoints using a synthetic CPU tagger. It does not
measure a production ML backend or network access, and peak
RSS also includes fixture preparation.

`captured_at` is fixed only in the fixture so comparisons
of BLOBs are not affected by sample execution time.

The CI measurement [run_history_2026-10-07.json](results/run_history_2026-10-07.json)
shows median off/on times of 139/167 ms for fresh inference,
5/33 ms for cache reuse, and 53/84 ms for resume.
Overhead includes opening/closing history, fingerprinting
code, and reading package metadata. History-write counts
are 7, 5 and 7, respectively. This is a small synthetic run,
so compare absolute time as well as the overhead percentage.

## Result-writing overhead with pinned evidence

`python -m benchmarks.benchmark_pinned_proofs --frames 256 --labels 12 --repeats 3`
measures the full scene+marker result write for four sources
while published, pending, and recovery proofs coexist, including
sparse dry-run. Every sample uses a new process and database.
The baseline disables only reuse of lazily loaded pinned
proofs within the call; the remaining code is identical.
Evidence preparation and hash verification are not timed.
The report includes wall/CPU time, process peak RSS
(including fixtures), snapshot decoding count and SQL writes.
All persistent results, archives, snapshots, recovery and
indexes must have identical hashes; dry-run additionally
preserves the source database byte for byte.

The CI measurement [pinned_proofs_2026-10-07.json](results/pinned_proofs_2026-10-07.json)
shows writer 412 → 283 ms and preview 405 → 279 ms
(medians of three repetitions). Snapshot decoding dropped
from 48 to 24, while SQL-write counts remained 40 for the
writer and 48 for preview. Entire-state verification hashes
match, and all eight pinned records remain archived.
The approximately 31% improvement applies to this synthetic
write with large proofs, excluding inference and API overhead.

`python -m benchmarks.benchmark_input_transport --reads 200 --urls 10000`
compares three separate deadline stages with one owned read worker using the same
mock HTTP/JSON payload. It also compares a URL/hash dictionary with zero-cache
media verification backed by temporary SQLite. The checked-in October 10 result
shows 600 -> 200 worker starts (986 -> 2,690 mocked reads/s), and Python peak
memory 1.97 MB -> 23 KB for 10,000 unique URLs. SQLite used 1.19 MB of temporary
database pages with a 64 KiB page cache and took 0.25 s versus 0.04 s for the
in-memory dictionary. These are controlled overhead measurements, not network
or end-to-end inference speedups. Media bytes stay in RAM; only hashed URL keys
and checksums spill. RunInputs also spills its manifest identity ledger, and
closes both ledgers after producers have joined.

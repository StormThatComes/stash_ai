# Reference benchmark results

The JSON files were moved without modifying their contents. They
describe earlier versions and test conditions documented elsewhere;
they are **not** results from the current execution.

- [SQL cleanup](cleanup_benchmark.json)
- [Performance](performance_results.json)
- [Merge](merge_performance_results.json)
- [Occurrence](occurrence_performance_results.json)
- [Merge diagnostics](diagnostic_benchmark_results.json)

[Shared snapshot evidence](shared_snapshot_proofs.json) measures
2026-10-02 against `main` after #90: four models, 400 frames,
28 labels and 453 markers (median of five repetitions).
Total offline merge dropped from 1.996 to 1.511 seconds, and
snapshot compaction, serialization and compression from 0.790
to 0.319 seconds. Marker JSON shrank from 44.29 to 14.90 MB,
and compressed BLOBs from 1.69 to 0.73 MB. Logical snapshot
and Stash-result hashes and publication counts remained identical.
The file also includes a one-label case with no shared evidence.
These are synthetic measurements without inference or network latency,
not a prediction for a full library.

[How to run new benchmarks](../README.md) ·
[Historical validation reports](../../docs/validation/README.md)

[Marker interval index](marker_interval_index.json) measures
2026-10-02 against `main` after #93, with five repetitions
of the same script. For 1,000 existing markers sharing one
primary tag, resynchronization fell from 0.913 to 0.061 seconds,
and matching alone from 0.855 to 0.031 seconds. For 125
point markers, sync fell from 1.242 to 0.012 seconds; cleanup
of 1,000 markers across 20 primary tags dropped from 0.179 to
0.065 seconds. A small control case is also included. Input,
logical snapshot and Stash-result hashes and counts match between
revisions. Timings cover SQLite and mocked Stash, excluding inference
and API latency.

[Popular-tag cascade](tag_cascade_2026-10-07.json) measures
2026-10-07 with 100,000 and 1,000,000 markers, indexed
relationships, and Stash v0.31.1-compatible WAL/NORMAL behavior.
Pre-detaching relationships in batches reduced the longest
transaction but increased total time by 37–49%, even without
API calls, hooks or object loading. Retained-state hashes
were identical. See the [analysis and decision](../../docs/cleanup_performance.md).

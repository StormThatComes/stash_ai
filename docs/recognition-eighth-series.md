# Recognition: reliable lifecycle and lower execution cost

Stacked PRs from the refreshed 30-item audit. Merge in order.

| PR | Audit items | Scope |
| --- | --- | --- |
| 1 | 1, 2, 5, 6, 7, 23 | Exclusive checkpoint/winner ownership, interrupted finalization, bounded shutdown, atomic exports and stable artifact-independent signatures. |
| 2 | 3, 4, 18, 24, 30 | Physical goal addresses and bounded, fair optional diagnostics. |
| 3 | 25, 26, 27 | End-to-end report profiling, partial failure metrics and accurate feasibility counters. |
| 4 | 9, 10, 11, 28 | Numerical cache economics, worker reuse and cached-job bypass. |
| 5 | 8, 12, 13, 17 | RAM admission, shared replay evidence, snapshot import and fingerprint reuse. |
| 6 | 14, 15, 16, 19 | Batched metadata, scan lookup index, pair-report reuse and compact trial output. |
| 7 | 20, 21, 22, 29 | Batched detection, overlapped pose decoding, frame ownership and incremental buffering. |

## Lifecycle

Checkpoint and winner files use the existing cross-platform advisory lock for
their entire owning solver invocation. Standalone `Journal` owners must call
`close()` before opening a replacement writer. A killed process releases the OS
lock; append-only durable events remain recoverable. Final report files publish
atomically and do not replace independently created files.

Completed trial selection survives interruption during final aggregation; the
report marks interrupted feasibility aggregation rather than inventing an empty
run. Cooperative worker shutdown has a bounded grace period, followed by process
termination for unresponsive isolated trials. Crop retention metadata is omitted
from source fingerprints while all matching measurements remain authoritative.

## Goal and diagnostic coverage

Physical goal addresses are shared by scoring, destination discovery and query
ownership. Optional pair-report body/hybrid failures mark `measurement_coverage`
incomplete and preserve the other routes; unrelated data errors still propagate.
Same-identity simulation distinguishes a reproduced merge from replay splitting
or incomplete replay. Query measurements account for actual witness-pair work
under a separate ceiling. Local body hints reserve work across goal sets/scenes,
rotate oversubscribed scenes between trials and expose per-scene coverage.

## End-to-end observability

`--profile-report` aliases `--profile-solver` and also works without `--solve`.
The JSON `report_performance` covers snapshot preparation, goal resolution,
fingerprinting and analysis up to output serialization. A final structured
`Merge report performance:` line on stderr includes serialization and atomic
publication; it reports its scope explicitly because a file cannot contain the
completed duration of its own publication. Profiling emits this final line even
with `--progress-style off`; otherwise that option remains silent.

Plain pair reports show stage-aware liveness without adding text to JSON stdout.
Failed trials retain partial stage/CPU/RSS profiles, plus typed resource budget,
limit and requested values where available. Feasibility failure counters include
only feasibility-phase attempts, while the solver retains all failed trials.

## Numerical cache and workers

Individual body dot products stay in RAM; shared disk entries are reserved for
reductions with at least 256 vector pairs and one million scalar products.
Shared hits are promoted to the process cache before subsequent disk lookups.
Warm matrix tiles bypass executor submission, and partially cached tiles submit
only missing products. Float32 reduction order remains unchanged.

Workers retain content-addressed numerical entries and the immutable goal index
between trials. Membership, plans and database-dependent entries are discarded;
a changed seed, goal object, execution contract or cache budget resets reuse.
Targeted numerical, cache and parallel solver tests pass (33 tests).

## Replay memory and preparation

Automatic worker admission uses 80% of available RAM, including exposed cgroup
limits. Configured ceilings are conservative rather than allocations: if even
one ceiling exceeds available RAM, a serial trial is still allowed and that
condition is explicitly reported. An explicit memory budget remains strict.
Volatile available-memory readings do not invalidate a checkpoint when its
effective worker plan stays the same.

Inner solver replays now read embedding bytes from the immutable seed through
indexed views, keeping quality, membership and other mutable metadata private.
Independent lookahead readers install the same views. Ordinary live runs retain
their copying path. Initial snapshot import suspends trusted destination triggers
transactionally and restores them before publication. Source fingerprints are
memoized only for the lifetime of a prepared, read-only snapshot.

## Catalogue and report costs

Body metadata is read in bounded owner/witness batches without loading vectors
or crops. Face ownership and cue lookups use catalogue-wide SQL passes; missing
original witnesses still block merges. A `scans(scope, namespace, scene)` index
lets current-head lookups traverse insertion order without a temporary sort and
is added when opening existing databases.

Pair reports share one scoped numerical/catalogue cache. Equivalent focus label
partitions reuse one body plan; callers receive independent report copies.
Intermediate solver body reports may replace `identities` with explicitly marked
`baseline-identities-delta-v1` data (`removed`, `changed`, and final count). The
baseline and selected winner keep complete lists. Deltas are lossless and used
only when smaller, including worker transport and checkpoint records. Ordinary
non-solver replay output retains complete lists.

## Scan and pose pipeline

YOLO person detection batches equal-sized frames under `pose.batch_size`, keeping
rectangular padding, frame order, thresholds and NMS settings. Both native YOLO
pose and ViTPose refinement use the queued path. Accelerator ViTPose inference
can overlap one CPU heatmap decode with the next model batch; model execution and
device transfers remain on the owner thread. CPU execution stays synchronous.

Prefetched frames transfer ownership to cue batches. Explicit leases keep a
frame alive for queued person collection without another pixel copy, including
abort and early-generator-close paths. Deferred record byte accounting updates
on append/take/abort instead of rescanning all buffered records per cue.

Geometry, batching, ownership and overlap scheduling have CPU regression tests.
This environment has no CUDA device; accelerator throughput requires measurement
on representative media and target hardware before making speedup claims.

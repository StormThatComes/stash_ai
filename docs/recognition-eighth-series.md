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

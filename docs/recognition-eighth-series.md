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

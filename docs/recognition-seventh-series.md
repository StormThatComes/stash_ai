# Recognition: solver safety and throughput

This seven-PR series covers the refreshed 30-item recognition audit. Each PR
builds on its predecessor; merge in the listed order.

| PR | Audit items | Scope |
| --- | --- | --- |
| 1 | 1, 2, 5, 24 | Resource failures, bounded optional diagnostics, detached report snapshots, complete input fingerprints. |
| 2 | 3, 4, 7, 26 | Compatible body rivals, local body proposals, opt-in physical continuity goals. |
| 3 | 8, 12, 14, 18 | Scoped replay inputs, private database preparation and resource admission. |
| 4 | 9, 10, 16, 19, 25 | Exact numerical reuse, rival search, dependency and cache integrity. |
| 5 | 13, 15, 20 | Worker results, durable incremental checkpoints and rolling scheduling. |
| 6 | 17, 21, 22, 23 | Scan buffering, model lifetime, detector and pose inference batches. |
| 7 | 6, 11, 27, 28, 29, 30 | Report cost controls, lifecycle progress and observability. |

## Solver safety

Non-baseline resource admission failures are recorded as failed trials in both
default serial and explicitly configured execution. Baseline failures and data
errors still abort: no completed result is invented.

Optional body/hybrid hints use independent diagnostic ceilings (4096 candidate
pairs, one million witness comparisons, 16 MiB of plan data). Exhaustion reports
incomplete `measurement_coverage`; it does not change production matching limits
or prevent a baseline replay. A missing hint does not prove infeasibility.

CLI merge reports copy a scoped, image-free snapshot and close the original read
transaction before analysis. Writers can then commit even in SQLite rollback
journal mode. Goals, bindings and analysis all use the same detached snapshot.
Report fingerprints use the complete scoped replay input contract, including
reference roles and identity aliases.

## Body matching and goals

Hard co-presence/face contradictions are removed before attachment and rival
ranking. Weak or unknown competitors remain eligible. The solver uses the same
hard-rival policy and observes local body scores, complete-link minima, support
and margins within a separate diagnostic budget. These hints only propose trials.

`recognition report-merge --performer-set 900,901 --solve --goal-policy identity-physical`
opts into face/body continuity. Only already accepted, compatible body links to
the same retained owner become alternative addresses of a frozen physical vote.
Both addresses assigned consistently still contribute one vote; conflicting
addresses contribute none. `identity` remains the default, and `strict` continues
to require every original sample. JSON records `continuity: face_or_body`; strict
sample diagnostics remain visible beside the identity objective.

## Replay input cost and admission

A report's scoped immutable disk snapshot is reused by its solver rather than
copied again. `execution.replay_snapshot_bytes` (64 GiB by default) controls disk
snapshot pages separately from `rebuild_memory_bytes`. Snapshot statistics are
computed once, not by three table scans on every trial.

Private trials keep only address/foreign-key skeletons for shared evidence;
quality, geometry, vectors and images are read through immutable indexed views.
Bulk preparation restores all dependency triggers and generations atomically.
Scoped history relations are filtered by SQL instead of whole-table Python scans.

When a source is available, solver admission uses conservative input-derived
reference bounds while retaining both private SQLite RAM ceilings (trial history
and inner production replay). The estimate is admission accounting, not a hard
process RSS limit. Explicit production reference and page limits still apply.

## Exact evidence reuse

Ordered float32 witness reductions and local dot products can be reused across
trials by content. Mean/minimum accumulation order, weights, complete original
cores and live thresholds remain unchanged. Rival search retains resolved owners
and repeats matrix passes only for unresolved owners after a hard veto.

Face veto banks are cached only after a complete read, within a byte limit.
Portrait configuration and crop-backfill artifacts no longer invalidate body
matching. Historical triggers are accepted only when they exactly match audited
DDL, then upgraded atomically by the writer; real evidence/owner changes still
invalidate generations. Body policy v10 invalidates old attachment decisions.

Shared numerical files carry checksums and shape/dtype contracts in the ledger.
Valid-format files with altered data fall back to exact computation. Verification
is reused only while the file identity and modification stamps remain unchanged.

## Worker transport and durable scheduling

Workers receive baseline predictions once and return exact assignment deltas,
including explicit unassignments. The coordinator reconstructs the same full
predictions for scoring. Completed raw checkpoint results are removed only after
the applied trial record is durable.

The ordered scheduler refills its bounded window after each retirement instead
of waiting for a whole batch. Results still apply in dispatch order; async mode
retains its completion-based behavior. Dispatch/retirement events preserve resume.

Checkpoint transitions append to a checksummed, fsynced `events.jsonl` chain in
the owned `.trials` directory. The main JSON snapshot is refreshed on solver exit,
including interruption. For a live checkpoint use
`solver_journal_events.load_document(path)` to fold newer events. A torn final
append is discarded on resume; corrupt complete events are rejected. Winner
YAML still updates atomically after its recovery record is durable.

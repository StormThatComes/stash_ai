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

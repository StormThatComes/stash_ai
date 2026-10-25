# Recognition follow-up: six stacked PRs

Base: main, commit 3c7726a78c316368af44954b42682f4630c1e005.
VTT only; one inference owner; automatic identity decisions.
Tuner expansion and multi-scenario tuning (items 19–20) are deferred.

1. Evidence and deduplication (1, 2, 3, 7).
2. Stable representatives and cheaper refresh (4, 5, 13).
3. Shared matching, pose and conflicting claims (6, 8, 10).
4. Dependency retries, convergence and resume (9, 14, 15).
5. CPU scoring, memory accounting and shared cohort inputs (11, 12, 16, 17).
6. Overlapped VTT pipeline and visibility batches (18).

## Current implementation

PR1 foundation implements timestamp/pixel bipartite support for identity births
and query votes, timestamp deduplication and representative improvement in view
cohorts, propagation of quality into historical/cohort recovery, the promotion
key compatibility fix and historical-pool duplicate configuration forwarding.
The matching/checkpoint engine version changes invalidate prior decisions.

Still required within PR1: complete support-counter migration (merge proof
caches, same-scene and cross-scene support), stable pre-JPEG fingerprints,
verified near-duplicate matching, cross-scene copy handling and complete physical
claim canonicalization. These are not claimed as implemented by the foundation.
PR2–PR6 are not implemented yet. Publish the foundation as a draft until these
remaining scopes and regression checks are complete.

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

## Implementation status

PR1: independent timestamp/pixel matching now covers birth, merge proofs,
same-scene support, cross-scene confirmation, cohort majority, local ordering,
query support and promoted views. A new nomination revalidates cached prefixes
when it can change the matching; unchanged prefixes remain reusable.
Pre-JPEG RGB fingerprints deduplicate exact copies across scenes. Private replay
verifies compression copies using textured RGB thumbnails, aspect and embedding;
perceptual hashes only shortlist, and copies never form transitive chains.
Legacy scans retain timestamp/JPEG fallback until their pixels are rescanned.
Better duplicates can replace view representatives without gaining support.
Physical duplicates are checked before anchor co-presence rejection.

Validation: 129 focused recognition regressions passed (evidence, merge cache,
scene/cross-scene, cohort assignment, views, promotion, rebuild, retained data).
No measured real-library accuracy or GPU throughput claim.

PR2–PR6: implementation follows on stacked branches.

PR2: the strict sampler reserves slots for three-vote rare looks using frozen
pivots and complete representative compatibility. Local identity origin stays
stable while a demonstrably better complete core can replace its seed once;
the validated replacement is protected against repeated drift. Canonical
resampling uses membership ranks plus a disposable shared row cache; local
refresh uses an indexed candidate union instead of correlated full-table probes.
Validation: 42 representative/index/rebuild tests and 19 final focused tests,
including rare-mode retention and one-time seed replacement, passed.

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

All six scopes are implemented on the stacked branches described below.

PR2: the strict sampler reserves slots for three-vote rare looks using frozen
pivots and complete representative compatibility. Local identity origin stays
stable while a demonstrably better complete core can replace its seed once;
the validated replacement is protected against repeated drift. Canonical
resampling uses membership ranks plus a disposable shared row cache; local
refresh uses an indexed candidate union instead of correlated full-table probes.
Validation: 42 representative/index/rebuild tests and 19 final focused tests,
including rare-mode retention and one-time seed replacement, passed.

PR3: identity-birth plausibility shares the assignment bank (including supported
witnesses and corroborated edge semantics). Query-only pose fallback uses a
continuous score premium; unknown/opposite pose needs three independent queries
and three supporting witnesses. Strict retries solve small cue assignment sets
jointly with all strict alternatives, fixed accepted claims and raw rival vetoes.
Equal or oversized assignment problems remain unresolved; attachments never train
new cores. Validation: 52 matching/pose/conflict/rebuild tests passed.

PR4: negative retries depend on the frozen witness/rival bank and the query's
own co-presence cue, rather than all library memberships. Strict retry passes
run to a deterministic fixed point with cancellation, cycle detection and a
bounded pass budget. Checkpoints retain local appearance ordering and final
commit plans (including unresolved-pool backfill). Final-plan reuse validates
selected raw inputs, historical context, binding context, engine and limits.
Validation: 58 checkpoint/rebuild tests passed, including interruption before
commit and recovery without rerunning historical analysis.

PR5: parallel scoring carries exact results and conservative rejection bounds
into ranking, avoiding a second matrix screen while distinguishing uncomputed
scores. Query support uses immutable matrix retrieval with exact scalar boundary
checks. Shared cohort workers consume read-only shared buffers without worker
SQLite insertion, preserving canonical priorities and coordinator-only writes.
The working cache lends idle space to queues, banks and Python metadata; active
reservations evict disposable cache entries before admission. Cohort/promotion
metadata and indexed rows are included. This is working-memory accounting, not
an OS RSS guarantee (allocator, libraries, SQLite and configured prefetch remain
separate). Validation: 71 scoring/cohort/query/cache tests and 35 final focused
tests passed, including shared-buffer ownership and exact matrix boundaries.

PR6: bounded owned CPU jobs overlap JPEG encoding with caller-owned GPU
inference. Visibility preparation runs on CPU and inference uses batches of up
to eight, with explicit capability fallback for scalar-only backends. Deferred
JPEG results are joined in cue order before committing a complete cue. Failure
and cancellation retain existing checkpoint boundaries and release owned images.
Validation: 54 pipeline/visibility/scanning tests passed, including a synchronized
CPU/GPU overlap test. A broader 172-test regression pass also passed after fixing
scalar visibility compatibility, duplicate proof short-circuiting, immutable
metadata handling and transaction-safe checkpoint dependency reads.

## Final validation

All six branches passed the complete recognition suite in isolated checkouts
(four shards each): PR1 1516, PR2 1518, PR3 1519, PR4 1521, PR5 1524,
PR6 1527 passed; seven environment/model skips per branch. The final raw-peer
guard added to PR3 and carried forward also passed its five focused conflict
tests and the complete PR6 suite. No real-library fragmentation or GPU timing
benchmark was performed. Merge in order, starting at main.

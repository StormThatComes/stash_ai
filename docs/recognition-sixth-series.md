# Recognition: recovery correctness and pipeline throughput

This series is stacked: recovery → modes → catalogues → incremental evidence →
throughput. Each branch includes its predecessor. Merge in that order.

| PR | Analysis items | Implementation |
| --- | --- | --- |
| Recovery | 1, 2, 3, 6, 7, 9 | Canonical co-presence competitors; best admissible historical core; independent face/body missing states; strict-face recovery with query support; explicit unknown body rivals; compact, byte-budgeted query claims without crop BLOBs. |
| Modes | 4, 5, 8, 10 | Supported body modes shared by queries, joint claims, merge proofs and historical vetoes; support built before representative truncation; report variants rebuild policy-specific banks. |
| Catalogues | 11, 12, 13, 20 | Bulk body/ready queries; batched historical-core reads; committed cache metadata no longer invalidates immutable banks; content-keyed float64 packs; one bounded numerical cache across independent replay trials. |
| Incremental evidence | 14, 15, 16 | Reuse unchanged centroid tiles and exact pair proofs; rebuild decisions with every live rival; co-presence stops at the first conflict; scoped, bounded persistent pixel aliases with transactional raw dependencies. |
| Throughput | 17, 18, 19 | GPU batch admission independent of CPU future count; account shared frame copies once; conditional HTTP uses metadata before loading cached BLOBs; portrait fingerprints depend on selected scan generations; ranking eviction uses bytes. |

## Correctness constraints

Face and body cosine scores are never mixed. A missing body does not invalidate
an otherwise complete strict-face proof. Historical cores compete under their
canonical owner: an inadmissible high-scoring core cannot conceal an admissible
core of the same identity. A known or unknown competing identity still blocks
an insufficiently separated attachment.

Supported body modes retain their independent evidence before representative
selection. Every historical body witness must match a supported opposing mode;
unsupported outliers are not silently discarded. Report policies use their own
mode/support banks.

The caches retain numerical evidence, not final ownership decisions. Pair proof
keys include complete endpoint evidence, matching settings and the current
co-presence result. Every plan reconstructs components and checks current rivals.
Replay trials retain independent databases and assignments.

Pixel cache records are disposable and checksummed. Their dependency stamp
includes raw scan generations and referenced thumbnail bytes. Assignment-only
changes do not invalidate the raw index. Reports can read cached aliases without
writing source data. Missing or changed thumbnails still fail verification.

## Resource controls

Existing reference, plan, working-memory and matrix-cache budgets remain active.
Pixel witness indexes use temporary SQLite storage, with an explicit logical
byte ceiling. Cache eviction changes execution cost, never candidate coverage.

`execution.portrait_ranking_cache_bytes` defaults to 16 MiB; zero disables the
complete-ranking cache. The former global 64-entry limit is removed. Cached
rankings still validate portrait retry states and fresh external image state.

HTTP cache validators are read without loading the old body. A 304 triggers
integrity verification of that body. Corruption or concurrent eviction triggers
an unconditional fetch under the original deadline.

## Validation and performance limits

Regression tests cover canonical rivals, historical mode guards, content cache
invalidation, rollback, alias scope, shared-frame accounting, 16-item inference
with two CPU workers, metadata-first HTTP and ranking eviction beyond 64 entries.
Existing hybrid, replay, collection, publication and report tests exercise the
combined paths.

This is exact bounded reuse, not approximate nearest-neighbor pruning. A cold
all-identity graph can still require quadratic comparisons. A changed centroid
tile is recomputed; unchanged tiles and proofs can be reused within the bounded
run cache. Pixel dependency validation still reads referenced thumbnail bytes.
No production speedup or recognition accuracy percentage is claimed without a
representative retained corpus and hardware benchmark.

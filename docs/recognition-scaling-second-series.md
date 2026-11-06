# Recognition scaling: second series

The target is 10,000 scenes, 300 VTT cues per scene, and approximately
4.2 million detections each of faces and bodies before admission. Changes in
this series preserve exact identity decisions unless an optional numerical
inference mode is explicitly enabled. GPU throughput requires real-model
measurement; unit regressions measure work counts and decision equivalence.

## SQL import, publication and retention

Fresh private replay imports temporarily suspend destination row triggers and
restore them before matching. Durable database triggers remain active. Restored
triggers explicitly belong to `main`, even when the vector-source view shadows
`observations` in `temp`.

Membership application batches proposed assignments, discovers changed scenes
with a join, and updates only changed owners. Original ownership remains
first-write-wins and the complete rebuild keeps its atomic commit boundary.

Thumbnail reclamation uses the exact indexed JSON expression. The correlated
lookup also removes the right-hand column's SQLite affinity so the expression
index can perform a search rather than scanning every observation.

## Shared face/body projections

Fresh face-associated bodies write their person projection directly as a body
address. Repair uses the same representation after validating retained features.
Crop backfill re-shares byte-identical images and vectors, including projections
materialized by copy-on-write triggers. Replacing or deleting a body still
materializes its old projections before the change; legacy schemas keep inline
payloads. No additional encoder inference is involved.

## Face reference pages

Read vector pages in ordered blocks of at most 256, preserving duplicates and
reporting missing pages explicitly. Frozen scoring validates layout from immutable
page metadata, reuses shard partitions and packed matrices, and passes matrix views
to CPU workers. Scalar finalists keep the original dot/mean/min reductions. Cold
independent scene replay no longer rebuilds an unused global reference catalogue.
Validation: 33 reference/scoring/replay tests passed, including bounded query counts
and exact equality with scalar scores. This is operation-count evidence, not a full
three-million-frame throughput benchmark.

## Shared rebuild resources

`execution.rebuild.sqlite_cache_bytes` raises the previous tiny page-cache caps
while charging both stores against the existing shared working-memory budget.
`scratch_directory` selects the parent of one private run directory; `scratch_bytes`
limits capacity shared by replay/reference/plan SQLite stores, hybrid witness files
and centroid indexes. SQLite leases include rollback-journal headroom and grow in
blocks; native executemany remains batched. Allocation failures preserve rollback,
and owned files are removed on close/exit. Persistent caches, checkpoints and SQLite
internal sort files remain separately budgeted (not covered by this scratch limit).
Validation covers aggregate admission, rejected-write position, rollback, cleanup,
configuration roundtrips, cold replay, checkpoint restore and global body matching.

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

`execution.rebuild.sqlite_cache_mb` raises the previous tiny page-cache caps
while charging both stores against the existing shared working-memory budget.
`scratch_directory` selects the parent of one private run directory; `scratch_mb`
limits capacity shared by replay/reference/plan SQLite stores, hybrid witness files
and centroid indexes. SQLite leases include rollback-journal headroom and grow in
blocks; native executemany remains batched. Allocation failures preserve rollback,
and owned files are removed on close/exit. Persistent caches, checkpoints and SQLite
internal sort files remain separately budgeted (not covered by this scratch limit).
Validation covers aggregate admission, rejected-write position, rollback, cleanup,
configuration roundtrips, cold replay, checkpoint restore and global body matching.

## Body matching

Complete-link appearance construction now accepts a cluster through a conservative
sphere lower bound only when every original float32 dot is provably above the
threshold. Borderline cases retain scalar comparisons. All original witnesses and
independent support remain present. Rival search prunes centroid subtrees against
current exact-search maxima with a rounding envelope; weak rivals remain eligible.
Compact planning evaluates metadata vetoes before embeddings and marks uncomputed
measurements explicitly; full reports retain complete scoring. Owner metadata and
witness selection are paged without building a full owner ID-to-row map.
Validation: 71 body/hybrid/scaling/parallel tests, including the exhaustive oracle.
A 180-vector coherent mode retains every vector with fewer than 180 scalar dots
instead of 16110; no end-to-end GPU speedup is claimed.

## Resume before raw replay import

When rebuild checkpoints are enabled, retain a bounded raw replay image before
strict matching. Resume first validates the database's audited dependency clock,
selection, installation, trusted schema, vector source and file checksum. Matching
input images are copied with SQLite backup before restoring existing strict/late
checkpoints, skipping per-observation SQL import and pixel canonicalization. Changed
or corrupt inputs fall back to cold import. The optional image uses the smaller of
checkpoint_bytes and replay_snapshot_bytes; it is separate from transient scratch.
Tests cover interrupted disk/RAM replay, corrupt images, raw mutations, changed
bindings, checkpoint corruption and the existing full-vs-resumed decision oracle.

## Inference pipeline

Sequential GPU policy can process bounded scene groups before switching from face
to body models (`execution.gpu.sequential_scene_batch_size`; legacy default 1,
shipped profiles 4). Inputs retain explicit ownership and share the run sprite
budget. FaceOcc batches eligible faces across owned cue frames without reassessing
quality; dense frames keep bounded per-cue preparation. ViTPose reuses CPU workers
and prepares the next batch while the coordinator infers the current batch; model
calls/device transfers stay on the coordinator and cleanup joins outstanding work.
RF-DETR transfers equal-sized input frames together and copies only person logits
to CPU, after validating every output class for nonfinite values.
Validation: 145 inference/config/lifecycle tests passed, 1 optional dependency test
skipped. Scheduling tests simulate device ownership; no real CUDA speedup measured.

## Optional mixed precision

`execution.gpu.rfdetr_precision` and `vitpose_precision` independently select fp32,
fp16 or bf16. Defaults remain fp32. Reduced precision requires its corresponding
backend and an available CUDA device; unsupported bf16 is rejected explicitly.
Forward autocast changes activations while weights remain float32, and output
geometry/decoding returns to float32. Non-default precision is part of retained
body/person evidence contracts; historical fp32 fingerprints remain unchanged.
Validation: 108 model/config/inference tests plus 7 precision contract/admission
checks passed on CPU. No real CUDA throughput or detection-quality calibration
was available; enable reduced precision only after comparing detection audits.

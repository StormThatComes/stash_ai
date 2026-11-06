# Recognition run at library scale

The implementation targets 10,000 scenes with roughly 300 VTT cues each. At
70% occupied cues and two detections of each type, that is about 4.2 million
face detections and 4.2 million body detections before association, filtering
and deduplication. These counts are not identity counts or benchmark timings.

## Execution and storage

- Face-associated person evidence can reference its authoritative body embedding
  and crop. Updating or removing that body first materializes the old projection;
  retained face evidence never silently acquires a different feature.
- The duplicate-pixel catalogue persists scan generations and exact representative
  addresses. Only changed scans and affected buckets are recomputed. Similarity
  remains non-transitive and uses the original verified witness.
- `execution.rebuild.storage: disk` is the default. Replay tables, reference vectors,
  body witness metadata and scene plans use private disk-backed storage. The
  `memory` mode retains the original RAM replay for small workloads.
- `execution.rebuild.memory_bytes` limits tracked working allocations. Disk images
  have a separate `replay_snapshot_bytes` budget. Reference telemetry reports the logical
  vector volume, including paged vectors; it is not a process RSS measurement.
- `execution.rebuild.pixel_index_bytes` defaults to 4 GiB. This is a disk index
  allowance, separate from the face reference RAM budget.
- `execution.rebuild.derived_cache_bytes` defaults to 4 GiB, divided into centroid
  (25%), appearance (50%) and other plan (25%) pools. Appearance vectors use
  checksum-verified binary chunks; selected evidence is loaded only after a miss.
- Enabling `execution.rebuild.checkpoint_scenes` also enables durable pure body
  scene plans, global body reports and hybrid reports. Source signatures are
  revalidated before reuse. Main assignments still commit atomically. Checkpoint
  files are optional caches and use `checkpoint_bytes`; they never authorize
  remote publication. The default remains disabled (0).

## Incremental work and exact decisions

Scene and owner dependency clocks cover inserts, updates, deletions, scan heads,
assignment changes and alias changes. A new independent scene does not invalidate
all existing owner banks. Reference sampling persists the complete pre-reduction
state, including evidence independence, and resumes only a canonical append.
Retractions, reordered inputs and changed sample limits take the full path.

Views are cached per owner with their complete source/support proofs and candidate
state. Hierarchical centroid bounds only nominate possible body pairs; they do not
accept merges. Blocked dot products can find an impossible original witness, but
the scalar float32 comparison confirms every early rejection. Near-threshold
cases retain the original exact path. Diagnostic reports retain exact minima;
compact run proofs label minima that are only upper bounds.

## Inference and publication

`execution.scan.detector_batch_size` is independent from face encoder batching.
Native face detector batching requires a compatible dynamic ONNX graph; pinned
static SCRFD graphs retain single-frame inference. Pose collection batches fresh,
enrichment and resumed work through the same bounded queue. Actual GPU throughput
still depends on the backend and available VRAM.

Strong HTTP validators persist separately from cached image bodies. An unchanged
sprite can validate a warm scan after its body was evicted. Actual image use still
verifies body integrity and falls back to a full fetch when necessary. Weak or
missing validators do not authorize this shortcut.

Publication verifies scenes and performer ownership in bounded batches. Only
already reconciled relationships without journal debt can skip publication work.
External removals, changed ownership and other-model links take the normal guarded
path. Any mutation invalidates the remaining prefetched batch.

Portrait rankings compose per-scene caches under the exact global reference
context. A changed scene can retain other scene rankings when that context is
unchanged. A changed rival/reference context invalidates them conservatively.
Winner pixel validation and manual portrait protection still run.

## Validation limits

Regression tests exercise rollback/resume, cache corruption, input invalidation,
canonical sampling equivalence, complete candidate recall and external Stash edits.
The 10,000-scene plan-spool test exercises metadata scale. A complete 3-million-cue
GPU run is not part of the unit suite; end-to-end speedup and peak RSS/VRAM require
measurement on the real library and configured models.

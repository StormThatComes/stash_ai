# Profile-guided recognition optimizations

The reference run processed 2,109 cues from ten fresh scenes in 264.81 seconds,
including 15.77 seconds of retention. Stage timings are inclusive host elapsed
times, not additive GPU kernel measurements.

## Retention

Retention selects obsolete scans and complete superseded person contracts before
examining their references. Empty target sets skip reference indexing and JSON
traversal. Empty obsolete scans and contracts are still removed. When targets
exist, references from every scope are checked before deletion or crop removal.

The disposable reference index now keeps UUID-shaped addresses and existing
legacy observation addresses instead of arbitrary descriptive text. Existing
version-1 indexes are rebuilt transactionally on their next writable refresh.
Non-UUID targets additionally use a full live JSON reference check: a legacy
address inserted after an unchanged source document cannot become unprotected.
Read-only previews fall back to live JSON when the index is stale.

With `execution.rebuild.profile: true`, the run records retention scan/contract
selection, target selection, index refresh, reference protection and contract
actions. Counters report empty target sets, target rows, indexed documents and
inserted index links. No observation identifiers are emitted in these metrics.

## Retained model and sprite memory

`execution.scan.retained_memory_mb` is the shared admission limit for retained
sprite images, verified model bytes and CPU-resident model weights. Existing
per-cache limits still apply. It is not an RSS limit: active models, transient
inference tensors, queues and rebuild allocations retain their own limits.
Owners evict their own entries or decline admission; they never call another
cache while holding a cache lock. This prevents lock inversion between decode
workers and model acquisition. Closing a scope releases every reservation.

`execution.models.verified_cache_mb` caps the verified-byte LRU within that shared
budget. Repeated ONNX sessions reuse immutable bytes verified against the same
pinned SHA256 and size; the first read still verifies the file. This saves read
and hash work, not ONNX session construction. SAM can now park its weights in CPU,
after resetting all image-specific predictor state. Model load, restore, park,
asset verification and ONNX session creation have separate timers; retained
memory peak and failed admissions are counters in the pipeline profile.

## Bounded SAM and masked PMPose batches

`execution.gpu.sam_frame_batch_size` (4) shares a SAM encoder forward between
images; `sam_prompt_batch_size` (8) batches people with equal prompt lengths in
one image. Prompt tokens are not padded. The adapter uses the private prompt and
prediction methods from the already hash-pinned SAM source revision, avoiding
transfer of unused low-resolution logits. Both batching levels halve on CUDA OOM,
retain ordering and propagate errors if even one item cannot run. Predictor image
state is reset on every exit, including failed forwards.

`refinement_queue_mb` (128) bounds the estimated retained CPU masks in a window.
Oversized frames use the existing scalar path; this budget excludes caller-owned
images, temporary mask-validation arrays and GPU feature tensors. The masked
second pass batches people across the window, including after deduplication, and
flushes at window boundaries. Existing mask ownership, drift, limb, overlap and
audit gates still apply. Calls without the batched adapter retain compatibility.

`pmpose_fused_flip` defaults to true: original and flipped crops share one forward,
then the same label reversal and averaging. Activation memory can double; adaptive
OOM handling reduces crop batch size. False restores two forwards. All tensor
validity checks share one scalar GPU readback; probability outputs share one CPU
transfer. New batch-size counters cover SAM images/prompts and PMPose, with
separate SAM readback and PMPose validation/probability-readback timers. Second-pass
people/batch counts and maximum window bytes expose whether larger windows help.

CPU deterministic-model tests compare scalar/batched geometry and record forward
counts, queue fallbacks and simulated OOM retries. These are correctness and
scheduling checks, not a GPU throughput benchmark. Detector, body encoder and
sequential scene batch defaults are unchanged.

## Cache writes and SQLite reads

Inference results are packed once into immutable payloads before ownership passes
to the cache writer. Groups of at most 32 entries share a writer future, quota
admission and SQLite `executemany`; the existing byte queue budget still applies.
Geometry is reconstructed once from that payload, giving each unique result its
own mutable arrays/containers. Duplicate inputs get independent copies. A model
reusing the same array for multiple outputs cannot make those results alias.
Read-after-write waits for the corresponding future. Failures drain reservations,
and `SQLITE_FULL` rolls back and recounts actual database state before reuse.

Read-only rebuild observation and candidate streams use `fetchmany(64)` instead
of per-row profiled `fetchone`. Candidate-index writes read 256 rows per block.
Cursor semantics elsewhere are unchanged, including cursors whose tables change
during iteration. The profiler's SQL fetch calls now count blocks rather than
individual rows; compare total rows and elapsed time. Cache pack/write timers,
written-entry/batch counters and queued-byte peak expose serialization tradeoffs.
Tests verify one write group for 12 results, shared-array isolation, concurrent
admission, rollback recovery and 17 fetches for a 1,000-row stream.

## Optional SAM/PMPose precision and sampled CUDA timings

`execution.gpu.pmpose_precision` and `sam_precision` accept `fp32`, `fp16` and
`bf16`. Both default to FP32. Reduced precision requires the corresponding active
CUDA model; CPU and unsupported BF16 configurations fail explicitly. Weights stay
FP32, autocast covers forwards, and PMPose averaging/decode returns to FP32. SAM
scores are converted to FP32 before NumPy, including the scalar fallback. Raw
PMPose cache keys include its non-default precision; full pose/body evidence
contracts include active SAM/PMPose precision. Face and raw detector caches remain
independent. FP32 contracts retain their previous signatures.

For GPU measurements, enable `execution.rebuild.profile: true` and set
`execution.gpu.profile_every_n_batches: 10` (0, the default, disables it). Pipeline
history then includes `cuda`: completed sampled event times by SAM encoder,
SAM prompt decoder and PMPose forward, sampled allocated/reserved VRAM, pending
samples and samples dropped when the 64-event queue is full. There is no forced
CUDA synchronization. Sampled stage times are not the whole run's elapsed time;
stage intervals can overlap, and sampled VRAM is not a process-wide peak.

Compare FP32 and each optional precision on the same frames using detection audit
before accepting it. Keep scene/cue order, thresholds and cache state comparable;
check mask ownership, keypoint drift, rejected observations and resulting identity
assignments, alongside time and VRAM. CPU tests verify contracts, dtype boundaries,
validation and event scheduling. Actual CUDA throughput and numerical quality
require a GPU run; neither is established by the CPU tests.

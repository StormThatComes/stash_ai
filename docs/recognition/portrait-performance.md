# Portrait performance

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

This page describes execution controls. Selection and image ownership are documented
in [portraits](portraits.md).

VTT portrait refresh can reuse a completed full ranking after verifying the
exact selected memberships/inputs, reference context, assignment proofs,
measurement contracts/bases, target and portrait metadata. Cached ranks do not
skip live ownership/current-image checks before publication or JPEG digest and
framing checks. Missing measurements, recognition-crop approximations, quality
errors and source failures keep the full evaluation/retry path. Corrupt or
changed descriptors also recompute. The cache is bounded by `execution.portraits.ranking_cache_bytes` (16 MiB by
default; zero disables it), rather than a fixed descriptor count. Oversized
results are not cached or truncated. Ranking still scores every eligible VTT
observation in RAM.

Changed VTT quality/candidate metadata writes are committed in batches bounded
by 128 writes/4 MiB, with a final flush before winner preparation/return.
Unchanged metadata does not start a write transaction. Failed tails roll back;
previous complete batches remain reusable. Publication intent, dispatch and
confirmation retain their separate durable transactions. A partial provenance
index covers scope/person/observation lookups without indexing malformed JSON.
Explicit scene selections drive portrait-target discovery from selected scans
in 256-scene chunks; all-scene discovery preserves the retained-scan contract.

`execution.portraits.prefetch` enables bounded VTT preparation (default true;
false keeps the serial path). The metadata phase shares two verified scene
sources through an isolated read owner, prefetches at most two upcoming
thumbnails, and retains at most eight decoded thumbnails. Queue allowances
scale from `execution.rebuild.memory_bytes`, capped at 256 MiB: thumbnail pixels
use at most one quarter, and detached next-target descriptors use one eighth
and at most 128 observations. Oversized sources/frames use original serial
acquisition; no working data spills to disk. Source caches and decoded sheets
have separate bounded item/pixel allowances; model/runtime memory is additional.

The coordinator snapshots descriptors, identity proofs and policy; a source
worker owns its read sessions, a preparation worker handles detached quality
results, and one model consumer serializes visibility/expression initialization,
inference and recovery. Model assets and warning rendering stay on the
coordinator. The next new target can prepare while the current verified image
is published; existing portrait targets keep demand-driven evaluation without
a second history prepass. All eligible observations still reach ordered ranking.
The queues do not predict assignments or mutate memberships.

Preflight performer/scene reads support validated GraphQL aliases in chunks of
32. Null or invalid IDs stay isolated, and an envelope error falls back to
separate proven reads. Initial ownership/image snapshots can be prefetched;
pending/unknown image reconciliation starts from fresh reads, and every
publication still refreshes live ownership and image bytes before dispatch.
Workers never reconcile publication rows or use the coordinator's SQLite.
Image copies remain owned until inference finishes even if their consumer is
interrupted. Shutdown cancels reads, joins owners and releases queued images.

Portrait-only FaceOcc work uses `execution.portraits.faceocc_batch_size` (default
4, range 1–8; 1 keeps scalar inference). Missing/explicitly refreshed measurements
are prepared from bounded detached observations and flushed by face count,
owned-image bytes, scene/batch end and cancellation. Expression inference keeps
its fixed single-face input contract. Unsupported/failed FaceOcc batches release
the session and retain scalar recovery; geometry errors stay local to a face.
Scanning/identity matching do not use portrait microbatches.

Fixed visibility grids/normalizers are immutable and reused; face-specific
buffers remain independently owned until inference completes. CPU mask reduction
retains the NumPy reference path. CUDA checks all logit shapes/finiteness and
reduces the same strict-positive mask inside each real-face ellipse on device,
returning integer counts rather than complete masks. Integer reduction is exact;
batched model inference can retain ordinary CPU/GPU floating-point variation.
The real-model CPU smoke compares scalar/batch fractions within 0.001 while
requiring identical face-region denominators and the established occlusion test.

VTT JPEG preparation uses one ordered, bounded CPU job after ranking; a failed
encoding exposes the next candidate without a shortlist limit. Verified pixels,
geometry and policy transfer to the job, and only the coordinator persists its
result. Small/oversized jobs retain serial encoding. New-target preparation may
overlap encoding once current quality work is complete; existing-prior checks
and publication guards still execute in order.

`execution.portraits.cpu_threads: 0` preserves current native settings and avoids
loading Torch solely to tune an otherwise cached refresh. Values 1–16 opt into a
phase-owned compute budget: 1 selects serial work; larger budgets divide native
Torch/BLAS and portrait ORT intra-op threads among source/encoding work. All
owners join before native settings are restored. This configures compute pools,
not an operating-system quota for coordinator/HTTP helper threads. GPU device
selection remains independent, and no additional model replicas are created.

`python -m benchmarks.benchmark_portrait_quality` probes the pinned FaceOcc
architecture with random weights in isolated CPU processes (not face accuracy).
One Linux/Python 3.14.7 probe of four 256×256 faces measured scalar/batch medians
0.414/0.380 s at one native thread, 0.238/0.234 s at two, and 0.221/0.201 s at four;
sampled peak RSS was 617–636 MB and mask-count deltas were zero on this fixture.
This is a scheduling probe, not a production or CUDA speedup forecast. The
microbatch tensor contains 3 MiB of inputs; four full masks would be 1 MiB, while
the logical returned device counts plus finite flag contain 33 bytes.

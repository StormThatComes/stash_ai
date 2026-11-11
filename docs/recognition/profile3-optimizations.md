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

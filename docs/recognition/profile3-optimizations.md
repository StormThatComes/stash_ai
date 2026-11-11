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

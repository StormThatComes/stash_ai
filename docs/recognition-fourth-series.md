# Recognition fourth series

Base: main c90698b049faa86818100a5192013b7eda145651.
VTT-only, automatic decisions, one inference owner. Every PR starts from the
previous PR. This ledger records implementation status, not a claim that pending
work is implemented.

| PR | Scope | Analysis items | Status |
|---|---|---|---|
| 1 | Representative selection and independent evidence | 1, 3, 8, 9 | Implemented; focused and recognition suite validated |
| 2 | Supplemental evidence and preserved appearance banks | 2, 4, 5, 10 | Implemented; 47 focused tests passed |
| 3 | Conflict resolution and automatic consolidation | 6, 7, 11 | Pending |
| 4 | Incremental dependencies, compact data and memory | 12, 19, 20 | Pending |
| 5 | Parallel matching and change-driven recovery | 13, 17, 18 | Pending |
| 6 | Shared VTT inputs and CPU/inference pipeline | 14, 15, 16 | Pending |

## PR1

A bounded probation bank allows later modes to collect independent support even
when earlier singletons occupied the original reference capacity. Pixel units
are deduplicated before diversity selection and when protected examples are
reintroduced. Better duplicates can return after exemplar eviction; the centroid
keeps its original independent witnesses, so duplicate replacement cannot
reweight it. Supplemental support now uses the same timestamp/pixel matching as
face evidence. Matching/checkpoint versions invalidate old derived decisions.

Focused validation: 62 tests passed, including late supported appearance,
cross-scene pixel copies and replacement after exemplar eviction. No real-corpus
accuracy or hardware throughput benchmark has been performed.

Full recognition validation: 1578 passed, 8 skipped across four shards after
correcting four run fixtures to supply genuinely independent images; the affected
26-test run module was rerun successfully. Skips require optional model/runtime
dependencies. Three unaffected shards passed in the initial run.

PR2 selects body witnesses independently from strict face representatives, retains
separate original face/body veto IDs, and recovers supported withheld queries
against frozen current and original appearance banks as attachments only. Head
association allows 5% detector disagreement while preserving reciprocal
uniqueness. Person input contract v2 intentionally recollects prior associations.

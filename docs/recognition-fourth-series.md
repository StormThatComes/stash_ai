# Recognition fourth series

Base: main c90698b049faa86818100a5192013b7eda145651.
VTT-only, automatic decisions, one inference owner. Every PR starts from the
previous PR. This ledger records implementation status, not a claim that pending
work is implemented.

| PR | Scope | Analysis items | Status |
|---|---|---|---|
| 1 | Representative selection and independent evidence | 1, 3, 8, 9 | Implemented; focused and recognition suite validated |
| 2 | Supplemental evidence and preserved appearance banks | 2, 4, 5, 10 | Implemented; 47 focused tests passed |
| 3 | Conflict resolution and automatic consolidation | 6, 7, 11 | Implemented; 34 focused tests passed |
| 4 | Incremental dependencies, compact data and memory | 12, 19, 20 | Implemented; 43 focused tests passed |
| 5 | Parallel matching and change-driven recovery | 13, 17, 18 | Implemented; 27 focused tests and 28 hybrid tests passed |
| 6 | Shared VTT inputs and CPU/inference pipeline | 14, 15, 16 | Implemented; full stack validation below |

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

PR3 uses complete, outsider-separated nonoverlapping subgroups, retries ambiguous
and co-present strict claims, and journals missing-scope consolidation requests.
The journal never expands publication scope: a later run selecting all required
scans automatically revalidates and drains it. No stale proof is executed.

PR4 adds schema v7 with content-addressed thumbnail BLOBs for new scans, keeping
legacy inline values readable. Replay and offline trials copy only referenced
BLOBs. Audited person/thumbnail triggers protect cached per-scan hybrid hashes.
Pixel comparisons share a content-keyed run cache; hybrid arrays reserve shared
working memory and support prefixes account for retained metadata. Migration,
read-only compatibility and direct SQL invalidation are covered by tests.

PR5 filters hybrid blocks in NumPy before Python enumeration, checks competitors
only on eligible incident identities, and schedules exact proofs on bounded CPU
workers. Frozen rank banks/exact query scores share the run cache. Private
change triggers restrict convergence retries to affected cues; a changed bank
invalidates the restriction. Full-state cycle detection remains a safety check.

PR6 passes the current VTT cue directly from face scanning to person collection.
Only missing cue indices are decoded on resume. Measured person geometry stays on
the inference owner; crop preparation and encoding use bounded CPU workers, with
prepared tensors batched across cues. A complete cue is published atomically,
and failed partial cues are recomputed on resume. Concurrent CPU jobs share an
owned read-only frame copy. Queue budgets remain conservative per job.

Integration regressions cover compact thumbnail copies for pilot/retained-history
replays, preservation of the primary missing-evidence error, and both outcomes
of the joint ambiguity gate: withheld without a complete joint explanation,
attachment-only with a unique explanation. Input digest fixtures changed because
thumbnail IDs now participate in dependency hashing.

Final recognition regression run: 1595 passed, 7 skipped across four shards.
The final cleanup adjustment was additionally checked with 49 focused pipeline,
SapiensID inference, scan, and prepared-pixel tests. Optional model smoke coverage
is run separately in GitHub Actions; no real-corpus accuracy or throughput gain
is claimed from the synthetic regressions.

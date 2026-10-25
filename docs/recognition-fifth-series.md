# Recognition fifth improvement series

The stack targets items 1–20 in the approved analysis. Source inputs remain VTT;
GPU inference has one owner and recovered queries never train strict banks.

PR2: Historical endpoint cores are retained separately for attachment queries.
Body reference quality uses measured body size, pose coverage and confidence.
Complete-link body modes may replace whole-bank comparisons only when every
witness belongs to an independently supported mode and every mode on both sides
has a supported counterpart. Singleton/outlier modes retain the strict fallback.
Original merged cores remain vetoes for later merges. All current identities
remain competitors, and appearances with one canonical label are ranked together.

PR3: Face-contradictory pairs do not remain body-only competitors. Hybrid
recovery also handles missing-body queries through strict face thresholds and
reranks canonical labels after merges. Newly created face identities can attach
to frozen pre-existing targets before publication; all their queried members
must agree, and new identities cannot train one another through this route.
Joint cue assignment uses bounded constraint search without truncating domains.
With complete body coverage, an ambiguous face solution may be resolved using a
separate body ranking while retaining final face/co-presence rival guards.

PR4: Reuse immutable hybrid catalogues under the run cache budget. Query recovery
uses blocked centroid nomination with conservative tolerance and exact witness
finalists. Scalar face-query results retain the original NumPy reduction and are
cached by immutable input bytes. Copresence joins run in blocks of 64 pairs;
oversized proofs alone fall back to serial computation. Schema 8 adds audited,
transactional raw-feature stamps, with migration from schema 7, so assignment
updates do not rehash unchanged face/body inputs. Pixel alias discovery is lazy
on warm paths. Birth recovery now requires every original member, including
members excluded by query eligibility, and excludes rejected cohorts from votes.

PR5: Persist bounded compressed VTT/sprite responses next to the recognition DB.
Reuse requires a strong HTTP ETag and successful conditional origin request;
verify cached bytes, discard corrupt payloads, honour no-store, and fall back to
full reads when validators are absent. `execution.media_cache_bytes: 0` disables
this disposable transport cache. Prepared body inputs use two bounded batches
so CPU preparation can overlap the single owner's GPU inference. Queued CPU
results retain reservations until consumed. Inline face/body evidence is written
with the cursor in one grouped transaction; failure replays from the previous
checkpoint. Fresh inline rows no longer need per-cue SQLite rereads.

PR6: The pending consolidation queue now has a consumer. By default it stays
within selected scenes. Set `sapiensid.consolidation.scope: retained` to allow
automatic replay expansion within the current Stash library and model namespace;
`scene_ids` optionally restricts it and `max_scenes` bounds the work. Both existing
and newly queued requests are revalidated against complete current scans and a
fresh frozen catalogue. Old queue scores never authorize a merge. Expanded scenes
receive publication debt; HTTP publication still uses the caller's selection.

Stored performer ownership is read in batches. Scene links are added in groups
of at most 32 using ADD, with all per-relationship intents committed before the
request and fresh post-write scene reads. Unknown batch outcomes are reconciled
by observation and never blindly retried. Manual pre-existing links retain their
unmanaged provenance. Matching format 53 invalidates earlier decision caches.

The six PRs are an ordered stack: evidence banks (#961), historical/body banks
(#962), decisions (#963), compute reuse (#964), input/checkpoints (#965), and
consolidation/publication. Targeted tests cover migrations, exact-score parity,
input corruption, CPU/GPU overlap, interrupted checkpoints, stale requests,
ownership changes, manual relationships and lost HTTP acknowledgements.

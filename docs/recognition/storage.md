# Recognition state and maintenance

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Schema and backups](#schema-and-backups)
- [Internal media and state](#internal-media-and-state)
- [Recognition database status and VACUUM](#recognition-database-status-and-vacuum)
- [Recognition evidence retention](#recognition-evidence-retention)
- [Orphan cleanup and switching recognition models](#orphan-cleanup-and-switching-recognition-models)

## Schema and backups

One face SQLite stores model/scan provenance, detections, normalized embeddings,
bounded crop BLOBs, accepted identities, Stash ownership/bindings, candidate
portraits, mutation records, progress and run history. Its independent schema
refuses tag/history/foreign databases and hard-linked aliases. Internal backup
uses SQLite's consistent backup API; no result/history/image sidecars are needed.
The source connection must have no active transaction before backup; callers
must explicitly commit or roll back first. Backups publish a complete staged
file exclusively and preserve any existing destination. A setup failure removes
the owned staging file; a process crash can leave a hidden staging file.

Opening state validates column defaults, required constraints and the installation
identity before migration or interrupted-run recovery. Unknown triggers and views
are refused. Close the SQLite owner on its creating thread: a rejected close or
a failed close that leaves SQLite live retains the writer lock.
Writer initialization upgrades supported schemas v1–v10 atomically to v11.
Existing observations, IDs, performer bindings, merge aliases, support and history
are preserved. Schema validation rejects incomplete or foreign databases before
migration. Read-only maintenance never performs this upgrade.
After acquiring this database's exclusive writer lock, a new owner marks prior
`running` recognition histories interrupted regardless of instance scope or
selected scene order. Their saved summaries/progress and pending or unknown
Stash intents remain intact. Runs started by the current owner are left active;
another writer cannot acquire the database while that owner holds it.

## Internal media and state

Each media acquisition has one elapsed budget covering redirects, headers and
body reads, including trickling responses. The video bridge shares one budget
across its seeks and cancels owned socket reads when closed. GraphQL mutations
also use one bounded attempt: expiry before dispatch is a known failed write;
after possible dispatch it is unknown and requires fresh-state reconciliation.
No elapsed timeout automatically retries a mutation. Owned HTTP workers have
shared bounded capacity; expired results cannot authorize subsequent writes.
Stored source URLs omit username/password userinfo, queries and fragments.
Recognition diagnostic URLs use the same sanitized source identity; authentication
still uses the original request/header path. Selected cached metadata and copied
retry-error snapshots are sanitized without reloading models or rewriting images.

Each selected scene verifies the VTT/sprite content before reusing observations.
The sprite cache keeps at most two active sheets. Completed cue checkpoints
survive an interrupted inference; incomplete
scans cannot publish identities. Partial sampling and rejected/no-face outcomes
remain explicit coverage information, not proof that a person is absent.

The detector and recognizer are initialized lazily and shared across VTT scenes.
By default, after VTT collection finishes, recognition closes the face, visibility,
person/pose and SapiensID models before identity rebuild and Stash publication.
ONNX sessions are released, Python garbage is collected, and unused PyTorch CUDA
cache is returned. Portraits lazily reload only the models they need. A fully
cached run does not load inference models just for cleanup. An inference failure
discards the failed backend so a later scene can initialize a fresh instance.

## Recognition database status and VACUUM

These offline commands operate on the **whole recognition database**, across
all scopes and model namespaces. They never contact Stash or load models:

```powershell
# Health, all table/index sizes, total disk use and estimated space savings
python .\stash_ai.py recognition db-status

# Preview without compacting
python .\stash_ai.py recognition db-vacuum --dry-run

# Execute VACUUM and report before/after disk use and actual savings
python .\stash_ai.py recognition db-vacuum

# An explicit database, or select storage.database using --face-config
python .\stash_ai.py recognition db-status --faces-db D:\stash\faces_state.sqlite3 --json
```

`db-report` and `db-compact` are aliases for `db-status` and `db-vacuum`,
matching the tagger's maintenance terminology. Both commands accept
`--face-config`, `--faces-db`, `--json` and `--progress-style off`.
The destination defaults to `storage.database` in the current recognition YAML.
The database must already exist; maintenance never initializes or migrates it.

Status includes SQLite/schema versions, `quick_check`, row counts and a table
ranking by occupied bytes. With SQLite's [dbstat](https://www.sqlite.org/dbstat.html)
extension, each table has separate data and index sizes (including overflow
pages and automatic indexes), their combined size, and unused bytes inside
occupied pages. Table/index totals, schema/other pages and freelist pages account
for the logical allocated database size. Physical disk totals separately include
the main file, WAL, SHM and rollback journal; WAL bytes are not allocated database
pages and must not be counted twice. Disposable media/cache companion databases are
outside this command's scope. Physical file sizes can change during a concurrent
writer, while SQL measurements belong to one consistent read snapshot.

Some SQLite builds lack `dbstat`. In that case the report explicitly marks
physical per-table sizes as unavailable and ranks tables by logical value bytes:
actual BLOB/UTF-8 lengths, with numbers measured in text form. This fallback
excludes indexes and SQLite overhead; it is **not a physical size estimate**.
Whole-database sizes, freelist estimates, row counts and VACUUM remain available.

The conservative VACUUM estimate is `freelist_count * page_size`. Unused bytes
inside occupied pages are shown separately, without promising that all can be
reclaimed. [VACUUM](https://www.sqlite.org/lang_vacuum.html) can also repack those
pages, so actual savings can differ. The final report measures total disk use
before/after; JSON additionally separates the logical allocated-byte reduction.
An active WAL can make disk savings differ from the main-file reduction.

VACUUM uses recognition's existing exclusive writer lock, validates the database
before and after, checks temporary disk headroom (up to twice the logical/main
database size plus existing WAL and 16 MiB), and checkpoints/truncates WAL.
Another recognition writer or a busy pre-VACUUM WAL checkpoint causes a refusal.
Afterward, health, table row counts and aggregate states are verified. It retains
all evidence, embeddings, crops, IDs, bindings, pending/unknown publications and
history; even stale `running` history rows are left unchanged. It does not invoke
the normal database opener's orphan recovery. Supported old schemas are compacted
in place without an upgrade. Status and `--dry-run` do not acquire the writer lock.
Progress reports the current phase and elapsed time; SQLite does not expose an
exact VACUUM completion percentage.

## Recognition evidence retention

Changing detection settings creates a new immutable scan. Person/pose settings
can also create multiple SapiensID contracts inside the same scan. Retention
limits obsolete evidence without deleting identity labels or Stash performers:

```powershell
# Read-only preview for the model and detection policy in the selected YAML
python .\stash_ai.py recognition db-prune --face-config facial_recognition.yml
# Apply a freshly recomputed plan under the exclusive recognition writer lock
python .\stash_ai.py recognition db-prune --face-config facial_recognition.yml --apply
# Optionally limit the operation; --scene-id is repeatable
python .\stash_ai.py recognition db-prune --scene-id 123 --scope YOUR_STASH_SCOPE --json
# Reclaim physical free pages separately, after checking the pruning report
python .\stash_ai.py recognition db-vacuum
```

`db-prune` defaults to preview; `--dry-run` is an explicit alias for that mode.
It accepts `--faces-db`, `--face-config`, `--scope`, `--scene-id`, `--json` and
`--progress-style`. It never contacts Stash or loads inference models. It requires
the current schema; it does not migrate an old database or recover run history.
The configured model namespace is the boundary, across all scopes unless limited.
A different active detection policy is reported as blocked, not silently pruned.

```yaml
version: 2
storage:
  retention:
    auto_prune: true
    keep_previous_scans: 1
    keep_previous_person_contracts: 1
    prune_obsolete_crops: true
```

Automatic pruning is **opt-in** (default `false`). It runs only after a fully
successful `recognition run`, including resource cleanup and finalized history,
and only for its selected scenes. Portrait-only commands, failed/partial runs,
and interrupted runs do not authorize pruning. The result is printed and saved
as the run's `retention` progress record. A maintenance failure is reported without
undoing completed recognition/publication work.

Automatic retention has its own `Recognition auto prune` phase. Its main bar
tracks planning, pruning and commit (3 steps); 100% is reached only after the
transaction is committed. A detail row counts inspected scenes/contracts or
completed deletion statements, while the status names the active operation
(including reference checks, crop clearing, commit and rollback). These are
work counters, not a percentage of database bytes or a guaranteed time estimate.
Redirected runs keep emitting the recognition heartbeat during long SQLite
operations. `--progress-style off` and `db-prune --json` remain quiet.

During planning the overall bar stays at `0/3` until the complete plan is ready.
JSON reference checks show a separate `Inspect references: table.column` bar
with completed/total source records, updated every 256 records. The planner
visits source documents in row order and looks up referenced observations by
their indexed IDs, avoiding repeated JSON traversal for every possible target.
References from other scenes/scopes are still checked before evidence is removed.

The active scan plus one previous complete scan are retained by default. Within
eligible retained scans, the configured complete person/body contract plus one
previous contract are retained. Counts can be set from 0 to 100. These are
retention targets, not hard database-size limits: incomplete scans, remembered
head creation markers, original identity ownership, memberships, merge support,
and SQL/JSON audit references can keep additional evidence. An active scan needs
a successful rebuild run before it may supersede older scans; unresolved
publications and queued scene reconciliation block the entire scene.

Unreferenced obsolete scans/contracts are removed. Identity-pinned old evidence
keeps its numerical vectors, geometry and origin bindings. With
`prune_obsolete_crops: true`, unnecessary image BLOBs may be cleared while current
membership/portrait/publication images remain protected. Retention does not alter
people, origins, memberships, merges, publications or run history. Shared matching
thumbnails are removed only when their deleted observations leave no references.

Within an obsolete body collection variant, only the individually referenced
observations are pinned. Unreferenced peers are deleted even when they share the
same cue or contract. Retention atomically removes that variant's completed-cue
markers, so its surviving audit references cannot masquerade as a complete
collection. Selecting it again requires recollection. Retired fragments remain
eligible for cleanup if their final references disappear; genuinely interrupted
collection attempts remain protected for resume. The configured previous-variant
allowance and opt-in `auto_prune` setting still apply.

Returning to a completely removed variant requires detection again. Returning
to an intact archived variant reuses its numerical evidence and restores missing crops
from verified VTT input without model inference. Until then, historical image
exports may report missing crops. Active and retained previous variants keep
their images. Existing body-crop retention and byte limits still apply.

The report separates deleted rows from cleared crops, lists scan protection
reasons, and estimates removed **logical evidence payload bytes**. It excludes
indexes, bookkeeping and page fragmentation, so it is not a promise of physical
file savings. Freed SQLite pages can be reused by subsequent writes; run
`db-vacuum` separately to shrink the file. Retention preserves additional pinned
history, and does not deduplicate active crop BLOBs or impose a hard disk quota.

## Orphan cleanup and switching recognition models

`recognition run` now permits other recognizer namespaces to have published
profiles. For **successfully rebuilt, selected scenes only**, publication attaches
the current recognizer's identities and detaches verified recognition-owned
performer links from *other* model namespaces. Old performer profiles, portraits,
local face evidence and links on unselected/failed scenes are retained. Existing
manual (non-recognition-owned) performers remain attached. Unlike ordinary
same-model stale-link cleanup, even a manually added association to an owned
**other-model** performer is detached for selected scenes. Ambiguous ownership or
uncertain unlink results fail the selected scene's publication rather than blindly
removing unrelated links. `recognition portraits` does not change scene links.

To inventory Stash profiles carrying recognition ownership tokens that have
**no matching identity in the selected face database**:

```powershell
python stash_ai.py recognition cleanup --faces-db faces_state.sqlite3
```

The default is read-only; an explicit apply deletes the verified orphan
**performer profiles** (which also removes their Stash associations):

```powershell
python stash_ai.py recognition cleanup --faces-db faces_state.sqlite3 --apply
```

This separate command never loads recognizer models, never scans scenes and
never runs implicitly on `recognition run`. The existing SQLite file is required:
a missing path does **not** create an empty database that might make every
profile appear orphaned. The inventory includes recognizer-owned profiles from
any installation represented by a valid recognition ownership token on this
Stash instance. Use the complete database for that instance: another independent
recognition installation will otherwise look orphaned. Profiles with manual or
unrecognized details, or with *any* identity token / performer binding still in
that database, are protected. Each deletion is checked against fresh Stash data
and recorded as a durable operation; uncertain dispatch is never blindly retried.
The dry run and apply record their counts in the face database's run history.

Schema 14 separates retained image payloads from frequently read face/body rows.
New crops use immutable SHA-256 addresses in `crop_payloads`; identical bytes are
stored once. Existing inline images remain readable and are relocated when their
scene is reused, without inference. Reports and portrait fallback resolve and
verify the address. Body projections keep their existing copy-on-write behavior.
Pruning collects unreferenced image payloads; reclaiming SQLite file space still
uses the explicit maintenance/VACUUM action.

Rebuilt durable candidates share policy and resource-admission documents in
`decision_contexts`. Reports expand the original document, and cache validation
compares the original limits. Schema migration creates the access paths without
rewriting all retained image data or starting a full VACUUM.

For libraries with millions of detections, `storage.crop_policy: selected` skips
JPEG/PNG encoding during collection while retaining the same embeddings,
geometry, raw pixel signatures and duplicate-evidence thumbnails. The default
`all` preserves fully offline image reports. Portrait selection can reconstruct
an omitted body crop from the verified VTT source without loading a model.
Switching back to `all` backfills missing crops when a scene is reused.

To reconstruct omitted images only for one requested identity report:

```powershell
python stash_ai.py recognition report-identity --performer-id 123 --output-dir report-123 --restore-crops
```

This flag reads the configured Stash instance. The report still leaves SQLite
unchanged; source scope, VTT fingerprint, cue and retained pixel signatures must
match. Without the flag, reports remain offline and mark omitted crops missing.
A changed or unavailable source cannot substitute new pixels for old evidence.

Supplementary collection checkpoints completed cues together using
`execution.scan.checkpoint_cues` and the scan queue byte budget. If a later cue
fails, only earlier completely inferred cues are committed and reused on retry.

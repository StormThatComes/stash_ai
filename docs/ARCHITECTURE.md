# Module boundaries

Two command-line Python entry points remain at repository root:
`stash_ai.py` and `update_stash_ai.py`. The `stash_ai.py` launcher
acquires the installation lock; `stash_ai/cli.py` coordinates CLI
dispatch and pipeline execution. Evidence/fusion and interval-union
algorithms live in `stash_ai/evidence/pipeline.py`.
The `stash_ai/publication/` package handles planning and publishing
shared `ai:` results. Its `markers.py` holds common marker-ownership,
payload, change-comparison and excluded-extra removal rules.
Modules and tests import these rules directly rather than
through CLI re-exports.

`commands/dispatch.py` invokes CLI modes and wraps `MergedRun.run()`
with a diagnostic context via `run_merged()`. Reconciliation calls
use `publication/merged.py` and `cleanup/scenes.py` directly, while
CLI starts cleanup through `CleanupRun.run()`. Code and tests
import the codec from `stash_ai/storage/codec.py`.

`stash_ai/runs/merged.py` / `MergedRun` owns the lifecycle of
a merged-output run: scanning, rethresholding, inference, publication,
and pruning. It opens and closes the Stash client and `StateStore`.
`runs/merged_model.py` / `MergedModelRun` manages one model's scene
eligibility, reuse/refresh, and per-scene inference. It shares
the store, cache, counters and report with `MergedRun`, loading
the tagger only after exhausting reuse options and closing it
in `finally`. An inference error affects one scene; a model-load
failure ends that model's loop.

The publication phase is implemented in
`stash_ai/publication/publisher.py` / `MergePublisher`. It checks
pending refresh and publication cache, selects sequential or parallel
execution, and centralizes publishing, counts and errors without
taking ownership of run resources. Optional
`stash_ai/publication/parallel.py` runs **read-only** evidence
replay in isolated processes; `MergePublisher` remains the single
writer for `TagResolver`, Stash, and `StateStore`. Dry-run and
profiling stay sequential, and worker failure never triggers
a second publication pass.

`stash_ai/publication/planning/` separates preparation of evidence,
tags and markers for each scene. `types.py` defines `MergeSyncPlan`;
`publication/merged.py` coordinates mutation order and snapshot
commit. Cleanup has its own lifecycle in
`stash_ai/cleanup/run.py` / `CleanupRun`. `cleanup/scenes.py`
reconciles a single scene, and `cleanup/evidence.py` re-evaluates
published evidence, including eligibility for changed evidence,
lazy `EvidenceReplay` with a signature index/replay cache, and
clipping to published intervals. `cleanup/state.py` contains cleanup
state and masks, `cleanup/client.py` reads tag catalogs, and
`cleanup/prune.py` handles ordinary tag pruning. Remaining
domain code includes `evidence/scenes.py`, `observations.py`,
`temporal/` and `occurrences.py`.

`stash_ai/publication/index.py` builds a per-scene immutable
index of existing markers by primary tag. Sorted starts and
a maximum-end tree reduce candidates even for nested intervals;
the final filter uses `merge.overlaps()` with a separate start
tolerance. Cleanup resolves markers by primary-tag name,
while planning uses IDs after refreshing the scene. For point
outputs, `PreviousMarkerRanges` indexes rounded starts from
published, pending and recovery state while retaining the largest
matching end. Normalization happens once per indexed marker.
Matched markers remain in the index so their manual extras
can contribute to subsequent fragments; the publication plan
still controls whether they can be matched again. These indexes
are memory-only and change neither snapshot format nor DB schema.

`runtime/settings.py` holds per-run settings, while
`runtime/context.py` binds them to explicitly injected services.
CLI creates a new context per invocation; domain modules never
import CLI. Shared `runtime/workers.py` opens isolated read-only
state and settings for merge and model-report worker processes
without loading CLI services.

`stash_ai/cli.py` coordinates one publication pipeline.
The `stash_ai/inference/` package owns cue/sprite retrieval,
image preparation, score filtering and resumable inference.
Fresh inference uses the temporal engine; frozen replay
also uses only that engine. Offline upgrades preserve old
contribution geometry, manual tag IDs, ownership and
pending/recovery intent while removing obsolete build options.
Cleanup rejects contributions marked `requires_rebase`;
fresh inference and merge replace them with current proofs.
Offline snapshot upgrades are supported from schema v18 onward.

Active `runs/separate.py` and `publication/separate.py`
modules and their planners were removed. Migrating
source-prefixed output to `ai:` requires one-time fresh
inference using the historical migration bridge.
Run-dependent functions take an explicit `app` argument,
and the JoyTag image-preparation callback binds it with
`functools.partial`.

`stash_ai/inference/inputs.py` defines `VerifiedMedia`,
`RunMedia`, `RunInputs` and `inspect_input()`. It checks
VTT/sprite inputs, constructs identities, and shares a
bounded RAM-only cache of media and manifests between models.
Code and tests import these symbols directly.
`inference/sprites.py` manages cues and sprite decoding;
`inference/preparation.py` performs image preprocessing
and bounded, ordered batching, including worker ownership
and image cleanup. With prefetch, the transformation window
covers frames across several batches, and backend inputs
are assembled before inference.

`inference/batches.py` runs ordered inference on one thread,
with a bounded CPU result queue and shared cancellation
of media preparation. `runs/merged_model.py` provides one
executor for model loading, inference across all scenes,
and closing on that same thread. A scene iterator awaits
its task but does not close the borrowed executor.
`inference/prediction.py` restores prediction state,
processes outputs/checkpoints in the SQLite-owner thread,
and finalizes the scene.

`inference/input_prefetch.py` verifies the next scene's
input in the background over isolated HTTP transport.
`inference/batch_stream.py` decodes its images and fills
the bounded `prefetch_batches` CPU-batch queue
(0 disables; maximum 128; default 4). Each active queue
has its own limit, and prepared inputs are outside
the RAM cache budget. Current and next scene share
one model-pass transformation pool; ending one scene
waits for its images without closing the pool.
Input verification and up to two CPU producers have
separate executors/HTTP sessions; a full current-scene
queue cannot occupy the executor needed by the next scene.
Synchronized `RunMedia`/`RunInputs` maps preserve one
snapshot of URL contents and manifests under parallel reads.

The reuse plan chooses candidates before inference without
extra SQLite reads. Actual reuse, peer invalidation and
checkpoint restoration happen on the main thread before
GPU execution. `runs/model_scene.py` stores state for one
authorized scene. When current-scene batches are consumed,
finalization can overlap with eager `BatchPredictions`
for the next scene on the same backend owner. Distinct
observers attribute history/proofs to the correct scenes.
Slow or failing next-scene input does not delay finalization;
future-scene errors are reported when that scene is reached.

SQLite writes remain ordered on the main thread.
The GPU result queue holds `result_prefetch_batches` batches
(1–128, default 32), plus one pending insertion or one
in-flight inference call. The limit is independent of
the input queue and backend ownership. Resume discards
early CPU look-ahead and runs only frames after the saved
checkpoint. Cancellation stops future work and discards
unprocessed results.

`inference/identity.py` constructs model identity.
`inference/checkpoints.py` defines `Checkpoint`, compact
state persistence/restoration and atomic inference completion.
`stash_ai/storage/progress.py` manages completion indexing,
result reuse and resume/refresh. `stash_ai/signatures.py`
defines shared `digest()` for input, checkpoint and
publication signatures. Canonical JSON, hashes, BLOB format
and schema remain stable, as does the atomic sequence of
writing results, removing checkpoints and completing refresh.
`inference` does not re-export helpers from unrelated modules.

`stash_ai/commands/` holds run-mode boundaries.
`arguments.py` defines the parser. `validation.py` splits
report, DB/maintenance and pipeline validation while
preserving error order and option normalization.
`dispatch.py` handles DB commands before runtime setup,
standalone commands before model loading, and explanation/
merge-only before vocabulary/weight access.
`explain.py` contains read-only tag diagnostics;
`hierarchy.py` handles shared hierarchy and standalone
synchronization. CLI imports modules directly without
legacy-path wrappers.

All models use one `tag_hierarchy.yml` through `--hierarchy`.
Bootstrap parses it once and checks coverage of the
selected vocabularies **before contacting Stash**.
Vocabulary metadata describes the union of all labels
in the file. `--merge-only` reads the same YAML without
access to model assets. A label is removed from the
shared hierarchy only if excluded across all model
policies; per-model exclusions still apply when
publishing or creating tags. Hierarchy changes
affect publication fingerprint, not inference identity.
`stash_ai/config.py` parses model configuration, exclusions
and hierarchy; `stash_ai/vtt.py` parses WebVTT cues.
CLI arguments, YAML, VTT and special run paths can
therefore be tested independently of normal inference.

`stash_ai/reports/review/report.py` coordinates offline
evidence replay for prioritized tag review.
`reports/review/scenes.py` and `markers.py` compute signals
for the two evidence kinds; `options.py` defines signal
meaning and settings. `accumulator.py` implements
`ReviewAccumulator`, deduplicating global/per-model counts,
reasons and example limits. The report loads no models,
does not contact Stash and skips evidence awaiting refresh.

`stash_ai/reports/model/report.py` builds offline
leave-one-model-out analysis from retained evidence.
`analysis.py` calculates scene baselines, support
attribution and counterfactuals; `comparison.py`
compares semantics and marker overlap. `report.py`
coordinates scene traversal and output rows.
`reports/model/workers.py` initializes isolated read-only
workers, divides memory budgets and reduces worker
results into the sequential report's format.

`stash_ai/models.py` defines `ModelSource`,
`ModelSettings` and `Vocabulary`, the source registry,
and model/prediction validation without depending
on loaders or inference adapters.
`stash_ai/backends/loader.py` contains the JoyTag
adapter and `load_tagger()` dispatch; model files
and tag vocabularies are handled in
`stash_ai/backends/vocabulary.py`.
`backends/joytag.py` validates/loads the pinned
upstream architecture. ONNX responsibilities are
split between image transformations in
`backends/onnx_preprocessing.py`, provider selection,
CPU fallback and I/O inspection in
`backends/onnx_session.py`, and inference adaptation
in `backends/onnx.py`. Code, tests and benchmarks
import types/helpers from their defining modules;
`backends/loader.py` does not re-export the old API.
Shared types and validation prevent loader/adapter
dependency cycles.

Generic components (`TagResolver`, hierarchy loader
and source cleanup) require explicit `ModelSource`;
JoyTag is not an implicit default namespace.
Internal contracts are explicit: `select_scores()`
accepts NumPy arrays; retention and inference
signatures use complete `ModelSettings`; and
`MarkerIntervalIndex` requires a built
`PreviousMarkerRanges`. Frame iterators and
batch preparation operate on PIL images: workers
copy their images and close them after transformation.
Test fixtures observe the same contracts.

## Stash communication

`stash_ai/stash_api/` groups the domain client, HTTP transport,
SQL scene-snapshot reads and tag resolution. `StashClient`
remains a cohesive subclass of `StashTransport`.
`sql.py` contains only reads, and `tags.py` manages the tag
catalog and hierarchy. Callers import directly from these
modules without legacy-path wrappers.

| Module / class | Responsibility |
|---|---|
| `stash_ai/stash_api/transport.py` / `StashTransport` | HTTP sessions, authentication, read retries, dry-run write prevention, GraphQL decoding and media redirects |
| `stash_ai/stash_api/client.py` / `StashClient` | Scene, tag and marker queries; enforcing Stash 0.31.1+; pagination and SQL-to-GraphQL fallback |

`StashClient` specializes `StashTransport`. The shared
transport maintains one connection configuration and
request report. Import the client from
`stash_ai.stash_api.client` and HTTP helpers directly
from `stash_ai.stash_api.transport`. Other packages
use equivalent direct imports, for example
`from stash_ai.storage.store import StateStore`.

The transport retries only reads, including
strictly allowlisted SQL operations that happen to be
wrapped as GraphQL mutations. It does not retry
writes. The `ApiKey` header is not sent with
cross-origin media redirects.

## SQL-based cleanup

`stash_ai/cleanup/sql_session.py` implements
allowlisted SQL/API transport, read/write metrics,
and keyset pagination. `stash_ai/cleanup/sql.py` /
`SQLCleanup` remains the domain layer: it discovers
candidates, removes marker relationships, handles
previews and pruning. It uses its `CleanupSqlSession`
directly for reads, writes, pagination and discovery
boundaries. SQL constants and read validation are
imported from `cleanup/sql_session.py`.

## Local state

`stash_ai/storage/` manages the database, result and
publication stores, serialization, metadata, snapshots,
preview overlays and completion indexes. Callers import
from specific package modules. This arrangement does
not change schema version, database identity, BLOB format,
or default SQLite path.

| Module / class | Responsibility |
|---|---|
| `stash_ai/storage/database.py` / `StateDatabase` | SQLite connection lifecycle and process lock, database ID/schema checks, base tables and opening the dry-run source |
| `stash_ai/storage/store.py` / `StateStore` | Facade owning lifecycle, cache and serialization; coordinates schema initialization and delegates persistence |
| `stash_ai/storage/results.py` / `ResultStore` | Inference results and source indexing |
| `stash_ai/storage/publications.py` / `PublicationStore` | Published/pending snapshots, durable publication intent and atomic promotion of prepared publications |
| `stash_ai/storage/initialization.py` / `InitializationStore` | New-database rebuild scope, adopted tag ownership and pre-publication completion barrier |
| `stash_ai/storage/preview.py` | Dry-run overlay views, overlay deletions and source-change detection |
| `stash_ai/storage/codec.py` | BLOB encoding and decoding |
| `stash_ai/storage/snapshots.py` | Versioned snapshots with shared `extra_support` proofs and reconstructed contributions on read |

`StateStore` owns a `StateDatabase`, `ResultStore` and
`PublicationStore`. Its `records()/put()/snapshot()/prepare()`
methods delegate to the responsible owner. Normal
writes use `ResultStore.put()` and its transaction.
Inference completion calls `ResultStore.put_in_transaction()`
in the same transaction as checkpoint and refresh-request
removal. `StateStore` closes its database even if
initialization fails. The process lock spans the entire
store lifetime, including requests to Stash.

Rethreshold validates and applies changes using compact
`result_metadata` rows.
`ResultStore.set_threshold(..., metadata_only=True)`
writes thresholds and completion indexes in one
transaction without reading frame BLOBs.
The same write path supports full records for
inference reuse. Dry-run requires the current
schema and uses an overlay over a read-only source.
Only `--db-update` upgrades old databases.

`StateStore.pack_snapshot()` interns repeated proof
objects by identity, without hashing large structures.
Unique proofs remain inline. The compact BLOB contains
`__stash_ai_snapshot__: 3`, `snapshot` and a `proofs`
array; `extra_support` values are complete proofs or
indexes into that array. `storage.snapshots.decode()`
validates references and returns an expanded structure
compatible with cleanup, recovery and DB checks.
Shared proofs are immutable for borrowing readers;
public reads still return copies separate from cache.
Historical snapshots and database versions are updated
only offline via `--db-update`. Pending promotion and
recovery archival copy stored BLOBs without re-encoding.

Standalone database commands live in `stash_ai/db/`:

| Module | Responsibility |
|---|---|
| `stash_ai/commands/database.py` | Dispatch `--db-report`, `--db-check`, `--db-update` and `--db-compact` before runtime initialization, importing maintenance functions directly |
| `stash_ai/db/update.py` | Explicit offline whole-database upgrade via existing migrations; read-only preflight and dry-run |
| `stash_ai/db/inspection.py` | Read DB state/storage and format aggregate reports |
| `stash_ai/db/checks.py` | Coordinate deep checks in one read transaction; format errors and warnings |
| `stash_ai/db/validation/` | Validators for schema, result/index/thresholds, checkpoints, payloads and cleanup/recovery relationships, with shared decoding/progress |
| `stash_ai/db/compaction.py` | Process lock, safe `VACUUM` preconditions and before/after aggregate checks |
| `stash_ai/db/common.py` | Shared supported-schema definitions, read-only SQLite connection and query helpers |

`stash_ai/reports/developer/report.py` uses schema
definitions and read helpers from `stash_ai.db.common`
but owns its anonymous JSON report format and output.
`developer/database.py` gathers database/checkpoint
aggregates; `developer/evidence.py` analyzes result
shape and compression; `distribution.py` holds
memory-bounded distributions. The `db` package does
not invoke migrations or change `StateStore` locking.

Anonymous diagnostics format 2 scans detailed evidence
only on the current schema. Supported v18–v20 databases
return aggregate facts and `upgrade_required` without
historical payload adapters. Current evidence counts
expose only scene/frames, inference origin and current
analysis format; obsolete values map to `other`.
Retired import tables are never parsed; DB checks
report their presence as a schema error.
Tag explanations return a scene-report list with
`format: 2` and one `record` per source rather than
a singleton `records` list.

`stash_ai/rethreshold/` separates local threshold
changes from evidence projection. `targets.py`
selects scene/model pairs from small indexes and
groups validation errors. `markers.py` checks marker
coverage and writes marker thresholds; `scenes.py`
plans scene updates and optionally marker updates.
`execution.py` coordinates all three modes: full
preflight, batch metadata backfill and abort on any
failure, followed by `Apply` in the existing
per-pair transaction. Marker-only still blocks
pending publication even for unchanged thresholds;
scene/combined modes allow pending no-op.
Check/Apply use metadata and preserve raw evidence
BLOBs while atomically writing threshold and index.

`evidence/thresholds.py` owns effective marker
thresholds, signatures and retention contracts;
`evidence/cache_views.py` handles scene projections
and converting observations to frames. Reuse calls
rethreshold plans directly without old wrappers.

The updater installs Python files from `stash_ai/`
and its subdirectories. It remains a standalone
stdlib script so interrupted-update recovery works
even with an incomplete application package.
`RemoteSnapshot` identifies one pinned commit,
and `UpdatePlan` contains operations, conflicts
and the future installation state. `prepare` writes
all files and backups before publishing a journal.
`PreparedUpdate` conveys the identity and directory
to the helper. `UpdateTransaction` validates
durable plans and owns apply/rollback through
`Phase` states. `run_helper` manages lock handoff,
removes tokens from the environment and tracks
process status. Metadata format 1 remains compatible
with the previous updater; none of these classes
requires importing the application package.
For legacy updater migration, see
[the main README](../README.md#updating-the-application).

YAML configuration, the default
`tagger_sources.sqlite3` database, and `models/`
remain next to the launchers. Default paths resolve
from the installation directory, not the terminal
working directory. Tests live in `tests/` and
benchmarks in `benchmarks/`.

Modules that can change merged publication for
the same policy, retained evidence and observed Stash
state must be included in `PUBLICATION_CODE_FILES`
in `stash_ai/publication/fingerprints.py`. Transport,
CLI, VTT parsing and maintenance/cleanup have
separate invalidation signals and are intentionally
excluded: refactoring them should not trigger a
library-wide re-merge. A changed publication
fingerprint does not invalidate raw inference cache.

## Cache coverage report

`stash_ai/reports/cache.py` streams compact result
metadata and checks indexes, refresh requests,
checkpoints and retention against selected YAML.
It reads from a read-only source with sparse preview
and generates advisory `argv` for inference,
rethreshold and merge without executing them.
It does not establish current media or model
identity; deep payload validation belongs to
`--db-check`.

`stash_ai/reports/` groups offline reports
without legacy-module re-exports.
`cache_assessment.py` checks retention, index and
scene-wide publication blockers.
`cache_coverage.py` aggregates counts and issues;
`cache_plan.py` selects actions and constructs
ordered command plans in batches of 50 scenes.
`cache.py` orchestrates streaming metadata reads,
preview-source verification and the existing
JSON output format.

### Reguły czasowe i obserwacje

`stash_ai/evidence/temporal/` oddziela politykę i jej historyczne snapshoty (`policy.py`),
rzadkie statystyki oraz kwalifikację scen (`statistics.py`), budowę markerów
i consensus seeds (`markers.py`) oraz kwalifikację dodatkowych tagów (`extras.py`).
`evidence/intervals.py` scala uporządkowane przedziały, `evidence/timeline.py`
udostępnia indeksowane zapytania o obserwacje, a `evidence/retention.py` wyznacza
progi retencji. Przeniesienie nie zmienia zapisanych dowodów ani reguł publikacji.

### Potok zachowanych dowodów

`stash_ai/evidence/pipeline.py` koordynuje kwalifikację scen, markerów i punktów
wystąpień. `source_markers.py` buduje wkład pojedynczych źródeł, `frames.py`
weryfikuje grupy klatek i przygotowuje obserwacje, a `observations.py` obsługuje
wielokrotne zapytania o przedziały. `shared_markers.py` łączy kompatybilne źródła
i odtwarza przycięte dowody; `scenes.py` kwalifikuje oraz odwołuje dowody scen.
`occurrences.py` oddziela kwalifikację kandydatów, budowę strumieni i wybór punktu.

`identity.py` definiuje tożsamość segmentów bez przejściowych override progów;
`labels.py` waliduje surowe etykiety i opisuje namespace AI. `cache_views.py`
i `thresholds.py` tworzą widoki kwalifikacji, a zapis zmian progów pozostaje
w `rethreshold/`. `inference/results.py` zapisuje kształt raw prediction outputs,
a `inference/reuse.py` dowodzi zgodności wcześniejszej inferencji przed zmianą progów.

### Runtime i raportowanie wykonania

`stash_ai/runtime/cache.py` zawiera cache LRU, a `memory.py` estymację utrzymanych
obiektów i odczyt RSS. `report.py` zachowuje licznikowy interfejs `RunReport`,
lecz `timing.py` zarządza oddzielnym `RunTimings` dla każdego raportu, jego
thread-local stosami i rozmiarem odpowiedzi API. `summary.py` renderuje dotychczasowe
podsumowanie, a `diagnostics.py` strumieniuje profil JSONL. Moduły runtime
pozostają poza fingerprintem kodu publikacji i nie wymagają zależności ML.

Grupy markerów rozwiązuje `publication/groups.py`; ich backfill uruchamia
`commands/marker_groups.py`, a pruning wraz z dry-run korektą liczników
obsługuje `cleanup/groups.py`. `cleanup/markers.py` usuwa wykluczone rdzenie
i extras historycznych wyników modeli, bez wyliczania nowych markerów.

Recovery jest rozdzielone między `storage/recovery.py` (lokalna archiwizacja
i odczyt intentów), `publication/recovery.py` (uzgodnienie dowodów z faktycznie
obecnymi wynikami Stash) oraz `commands/recovery.py` (tryb abandon-pending).
`storage/merge_checks.py` odpowiada za transakcyjnie unieważniany indeks publikacji,
`storage/segments.py` za pożyczone segmenty i ich memoizowaną tożsamość, a
`publication/fingerprints.py` za podpis kodu, polityki, katalogu i stanu sceny.
Ścieżki kodu do hashowania są liczone względem pakietu, niezależnie od cwd.

`commands/bootstrap.py` przygotowuje runtime, wybrane modele, polityki oraz
katalogi i hierarchie w dotychczasowej kolejności CLI. Szybkie tryby bazy
kończą się przed bootstrapem, a wszystkie katalogi są walidowane przed
ładowaniem wag i zapisami Stash. `runs/merged.py` zarządza wspólnym
lifecycle Stash dla wybranych modeli. `inference/scene_predictions.py` oddziela
wybór zachowanych etykiet od kwalifikacji temporal; finalizacja w `prediction.py`
zachowuje raw evidence, limity i wykluczenia przed publikacją.


Evidence ma jawne kontrakty słownikowe w `evidence/contracts.py`, w tym warianty
rekordów scene/frames/spans i wyników pochodnych. `interval_geometry.py` scala
geometrię bez algorytmów temporal; `intervals.py` dokłada kwalifikację extras.
Nazwy zarezerwowane i klasyfikacja occurrence żyją w `marker_identity.py`.
`inference/input_identity.py` współdzieli manifest cue i historyczny format
podpisów między preflightem, cache, reuse i inferencją.

Storage rozdziela `ResultStore`, `MetadataStore` i `LegacyStore`; `StateStore`
pozostaje właścicielem połączenia, cache i granic transakcji. Katalog
`storage/schema.py` zasila diagnostykę, widoki preview i triggery invalidacji,
oraz jawnie ustala kolejność inicjalizacji funkcji i migracji.
Re-threshold korzysta z `ThresholdRequest`, `RecordAccess` i `ThresholdPlan`.
Executor sprawdza wszystkie cele przed zapisami, a apply używa tych samych
planów metadanych bez ponownego odczytu lub dekodowania frame BLOB.

Planowanie publikacji przekazuje nazwane `PreparedEvidence`, `ResolvedEvidence`,
`MarkerCatalog` i `SeparateMarkerPlan`. Resolver kończy I/O katalogu przed
dopasowaniem markerów; payloady i plan create/update/delete powstają z gotowych
danych. Wspólny `evidence/frozen_replay.py` wybiera dokładne podpisy opublikowanej
evidence i odtwarza zamrożoną politykę. Recovery osobno rekonstruuje obserwowane
snapshoty scen i markerów, po czym koordynator zapisuje baseline.

`stash_api/scene_reads.py` koordynuje SQL snapshoty, pola mediów GraphQL
i jednokierunkowy fallback po błędzie, wspólny dla find, batch i page.
`cleanup/sql.py` ustala lifecycle i politykę cleanupu; `sql_discovery.py`
wykonuje ograniczone SELECT-y, `sql_mutations.py` rewaliduje partie i potwierdza
zapisy natywnym API, a `sql_preview.py` modeluje skutki dry-run bez zapisu źródła.

`commands/modes.py` ustala jeden `Command` przed normalizacją domyślnych
opcji. Dispatch, grupy trybów i deklaratywne reguły raportów korzystają z tego
wyboru, zachowując kolejność walidacji. `reports/runtime.py` współdzieli
izolowane ustawienia, polityki i read-only store, a `reports/io.py` zapisuje
nowy JSON z wyłącznym otwarciem pliku. Wybór scen i specyfika raportów pozostają
w ich modułach.


Runtime read contracts use declared optional state: StateStore.report and
FrameIndex.report default to None, preview/recovery fields are initialized by
StateStore, and TagResolver owns its lazy marker-group cache. Threshold plans
select either full batch evidence or derived metadata; the unused model-only
plan mode is removed. Native marker geometry is imported from interval_geometry.

Shared marker extras consume ObservationIndex. FrameIndex owns indexes used
throughout a scene, while clipped cleanup replay closes its temporary index
even when qualification fails. Closing drops query cache references without
changing the proof objects retained by result spans.

Schema upgrades run only through `--db-update`. Runtime writers and dry-run readers require the current schema; fresh stores are created at the current version. Preview always attaches a current database read-only and writes into a sparse temporary overlay. The offline updater visits every scope, backfills derived result metadata, and preserves raw evidence and unfinished plans. Metadata invalidated or corrupted later is still rebuilt from authoritative evidence.

Schema v25 accepts offline upgrades from v18 onward. Earlier databases must first run the updater at commit `c5bcd887bafe76b7ee2a9de944666799c9a0fc30`. Supported input schemas already store one inference object and one numeric threshold per (scope, scene, model, output). List histories, JSON threshold arrays and their migrations have been removed. Current metadata format 5 is rebuilt only after authoritative evidence conversion. Fresh base schemas are marked current so feature-initialization failures can be repaired with `--db-update`.

Fresh-state rebuilding uses a persisted `db_initializations` manifest per Stash scope and `db_init_scenes` ownership/completion rows. `--db-init` first refuses any populated state table across all scopes, then records the selected models and scene selector. Enrollment atomically captures existing AI leaf tag associations and seeds refresh requests for both outputs of every selected model. A successful scan freezes enrollment. Pending scenes authorize the missing-source guard exception; completed scenes and scenes outside the manifest do not. Publication requires signed result/index pairs and no refresh requests for every selected model/output, including valid zero-detection results. Planning persists any additional AI associations from the fresh scene read before API mutations. Snapshot promotion and init completion share one SQLite transaction; failed API writes keep durable intent and ownership for retry. Dry-run uses the existing read-only source plus temporary overlay. Upgrading an existing DB creates empty manifest tables without granting adoption permission.

Runtime snapshot decoding accepts only envelope v7. The offline v18+ reader handles v4/v5/v6/v7 and shares mechanical proof-reference expansion with runtime. One publication pass normalizes frozen policies, retires obsolete algorithms, converts source contexts to scalar signatures, and removes unused `retired_build_options`. Geometry, ownership, source pins, cleanup generations and recovery operation IDs remain unchanged. Current pinned inference archives remain available for frozen replay. Migration preserves shared proof identity and rejects unknown contracts atomically. Schema v25 removes `max_tags` and `MAX_MARKERS_PER_SCENE` from frozen build options, including nested scene evidence and pending/recovery snapshots; current replay has no per-scene output budgets. Existing geometry and contribution lists stay intact during the offline upgrade, and the next merge can publish additional qualified results from unchanged inference evidence. The remaining frame tag limit is independent of scene output counts.

Scene accumulator checkpoints carry a complete analysis format. Its version participates in the checkpoint signature, so old unfinished batches restart for the affected scene/model. Active inference requires complete format-5 evidence. Upgrading v18 retires obsolete active records and requests reinference, retaining exact proofs referenced by published, pending or recovery state.

The current package publishes only the AI namespace and validates hierarchy YAML v2 and native marker ranges. The namespace, hierarchy-v1 and ordinary-point migrators and their historical namespace definitions have been retired. These tools remain available in the immutable bridge commit `9fcaa52502cd487455457547860b5db7d8710ad8`; README documents an isolated checkout with the original database and configuration. Complete interrupted namespace conversion using that bridge before returning to current publication. Ordinary occurrences remain native points, and current ownership and recovery guards remain active. Pre-v18 databases use the separate schema-upgrade bridge described above. Historical namespaces survive only as foreign-output fixtures in tests.

New result writes require an explicit output mode, metadata dictionary and item list before changing SQL state. Scene inference capture requires retained frame analysis. Missing marker retention floors are insufficient in both cache reports and publication validation. Frame indexes use current evidence signatures and require complete explicit core and scene-tag policy maps; supported result threshold overrides are scalar numeric values. Spawned merge workers carry explicit publication policy, and HTTP timing uses the native Response body contract.

Stash credentials have no bundled fallback: an unset STASH_API_KEY sends no ApiKey header. Pruning requires the native studio/group usage counters guaranteed by Stash 0.31.1+, checks all other exposed usage counters, and omits the deprecated movie_count alias of group_count. It never probes the retired Movie API.

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

## Komunikacja ze Stash

Pakiet `stash_ai/stash_api/` grupuje klienta domenowego, transport HTTP,
odczyt snapshotów scen przez SQL i resolver tagów. `StashClient` pozostaje
spójną klasą z dziedziczeniem po `StashTransport`; `sql.py` zawiera wyłącznie
odczyty, a `tags.py` zarządza katalogiem i hierarchią tagów. Wywołania importują
bezpośrednio te moduły, bez wrapperów pod dawnymi ścieżkami.

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/stash_api/transport.py` / `StashTransport` | Sesje HTTP, uwierzytelnianie, retry odczytów, blokada zapisu podczas dry-run, dekodowanie GraphQL i przekierowania mediów. |
| `stash_ai/stash_api/client.py` / `StashClient` | Zapytania dotyczące scen, tagów i markerów, wymaganie Stash 0.31.1+, stronicowanie oraz fallback z SQL na GraphQL. |

`StashClient` jest specjalizacją `StashTransport`. Wspólny transport utrzymuje
jedną konfigurację połączenia i jeden raport żądań. Klient jest importowany z
`stash_ai.stash_api.client`, a funkcje pomocnicze HTTP bezpośrednio z
`stash_ai.stash_api.transport`. Pozostałe moduły używają analogicznych
importów pakietowych, np. `from stash_ai.storage.store import StateStore`.

Transport ponawia tylko odczyty, w tym ściśle dozwolone operacje SQL opakowane
w mutację GraphQL. Zapisy nie są ponawiane. Nagłówek `ApiKey` nie jest przesyłany
przy przekierowaniu mediów do innego origin.

## Cleanup przez SQL

`stash_ai/cleanup/sql_session.py` zawiera ograniczony allowlistą transport SQL/API,
metryki odczytów i zapisów oraz keyset pagination. `stash_ai/cleanup/sql.py` /
`SQLCleanup` pozostaje warstwą domenową: wybiera kandydatów, usuwa markery i
powiązania, obsługuje preview i pruning. Korzysta bezpośrednio ze swojej
`CleanupSqlSession` do odczytów, zapisów, stronicowania i granic discovery.
Stałe SQL i walidator odczytów są importowane bezpośrednio z
`cleanup/sql_session.py`.


## Lokalny stan

Pakiet `stash_ai/storage/` obejmuje bazę, magazyny wyników i publikacji,
serializację, metadata, snapshoty, nakładkę preview oraz indeks ukończenia.
Importy wskazują bezpośrednio moduły pakietu. Przeniesienie nie zmienia wersji
schematu, identyfikatora bazy, formatu BLOB-ów ani domyślnej lokalizacji SQLite.

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/storage/database.py` / `StateDatabase` | Cykl życia połączenia SQLite i blokady procesu, sprawdzenie identyfikatora/wersji bazy, podstawowe tabele i otwarcie źródła dry-run. |
| `stash_ai/storage/store.py` / `StateStore` | Façade i właściciel lifecycle/cache/serializacji; koordynuje inicjalizację schematów oraz deleguje trwałe dane do wyspecjalizowanych magazynów. |
| `stash_ai/storage/results.py` / `ResultStore` | Wyniki inferencji i indeksowanie źródeł. |
| `stash_ai/storage/publications.py` / `PublicationStore` | Snapshoty published/pending, trwały zamiar publikacji i atomowe zatwierdzanie przygotowanych publikacji. |
| `stash_ai/storage/initialization.py` / `InitializationStore` | Zakres odbudowy nowej bazy, przejęta własność tagów i bariera kompletności przed publikacją. |
| `stash_ai/storage/preview.py` | Widoki nakładki dry-run, usunięcia w nakładce i wykrywanie zmian źródła. |
| `stash_ai/storage/codec.py` | Kodowanie i dekodowanie BLOB-ów. |
| `stash_ai/storage/snapshots.py` | Wersjonowana reprezentacja snapshotów ze współdzielonymi dowodami `extra_support` i odtwarzanie contributions przy odczycie. |

`StateStore` posiada jeden `StateDatabase`, jeden `ResultStore` i jeden
`PublicationStore`; udostępnia metody `records()/put()/snapshot()/prepare()`
delegujące do właściciela danej operacji. `ResultStore.put()` zarządza transakcją
zwykłego zapisu. Zakończenie inferencji używa `ResultStore.put_in_transaction()`
we wspólnej transakcji z usunięciem checkpointu i żądań refresh. `StateStore`
zamyka bazę również przy błędzie inicjalizacji.
Blokada procesu obejmuje cały czas życia magazynu, także żądania do Stash.
Rethreshold waliduje i stosuje zmiany z małych rekordów `result_metadata`;
`ResultStore.set_threshold(..., metadata_only=True)` zapisuje próg oraz indeks
ukończenia z tego planu w jednej transakcji, bez odczytu BLOB-ów klatek.
Pełne rekordy pozostają obsługiwane przez tę samą ścieżkę zapisu dla reuse inferencji.
Dry-run wymaga aktualnego schematu i odczytuje bazę przez nakładkę ze źródłem
otwartym tylko do odczytu. Starszą bazę aktualizuje wyłącznie `--db-update`.

`StateStore.pack_snapshot()` internuje powtarzające się obiekty dowodów według
tożsamości, bez hashowania dużych struktur. Unikatowe dowody pozostają inline.
Kompaktowy BLOB zawiera envelope `__stash_ai_snapshot__: 3`, `snapshot` oraz
tablicę `proofs`; wartości `extra_support` są pełnymi dowodami lub indeksami tej
tablicy. `storage.snapshots.decode()` waliduje referencje i zwraca dotychczasową,
rozwiniętą strukturę także dla cleanupu, recovery i kontroli bazy. Dowody są
współdzielone i niemutowalne dla pożyczających je czytelników; odczyt publiczny
nadal zwraca kopię oddzieloną od cache. Zmiany historycznych snapshotów i wersji
bazy odbywają się wyłącznie offline przez `--db-update`. Promocja pending i
archiwizacja recovery kopiują już zapisany BLOB bez ponownej serializacji.

Obsługa samodzielnych komend bazy znajduje się w pakiecie `stash_ai/db/`:

| Moduł | Odpowiedzialność |
| --- | --- |
| `stash_ai/commands/database.py` | Dispatch komend `--db-report`, `--db-check`, `--db-update`, `--db-compact` przed inicjalizacją runtime. Funkcje utrzymania bazy są importowane bezpośrednio z modułów poniżej. |
| `stash_ai/db/update.py` | Jawna, lokalna aktualizacja całej bazy do najnowszego schematu przez istniejące migracje stanu; odczytowy preflight i dry-run. |
| `stash_ai/db/inspection.py` | Odczyt stanu i zajętości bazy oraz formatowanie raportu agregatów. |
| `stash_ai/db/checks.py` | Koordynacja głębokiej kontroli bazy w jednej transakcji odczytu oraz format raportu błędów i ostrzeżeń. |
| `stash_ai/db/validation/` | Osobne walidatory schematu i domen, wyników/indeksów/progów, checkpointów oraz payloadów i relacji cleanup/recovery; wspólne dekodowanie i rozliczanie postępu. |
| `stash_ai/db/compaction.py` | Blokada procesu, warunki bezpiecznego VACUUM i porównanie agregatów przed i po kompaktowaniu. |
| `stash_ai/db/common.py` | Wspólne definicje obsługiwanego schematu, połączenie SQLite tylko do odczytu i helpery zapytań. |

`stash_ai/reports/developer/report.py` korzysta z definicji schematu i helperów odczytu
z `stash_ai.db.common`; nadal odpowiada za własny format anonimowego raportu JSON
i zapis JSON. `developer/database.py` zbiera agregaty bazy i checkpointów,
`developer/evidence.py` analizuje kształt i kompresję wyników, a `distribution.py`
przechowuje ograniczone pamięciowo rozkłady. Pakiet `db` nie uruchamia migracji ani nie
zmienia protokołu blokady używanego przez `StateStore`.

Anonymous diagnostics format 2 scans detailed evidence only on the current schema.
Supported v18–v20 databases receive aggregate database facts and `upgrade_required`,
without historical payload adapters. Current evidence counters expose only scene/frames,
inference origin and the current analysis format; obsolete values become `other`.
Retired import tables are never parsed. DB checks report their presence as a schema error.
Tag explanations retain the scene-report list, mark each report `format: 2`, and expose
one `record` per source instead of a singleton `records` list.

`stash_ai/rethreshold/` oddziela lokalną zmianę progów od projekcji evidence.
`targets.py` wybiera pary scena/model z małych indeksów i grupuje przyczyny błędów;
`markers.py` waliduje pokrycie markerów i zapisuje ich próg, a `scenes.py` planuje
zmiany scen oraz opcjonalnie markerów. `execution.py` wspólnie obsługuje wszystkie
trzy tryby: pełny preflight w batchu backfill metadanych, przerwanie przy dowolnym
błędzie, a dopiero potem Apply z dotychczasową transakcją dla pojedynczej pary.
Marker-only nadal blokuje pending publication także przy niezmienionym progu;
scene/combined dopuszcza pending no-op. Check i Apply korzystają z metadanych,
zachowując surowe BLOB-y evidence oraz atomowy zapis progu i indeksu ukończenia.
`evidence/thresholds.py` zachowuje efektywne progi markerów, podpisy i kontrakty retencji,
a `evidence/cache_views.py` projekcję scen i konwersję obserwacji do klatek. Reuse korzysta
bezpośrednio z planów w `rethreshold`, bez wrapperów pod dawnymi ścieżkami.

Updater instaluje pliki Pythona z pakietu `stash_ai/`, również z jego podkatalogów.
Pozostaje samodzielnym skryptem stdlib, aby odzyskiwanie przerwanej aktualizacji
działało także przy niekompletnym pakiecie. `RemoteSnapshot` opisuje jeden przypięty
commit, a `UpdatePlan` zawiera operacje, konflikty i następny stan instalacji.
`prepare` zapisuje komplet plików i backupów przed publikacją journalu;
`PreparedUpdate` przekazuje helperowi jego tożsamość i katalog. `UpdateTransaction`
waliduje trwały plan oraz skupia apply i rollback, z fazami `Phase`.
`run_helper` obsługuje przekazanie blokady, usunięcie tokenów z environment i status
procesu. Format 1 metadanych pozostaje zgodny z wcześniejszym updaterem;
żaden z tych obiektów nie wymaga importu pakietu aplikacji. Migrację starego updatera opisuje
[README](../README.md#aktualizacje).

YAML-e, domyślna baza `tagger_sources.sqlite3` i katalog `models/` pozostają
obok launcherów. Domyślne ścieżki są wyznaczane względem katalogu instalacji,
nie katalogu roboczego terminala. Testy znajdują się w `tests/`, a benchmarki
w `benchmarks/`.

Moduły, które mogą zmienić wynik merged publication przy tej samej efektywnej
polityce, zapisanym evidence i obserwowanym stanie Stash, trzeba uwzględniać w
`PUBLICATION_CODE_FILES` w `stash_ai/publication/fingerprints.py`. Warstwy transportu,
CLI, parser VTT oraz tryby maintenance/cleanup mają własne sygnały invalidacji
i celowo nie należą do tego fingerprintu, aby ich refaktory nie wymuszały
re-merge całej biblioteki. Zmiana publication fingerprintu nie unieważnia
surowego cache inferencji.

## Raport kompletności cache

`stash_ai/reports/cache.py` strumieniuje małe metadane wyników i sprawdza indeks,
refresh requests, checkpointy oraz retencję względem wybranego YAML. Raport
korzysta z read-only źródła i sparse preview; generuje advisory `argv` dla
inferencji/rethreshold/merge bez ich uruchamiania. Nie dowodzi aktualności mediów
ani modelu; głęboki test payloadów pozostaje w `--db-check`.

Pakiet `stash_ai/reports/` grupuje raporty offline bez re-eksportów starych modułów.
`cache_assessment.py` ocenia retencję, indeks i blokady publikacji całej sceny;
`cache_coverage.py` agreguje liczniki oraz problemy, a `cache_plan.py` wybiera akcje
i buduje uporządkowany plan komend w paczkach po 50 scen. `cache.py` koordynuje
strumieniowy odczyt metadanych, kontrolę źródła preview i zapis istniejącego formatu JSON.

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

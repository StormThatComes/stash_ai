# Granice modułów

W katalogu głównym pozostają dwa pliki Pythona uruchamiane z linii poleceń:
`stash_ai.py` i `update_stash_ai.py`. Launcher `stash_ai.py` przejmuje blokadę
instalacji, a `stash_ai/cli.py` koordynuje CLI i uruchomienie pipeline.
Algorytmy evidence/fusion i union przedziałów pozostają w `stash_ai/evidence/pipeline.py`.
Pakiet `stash_ai/publication/` grupuje planowanie i publikację wspólnych wyników `ai:`.
`markers.py` zawiera wspólne reguły własności markerów, payloadów, porównywania
zmian i usuwania wykluczonych extras. Moduły i testy importują je bezpośrednio,
bez ponownego eksportowania przez CLI.
`commands/dispatch.py` wywołuje tryby CLI oraz obejmuje `MergedRun.run()`
kontekstem diagnostyki przez `run_merged()`. Wywołania
reconciliation korzystają bezpośrednio z `publication/merged.py` i `cleanup/scenes.py`; CLI
uruchamia cleanup przez `CleanupRun.run()`. Kod i testy importują codec
bezpośrednio z `stash_ai/storage/codec.py`.
Stanowy lifecycle jednego przebiegu merged-output (scan, rethreshold, inference,
publikacja i pruning) obsługuje `stash_ai/runs/merged.py` / `MergedRun`, który otwiera
oraz zamyka Stash i StateStore. `runs/merged_model.py` / `MergedModelRun` odpowiada
za przebieg jednego modelu: kwalifikację scen, reuse/refresh oraz inferencję
pojedynczej sceny. Współdzieli store, cache, liczniki i raport z `MergedRun`,
a tagger ładuje dopiero po wyczerpaniu możliwości reuse i zamyka w `finally`.
Błąd inferencji dotyczy jednej sceny; błąd ładowania modelu kończy jego pętlę.
Faza publikacji znajduje się w
`stash_ai/publication/publisher.py` / `MergePublisher`: sprawdza pending refresh i cache
publikacji, wybiera tryb szeregowy lub równoległy oraz wspólnie obsługuje
publikację, liczniki i błędy. Korzysta z zasobów przebiegu bez przejmowania ich
lifecycle. Opcjonalny `stash_ai/publication/parallel.py` uruchamia wyłącznie read-only
replay evidence w izolowanych procesach; `MergePublisher` zachowuje pojedynczy
writer dla TagResolver, Stash i StateStore. Dry-run i profilowanie pozostają
szeregowe, a awaria workerów nie uruchamia drugiego przebiegu publikacji.
Planowanie jednej sceny znajduje się w `stash_ai/publication/planning/`: osobne
moduły przygotowują evidence, tagi i markery.
`types.py` definiuje `MergeSyncPlan`. Koordynator `publication/merged.py`
zachowuje kolejność mutacji i zatwierdzania snapshotów. Cleanup ma
osobny lifecycle w `stash_ai/cleanup/run.py` / `CleanupRun`. Rekonsyliację jednej
sceny zawiera `cleanup/scenes.py`, a re-ewaluację opublikowanego evidence
`cleanup/evidence.py`: kwalifikacja zmienionych dowodów, leniwy `EvidenceReplay`
z indeksem sygnatur i cache replay oraz clipping do opublikowanych przedziałów.
Stan i maski cleanupu znajdują się w `cleanup/state.py`, a odczyt katalogu tagów
w `cleanup/client.py`. `cleanup/prune.py` obsługuje pruning zwykłych tagów. Pozostała logika domenowa znajduje się m.in. w `stash_ai/evidence/scenes.py`,
`stash_ai/evidence/observations.py`, `stash_ai/evidence/temporal/` i `stash_ai/evidence/occurrences.py`.

`stash_ai/publication/index.py` buduje dla jednej sceny niemutowalny indeks istniejących
markerów według primary tagu. Posortowane początki i drzewo maksymalnych końców
ograniczają kandydatów także dla zagnieżdżonych przedziałów; końcowy filtr używa
dotychczasowego `merge.overlaps()`, z osobną tolerancją początku. Cleanup wyszukuje
według nazwy primary, a planowanie po odświeżeniu sceny według jego ID. Dla outputu
punktowego `PreviousMarkerRanges` indeksuje zaokrąglone początki z published,
pending i recovery oraz zachowuje największy pasujący koniec. Normalizacja odbywa
się raz na marker w indeksie. Wybrane markery pozostają w indeksie, aby ich ręczne
extras mogły trafić do kolejnych fragmentów; dostępność do ponownego dopasowania
nadal kontroluje plan publikacji. Indeksy żyją wyłącznie w pamięci, bez zmiany
formatu snapshotów lub schematu bazy.

`runtime/settings.py` zawiera ustawienia jednego uruchomienia, a
`runtime/context.py` łączy je z jawnymi, wstrzykiwanymi usługami. CLI tworzy nowy
kontekst przy każdym wywołaniu; moduły domenowe nie importują CLI. Wspólny
`runtime/workers.py` otwiera izolowany, read-only store i dane ustawień dla
procesów merge oraz model report, bez ładowania usług CLI.

`stash_ai/cli.py` koordynuje jeden pipeline publikacji. Pobieranie cue/sprite,
przygotowanie obrazów, filtrowanie score i resumowalna inferencja znajdują się
w `stash_ai/inference/`. Nowa inferencja używa silnika temporalnego.
Frozen replay uses only the temporal engine. Schema v17 retains old contribution geometry,
manual tag IDs, ownership and pending/recovery intent, but retires old build options under
`retired_build_options`. Cleanup refuses contributions marked `requires_rebase`; fresh
inference and merge replace them with current proofs. Offline snapshot migrations remain available.
Usunięto aktywne moduły `runs/separate.py`, `publication/separate.py` i ich
plannery. Source-prefixed wyniki przechodzą jednorazowo do ai: przez świeżą inferencję.
Funkcje zależne od ustawień przebiegu otrzymują `app` jako jawny argument,
a callback przygotowania obrazu JoyTag wiąże go przez `functools.partial`.
`stash_ai/inference/inputs.py` zawiera `VerifiedMedia`, `RunMedia`, `RunInputs`
i `inspect_input()`: weryfikuje wejścia VTT/sprite, tworzy ich sygnatury i
współdzieli ograniczony cache mediów oraz manifestów między modelami wyłącznie
w RAM. Kod i testy importują je bezpośrednio z tego modułu.
`inference/sprites.py` odpowiada za cue i dekodowanie sprite, a
`inference/preparation.py` za preprocessing obrazów oraz ograniczone, uporządkowane
batchowanie z własnością i zamykaniem obrazów w workerach. `inference/prediction.py`
odtwarza stan predykcji, wykonuje batche i finalizuje wyniki sceny.
`inference/identity.py` buduje tożsamość modeli; `inference/checkpoints.py` zawiera
`Checkpoint`, zapis i odtworzenie kompaktowego stanu oraz atomowe zakończenie inferencji.
`stash_ai/storage/progress.py` obsługuje indeks ukończenia, ponowne użycie wyników i resume/refresh.
`stash_ai/signatures.py` definiuje wspólny `digest()` dla sygnatur wejść, checkpointów
i publikacji. Kanoniczny JSON, hashe, format BLOB-ów oraz schemat bazy pozostają takie same;
zapis wyników, usunięcie checkpointu i zakończenie refresh nadal są atomowe.
Pakiet `inference` nie re-eksportuje helperów; wywołania używają modułów, które je definiują.
Pakiet `stash_ai/commands/` grupuje granice trybów uruchomienia:
`arguments.py` definiuje parser, a `validation.py` oddziela walidację raportów,
DB/maintenance i pipeline, zachowując kolejność błędów oraz normalizację opcji.
`dispatch.py` rozdziela komendy DB przed konfiguracją runtime, komendy samodzielne
przed ładowaniem modeli oraz explain/merge-only przed dostępem do vocabularies i wag.
`explain.py` zawiera odczytową diagnostykę tagów; `hierarchy.py` kompozycję hierarchii
i jej samodzielną synchronizację, a `namespace_migration.py` rozpoznanie starych namespace'ów, source check oraz jednorazowe przejście starych wyników. CLI używa bezpośrednich importów bez wrapperów pod dawnymi ścieżkami.
Parsowanie konfiguracji,
wykluczeń i hierarchii znajduje się w `stash_ai/config.py`, a parser cue WebVTT
w `stash_ai/vtt.py`. Dzięki temu format argumentów, YAML i VTT oraz specjalne
ścieżki uruchomienia można testować niezależnie od normalnego pipeline inferencji.

`stash_ai/reports/review/report.py` koordynuje odtwarzanie lokalne evidence do raportu priorytetów przeglądu
tagów. `reports/review/scenes.py` i `markers.py` obliczają sygnały dla obu rodzajów evidence.
`reports/review/options.py` definiuje opcje i opis znaczenia sygnałów.
`reports/review/accumulator.py` zawiera `ReviewAccumulator`, który zarządza
deduplikacją liczników globalnych i per model, powodami oraz limitami przykładów. Raport nie
korzysta z modeli ani połączenia ze Stash i pomija evidence z pending refresh.

`stash_ai/reports/model/report.py` buduje offline raport leave-one-model-out z zachowanego
evidence. `analysis.py` oblicza baseline sceny, attribution supportu i counterfactual
bez jednego modelu, a `comparison.py` porównuje semantykę oraz nakładanie markerów.
`report.py` koordynuje iterację scen i formatowanie finalnych wierszy.
`reports/model/workers.py` odpowiada za izolowane procesy odczytu, ich inicjalizację,
podział budżetu pamięci oraz redukcję wyników do tego samego formatu co ścieżka serialna.

`stash_ai/models.py` definiuje wspólne typy `ModelSource`, `ModelSettings` i
`Vocabulary`, katalog źródeł oraz walidację wyboru modeli i wyników predykcji.
Moduł nie zależy od loaderów ani adapterów inferencji. `stash_ai/backends/loader.py`
zawiera adapter JoyTag i wybór adaptera przez `load_tagger()`. Obsługa plików
modelu oraz parsowanie słowników tagów znajduje się w `stash_ai/backends/vocabulary.py`.
`backends/joytag.py` weryfikuje i ładuje przypiętą architekturę autora.
ONNX jest rozdzielone na transformacje obrazów w `backends/onnx_preprocessing.py`,
wybór providerów, fallback CPU i inspekcję wejść/wyjść w `backends/onnx_session.py`
oraz adapter inferencji w `backends/onnx.py`. Kod, testy i benchmarki importują typy oraz helpery
bezpośrednio z modułów, które je definiują; `backends/loader.py` nie re-eksportuje
starego API. Wspólne typy i walidacja usuwają cykle między loaderami i adapterami.
Genericzne komponenty (`TagResolver`, loader hierarchii i cleanup źródłowy) wymagają jawnego `ModelSource`; JoyTag nie jest ich ukrytym
domyślnym namespace'em.

Kontrakty wewnętrzne są jawne: `select_scores()` przyjmuje tablicę NumPy,
retencja i podpisy inferencji korzystają z pełnego `ModelSettings`, a
`MarkerIntervalIndex` wymaga gotowego `PreviousMarkerRanges`. Iterator klatek
i przygotowanie batchy operują na obrazach PIL; worker tworzy własną kopię
i zamyka ją po transformacji. Fixture testowe zachowują te same kontrakty.

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
| `stash_ai/db/__init__.py` | Dotychczasowe importy `stash_ai.db` i dispatch komend `--db-report`, `--db-check`, `--db-update`, `--db-compact`. |
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

Schema v10 stores per-segment effective thresholds as JSON arrays in `result_thresholds.thresholds`; null entries use the segment's original inference threshold. Evidence BLOBs and derived metadata contain no publication overrides. The offline migration strips retired refinement and inline overrides once, preserving evidence signatures and mixed history. New inference always declares its actual marker capture floor and an empty marker blocklist. Current-format retention is required for both publication and frozen cleanup replay.

Schema v12 uses snapshot envelope v3 for every new publication, including snapshots with no shared proofs. Generic compressed JSON decoding never dispatches by a magic key. Snapshot decoding requires complete frozen marker/scene build options and every source's exclusion list, also inside occurrence scene evidence. Offline migration handles plain JSON and v1/v2 envelopes, completes historical empty maps and policies, and folds retired blocklists into frozen tag exclusions. Numeric marker settings and scene tag caps must already be saved; migration never substitutes today's runtime settings. Pending promotion still copies the encoded intent, and migration preserves proof references, recovery operation IDs, ownership and old False flags.

Scene accumulator checkpoints carry a complete analysis format. Its version participates only in the checkpoint signature, so old unfinished batches restart for the affected scene/model. Active inference requires complete format-4 evidence; obsolete active results are retired by schema 15 and require reinference. Historical publication proofs remain pinned outside the active cache until rebased.

Source-owned Stash conversion runs through `--migrate-legacy-only` using fresh inference and normal durable publication plans. The isolated namespace planner preserves manual objects and extras, reuses matching marker IDs and removes old source associations after new output is ready. Repeated conversion skips completed scenes; pending plans allow retries after partial API success. Schema 16 retires imported active evidence and drops its ledger. Normal publication has no imported-evidence or ledger branches.

New result writes require an explicit output mode, metadata dictionary and item list before changing SQL state. Scene inference capture requires retained frame analysis. Missing marker retention floors are insufficient in both cache reports and publication validation. Runtime index signatures accept only the output's own override; mixed historical policies are inspected only by offline DB validation. Spawned merge workers always carry both declared threshold maps, and HTTP timing uses the native Response body contract.

Stash credentials have no bundled fallback: an unset STASH_API_KEY sends no ApiKey header. Pruning requires the native studio/group usage counters guaranteed by Stash 0.31.1+, checks all other exposed usage counters, and omits the deprecated movie_count alias of group_count. It never probes the retired Movie API.

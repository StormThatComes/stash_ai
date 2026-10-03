# Granice modułów

W katalogu głównym pozostają dwa pliki Pythona uruchamiane z linii poleceń:
`stash_ai.py` i `update_stash_ai.py`. Launcher `stash_ai.py` przejmuje blokadę
instalacji, a `stash_ai/cli.py` koordynuje CLI i uruchomienie pipeline.
Algorytmy fusion i entry point `run_merged()` pozostają w `stash_ai/merge.py`.
`run_merged()` obejmuje `MergedRun.run()` kontekstem diagnostyki. Wywołania
reconciliation korzystają bezpośrednio z `merge_sync` i `merge_cleanup`; CLI
uruchamia cleanup przez `CleanupRun.run()`. Kod i testy importują codec
bezpośrednio z `stash_ai/state_codec.py`.
Stanowy lifecycle jednego przebiegu merged-output (scan, rethreshold, inference,
publikacja i pruning) obsługuje `stash_ai/merge_run.py` / `MergedRun`, który otwiera
oraz zamyka Stash i StateStore. Faza publikacji znajduje się w
`stash_ai/merge_publish.py` / `MergePublisher`: sprawdza pending refresh i cache
publikacji, wybiera tryb szeregowy lub równoległy oraz wspólnie obsługuje
publikację, liczniki i błędy. Korzysta z zasobów przebiegu bez przejmowania ich
lifecycle. Opcjonalny `stash_ai/merge_parallel.py` uruchamia wyłącznie read-only
replay evidence w izolowanych procesach; `MergePublisher` zachowuje pojedynczy
writer dla TagResolver, Stash i StateStore. Dry-run i profilowanie pozostają
szeregowe, a awaria workerów nie uruchamia drugiego przebiegu publikacji.
Planowanie i publikację zmian jednej sceny rozdziela `stash_ai/merge_sync.py`;
`MergeSyncPlan` przechowuje wyłącznie plan snapshotów i mutacji Stash, a kolejne
funkcje obsługują przygotowanie evidence, tagi, markery i publikację. Cleanup ma
osobny lifecycle w `stash_ai/merge_cleanup.py` / `CleanupRun`; ten moduł zawiera
też re-ewaluację wcześniej opublikowanego evidence oraz rekonsyliację pojedynczej
sceny. Pozostała logika domenowa znajduje się m.in. w `stash_ai/scene.py`,
`stash_ai/fusion.py`, `stash_ai/temporal.py` i `stash_ai/occurrences.py`.

`stash_ai/marker_index.py` buduje dla jednej sceny niemutowalny indeks istniejących
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

`stash_ai/cli.py` koordynuje uruchomienie pipeline. Lifecycle jednego modelu
w trybie `--output-mode separate` obsługuje `stash_ai/separate_run.py`; lifecycle
jest podzielony na synchronizację hierarchii, przetwarzanie pojedynczej sceny
oraz raportowanie. Pobieranie cue/sprite, przygotowanie obrazów, filtrowanie
score i resumowalny lokalny pipeline inferencji znajdują się w
`stash_ai/inference/`. Reconciliation dla `--output-mode separate` (własność
markerów, payloady, plan create/update/delete, cleanup wykluczeń i publikacja
sceny) znajduje się w `stash_ai/separate_sync.py`. Wywołania i testy korzystają
bezpośrednio z tych modułów; helpery inferencji i reconciliation nie są
ponownie eksportowane przez `cli.py`. Funkcje zależne od ustawień przebiegu
otrzymują `app` jako jawny argument, a callback przygotowania obrazu JoyTag
wiąże ten argument przez `functools.partial`.
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
`stash_ai/progress.py` obsługuje indeks ukończenia, ponowne użycie wyników i resume/refresh.
`stash_ai/signatures.py` definiuje wspólny `digest()` dla sygnatur wejść, checkpointów
i publikacji. Kanoniczny JSON, hashe, format BLOB-ów oraz schemat bazy pozostają takie same;
zapis wyników, usunięcie checkpointu i zakończenie refresh nadal są atomowe.
Pakiet `inference` nie re-eksportuje helperów; wywołania używają modułów, które je definiują.
Definicja opcji CLI i walidacja kombinacji flag są
w `stash_ai/cli_args.py`, a dispatch samodzielnych trybów DB/diagnostics/recovery/
prune/model-report/backfill w `stash_ai/cli_modes.py`. Parsowanie konfiguracji,
wykluczeń i hierarchii znajduje się w `stash_ai/config.py`, a parser cue WebVTT
w `stash_ai/vtt.py`. Dzięki temu format argumentów, YAML i VTT oraz specjalne
ścieżki uruchomienia można testować niezależnie od normalnego pipeline inferencji.

`stash_ai/reports/tag_review.py` odtwarza lokalne evidence do raportu priorytetów przeglądu
tagów. `reports/review_accumulator.py` zawiera `ReviewAccumulator`, który zarządza
deduplikacją liczników globalnych i per model, powodami oraz limitami przykładów. Raport nie
korzysta z modeli ani połączenia ze Stash i pomija evidence z pending refresh.

`stash_ai/reports/model.py` buduje offline raport leave-one-model-out z zachowanego
evidence. `build_report()` koordynuje osobne fazy baseline sceny, attribution
supportu, counterfactual bez jednego modelu oraz formatowanie finalnych wierszy.
`reports/model_workers.py` odpowiada za izolowane procesy odczytu, ich inicjalizację,
podział budżetu pamięci oraz redukcję wyników do tego samego formatu co ścieżka serialna.

`stash_ai/models.py` definiuje wspólne typy `ModelSource`, `ModelSettings` i
`Vocabulary`, katalog źródeł oraz walidację wyboru modeli i wyników predykcji.
Moduł nie zależy od loaderów ani adapterów inferencji. `stash_ai/taggers.py`
zawiera adapter JoyTag i wybór adaptera przez `load_tagger()`. Obsługa plików
modelu oraz parsowanie słowników tagów znajduje się w `stash_ai/vocabulary.py`,
a preprocessing, konfiguracja sesji i inference modeli ONNX w
`stash_ai/onnx_tagger.py`. Kod, testy i benchmarki importują typy oraz helpery
bezpośrednio z modułów, które je definiują; `taggers.py` nie re-eksportuje
starego API. Wspólne typy i walidacja usuwają cykle między loaderami i adapterami.
Genericzne komponenty (`TagResolver`, loader hierarchii i separate-output
reconciliation) wymagają jawnego `ModelSource`; JoyTag nie jest ich ukrytym
domyślnym namespace'em.

Kontrakty wewnętrzne są jawne: `select_scores()` przyjmuje tablicę NumPy,
retencja i podpisy inferencji korzystają z pełnego `ModelSettings`, a
`MarkerIntervalIndex` wymaga gotowego `PreviousMarkerRanges`. Iterator klatek
i przygotowanie batchy operują na obrazach PIL; worker tworzy własną kopię
i zamyka ją po transformacji. Fixture testowe zachowują te same kontrakty.

## Komunikacja ze Stash

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/transport.py` / `StashTransport` | Sesje HTTP, uwierzytelnianie, retry odczytów, blokada zapisu podczas dry-run, dekodowanie GraphQL i przekierowania mediów. |
| `stash_ai/client.py` / `StashClient` | Zapytania dotyczące scen, tagów i markerów, wymaganie Stash 0.31.1+, stronicowanie oraz fallback z SQL na GraphQL. |

`StashClient` jest specjalizacją `StashTransport`. Wspólny transport utrzymuje
jedną konfigurację połączenia i jeden raport żądań. Klient jest importowany z
`stash_ai.client`, a funkcje pomocnicze HTTP bezpośrednio z
`stash_ai.transport`. Pozostałe moduły używają analogicznych
importów pakietowych, np. `from stash_ai.state import StateStore`.

Transport ponawia tylko odczyty, w tym ściśle dozwolone operacje SQL opakowane
w mutację GraphQL. Zapisy nie są ponawiane. Nagłówek `ApiKey` nie jest przesyłany
przy przekierowaniu mediów do innego origin.

## Cleanup przez SQL

`stash_ai/cleanup_sql_session.py` zawiera ograniczony allowlistą transport SQL/API,
metryki odczytów i zapisów oraz keyset pagination. `stash_ai/cleanup_sql.py` /
`SQLCleanup` pozostaje warstwą domenową: wybiera kandydatów, usuwa markery i
powiązania, obsługuje preview i pruning. Korzysta bezpośrednio ze swojej
`CleanupSqlSession` do odczytów, zapisów, stronicowania i granic discovery.
Stałe SQL i walidator odczytów są importowane bezpośrednio z
`cleanup_sql_session.py`.


## Lokalny stan

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/state_db.py` / `StateDatabase` | Cykl życia połączenia SQLite i blokady procesu, sprawdzenie identyfikatora/wersji bazy, podstawowe tabele i otwarcie źródła dry-run. |
| `stash_ai/state.py` / `StateStore` | Façade i właściciel lifecycle/cache/serializacji; koordynuje inicjalizację schematów oraz deleguje trwałe dane do wyspecjalizowanych magazynów. |
| `stash_ai/state_results.py` / `ResultStore` | Wyniki inferencji, indeksowanie źródeł i importy legacy. |
| `stash_ai/state_publications.py` / `PublicationStore` | Snapshoty published/pending, trwały zamiar publikacji i atomowe zatwierdzanie przygotowanych publikacji. |
| `stash_ai/preview.py` | Widoki nakładki dry-run, usunięcia w nakładce i wykrywanie zmian źródła. |
| `stash_ai/state_codec.py` | Kodowanie i dekodowanie BLOB-ów. |
| `stash_ai/state_snapshot.py` | Wersjonowana reprezentacja snapshotów ze współdzielonymi dowodami `extra_support` i odtwarzanie contributions przy odczycie. |

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
Dry-run korzysta z tymczasowej bazy: współczesne schematy odczytuje
przez nakładkę ze źródłem otwartym tylko do odczytu, a starsze kopiuje przed
migracją. Schemat v9 również korzysta z nakładki odczytu.

`StateStore.pack_snapshot()` internuje powtarzające się obiekty dowodów według
tożsamości, bez hashowania dużych struktur. Unikatowe dowody pozostają inline.
Kompaktowy BLOB zawiera envelope `__stash_ai_snapshot__: 1`, `snapshot` oraz
tablicę `proofs`; wartości `extra_support` są pełnymi dowodami lub indeksami tej
tablicy. `state_codec.decode()` waliduje referencje i zwraca dotychczasową,
rozwiniętą strukturę także dla cleanupu, recovery i kontroli bazy. Dowody są
współdzielone i niemutowalne dla pożyczających je czytelników; odczyt publiczny
nadal zwraca kopię oddzieloną od cache. Zapis nowego formatu i podniesienie wersji
bazy do 9 odbywają się w jednej transakcji. Stare snapshoty oraz inne BLOB-y
zachowują dotychczasowy format. Promocja pending i archiwizacja recovery kopiują
już zapisany BLOB bez ponownej serializacji.

Obsługa samodzielnych komend bazy znajduje się w pakiecie `stash_ai/db/`:

| Moduł | Odpowiedzialność |
| --- | --- |
| `stash_ai/db/__init__.py` | Dotychczasowe importy `stash_ai.db` i dispatch komend `--db-report`, `--db-check`, `--db-update`, `--db-compact`. |
| `stash_ai/db/update.py` | Jawna, lokalna aktualizacja całej bazy do najnowszego schematu przez istniejące migracje stanu; odczytowy preflight i dry-run. |
| `stash_ai/db/inspection.py` | Odczyt stanu i zajętości bazy oraz formatowanie raportu agregatów. |
| `stash_ai/db/checks.py` | Głęboka kontrola BLOB-ów, indeksów, checkpointów i relacji stanu w jednej transakcji odczytu; raport błędów i ostrzeżeń. |
| `stash_ai/db/compaction.py` | Blokada procesu, warunki bezpiecznego VACUUM i porównanie agregatów przed i po kompaktowaniu. |
| `stash_ai/db/common.py` | Wspólne definicje obsługiwanego schematu, połączenie SQLite tylko do odczytu i helpery zapytań. |

`stash_ai/reports/developer.py` korzysta z definicji schematu i helperów odczytu
z `stash_ai.db.common`; nadal odpowiada za własny format anonimowego raportu JSON
i ograniczone pamięciowo rozkłady. Pakiet `db` nie uruchamia migracji ani nie
zmienia protokołu blokady używanego przez `StateStore`.

Updater instaluje pliki Pythona z pakietu `stash_ai/`, również z jego podkatalogów.
Pozostaje samodzielnym skryptem stdlib, aby odzyskiwanie przerwanej aktualizacji
działało także przy niekompletnym pakiecie. Migrację starego updatera opisuje
[README](../README.md#aktualizacje).

YAML-e, domyślna baza `tagger_sources.sqlite3` i katalog `models/` pozostają
obok launcherów. Domyślne ścieżki są wyznaczane względem katalogu instalacji,
nie katalogu roboczego terminala. Testy znajdują się w `tests/`, a benchmarki
w `benchmarks/`.

Moduły, które mogą zmienić wynik merged publication przy tej samej efektywnej
polityce, zapisanym evidence i obserwowanym stanie Stash, trzeba uwzględniać w
`PUBLICATION_CODE_FILES` w `stash_ai/performance.py`. Warstwy transportu,
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

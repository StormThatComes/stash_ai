# Granice modułów

W katalogu głównym pozostają dwa pliki Pythona uruchamiane z linii poleceń:
`stash_ai.py` i `update_stash_ai.py`. Launcher `stash_ai.py` przejmuje blokadę
instalacji, a `stash_ai/cli.py` koordynuje CLI i uruchomienie pipeline.
Algorytmy fusion i kompatybilne entry pointy pozostają w `stash_ai/merge.py`.
Stanowy lifecycle jednego przebiegu merged-output (scan, rethreshold, inference,
publikacja i pruning) obsługuje `stash_ai/merge_run.py` / `MergedRun`.
Planowanie i publikację zmian jednej sceny rozdziela `stash_ai/merge_sync.py`;
`MergeSyncPlan` przechowuje wyłącznie plan snapshotów i mutacji Stash, a kolejne
funkcje obsługują przygotowanie evidence, tagi, markery i publikację. Cleanup ma
osobny lifecycle w `stash_ai/merge_cleanup.py` / `CleanupRun`; ten moduł zawiera
też re-ewaluację wcześniej opublikowanego evidence oraz rekonsyliację pojedynczej
sceny. Pozostała logika domenowa znajduje się m.in. w `stash_ai/scene.py`,
`stash_ai/fusion.py`, `stash_ai/temporal.py` i `stash_ai/occurrences.py`.

`stash_ai/cli.py` pozostaje warstwą orkiestracji i kompatybilnym miejscem eksportu
helperów używanych przez testy. Lifecycle jednego modelu w trybie
`--output-mode separate` obsługuje `stash_ai/separate_run.py`; lifecycle jest
podzielony na synchronizację hierarchii, przetwarzanie pojedynczej sceny oraz
raportowanie, a moduł wywołuje helpery przez obiekt `app`, dzięki czemu
dotychczasowe punkty monkeypatch w `cli.py` pozostają aktywne. Pobieranie cue/sprite, przygotowanie obrazów,
filtrowanie score i resumowalny lokalny pipeline inferencji znajdują się w
`stash_ai/inference.py`; kompatybilne wrappery pozostają w `cli.py`.
Definicja opcji CLI i walidacja kombinacji flag są
w `stash_ai/cli_args.py`, a dispatch samodzielnych trybów DB/diagnostics/recovery/
prune/model-report/backfill w `stash_ai/cli_modes.py`. Parsowanie konfiguracji,
wykluczeń i hierarchii znajduje się w `stash_ai/config.py`, a parser cue WebVTT
w `stash_ai/vtt.py`. Dzięki temu format argumentów, YAML i VTT oraz specjalne
ścieżki uruchomienia można testować niezależnie od normalnego pipeline inferencji.

`stash_ai/taggers.py` pozostaje publiczną façade dla definicji modeli i adapterów
inferencji. Obsługa plików modelu oraz parsowanie słowników tagów znajduje się w
`stash_ai/vocabulary.py`, a preprocessing, konfiguracja sesji i inference modeli
ONNX w `stash_ai/onnx_tagger.py`. Helpery i `OnnxTagger` są re-exportowane z
`taggers.py`, aby zachować istniejące importy w kodzie i testach.

## Komunikacja ze Stash

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/transport.py` / `StashTransport` | Sesje HTTP, uwierzytelnianie, retry odczytów, blokada zapisu podczas dry-run, dekodowanie GraphQL i przekierowania mediów. |
| `stash_ai/client.py` / `StashClient` | Zapytania dotyczące scen, tagów i markerów, wykrywanie możliwości API, stronicowanie oraz fallback z SQL na GraphQL. |

`StashClient` jest specjalizacją `StashTransport`. Wspólny transport utrzymuje
jedną konfigurację połączenia i jeden raport żądań. Klient i funkcje pomocnicze
są importowane z `stash_ai.client`. Pozostałe moduły używają analogicznych
importów pakietowych, np. `from stash_ai.state import StateStore`.

Transport ponawia tylko odczyty, w tym ściśle dozwolone operacje SQL opakowane
w mutację GraphQL. Zapisy nie są ponawiane. Nagłówek `ApiKey` nie jest przesyłany
przy przekierowaniu mediów do innego origin.

## Cleanup przez SQL

`stash_ai/cleanup_sql_session.py` zawiera ograniczony allowlistą transport SQL/API,
metryki odczytów i zapisów oraz keyset pagination. `stash_ai/cleanup_sql.py` /
`SQLCleanup` pozostaje warstwą domenową: wybiera kandydatów, usuwa markery i
powiązania, obsługuje preview i pruning. Cienkie metody delegujące w `SQLCleanup`
zachowują dotychczasowy kontrakt dla call-site'ów i testów.


## Lokalny stan

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/state_db.py` / `StateDatabase` | Cykl życia połączenia SQLite i blokady procesu, sprawdzenie identyfikatora/wersji bazy, podstawowe tabele i otwarcie źródła dry-run. |
| `stash_ai/state.py` / `StateStore` | Façade i właściciel lifecycle/cache/serializacji; koordynuje inicjalizację schematów oraz deleguje trwałe dane do wyspecjalizowanych magazynów. |
| `stash_ai/state_results.py` / `ResultStore` | Wyniki inferencji, indeksowanie źródeł i importy legacy. |
| `stash_ai/state_publications.py` / `PublicationStore` | Snapshoty published/pending, trwały zamiar publikacji i atomowe zatwierdzanie przygotowanych publikacji. |
| `stash_ai/preview.py` | Widoki nakładki dry-run, usunięcia w nakładce i wykrywanie zmian źródła. |
| `stash_ai/state_codec.py` | Kodowanie i dekodowanie BLOB-ów. |

`StateStore` posiada jeden `StateDatabase`, jeden `ResultStore` i jeden
`PublicationStore`; zachowuje dotychczasowe metody `records()/put()/snapshot()/prepare()`
jako kompatybilne delegacje. Zamyka bazę również przy błędzie inicjalizacji.
Blokada procesu obejmuje cały czas życia magazynu, także żądania do Stash.
Dry-run korzysta z tymczasowej bazy: współczesne schematy odczytuje
przez nakładkę ze źródłem otwartym tylko do odczytu, a starsze kopiuje przed
migracją. Format zapisanych danych i kolejność migracji pozostają bez zmian.

Obsługa samodzielnych komend bazy znajduje się w pakiecie `stash_ai/db/`:

| Moduł | Odpowiedzialność |
| --- | --- |
| `stash_ai/db/__init__.py` | Dotychczasowe importy `stash_ai.db` i dispatch komend `--db-report`, `--db-check`, `--db-compact`. |
| `stash_ai/db/inspection.py` | Odczyt stanu i zajętości bazy oraz formatowanie raportu agregatów. |
| `stash_ai/db/checks.py` | Głęboka kontrola BLOB-ów, indeksów, checkpointów i relacji stanu w jednej transakcji odczytu; raport błędów i ostrzeżeń. |
| `stash_ai/db/compaction.py` | Blokada procesu, warunki bezpiecznego VACUUM i porównanie agregatów przed i po kompaktowaniu. |
| `stash_ai/db/common.py` | Wspólne definicje obsługiwanego schematu, połączenie SQLite tylko do odczytu i helpery zapytań. |

`stash_ai/developer_report.py` korzysta z definicji schematu i helperów odczytu
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

Moduły wpływające
na publikację trzeba uwzględniać w `PUBLICATION_CODE_FILES` w `stash_ai/performance.py`,
aby zmiana kodu unieważniała wcześniejsze sprawdzenia merge. To nie unieważnia
surowego cache inferencji.

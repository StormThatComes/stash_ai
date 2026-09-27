# Granice modułów

W katalogu głównym pozostają dwa pliki Pythona uruchamiane z linii poleceń:
`stash_ai.py` i `update_stash_ai.py`. Launcher `stash_ai.py` przejmuje blokadę
instalacji, a `stash_ai/cli.py` koordynuje CLI i uruchomienie pipeline.
Logika scalania pozostaje
w `stash_ai/merge.py` oraz modułach domenowych, m.in. `stash_ai/scene.py`,
`stash_ai/fusion.py`, `stash_ai/temporal.py` i `stash_ai/occurrences.py`.

`stash_ai/cli.py` pozostaje warstwą orkiestracji i kompatybilnym miejscem eksportu
helperów używanych przez testy. Parsowanie konfiguracji, wykluczeń i hierarchii
znajduje się w `stash_ai/config.py`, a parser cue WebVTT w `stash_ai/vtt.py`.
Dzięki temu format YAML/VTT może być testowany niezależnie od dispatchu CLI i pipeline inferencji.

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

## Lokalny stan

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_ai/state_db.py` / `StateDatabase` | Cykl życia połączenia SQLite i blokady procesu, sprawdzenie identyfikatora/wersji bazy, podstawowe tabele i otwarcie źródła dry-run. |
| `stash_ai/state.py` / `StateStore` | Cache, wyniki inferencji, importy legacy, trwały zamiar publikacji i zatwierdzanie snapshotów. Koordynuje inicjalizację schematów poszczególnych funkcji. |
| `stash_ai/preview.py` | Widoki nakładki dry-run, usunięcia w nakładce i wykrywanie zmian źródła. |
| `stash_ai/state_codec.py` | Kodowanie i dekodowanie BLOB-ów. |

`StateStore` posiada jeden `StateDatabase` i zamyka go również przy błędzie
inicjalizacji. Blokada procesu obejmuje cały czas życia magazynu, także żądania
do Stash. Dry-run korzysta z tymczasowej bazy: współczesne schematy odczytuje
przez nakładkę ze źródłem otwartym tylko do odczytu, a starsze kopiuje przed
migracją. Format zapisanych danych i kolejność migracji pozostają bez zmian.

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

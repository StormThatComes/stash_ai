# Granice modułów

`stash_ai.py` koordynuje CLI i uruchomienie pipeline. Logika scalania pozostaje
w `stash_merge.py` oraz modułach domenowych, m.in. `stash_scene.py`,
`stash_fusion.py`, `stash_temporal.py` i `stash_occurrences.py`.

## Komunikacja ze Stash

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_transport.py` / `StashTransport` | Sesje HTTP, uwierzytelnianie, retry odczytów, blokada zapisu podczas dry-run, dekodowanie GraphQL i przekierowania mediów. |
| `stash_client.py` / `StashClient` | Zapytania dotyczące scen, tagów i markerów, wykrywanie możliwości API, stronicowanie oraz fallback z SQL na GraphQL. |

`StashClient` jest specjalizacją `StashTransport`. Wspólny transport utrzymuje
jedną konfigurację połączenia i jeden raport żądań. Publiczne metody klienta
oraz importy pomocniczych funkcji URL i redakcji błędów pozostają dostępne
z `stash_client`.

Transport ponawia tylko odczyty, w tym ściśle dozwolone operacje SQL opakowane
w mutację GraphQL. Zapisy nie są ponawiane. Nagłówek `ApiKey` nie jest przesyłany
przy przekierowaniu mediów do innego origin.

## Lokalny stan

| Moduł / klasa | Odpowiedzialność |
| --- | --- |
| `stash_state_db.py` / `StateDatabase` | Cykl życia połączenia SQLite i blokady procesu, sprawdzenie identyfikatora/wersji bazy, podstawowe tabele i otwarcie źródła dry-run. |
| `stash_state.py` / `StateStore` | Cache, wyniki inferencji, importy legacy, trwały zamiar publikacji i zatwierdzanie snapshotów. Koordynuje inicjalizację schematów poszczególnych funkcji. |
| `stash_preview.py` | Widoki nakładki dry-run, usunięcia w nakładce i wykrywanie zmian źródła. |
| `stash_state_codec.py` | Kodowanie i dekodowanie BLOB-ów. |

`StateStore` posiada jeden `StateDatabase` i zamyka go również przy błędzie
inicjalizacji. Blokada procesu obejmuje cały czas życia magazynu, także żądania
do Stash. Dry-run korzysta z tymczasowej bazy: współczesne schematy odczytuje
przez nakładkę ze źródłem otwartym tylko do odczytu, a starsze kopiuje przed
migracją. Format zapisanych danych i kolejność migracji pozostają bez zmian.

Nowe moduły `stash_*.py` są instalowane przez istniejący updater. Moduły wpływające
na publikację trzeba uwzględniać w `PUBLICATION_CODE_FILES` w `stash_performance.py`,
aby zmiana kodu unieważniała wcześniejsze sprawdzenia merge. To nie unieważnia
surowego cache inferencji.

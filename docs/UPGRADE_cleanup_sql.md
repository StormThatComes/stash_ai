# Cleanup dużej kolekcji — 14.09.2026

Zmiana przygotowana dla kolekcji około 2,5 mln markerów. Dotyczy cleanupu;
inferencja modeli, progi rozpoznawania i zawartość YAML pozostają takie jak
w przesłanej paczce. Zachowaj dotychczasową bazę `tagger_sources.sqlite3`.

## Uruchomienie

Wprowadź zmiany w dotychczasowym katalogu projektu. Skopiuj komplet modułów,
w tym nowy `stash_cleanup_sql.py`. Jeżeli od wysłania ZIP-a zmieniłeś ustawienia
w `stash_ai.py` lub YAML, przenieś również te zmiany do używanej wersji.

```powershell
# Podgląd; bez zmian w Stash i trwałej bazie wyników:
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql --dry-run

# Wykonanie:
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql
```

Jeżeli baza wyników leży gdzie indziej, dodaj dotychczasowe `--state-db PATH`.
Po dopisaniu nowego tagu do YAML wykonaj tę samą komendę; nie potrzeba ponownej
inferencji ani budowania indeksu wszystkich markerów przez skrypt.

`--cleanup-strategy sql` celowo zatrzymuje pracę, jeśli wymagane API jest
niedostępne. Domyślne `auto` wybiera SQL, a bez niego przegląda sceny. Nie wraca
automatycznie do dawnego globalnego `findSceneMarkers` z filtrem tagów.

```powershell
python .\stash_ai.py --models all --cleanup-markers-only --cleanup-strategy sql
python .\stash_ai.py --models all --cleanup-scene-tags-only --cleanup-strategy sql
```

Tryby częściowe usuwają wyłącznie powiązania odpowiedniego rodzaju i zachowują
definicje tagów. Pełne wykluczenie globalne zachowuje wcześniejszą semantykę:
usuwa marker z zakazanym tagiem głównym, a następnie definicję tagu wraz z jej
powiązaniami, również do innych obiektów Stash. Core-only nadal wymaga
rozpoznania markera jako zarządzanego przez skrypt i zachowuje jego ręczny
odpowiednik oraz dozwolone wykorzystanie tagu jako dodatku.

## Co zmieniono

1. **Odczyt ID przez `querySQL`.** Wyłącznie stałe, parametryzowane `SELECT`.
   Skrypt nie wykonuje SQL `DELETE`, `UPDATE`, `CREATE INDEX` ani `execSQL` na
   bazie Stash. Usuwanie, odpinanie tagów, hooki i sprzątanie plików markerów
   obsługuje oficjalne API Stash.
2. **Brak globalnych `COUNT` i `OFFSET`.** Markery wybierane są osobno dla
   konkretnego `primary_tag_id`, przez istniejący indeks, od ostatniego ID.
   Relacje tagów używają kursora `rowid`, zgodnego z kolejnością ich indeksu
   `tag_id`. Odczyt jest ograniczony górną granicą uchwyconą na początku przebiegu.
3. **Globalny zakaz nie skanuje dodatków wszystkich markerów.** Po usunięciu
   markerów z zakazanym tagiem głównym natywne usunięcie definicji sprząta
   pozostałe relacje. Reguły `by_model` wyszukują ID potrzebnych scen bez
   pobierania pełnych obiektów markerów; dopiero te sceny są przeliczane.
4. **Strumieniowe operacje zbiorcze.** Nie powstaje lista milionów markerów do
   usunięcia. Każda partia przechodzi ponowną weryfikację aktualnego tagu i,
   dla core-only, oznaczeń właściciela. Tryby częściowe używają zbiorczego
   `tag_ids: {mode: REMOVE}`. Tag usuwany jest pojedynczo, ponieważ może mieć
   bardzo dużo powiązań.
5. **Lżejszy `dry-run`.** Dla bazy stanu w formacie 5/6 surowe wyniki są
   dołączone tylko do odczytu; zmienione publikacje trafiają do tymczasowej bazy SQLite.
   Nie ma kopii całej wielogigabajtowej bazy wyników w RAM. Starsze formaty
   zachowują dotychczasową ścieżkę podglądu migracji.
6. **Lżejsze sprzątanie pustych definicji.** Najpierw indeksowane `EXISTS`
   odrzucają tagi używane przez sceny lub markery. Dopiero potencjalnie puste
   tagi przechodzą pełne sprawdzenie wszystkich liczników użycia i dzieci
   przez Stash, w tym obrazów i pozostałych obsługiwanych obiektów.

W kodzie Stash `findSceneMarkers` bez listy `ids` uruchamia zarówno zapytanie
zliczające, jak i pobranie strony. Samo usunięcie pola `count` z GraphQL nie
eliminuje zliczania. Filtr tagów łączy trafienia tagów głównych i dodatków,
a pola relacji markera uruchamiają dalsze odczyty. Nowa ścieżka omija te koszty.

## Rozmiary partii i wznowienie

| Opcja | Domyślnie | Zastosowanie |
|---|---:|---|
| `--cleanup-page-size` | 2000 | Maksymalna liczba wierszy ID w jednym SELECT; zakres 1–10000. |
| `--cleanup-write-batch-size` | 200 | Maksymalna liczba scen/markerów w jednej mutacji; zakres 1–1000. |
| `--cleanup-strategy` | auto | `auto` wybiera SQL, a bez niego przebieg po scenach. `sql` lub `scenes` wymusza konkretną ścieżkę. |

Jeżeli timeout przeniesie się na **usuwanie**, użyj mniejszej partii:

```powershell
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql --cleanup-write-batch-size 50
```

Mutacje nie są automatycznie ponawiane: po timeoutcie serwer mógł już zapisać
zmianę. Wznowienie tej samej komendy, z tym samym trybem, YAML i bazą stanu,
ponownie znajduje pozostałe obiekty. Nie odtwarza już skasowanych ID. Można
zmienić rozmiar partii. Przerwany pełny cleanup poprzedniej wersji można
dokończyć przez `--cleanup-strategy sql`, bez zmiany blacklist. Strategia
`--cleanup-strategy legacy` została usunięta; w istniejących poleceniach
zastąp ją przez `sql` lub usuń opcję, aby użyć domyślnego `auto`. Zachowany
format stanu pozwala wznowić również cleanup rozpoczęty przez `legacy`.

Aby zamiast wznowienia **porzucić pozostałą część planu i zmienić blacklisty**,
po zatrzymaniu procesu dodaj `--abandon-pending` do nowej komendy. Można też
uruchomić tę opcję samodzielnie. Zmiany wykonane w Stash pozostają; wyniki
inferencji i dane potrzebne do odzyskania własności nie są kasowane.
Szczegóły i zgodność baz 5/6/7: [UPGRADE_abandon_pending.md](UPGRADE_abandon_pending.md).

Szybki globalny cleanup ograniczony do jednego rodzaju wyjścia wprowadza maski
zależne od rodzaju publikacji i podnosi format bazy stanu do 6. Starszy skrypt
nie powinien być następnie używany z tą bazą. Pełny cleanup istniejącej bazy 5
nie wymaga tego podniesienia.

W trakcie cleanupu unikaj innych zadań zmieniających te same obiekty. Lokalna
blokada i ponowna weryfikacja pozostają aktywne, ale API nie daje atomowej
operacji „sprawdź aktualną nazwę i usuń tylko jeśli nadal pasuje”. Obiekty dodane
w trakcie przebiegu mogą wymagać kolejnego uruchomienia.

## Weryfikacja i pomiar

Przed zmianami przeszło 338 testów. Wynik pełnej końcowej walidacji znajduje się
w [cleanup_validation.txt](validation/cleanup_validation.txt). Nowe testy sprawdzają m.in. realne zapytania SQLite,
stronicowanie podczas usuwania, ochronę ręcznych markerów, zakres trybów
częściowych, wznowienie po zatwierdzonej mutacji i utraconej odpowiedzi,
zmianę nazwy między odczytem i zapisem oraz brak trwałych zmian w podglądzie.

[cleanup_benchmark.json](../benchmarks/results/cleanup_benchmark.json) zawiera wynik uruchomionego testu na **2 500 000
markerów i 5 000 000 relacji dodatkowych tagów**. W tym celowo wymagającym
przykładzie zakazany tag jest dodatkiem każdego markera, ale głównym tagiem
tylko 25 000 markerów:

| Wyszukiwanie | Wiersze markerów | Strony odczytu |
|---|---:|---:|
| Dotychczasowy filtr primary + extras, po 1000 | 2 500 000 | 2500, z ponownym COUNT |
| Nowe wyszukiwanie primary, po 2000 | 25 000 | 13, bez COUNT |

Plany SQLite potwierdziły wyszukiwanie po indeksach bez sortowania całego
zbioru trafień. Lokalne SELECT-y zajęły około 0,011 s dla globalnego zakazu
i 0,0869 s przy dodatkowej kontroli właściciela. To **pomiar syntetyczny samego
odczytu**, bez HTTP, hooków, plików i mutacji Stash. Nie jest prognozą całego
czasu cleanupu Twojej instalacji ani obietnicą usunięcia 2,5 mln markerów w takim czasie.

```powershell
python -m benchmarks.benchmark_cleanup_sql --markers 2500000 > cleanup_benchmark.json
python -m pytest -q
```

Benchmark tworzy wyłącznie własną tymczasową bazę i nie łączy się ze Stash.
Raport cleanupu pokazuje `cleanup_sql_reads`, `cleanup_rows_read`,
`cleanup_sql_read_ms`, `cleanup_api_writes` i `cleanup_api_write_ms`.
Liczniki `marker_tag_removals` / `scene_tag_removals` oznaczają operacje
obiekt–tag; ten sam obiekt może wystąpić dla kilku usuwanych tagów.

Nie wykonywano połączenia z Twoim Stash ani pomiaru jego faktycznych mutacji.
Rzeczywisty czas może być zdominowany przez natywne usuwanie, hooki lub dysk.

Implementację API i indeksy zweryfikowano w źródłach `stashapp/stash`,
commit `b872e360f05963f59d45605d1e638de88511dc47`:

- `internal/api/resolver_query_find_scene_marker.go`
- `pkg/sqlite/repository.go` i `pkg/sqlite/scene_marker_filter.go`
- `pkg/sqlite/migrations/1_initial.up.sql`
- `internal/api/resolver.go` i `internal/api/resolver_mutation_scene.go`
- `graphql/schema/types/scene-marker.graphql`

Repozytorium: https://github.com/stashapp/stash

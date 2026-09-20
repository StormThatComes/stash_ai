# Kolejna optymalizacja merge

Ta poprawka odpowiada na zgłoszone około 2 s na scenę. Bez logu rzeczywistego
przebiegu nie można przypisać całego tego czasu do jednej przyczyny. Profil
kodu potwierdził jednak powtarzane skanowanie klatek i kosztowne liczenie
rozmiaru wyników zapytań. Odczyty GraphQL dodatkowo uruchamiały resolvery
tagów dla każdego markera. Poprawka obejmuje oba obszary i dodaje pomiary.

## Co się zmieniło

- **Indeks wystąpień dodatków.** Dla zgodnej osi VTT zapisuje pozycje wykryć,
  przedziały pokrycia i ciągi kolejnych próbek. Zapytania o przedział wybierają
  odpowiednie pozycje przez wyszukiwanie binarne i przycinają gotowe ciągi.
  Nie przechodzą ponownie przez wszystkie klatki i źródła. Nietypowe,
  nakładające się stare osie czasu zachowują dotychczasowy algorytm.
- **Ponowne użycie kwalifikacji i tańszy cache.** Dodatki tego samego
  przedziału są kwalifikowane raz. Rozmiar znanego schematu jest szacowany
  konserwatywnie, bez przechodzenia po każdym obiekcie. Cache zapytań
  ograniczony do danej osi jest zwalniany po obliczeniu sceny; wynikowe
  wkłady zachowują potrzebne dane potwierdzające wykrycie.
- **Zbiorczy odczyt Stash.** Dostępne API `querySQL` odczytuje tagi i markery
  scen przez stałe, parametryzowane SELECT-y. Nie uruchamia osobnych resolverów
  GraphQL dla tagów każdego markera. Wewnętrzne podzapytania korzystają z
  indeksów `scene_id` i `scene_marker_id`.
- **Lżejszy `--merge-only`.** Sceny są czytane przez SQL po rosnącym ID,
  z ograniczonym rozmiarem strony, bez COUNT/OFFSET i bez pobierania ścieżek
  VTT/sprite. Przy zwykłej inferencji ścieżki pozostają pobierane przez API,
  a tagi i markery są dołączane zbiorczo. Bez dostępnego SQL używany jest
  GraphQL. Błąd odczytu także przełącza przebieg na GraphQL.
- **Jedna transakcja mniej na publikację.** Zatwierdzenie `pending` i zapis
  sprawdzenia zgodności publikacji odbywają się w tej samej transakcji SQLite.
  Zamiar publikacji nadal jest zapisany przed mutacjami Stash.

Wszystkie zapisy Stash nadal wykonuje jego API. SQL służy wyłącznie do
odczytu; podgląd dopuszcza jedynie konkretne SELECT-y z kodu. Reguły
kwalifikacji, YAML, progi i format zapisanych wykryć pozostają bez zmian.
Domyślnie nadal działa jeden model na GPU, cztery wątki przygotowania CPU
i cel 32 GiB RAM opisany w `UPGRADE_performance.md`.

## Jak sprawdzić wynik na swojej kolekcji

Podmień komplet modułów z archiwum, zachowując bazę i własne YAML.
Nowe moduły to `stash_timeline.py` i `stash_merge_sql.py`.
Pierwszy merge po zmianie kodu ponownie sprawdzi publikacje; nie wymaga
inferencji. Drugi identyczny przebieg może pomijać niezmienione sceny.

```powershell
python .\stash_ai.py --merge-only

# Szczegółowe czasy jednej sceny; zastąp 123 jej identyfikatorem.
python .\stash_ai.py --merge-only --scene-id 123 --verbose
```

Podsumowanie automatycznie pokazuje `Timings (exclusive)`. Czasy etapów
nie zawierają czasu zagnieżdżonych etapów, więc odczyt Stash nie jest
ponownie doliczany do planowania merge.

| Pole | Co obejmuje |
|---|---|
| `merge_compute` | Obliczenia wkładów modeli i tagów, z wyłączeniem odczytów stanu |
| `merge_plan` | Unia, dopasowanie markerów i plan zmian, z wyłączeniem mierzonych odczytów/zapisów |
| `state_read` | Odczyt, dekodowanie i cache lokalnych wyników/snapshotów |
| `state_write` | Kodowanie i transakcje lokalnych snapshotów |
| `stash_read` | Czas żądań odczytu API, także SELECT-y i katalog tagów |
| `stash_write` | Czas mutacji API Stash |
| `bootstrap` | Import dawnych wyników, poza osobno mierzonymi operacjami stanu |

To pomiary wskazanych etapów, nie pełnego czasu startu skryptu. Czas odczytów
API w podsumowaniu obejmuje również początkowe czytanie kolekcji. `--verbose`
dodaje rozbicie dla każdej przeliczonej sceny.

Liczniki wyjaśniają wybór ścieżki:

- `merge_sql_enabled=1`: udało się uruchomić zbiorczy odczyt SQL.
- `merge_graphql_fallback`: brak SQL lub powrót do GraphQL po błędzie.
- `merge_reused`: sceny, dla których pominięto przeliczanie i publikację.
- `merge_cache_miss`: brak sprawdzenia zgodności albo jego unieważnienie
  przez zapis wyników, oczekującą publikację lub cleanup.
- `merge_policy_changed`: zmieniły się ustawienia, kod lub sprawdzana hierarchia.
- `merge_stash_changed`: zmieniły się tagi albo markery sceny.

Wzrost `merge_reused` przy powtórzeniu oznacza działanie szybkiej ścieżki.
Jeśli nadal wychodzi około 2 s na przeliczoną scenę, wiersz `Timings` oraz
liczniki `merge_*` pozwolą ustalić, który etap to powoduje.

## Weryfikacja

W identycznym teście offline (400 klatek, 28 etykiet, cztery źródła) mediana
czasu obliczeń spadła z **1,562 s w poprzedniej paczce do 0,979 s** — o **37,3%**.
Trzy powtórzenia na wersję, uruchamiane kolejno w tym samym środowisku.
Obie wersje zwracają 1812 wkładów i 453 markery oraz identyczny hash pełnego
wyniku. Szczegóły: [merge_performance_results.json](../benchmarks/results/merge_performance_results.json).

Pomiar nie obejmuje rzeczywistego serwera Stash ani GPU. Nie przesądza,
ile potrwa scena na Twojej maszynie; korzyść ze zbiorczych odczytów SQL
trzeba odczytać z nowego raportu podczas rzeczywistego przebiegu.

406 testów offline obejmuje dotychczasowe zachowania oraz porównanie indeksu
z algorytmem referencyjnym dla zakresów, luk, różnych źródeł i nakładających
się osi czasu. Testy SQL porównują snapshoty z odczytem natywnych pól,
sprawdzają użycie indeksów, brak zapisów w dry-run i powrót do GraphQL.

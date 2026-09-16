# Wydajność pipeline — 32 GiB RAM, jeden model na GPU

Dokument opisuje poprzedni etap optymalizacji. Najnowsze zmiany merge,
odczyty SQL i aktualne pomiary opisuje `UPGRADE_merge.md`.

Ta aktualizacja ogranicza kopiowanie i ponowne przetwarzanie danych w merge,
zapisach SQLite oraz przygotowaniu inferencji. Zachowuje reguły kwalifikacji,
progi, etykiety, czasy markerów i informacje o modelach źródłowych. Wartości
wszystkich ośmiu plików YAML pozostały identyczne; zmieniły się tylko komentarze
wydajnościowe w `tagger_models.yml`.

## Uruchomienie

Rozpakuj wszystkie moduły z paczki obok istniejącej bazy `tagger_sources.sqlite3`.
Zachowaj bazę oraz własne YAML. Domyślne ustawienia nowej wersji to:

| Ustawienie | Domyślnie | Znaczenie |
|---|---:|---|
| `--ram-budget-gb` | 32 | Cel pamięci procesu, używany do sterowania cache |
| `--ram-cache-gb` | 24 | Maksymalny wspólny cache wyników, indeksów, VTT i sprite’ów |
| `--prepare-workers` | 4 | Równoległe transformacje klatek na CPU |
| `--prefetch-batches` | 4 | Maksymalnie cztery przygotowane paczki przed inferencją |

```powershell
# Zwykłe uruchomienie; ukończone wyniki modeli są ponownie używane.
python .\stash_ai.py --models all

# Tylko scalenie zapisanych wyników, bez uruchamiania modeli.
python .\stash_ai.py --merge-only

# Po dodaniu wykluczeń do YAML — dotychczasowy szybki cleanup.
python .\stash_ai.py --models all --cleanup-only
```

Pierwszy merge po aktualizacji buduje małe sprawdzenia zgodności publikacji.
Kolejny może już pomijać przeliczenia. Nie trzeba usuwać bazy, robić `--restart`
ani ponownie uruchamiać modeli z powodu tej aktualizacji.

## Zmiany

1. **Pomijanie niezmienionego merge.** Każda poprawnie opublikowana scena dostaje
   mały zapis `merge_checks`: hash ustawień/kodu oraz zaobserwowanego stanu
   tagów i markerów. Jeśli się zgadzają, następny przebieg pomija import dawnych
   wyników, rozpakowanie wykryć, obliczenia unii i zapis snapshotów. Zmiana
   wyników modelu, oczekująca publikacja, cleanup, YAML lub wyjść Stash wymusza
   ponowne uzgodnienie. Przy włączonej hierarchii sprawdzane są też definicje
   i rodzice tagów. Nie pomija się automatycznie scen po nieudanej publikacji.

2. **Mniej kopiowania podczas unii markerów.** Wewnętrzne obliczenia odwołują
   się do tych samych niezmienianych danych potwierdzających wykrycie.
   Powtarzające się obiekty potwierdzeń są agregowane raz, bez wielokrotnego
   kopiowania ich list przedziałów, próbek i źródeł. Publiczne odczyty z bazy
   nadal zwracają niezależne, modyfikowalne dane. Wewnętrzne hashe segmentów
   i grup klatek są ponownie używane podczas pracy z tym samym rekordem.

3. **Lżejszy zapis SQLite.** Publikacja najpierw zapisuje `pending`, a po sukcesie
   Stash przenosi ten sam BLOB do `published`. Znika drugie kodowanie JSON/zlib
   całego snapshotu. Zastępowanie wyników nie wykonuje porównań potrzebnych
   wyłącznie do dopisywania historii. Indeks częściowy przyspiesza wykrywanie
   oczekujących publikacji. Triggery unieważniają cache merge w tej samej
   transakcji co dane; wycofanie transakcji wycofuje również unieważnienie.

4. **Wspólne wejście dla kolejnych modeli.** W trybie `merged` pobrane VTT,
   sprite’y i sprawdzone manifesty są przechowywane w jednym cache RAM.
   Modele korzystają z tego samego wejścia bez ponownego pobierania plików,
   które nadal mieszczą się w cache. Po usunięciu wpisu z cache ponowne pobranie
   porównuje hash. Zmiana danych w trakcie przebiegów zgłasza błąd zamiast
   łączyć niespójne wejścia. Nowe uruchomienie ponownie weryfikuje pliki.
   Sygnatury ukończonych inferencji i checkpointów pozostają zgodne.

5. **Równoległe przygotowanie CPU.** Cztery wątki wykonują transformacje
   niezależnych kopii klatek. Odczyt arkuszy i kolejność wyników pozostają
   kontrolowane; zamknięcie lub błąd zwalnia obrazy i wątki. Wywołania inferencji
   pozostają sekwencyjne, a poprzedni model jest zamykany przed wczytaniem
   następnego. Nie dodano równoległego uruchamiania modeli na RTX 4080 Super.

6. **Filtrowanie wyników bez pełnych list Python.** NumPy wybiera indeksy
   spełniające próg; do list trafiają tylko odpowiednie etykiety i wyniki.
   Porównanie zachowuje dotychczasowe granice float32/float64 i kolejność przy
   równych wynikach. Nie zmienia precyzji modeli ani progów.

7. **Podgląd bez kopii całej bazy.** Każdy `--dry-run`, także merge i inferencja,
   dołącza bazę formatu 5/6 tylko do odczytu. Zmienione rekordy trafiają do
   tymczasowej bazy SQLite z ograniczonym cache stron. Duży podgląd może użyć
   dysku tymczasowego, zamiast zajmować RAM rozmiarem całej bazy. Źródło jest
   zachowane; tymczasowy stan znika przy zamknięciu. Wznowienia widzą usunięcia
   checkpointów i żądań odświeżenia również w tej warstwie. Starsze formaty
   nadal wymagają kopii na czas migracji, ale trafia ona do bazy tymczasowej.
   Wykryta zmiana źródła przez inny proces przerywa podgląd.

## Pamięć i granice optymalizacji

Cache przydziela pamięć na żądanie i nie przekracza 75% `--ram-budget-gb`.
Rozmiary są szacowane dla rozpakowanych obiektów, także atrybutów manifestów,
a nie dla skompresowanych BLOB-ów. Jedna wspólna pula zapobiega przydzielaniu
osobnych 24 GiB dla wyników, indeksów i obrazów. Na Windows/Linux kontrola RSS
usuwa wpisy pod presją pamięci, przy domyślnym progu 28 GiB.

**32 GiB jest celem zarządzania cache, a nie twardym limitem systemowym.**
Domyślnie 8 GiB pozostaje poza cache na model, aktywną scenę, obrazy i SQLite.
Alokacji pojedynczej bardzo dużej sceny lub biblioteki modelu nie da się
zablokować samym limitem cache. Raport pokazuje szczyt cache, próbkowane RSS
i liczbę redukcji pod presją. Nie dodano trwałego cache mediów na dysku.

Początkowy odczyt scen wraz z tagami i markerami nadal jest potrzebny do
wykrycia zmian w Stash i importu dawnych wyników. Niezmieniona scena pomija
następnie kosztowny etap merge oraz drugi odczyt przed publikacją. Zmiany
wykonane w Stash już po początkowym odczycie pominiętej sceny będą wykryte
przy następnym uruchomieniu. Globalny YAML wpływający na politykę unieważnia
sprawdzenia dla całego zakresu; szybki cleanup nadal korzysta z selektywnej
ścieżki SQL opisanej w `UPGRADE_cleanup_sql.md`.

Nie wprowadzono równoległych zapisów do Stash ani zmian jego pliku bazy.
Istniejące SELECT-y cleanupu wykorzystują indeksy i paginację po kluczu;
zmiany wykonuje API Stash. Format surowych wyników i snapshotów pozostaje
zgodny z poprzednią paczką. Małe indeksy pochodne nie zastępują danych źródłowych.

## Pomiary i testy

Offline, w tym samym środowisku CPU, trzy powtórzenia na wersję, mediana:

| Próba | Poprzednia paczka | Ta paczka | Przyspieszenie |
|---|---:|---:|---:|
| Budowanie wkładów modeli i unia: 400 klatek, 28 etykiet, 4 modele | 4,633 s | 1,538 s | 3,01× |
| Filtrowanie macierzy 512 × 20 000 wyników, rzadkie trafienia | 0,382 s | 0,026 s | 14,58× |

Pierwsza próba daje w obu wersjach 1812 wkładów i 453 markery. SHA-256 pełnej
kanonicznej reprezentacji wyników, łącznie z danymi potwierdzającymi wykrycia,
jest identyczny. Druga próba również ma identyczny hash wyników. Surowe czasy
i hashe są w `performance_results.json`; skrypt `benchmark_performance.py`
pozwala powtórzyć porównanie, wskazując poprzedni katalog przez `--project`.

Te pomiary nie obejmują GPU, sieci ani całej rzeczywistej kolekcji Stash.
Nie wyznaczają czasu obsługi 2,5 mln markerów. Filtrowanie jest tylko częścią
inferencji, więc 14,58× nie oznacza takiego przyspieszenia pracy modelu.
Oszczędności pobrań zależą od tego, jaka część mediów mieści się w RAM.

```powershell
python -m pytest -q
python .\benchmark_performance.py
python .\benchmark_performance.py --project C:\poprzednia_paczka\stash_ai
```

Przechodzą 392 testy offline. Nowe testy sprawdzają m.in. pomijanie odczytu
wykryć i publikacji niezmienionej sceny, unieważnianie po zmianie markerów,
YAML, hierarchii, wyników i cleanupu, atomowe zatwierdzanie snapshotów,
podgląd dużych BLOB-ów bez kopiowania, wznowienia, spójność wejścia między
modelami, kolejność i zwalnianie obrazów w wątkach oraz dokładność progów.
W osobnym teście cztery modele pobierają ten sam VTT i sprite po jednym razie,
zachowując dotychczasowe sygnatury wejść.

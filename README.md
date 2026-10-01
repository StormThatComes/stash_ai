# Stash AI

Automatyczne tagowanie scen i tworzenie markerów czasowych dla [Stash](https://github.com/stashapp/stash).

Stash AI używa czterech modeli: JoyTag (`jt:`), WD EVA02-Large v3 (`wd:`),
DanbooruTagQuery L/16 (`dtq:`) i OppaiOracle V1.1 (`op:`). Modele są
uruchamiane kolejno — w GPU znajduje się jeden model naraz.

Domyślny tryb `merged` publikuje wspólne tagi `ai:`, łączy zgodne wyniki modeli
i zachowuje osobne źródła oraz pewności w lokalnej bazie SQLite. Wejście wizualne
pochodzi wyłącznie z VTT i sprite'ów Stash; aplikacja nie otwiera plików wideo.

## Wymagania

- Python **3.14.7** — to jedyna wspierana i testowana wersja;
- Stash z wygenerowanymi sprite'ami/VTT;
- zależności z `requirements.txt`;
- CUDA jest zalecana; CPU działa, ale jest znacznie wolniejsze.

Dla CPU zamień `onnxruntime-gpu` na zgodny `onnxruntime>=1.21,<2`.

## Szybki start

```powershell
pip install -r requirements.txt

$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<własny klucz ze Stash>"

python .\stash_ai.py --download-models --models all
python .\stash_ai.py --models all --scene-id 123 --dry-run --verbose
python .\stash_ai.py --models all --scene-id 123
```

Ustaw `STASH_API_KEY` jawnie w środowisku. W kodzie nadal istnieje historyczny
fallback klucza; należy traktować go jako skompromitowany i nie używać.
Szczegóły: [SECURITY.md](SECURITY.md).

## Najczęstsze tryby

```powershell
# Zwykła inferencja; poprawny cache i checkpointy są używane automatycznie.
python .\stash_ai.py --models all

# Świeża inferencja tylko dla wybranego zakresu.
python .\stash_ai.py --models wd --scene-id 123 --restart

# Ponowne scalenie zapisanych wyników bez uruchamiania modeli.
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Równoległe read-only przygotowanie evidence; publikacja do Stash pozostaje sekwencyjna.
python .\stash_ai.py --merge-only --merge-workers 4
# --dry-run i --profile-merge celowo używają ścieżki sekwencyjnej.

# Zmiana progów z zachowanego cache, bez modeli.
python .\stash_ai.py --models all --merge-only --rethreshold-markers
python .\stash_ai.py --models all --merge-only --rethreshold-scenes --rethreshold-markers

# Cleanup wykluczeń.
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only

# Naprawa grup istniejących markerów.
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run

# Lokalny podgląd źródeł tagu.
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123

# Usuwanie nieużywanych definicji tagów.
python .\stash_ai.py --prune-empty-tags-only --dry-run

# Usuwanie niepotrzebnej historii importu starych source-prefixed tagów/markerów.
python .\stash_ai.py --prune-legacy-imports --dry-run
python .\stash_ai.py --prune-legacy-imports

# Zdrowie, spójność logiczna i kompaktowanie lokalnej bazy stanu.
python .\stash_ai.py --db-report
python .\stash_ai.py --db-check
python .\stash_ai.py --db-compact --dry-run
python .\stash_ai.py --db-compact

# Lokalny, anonimowy raport diagnostyczny do ręcznego przekazania developerowi.
python .\stash_ai.py --anonymous-diagnostics .\stash-ai-diagnostics.json

# Użyteczność modeli i leave-one-model-out ablation bez inferencji.
# Domyślnie używa do 4 procesów CPU; --model-report-workers 1 wymusza tryb sekwencyjny.
python .\stash_ai.py --model-report .\model-utility.json
python .\stash_ai.py --model-report .\model-utility.json --model-report-workers 8
python .\stash_ai.py --model-report .\model-utility-scene-123.json --scene-id 123

# Porzucenie niedokończonego planu merge/cleanup.
python .\stash_ai.py --abandon-pending
```

Zmiana progów zapisuje małe wiersze w tabeli `result_thresholds`; skompresowane
wyniki inferencji i evidence klatek pozostają bez zmian. Dotyczy to także
automatycznego użycia cache po zmianie progów w YAML. Odczyt wyników, merge,
cleanup i explain używają efektywnych progów z tej tabeli. Stare progi zapisane
wewnątrz wyników są nadal odczytywane, bez przepisywania istniejącej bazy.
Pierwsza zmiana progu oznacza bazę jako schemat v8, którego starsze wersje
programu nie obsługują. `--dry-run` zapisuje progi wyłącznie w tymczasowej bazie.

Pełną listę opcji pokazuje:

```powershell
python .\stash_ai.py --help
```

## Jak działa stan i cache

Domyślna baza `tagger_sources.sqlite3` przechowuje surowe dowody modeli,
checkpointy, opublikowane snapshoty i informacje potrzebne do bezpiecznego
wznowienia. **Zachowuj ją w backupie.** Nie zawiera obrazów ani klucza API.

W trybie `merged` domyślne `resume`:
- pomija ukończoną inferencję, gdy model i wejście są zgodne;
- wznawia częściowy checkpoint;
- może przeliczyć merge bez GPU po zmianie reguł publikacji;
- przy zmianie progów wykorzystuje zachowane obserwacje, jeśli ich retencja jest
  wystarczająca.

`--restart` wymusza świeżą inferencję dla wybranych modeli/scen. Nie używaj go
do zwykłego przeliczenia merge, cleanupu ani zmiany progów.

Domyślnie `REPLACE_SCENE_TAGS=True` i `REPLACE_MARKERS=True`. Wyłączenie
którejś flagi zachowuje historię danego rodzaju wyniku; ponowne włączenie
normalizuje historię przy bezpiecznym reuse cache.

### Czyszczenie historii legacy imports

`--prune-legacy-imports` usuwa wyłącznie wpisy `legacy_imports`, które nie są
już potrzebne do ochrony przed ponownym bootstrapem starych tagów/markerów
`jt:`, `wd:`, `dtq:` i `op:`. Polecenie łączy się ze Stash, ale nie ładuje
modeli ani nie uruchamia inferencji.

Każdy zapisany import jest porównywany z aktualnymi source-prefixed obiektami w
Stash. Wpis pozostaje, jeśli dokładnie zaimportowany wcześniej tag lub marker
nadal istnieje. Jeśli taki obiekt zniknął (albo cała scena została usunięta),
ledger można usunąć bez zmiany `results`, `result_index` ani snapshotów.
`--dry-run` pokazuje liczbę i rozmiar rekordów kwalifikujących się do usunięcia.
Komenda odmawia pracy przy pending publication planie.

Samo usunięcie `legacy_imports` nie unieważnia już `merge_checks`: nowy legacy
evidence i tak zapisuje `results`, a ten zapis nadal prawidłowo unieważnia cache.
Po większym czyszczeniu można użyć `--db-compact`, aby zwrócić zwolnione strony
SQLite do systemu plików.

### Zdrowie i kompaktowanie bazy stanu

`--db-check` wykonuje głęboki, wyłącznie odczytowy test spójności całej bazy
(we wszystkich scope'ach). Oprócz pełnego SQLite `integrity_check` sprawdza m.in.
zgodność `results` z `result_index` (łącznie z zapisaną sygnaturą ostatniej
inferencji), dekodowalność trwałych BLOB-ów, dozwolone model/kind, relacje
checkpoint/frames/analysis, wymiary checkpointów oraz generacje cleanup/recovery.
Waliduje też zawartość wyników scen, klatek i zakresów markerów: wymagane pola,
tagi, czasy, prawdopodobieństwa oraz zapisane statystyki czasowe. Sprawdza klatki
checkpointów (również `scene_tags`) i wartości float32: odrzuca `NaN`,
nieskończoności oraz wyniki poza `[0, 1]`. Zachowuje obsługę starszych wyników bez
statystyk i metadanych retencji, a także importowanych markerów punktowych i
brakujących historycznych ocen (`score: null`).
Osierocone lub historyczne rekordy, które nie muszą oznaczać uszkodzenia, są
raportowane jako warnings. Pending publication/cleanup i refresh requests są
pokazywane jako stan roboczy, a nie automatycznie jako błędy.

Komenda nie łączy się ze Stash, nie ładuje modeli i niczego nie naprawia ani nie
modyfikuje. Podczas głębokiego skanu pokazuje pasek postępu z aktualną fazą
(np. SQLite integrity, results/index, state payloads, checkpoints). Zwraca exit
code `0`, gdy nie znaleziono błędów logicznych, oraz `1`, gdy wykryto
niespójność; same warnings nie powodują błędu.

`--db-report` nie łączy się ze Stash ani nie ładuje modeli. Pokazuje wersję
schematu, wynik SQLite `quick_check`, rozmiar pliku i sidecarów WAL/SHM, wolne
strony, zagregowane rekordy `results` według modelu/rodzaju oraz liczniki
checkpointów, refresh requests, pending/recovery i merge checks dla bieżącego
scope. Nie wypisuje nazw tagów, ID scen ani zapisanych score'ów.

`--db-compact` wykonuje SQLite `VACUUM` pod tym samym lockiem co zwykły writer.
Przed operacją ponownie sprawdza integralność, dostępne miejsce na dysku oraz brak
niedokończonego merge/cleanup. `--dry-run` wykonuje tylko preflight i pokazuje
szacunkowy odzysk z wolnych stron. Recovery evidence i checkpointy są zachowywane.

### Anonimowy raport dla developera

`--anonymous-diagnostics NEW.json` tworzy lokalny raport przeznaczony do
ręcznego przejrzenia i ewentualnego przekazania developerowi. Aplikacja **nigdy
nie wysyła go automatycznie** i odmawia nadpisania istniejącego pliku.

Raport obejmuje całą wskazaną bazę stanu, ale nie eksportuje scene IDs, wartości
`scope`, URL-i, ścieżek plików, nazw tagów, score'ów, timestampów markerów,
sygnatur ani hashy wejścia. Nieznane nazwy modeli/rodzajów są zastępowane
`other`, a niestandardowe tabele są raportowane tylko jako liczba.

Dla optymalizacji schematu raport zawiera m.in.:
- wersje Pythona/SQLite i rodzinę systemu operacyjnego;
- page size/count, freelist, rozmiar DB/WAL oraz liczbę scope'ów i scen;
- liczbę rekordów znanych tabel i rozkłady rozmiarów BLOB-ów;
- per model/rodzaj: percentyle rozmiaru compressed JSON, compression ratio,
  segmenty/rekord, items/segment, detekcje/klatkę i scene observations;
- zagregowany stan checkpointów, result index i refresh requests.

Raport jest pozbawiony bezpośrednich identyfikatorów i treści, ale dokładne
agregaty rozmiaru/liczebności nadal opisują konkretną instalację. Przed
udostępnieniem warto przejrzeć wygenerowany JSON.

### Model Utility / Ablation Report

`--model-report NEW.json` działa wyłącznie na zapisanym evidence i nie łączy się
ze Stash ani nie uruchamia modeli. Dla każdej sceny najpierw odtwarza normalny
wynik wszystkich zapisanych modeli, a następnie cztery kontrfaktyczne wyniki
`all minus one model`. Oznacza to, że po usunięciu modelu od początku
przeliczane są consensus seeds, temporal rules, shared scene tags, marker extras
i Occurrence — raport nie filtruje jedynie gotowych markerów.

Per model raport pokazuje m.in.:
- liczbę scen z zapisanym evidence i scen, w których pominięcie modelu zmienia wynik;
- finalne scene tags i marker components wspierane przez model;
- scene tags, markery, extras i Occurrence utracone po jego usunięciu;
- markery, które pozostają, ale zmieniają zakres lub extras;
- procent scen z evidence, w których model ma mierzalny marginalny wpływ.

Zmiana samego badge'a źródeł markera, np. `[jt+wd]` na `[wd]`, nie jest liczona
jako zmiana semantyczna. Dzięki temu raport mierzy wpływ na tagowanie, zakresy i
extras, a nie samą obecność provenance. `--scene-id` może ograniczyć analizę
do wybranych scen.

## Wynik w Stash

W trybie `merged`:

| Element | Przykład |
|---|---|
| Tag sceny / core markera | `ai:blue_hair` |
| Tytuł markera | `[AI] blue_hair [jt+wd+op]` |
| Tag własności | `ai:marker:stash-taggers` |
| Punkt Occurrence | dodatkowo `ai:marker:occurrence` |
| Grupa zwykłego markera | `m:blue_hair` |
| Grupa Occurrence | `m:blue_hair:occurance` |
| Korzeń grup markerów | `ai_markers` |
| Kategorie hierarchii | `ai:category:...` |

Pisownia `:occurance` w nazwach grup jest celowa i pozostaje zgodna z istniejącym
formatem.

`--output-mode separate` zachowuje prefiksy poszczególnych modeli i nie używa
wspólnego mechanizmu publikacji `ai:`.

## Konfiguracja

| Plik / opcja | Rola |
|---|---|
| `tagger_models.yml` | modele, katalogi, progi, batch size |
| `temporal_policy.yml` | reguły temporalne, consensus, extras i Occurrence |
| `scene_tag_exclusions.yml` | globalne wykluczenia tagów |
| `marker_core_exclusions.yml` | wykluczenia głównych tagów markerów |
| `*_hierarchy.yml` | hierarchie kategorii modeli |
| `--state-db` | lokalizacja trwałej bazy stanu |

Aktualne domyślne limity procesu:
- `--ram-budget-gb 32`;
- `--ram-cache-gb 24` — cache jest przydzielany na żądanie i ograniczony do
  maksymalnie 75% budżetu;
- `--prepare-workers 4`;
- `--merge-workers 4` — równoległy CPU replay evidence; zapisy Stash/SQLite pozostają sekwencyjne;
- `--prefetch-batches 4`.

## Aktualizacje

`update_stash_ai.py` aktualizuje zarządzane pliki aplikacji bez Gita i bez
uruchamiania `pip`. Wymaga tokena GitHub z dostępem `Contents: read` do
prywatnego repozytorium.

Przy pierwszym przejściu ze starego układu (moduły `stash_*.py` obok launchera)
uruchom updater **dwa razy**. Stara wersja pobiera tylko pliki z katalogu
głównego; drugie uruchomienie nowego updatera pobierze pakiet `stash_ai/`.
Zachowaj te same potrzebne opcje, np. `--force-code`. Przed ukończeniem migracji
launcher wyświetla instrukcję ponownego uruchomienia updatera. Przy `git pull`
lub użyciu najnowszego updatera od początku wystarczy jedna aktualizacja.

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"

python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run
python .\update_stash_ai.py
```

Updater:
- pobiera pliki z jednego SHA `main`;
- instaluje launchery, pakiet `stash_ai/` oraz YAML-e; usuwa zastąpione moduły z katalogu głównego;
- zachowuje lokalnie zmienione YAML-e i tworzy propozycje `.incoming`;
- odmawia nadpisania zmienionego kodu bez `--force-code`;
- nie aktualizuje modeli, `.env`, SQLite ani zależności Pythona;
- używa blokady instalacji i dziennika odzyskiwania.

Po przerwaniu aktualizacji użyj:

```powershell
python .\update_stash_ai.py --recover
```

## Bezpieczeństwo

- używaj własnego, zrotowanego `STASH_API_KEY` z environment;
- nie commituj tokenów GitHub ani kluczy Stash;
- read-only GraphQL i media GET mogą być ponawiane po błędach przejściowych;
- mutacje GraphQL **nie są automatycznie retry'owane**, ponieważ serwer mógł
  wykonać zapis mimo utraconej odpowiedzi;
- `--dry-run` nie zapisuje do Stash ani trwałej bazy źródłowej.

## Testy i CI

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

CI testuje Python 3.14.7. Pełny zestaw testów działa na Linux, a updater jest
dodatkowo sprawdzany na Windows. Testy używają atrap Stash i modeli; nie wymagają
produkcyjnego serwera.

## Dokumentacja

- [Aktualny podręcznik](docs/README_multimodel.md)
- [Indeks dokumentacji](docs/README.md)
- [Cleanup SQL](docs/UPGRADE_cleanup_sql.md)
- [Wydajność i pamięć](docs/UPGRADE_performance.md)
- [Merge i szybka ścieżka](docs/UPGRADE_merge.md)
- [Wykluczenia i cleanup](docs/UPGRADE_exclusions.md)
- [Occurrence](docs/UPGRADE_occurrences.md)
- [Grupy markerów](docs/UPGRADE_marker_groups.md)
- [Porzucanie przerwanych operacji](docs/UPGRADE_abandon_pending.md)
- [Profilowanie merge](docs/UPGRADE_diagnostics.md)
- [Benchmarki](benchmarks/README.md)

Pliki w `docs/validation/` są historycznymi wynikami konkretnych walidacji i
benchmarków. Nie opisują bieżącej liczby testów ani aktualnego kontraktu aplikacji.

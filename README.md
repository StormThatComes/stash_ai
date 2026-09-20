# Stash AI

Automatyczne tagi scen i markery czasowe dla [Stash](https://github.com/stashapp/stash).
Cztery modele tagujące pracują po kolei (jeden w GPU naraz). Domyślny tryb `merged` zapisuje wspólne tagi `ai:` i łączy nakładające się przedziały tej samej etykiety.

Wejście pochodzi **wyłącznie** z VTT i sprite’ów Stash. Skrypt nie otwiera plików wideo.

## Spis treści

- [Co robi](#co-robi)
- [Wymagania](#wymagania)
- [Szybki start](#szybki-start)
- [Najczęstsze komendy](#najczęstsze-komendy)
- [Konfiguracja](#konfiguracja)
- [Aktualizacje](#aktualizacje)
- [Bezpieczeństwo](#bezpieczeństwo)
- [Testy](#testy)
- [Struktura projektu](#struktura-projektu)
- [Benchmarki](#benchmarki)
- [Dokumentacja szczegółowa](#dokumentacja-szczegółowa)

## Co robi

| Element | Tryb domyślny (`merged`) |
|---|---|
| Tag sceny / główny tag markera | `ai:blue_hair` |
| Tytuł markera | np. `[AI] blue_hair [jt+wd+op]` |
| Tag własności markera | `ai:marker:stash-taggers` |
| Grupa zwykłych markerów / Occurance | `m:blue_hair` / `m:blue_hair:occurance` |
| Rodzic wszystkich tagów grup | `ai_markers` |
| Hierarchia | `ai:category:…`, korzeń `ai:category:AI` |
| Pewności i źródła modeli | lokalny `tagger_sources.sqlite3` |

Modele: JoyTag (`jt:`), WD EVA02-Large v3 (`wd:`), DanbooruTagQuery L/16 (`dtq:`), OppaiOracle V1.1 (`op:`).
`--models` wybiera inferencję; wcześniejszy wkład pozostałych modeli nadal wchodzi do scalania.

Tryb `--output-mode separate` zostawia oryginalne prefiksy i nie rusza wyników `ai:`.

## Wymagania

- Python 3.10+
- Działający Stash z wygenerowanymi sprite’ami / VTT
- GPU z CUDA jest zalecane; CPU działa, ale jest wolniejsze
- zależności z `requirements.txt` (na CPU zamień `onnxruntime-gpu` na `onnxruntime`)

## Szybki start

```powershell
pip install -r requirements.txt
copy .env.example .env
# uzupełnij STASH_URL i STASH_API_KEY

$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<klucz ze Stash>"

python .\stash_ai.py --download-models --models all
python .\stash_ai.py --models all --scene-id 123 --dry-run --verbose
python .\stash_ai.py --models all --scene-id 123
```

Klucz API bierz **wyłącznie** ze zmiennej środowiskowej. Nie commituj `.env`. Szczegóły: [SECURITY.md](SECURITY.md).

## Najczęstsze komendy

```powershell
# Inferencja / wznowienie (domyślnie reuse + checkpoint)
python .\stash_ai.py --models all
python .\stash_ai.py --models all --scene-id 123 --restart

# Scalanie bez modeli (migracja i ponowne reguły)
python .\stash_ai.py --merge-only --dry-run
python .\stash_ai.py --merge-only --scene-id 123

# Uzupełnienie grup istniejących markerów bez inferencji i zmiany czasów
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run --verbose
python .\stash_ai.py --sync-marker-groups-only

# Podgląd pochodzenia tagu, offline
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123

# Cleanup i puste tagi
python .\stash_ai.py --cleanup-only --dry-run
python .\stash_ai.py --prune-empty-tags-only --dry-run

# Po przerwaniu cleanupu/merge: porzuć stary plan i użyj zmienionych YAML-i
python .\stash_ai.py --cleanup-only --abandon-pending --dry-run
python .\stash_ai.py --cleanup-only --abandon-pending

# Samo porzucenie planu, bez zapisów do Stash i bez inferencji
python .\stash_ai.py --abandon-pending
```

`joytag_stash_markers.py` to tylko stara nazwa launchera — używaj `stash_ai.py`.

## Konfiguracja

| Plik | Rola |
|---|---|
| `tagger_models.yml` | modele, ścieżki, progi, batch |
| `temporal_policy.yml` | histereza, wspólne klatki, dodatki markerów |
| `scene_tag_exclusions.yml` | globalna blacklista tagów |
| `marker_core_exclusions.yml` | wykluczenia głównych tagów markerów |
| `*_hierarchy.yml` | hierarchie kategorii per model |
| `--state-db` | trwała baza źródeł (domyślnie `tagger_sources.sqlite3`) |

Zachowuj `tagger_sources.sqlite3` w kopiach zapasowych. Nie zawiera obrazów ani klucza API.

## Aktualizacje

`update_stash_ai.py` pobiera aplikację bezpośrednio z `main` prywatnego repozytorium
`StormThatComes/stash_ai`. Wymaga tylko Pythona 3.10+; nie wymaga Gita ani bibliotek
modeli. Pobiera wyłącznie pliki `stash_*.py`, `joytag_stash_markers.py`, sam updater
i pliki `*.yml` z katalogu głównego. Nie pobiera testów, benchmarków, `.github`,
dokumentacji ani `requirements.txt` i nie uruchamia `pip`.

Do pierwszego uruchomienia pobierz `update_stash_ai.py` z repozytorium po zalogowaniu
do GitHub i umieść obok aplikacji. **Zatrzymaj wszystkie stare procesy aplikacji**:
wersje sprzed dodania updatera nie respektują jeszcze blokady instalacji.
Utwórz fine-grained PAT z dostępem tylko do tego repozytorium i `Contents: read`.
Token podawaj przez `GH_TOKEN` lub `GITHUB_TOKEN`, np. z menedżera sekretów.
Nie jest to klucz API Stash; połączenie GitHub w ChatGPT nie uwierzytelnia lokalnego skryptu.

```powershell
# GH_TOKEN powinien być już ustawiony w środowisku; nie wpisuj tokena do kodu.
python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run

# Pierwsza aktualizacja starej instalacji: jawna zgoda na zastąpienie nieznanego kodu.
python .\update_stash_ai.py --force-code

# Kolejne aktualizacje (zwykle bez dodatkowych flag).
python .\update_stash_ai.py

# Opcjonalnie zastąp również lokalnie zmienione YAML-e wersjami z main.
python .\update_stash_ai.py --overwrite-configs

# Po przerwaniu instalacji, offline i bez tokena.
python .\update_stash_ai.py --recover
```

Domyślnym celem jest katalog skryptu, niezależnie od bieżącego katalogu terminala.
Można wskazać inny istniejący katalog przez `--install-dir PATH`.

| Sytuacja | Zachowanie |
|---|---|
| Brak pliku lub brak lokalnych zmian od ostatniej instalacji | Instalacja wersji z `main` |
| Lokalnie zmieniony YAML / istniejący YAML bez historii | Zachowanie pliku; propozycja w `nazwa.yml.incoming` |
| Zmieniony kod Python / istniejący kod bez historii, różny od `main` | Przerwanie przed instalacją; można użyć `--force-code` |
| Lokalnie zmieniona propozycja `.incoming` | Przerwanie; przenieś propozycję przed ponowieniem |
| Moduł usunięty na `main` | Usunięcie tylko kodu wcześniej zarządzanego przez updater |
| YAML usunięty na `main` | Zachowanie lokalnej konfiguracji |

Zaakceptowanie propozycji oznacza ręczne przeniesienie jej zawartości do właściwego
YAML-a. Updater nie scala automatycznie list tagów. Zmiana `requirements.txt` jest
sygnalizowana na podstawie metadanych; sprawdź wymagania na GitHub i dostosuj
środowisko ręcznie, z uwzględnieniem swojej wersji Torch/CUDA/ONNX Runtime.

Wszystkie pliki jednej aktualizacji pochodzą z jednego SHA `main`. Updater sprawdza
hash każdego pobranego pliku i składnię Pythona przed instalacją. Nie wykonuje testów
ani importów aplikacji. Nie gwarantuje to zgodności zachowanego lokalnego YAML-a
z przyszłymi zmianami jego formatu — komunikaty `KEEP` wymagają przeglądu konfiguracji.

Aplikacja trzyma wspólną blokadę instalacji od startu do zakończenia. Updater wymaga
blokady wyłącznej; nie zatrzymuje procesów ani nie przerywa merge/cleanup.
`--check` i `--dry-run` mogą działać równolegle z aplikacją. Blokada jest zwalniana
przez system po zakończeniu procesu; nie usuwaj ręcznie pliku blokady.

Instalację, w tym podmianę updatera, wykonuje pomocnik uruchomiony z lokalnej kopii.
Kopie starych plików i plan pozostają w `.stash-ai-update/transactions/ID/`.
Pojedyncze podmiany są atomowe; cała instalacja wielu plików korzysta z dziennika.
Przerwany zapis blokuje start aplikacji do wykonania `--recover`. Odzyskiwanie
porzuca nierozpoczęty plan, cofa rozpoczętą instalację albo finalizuje zakończoną.
Uszkodzona kopia lub zewnętrzna zmiana pliku podczas odzyskiwania zatrzymuje operację
i zachowuje dziennik. Jeśli główny updater nie daje się uruchomić, użyj kopii:

```powershell
python .\.stash-ai-update\transactions\ID\worker.py --install-dir . --recover
```

Nie są zmieniane modele, `.env`, SQLite ani stan pending. Odzyskiwanie instalacji
nie cofa migracji bazy wykonanych później przez aplikację. Kopie pozostają do ręcznego
usunięcia po sprawdzeniu aktualizacji; nie usuwaj kopii wskazanej przez aktywny dziennik.
Kody wyjścia: `0` sukces/podgląd, `1` błąd, `2` konflikty w podglądzie, `130` przerwanie.
Podgląd nie zapisuje aplikacji ani historii aktualizacji, ale może utworzyć pusty plik blokady.

## Bezpieczeństwo

- `STASH_API_KEY` tylko z środowiska / lokalnego `.env` (w `.gitignore`)
- po wycieku klucza w historii gita **rotuj** go w Stash
- mutacje GraphQL nie są ponawiane; `--dry-run` nic nie zapisuje w Stash ani w bazie źródeł

## Testy

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Testy są w `tests/`; `pytest.ini` ustawia katalog kolekcji i ścieżkę importów.
Wspólne makiety są w `tests/helpers/`, a współdzielone fixtures w `tests/conftest.py`.
Można uruchomić pojedynczy plik, np. `python -m pytest -q tests/test_recovery.py`.
Sam updater: `python -m pytest -q tests/test_updater.py` — wymaga tylko `pytest`.

Przy pull requestach testy odpala GitHub Actions (`.github/workflows/tests.yml`).
Testy mockują Stash i wagi — nie pobierają modeli.

## Struktura projektu

| Lokalizacja | Zawartość |
|---|---|
| Katalog główny | aplikacja `stash_*.py`, launchery, updater, YAML-e i requirements |
| `tests/` | testy regresji, fixtures i wspólne pomocniki |
| `benchmarks/` | skrypty pomiarowe offline |
| `benchmarks/results/` | zachowane, referencyjne wyniki pomiarów |
| `docs/` | szczegółowy podręcznik i notatki aktualizacji |
| `docs/validation/` | historyczne raporty walidacji |
| `.github/workflows/` | testy CI |

Polecenia w dokumentacji zakładają uruchamianie z katalogu głównego repozytorium.
Układ plików aplikacji i konfiguracji pozostaje zgodny z updaterem.

## Benchmarki

Pomiary wykonują się na syntetycznych danych, bez połączenia ze Stash i bez GPU.
Po zainstalowaniu `requirements-dev.txt` można wykonać krótkie sprawdzenie:

```powershell
python -m benchmarks.benchmark_performance --frames 40 --labels 8 --repeats 1
python -m benchmarks.benchmark_merge_pipeline --frames 40 --labels 8 --repeats 1
python -m benchmarks.benchmark_occurrences --frames 40 --labels 8 --repeats 1
python -m benchmarks.benchmark_cleanup_sql --markers 1000 --legacy-time-budget 1
```

Obsługiwane jest też bezpośrednie uruchomienie pliku, np.
`python benchmarks/benchmark_occurrences.py`. Benchmarki wydajności i merge przyjmują
`--project PATH`, aby porównać bieżący kod z wcześniejszą kopią aplikacji.
Znaczenie wyników i większe zestawy danych opisuje [instrukcja benchmarków](benchmarks/README.md).

## Dokumentacja szczegółowa

Pełny opis scalania, reguł temporalnych, RAM i wznawiania:

- [README_multimodel.md](docs/README_multimodel.md)

Notatki aktualizacji:

- [UPGRADE_cleanup_sql.md](docs/UPGRADE_cleanup_sql.md) — cleanup SQL
- [UPGRADE_merge.md](docs/UPGRADE_merge.md) — przyspieszenie merge
- [UPGRADE_performance.md](docs/UPGRADE_performance.md) — cache i 32 GiB RAM
- [UPGRADE_occurrences.md](docs/UPGRADE_occurrences.md) — punkty wykrycia bez markera
- [UPGRADE_marker_groups.md](docs/UPGRADE_marker_groups.md) — grupy `m:` pod `ai_markers` i cleanup
- [UPGRADE_abandon_pending.md](docs/UPGRADE_abandon_pending.md) — porzucanie przerwanych operacji i zmiana YAML-i
- [UPGRADE_diagnostics.md](docs/UPGRADE_diagnostics.md) — diagnostyka wolnego merge
- [UPGRADE_exclusions.md](docs/UPGRADE_exclusions.md) — blacklisty

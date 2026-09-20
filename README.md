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
- [Bezpieczeństwo](#bezpieczeństwo)
- [Testy](#testy)
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

## Bezpieczeństwo

- `STASH_API_KEY` tylko z środowiska / lokalnego `.env` (w `.gitignore`)
- po wycieku klucza w historii gita **rotuj** go w Stash
- mutacje GraphQL nie są ponawiane; `--dry-run` nic nie zapisuje w Stash ani w bazie źródeł

## Testy

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

Przy pull requestach to samo odpala GitHub Actions (`.github/workflows/tests.yml`).
Testy mockują Stash i wagi — nie pobierają modeli.

## Dokumentacja szczegółowa

Pełny opis scalania, reguł temporalnych, RAM i wznawiania:

- [README_multimodel.md](README_multimodel.md)

Notatki aktualizacji:

- [UPGRADE_cleanup_sql.md](UPGRADE_cleanup_sql.md) — cleanup SQL
- [UPGRADE_merge.md](UPGRADE_merge.md) — przyspieszenie merge
- [UPGRADE_performance.md](UPGRADE_performance.md) — cache i 32 GiB RAM
- [UPGRADE_occurrences.md](UPGRADE_occurrences.md) — punkty wykrycia bez markera
- [UPGRADE_marker_groups.md](UPGRADE_marker_groups.md) — grupy `m:` pod `ai_markers` i cleanup
- [UPGRADE_diagnostics.md](UPGRADE_diagnostics.md) — diagnostyka wolnego merge
- [UPGRADE_exclusions.md](UPGRADE_exclusions.md) — blacklisty

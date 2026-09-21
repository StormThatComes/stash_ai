# Usunięte warstwy zgodności i JoyTag z Hugging Face

Uruchamiaj `python stash_ai.py`. Stary launcher `joytag_stash_markers.py` oraz
blokada wycofanego modułu wideo `stash_refine.py` zostały usunięte z projektu.
Usunięto także nieużywany wrapper `load_marker_core_exclusions()` i helper
`stash_performance.remember()` używany wyłącznie przez testy.

## Aktualizacja istniejącej instalacji

- Zmień nazwę launchera w poleceniach i harmonogramach na `stash_ai.py`.
- Usuń `code_dir` z sekcji `joytag` własnego `tagger_models.yml`.
  `JOYTAG_CODE_PATH` i checkout źródeł nie są już używane.
- `model_dir` oraz zmienne `*_MODEL_PATH` nadal wybierają katalogi pobierania.
  Domyślnie JoyTag używa `models/joytag` obok YAML, bez dawnej ścieżki Windows.
- Updater usuwa obydwa wycofane pliki aplikacji z kopią zapasową. Historia
  wcześniejszych instalacji i cofanie przerwanych aktualizacji pozostają obsługiwane.
  Zmienione lub nieśledzone kopie wymagają istniejącej opcji `--force-code`.
  Przy ręcznej aktualizacji usuń te dwa pliki samodzielnie.
- Własne YAML pozostają chronione przez updater; propozycja `.incoming` nie
  zmienia aktywnej konfiguracji automatycznie.

## JoyTag

Wagi i metadane pochodzą z `fancyfeast/joytag`, rewizja
`6b7f16331a6ccf0fdce37d5a9564715f6e772b22`: `model.safetensors`, `config.json`
oraz `top_tags.txt`. Architektura PyTorch pochodzi z HF Space tego samego autora,
rewizja `bce35c9afc4dbe5eb71ee836939c84d4614eec96`, plik `Models.py`.
Kod loadera jest weryfikowany przez SHA-256, a następnie ładowany bez modyfikacji
`sys.path` i bez importowania lokalnego modułu `Models`.

```powershell
python .\stash_ai.py --models joytag --download-models
```

Polecenie pobiera brakujące zasoby i przygotowuje loader w cache Hugging Face.
Nie wykonuje pobranego kodu ani nie ładuje modelu. Zwykła inferencja pobiera
brakujące zasoby automatycznie. Tryby `--offline` i `download_missing: false`
wymagają już pobranych plików, w tym loadera w cache HF.

`model.pt` nie jest używany ani usuwany. Jeśli istnieje tylko ten format,
aplikacja pobierze `model.safetensors`; offline zgłosi brak wymaganych wag.
Walidacja hierarchii nadal potrzebuje jedynie słownika, a cleanup i merge-only
nie pobierają ani nie wykonują loadera.

Tożsamość inferencji JoyTag uwzględnia teraz przypiętą rewizję i hash loadera
zamiast plików lokalnego checkoutu. Pierwsze zwykłe uruchomienie może więc
uznać wcześniejsze wyniki/checkpointy JoyTag za wymagające nowej inferencji.
`--merge-only` nadal wykorzystuje zachowane wyniki bez uruchamiania modeli;
tożsamość pozostałych trzech modeli nie zmienia się z powodu loadera JoyTag.

Nie zmieniono algorytmów markerów, trybu `separate`, formatu bazy ani obsługi
wcześniejszych publikacji i przerwanych operacji.

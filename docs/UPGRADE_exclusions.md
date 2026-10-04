# Wykluczenia i cleanup

Wykluczenia są stosowane podczas normalnego merge oraz w osobnych trybach
cleanupu bez inferencji.

## Pliki

`scene_tag_exclusions.yml` zawiera globalne i per-model reguły tagów scen.
Globalny zakaz dotyczy również marker cores/extras i tworzenia hierarchii.

`marker_core_exclusions.yml` wyklucza główne tagi markerów bez globalnego
zakazu użycia etykiety w innych rolach.

Opcja kodowa `MARKER_TAG_BLOCKLIST` została wycofana. Używaj powyższych plików
YAML. Core-only pozostawia etykietę dostępną jako dodatek i tag sceny, a reguły
`scene_tag_exclusions.yml` obejmują sceny oraz główne i dodatkowe tagi markerów.
Osobny zakaz obejmujący wszystkie role markerów z zachowaniem tagu sceny
nie jest już dostępny.

Cache zapisany z pustą blocklistą jest nadal użyteczny. Niepusta lub nieznana
historyczna blocklista wymaga `--restart` dla wskazanych scen/modeli przed
zwykłym merge lub rethreshold markerów. Cleanup może nadal cofać opublikowane
dowody bez inferencji, z zachowaniem ich historycznych reguł i granic.

## Aktualizacja starych nazw

Konfiguracja i CLI używają jednej nazwy dla każdej opcji:

| Dawna nazwa | Aktualna nazwa |
| --- | --- |
| YAML `excluded_tags` | `excluded_scene_tags` |
| `joytag_marker_core_exclusions.yml` wybierany automatycznie | `marker_core_exclusions.yml` |
| `--model` | `--models` |
| `--tag-exclusions` | `--scene-tag-exclusions` |
| `--sync-marker-core-only` | `--cleanup-markers-only` |

Zmień klucz listy w YAML, pozostawiając jej zawartość i `by_model` bez zmian.
Przenieś własny plik wykluczeń markerów pod aktualną nazwę albo wskaż go
jawnie przez `--marker-core-exclusions PATH`. Jeśli masz oba pliki, wybierz
własną politykę zamiast nadpisywać ją pustym plikiem dostarczonym z aplikacją.
Zaktualizuj argumenty w skryptach i harmonogramach. CLI wymaga pełnych nazw
opcji, bez skrótów. Wycofane aliasy i klucz YAML zgłaszają błąd.

## Cleanup

```powershell
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only
```

Tryby częściowe:

```powershell
python .\stash_ai.py --models all --cleanup-markers-only
python .\stash_ai.py --models all --cleanup-scene-tags-only
```

Cleanup nie uruchamia modeli i nie pobiera sprite'ów.

## Globalne wykluczenie

Globalny wpis może usunąć zakazany tag `ai:/jt:/wd:/dtq:/op:` ze scen i
markerów, a w pełnym cleanupie również samą definicję tagu. Usunięcie definicji
przez Stash usuwa jej relacje do innych obiektów.

To zachowanie jest celowo silniejsze niż core-only. Przed dodaniem globalnej
blacklisty upewnij się, że dana definicja nie jest potrzebna ręcznie.

## Core-only

`marker_core_exclusions.yml` usuwa tylko zarządzane markery mające zakazany
core. Definicja tagu pozostaje i może nadal działać jako dodatkowa etykieta.

Ręczne markery nie są przejmowane tylko dlatego, że mają podobną nazwę tagu.

## Stan i odtwarzanie

Surowe evidence modeli pozostaje w `tagger_sources.sqlite3`. Cleanup zapisuje
kompaktową informację potrzebną do prawidłowego odczytu wcześniejszych
publikacji, bez masowego przepisywania wszystkich wyników.

Po zdjęciu wykluczenia zwykły `--merge-only` może ponownie opublikować wynik z
zachowanego evidence, jeśli nie został on ręcznie utracony w sposób, którego
aplikacja nie potrafi odtworzyć.

## Przerwanie

Niedokończony cleanup należy:
- wznowić z tymi samymi regułami; albo
- porzucić przez `--abandon-pending` przed użyciem nowych reguł.

Nie zmieniaj polityk pending cleanupu bez jednej z tych operacji.

## Puste definicje

Opcjonalnie:

```powershell
python .\stash_ai.py --models all --cleanup-only --prune-empty-tags
python .\stash_ai.py --prune-empty-tags-only --dry-run
```

Pruning usuwa tylko zweryfikowane nieużywane liście i respektuje zarządzane
grupy markerów.

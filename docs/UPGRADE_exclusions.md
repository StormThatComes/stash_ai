# Wykluczenia i cleanup

Wykluczenia są stosowane podczas normalnego merge oraz w osobnych trybach
cleanupu bez inferencji.

## Pliki

`scene_tag_exclusions.yml` zawiera globalne i per-model reguły tagów scen.
Globalny zakaz dotyczy również marker cores/extras i tworzenia hierarchii.

`marker_core_exclusions.yml` wyklucza główne tagi markerów bez globalnego
zakazu użycia etykiety w innych rolach.

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

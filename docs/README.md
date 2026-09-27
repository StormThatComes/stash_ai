# Dokumentacja Stash AI

Dokumentacja jest podzielona na bieżące instrukcje użytkownika oraz historyczne
artefakty pomiarowe.

## Bieżąca dokumentacja

- [Szybki start i konfiguracja](../README.md)
- [Aktualny podręcznik](README_multimodel.md)
- [Granice modułów i odpowiedzialności klas](ARCHITECTURE.md)
- [Cleanup SQL](UPGRADE_cleanup_sql.md)
- [Wydajność i pamięć](UPGRADE_performance.md)
- [Merge i szybka ścieżka](UPGRADE_merge.md)
- [Wykluczenia i cleanup](UPGRADE_exclusions.md)
- [Punkty Occurrence](UPGRADE_occurrences.md)
- [Grupy markerów](UPGRADE_marker_groups.md)
- [Porzucanie przerwanych operacji](UPGRADE_abandon_pending.md)
- [Profilowanie merge](UPGRADE_diagnostics.md)
- [Benchmarki](../benchmarks/README.md)

Nazwy plików `UPGRADE_*.md` są historyczne. Treść tych dokumentów opisuje
aktualne zachowanie odpowiadających im funkcji, a nie instrukcję instalacji
konkretnego starego wydania.

## Historyczne artefakty

Pliki w [validation/](validation/README.md) oraz zachowane wyniki w
`benchmarks/results/` dokumentują konkretne wcześniejsze pomiary. Mogą być
przydatne do regresji, ale wartości czasu i liczby testów nie są bieżącą
specyfikacją projektu.

Kod aplikacji jest w pakiecie `stash_ai/`. W katalogu głównym pozostają
launchery `stash_ai.py`, `update_stash_ai.py`, YAML-e i domyślna baza
`tagger_sources.sqlite3`. Updater nie pobiera dokumentacji, testów ani benchmarków.

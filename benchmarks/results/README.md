# Referencyjne wyniki pomiarów

Pliki JSON przeniesiono bez zmiany zawartości. Dotyczą wcześniejszych wersji i
warunków pomiaru opisanych w dokumentacji; nie są wynikiem bieżącego uruchomienia.

- [Cleanup SQL](cleanup_benchmark.json)
- [Wydajność](performance_results.json)
- [Merge](merge_performance_results.json)
- [Occurance](occurrence_performance_results.json)
- [Diagnostyka merge](diagnostic_benchmark_results.json)

[Współdzielone dowody snapshotów](shared_snapshot_proofs.json) to pomiar z
2026-10-02 względem `main` po #90: 4 modele, 400 klatek, 28 etykiet i 453 markery,
mediana 5 prób. Pełny merge offline skrócił się z 1,996 do 1,511 s, a kompaktowanie,
serializacja i kompresja snapshotów z 0,790 do 0,319 s. JSON markerów zmalał
z 44,29 do 14,90 MB, skompresowany BLOB z 1,69 do 0,73 MB. Hashe logicznych
snapshotów, wyników Stash i liczniki publikacji są identyczne. Plik zawiera również
wariant z jedną etykietą, bez współdzielonych dowodów. Są to syntetyczne pomiary
bez inferencji i opóźnień sieciowych, nie prognoza czasu całej biblioteki.

[Instrukcja nowych pomiarów](../README.md) ·
[Historyczne raporty walidacji](../../docs/validation/README.md)

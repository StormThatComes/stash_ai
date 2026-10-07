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

[Indeks przedziałów markerów](marker_interval_index.json) to pomiar z 2026-10-02
względem `main` po #93, mediana 5 prób tym samym skryptem. Dla 1000 istniejących
markerów jednego primary tagu ponowny sync skrócił się z 0,913 do 0,061 s, a samo
dopasowanie z 0,855 do 0,031 s. Dla 125 markerów punktowych sync skrócił się
z 1,242 do 0,012 s; cleanup 1000 markerów w 20 primary tagach z 0,179 do 0,065 s.
Plik zawiera też mały wariant kontrolny. Hashe wejścia, logicznych snapshotów,
wyników Stash i liczniki są zgodne między wersjami. Pomiar obejmuje SQLite i makietę
Stash, bez inferencji oraz opóźnień API.

[Kaskada popularnego tagu](tag_cascade_2026-10-07.json) to pomiar z 2026-10-07:
100 tys. i milion markerów, indeksowane relacje oraz WAL/NORMAL zgodne ze Stash
v0.31.1. Wcześniejsze odpinanie relacji partiami skróciło największą transakcję,
ale zwiększyło całkowity czas o 37–49% nawet bez API, hooków i ładowania obiektów.
Hashe zachowanego stanu są identyczne. [Opis i decyzja](../../docs/cleanup_performance.md).

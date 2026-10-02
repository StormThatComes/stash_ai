# Benchmarki offline

Uruchamiaj z katalogu głównego repozytorium, po instalacji `requirements-dev.txt`.
Skrypty używają syntetycznych danych, lokalnej tymczasowej bazy lub makiety Stash;
nie łączą się ze Stash i nie uruchamiają modeli GPU.

```powershell
python -m benchmarks.benchmark_performance --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_marker_matching --markers 1000 --labels 1 --repeats 5
python -m benchmarks.benchmark_retained_merge --frames 400 --labels 12 --weak-labels 108 --repeats 5
python -m benchmarks.benchmark_occurrences --frames 500 --labels 100 --repeats 5
python -m benchmarks.benchmark_cleanup_sql --markers 2500000 --legacy-time-budget 5
python -m benchmarks.benchmark_rethreshold --scenes 6 --frames 600 --labels 12 --repeats 5
```

| Skrypt | Co mierzy |
|---|---|
| `benchmark_performance` | budowanie markerów i scalanie, z hashem wyniku |
| `benchmark_merge_pipeline` | merge ze stanem SQLite, etapami i makietą API |
| `benchmark_marker_matching` | ponowny merge na istniejących markerach; czas całego sync i dopasowania, hashe logicznych snapshotów oraz Stash |
| `benchmark_retained_merge` | obserwacje współdzielone, markery temporalne jednego modelu i merge CPU przy szerokiej retencji wyników, z hashami każdego etapu |
| `benchmark_occurrences` | wybór reprezentatywnych punktów Occurance |
| `benchmark_cleanup_sql` | odkrywanie kandydatów przez SQL; bez kosztu API i usuwania |
| `benchmark_rethreshold` | zmiana progów, czas/dekodowania Check i Apply, zapisy evidence oraz hashe wyników i result_index; bez merge/API |

Każdy skrypt obsługuje też bezpośrednie uruchomienie przez ścieżkę, np.
`python benchmarks/benchmark_performance.py`. Bez `--project` benchmark wskazuje
katalog aplikacji, niezależnie od bieżącego katalogu terminala.

Do porównań z poprzednią kopią kodu:

```powershell
python -m benchmarks.benchmark_performance --project C:\poprzednia_paczka\stash_ai
python -m benchmarks.benchmark_merge_pipeline --project C:\poprzednia_paczka\stash_ai
python -m benchmarks.benchmark_rethreshold --project C:\poprzednia_paczka\stash_ai
```

Wskazany katalog musi zawierać launcher `stash_ai.py` i pakiet `stash_ai/`.
Narzędzia pomiarowe oraz makieta Stash pochodzą z bieżącego repozytorium,
więc porównanie nie wymaga testów w starej instalacji. Porównuj identyczne dane
i parametry; przy zmianie zachowania aplikacji mogą się różnić również hashe.

`benchmark_merge_pipeline` pokazuje rozmiar JSON i skompresowanego BLOB-a każdego
snapshotu. Przy zmianie formatu zapisu porównuj `input_sha256`,
`logical_snapshot_sha256`, `stash_sha256` i liczniki operacji; `snapshot_sha256`
obejmuje surowe BLOB-y, więc może się zmienić przy identycznym wyniku logicznym.
Dekodowanie i hashowanie do tej weryfikacji odbywa się poza pomiarem czasu.

`benchmark_marker_matching` najpierw publikuje rozłączne przedziały, a następnie
mierzy ponowny sync na tych samych markerach. `--labels` rozkłada je na wiele
primary tagów, `--point-only` symuluje Stash bez końców markerów, a `--cleanup`
obejmuje również sprawdzanie obecności markerów przed cleanupem. Także ten skrypt
obsługuje `--project` i `--output`. Przy porównaniu wersji użyj tego samego skryptu
i parametrów; hashe oraz liczniki muszą być identyczne. Weryfikacja jest poza
pomiarem; wynik obejmuje lokalny zapis snapshotu, bez inferencji i opóźnień API.

`benchmark_rethreshold` domyślnie mierzy wyniki zapisane przez wskazaną wersję
aplikacji, z gotowymi metadanymi walidacji, jeśli je obsługuje. Dodaj
`--mode scene` lub `--mode markers`, aby zmierzyć zmianę tylko progów scen
lub markerów; domyślny `--mode combined` zmienia oba progi. Dodaj
`--legacy-metadata`, aby zmierzyć pierwsze uzupełnianie metadanych starego cache.
Opcja czyści tylko pochodne metadane w tymczasowej bazie benchmarku.
Raport osobno pokazuje dekodowania evidence i czas planowania w `Check`/`Apply`;
całkowity czas obejmuje też transakcje i zapisy progów.
Hashe obejmują wejściowe BLOB-y, efektywne wyniki i indeks ukończenia.

Skrypty drukują JSON na stdout. `benchmark_merge_pipeline` i `benchmark_occurrences`
przyjmują też `--output PATH`. Do nowego pomiaru użyj nowej nazwy pliku:

```powershell
python -m benchmarks.benchmark_occurrences --output benchmarks/results/local_occurrences.json
```

Lokalne wyniki `local_*` są ignorowane przez Git. Zachowane wcześniejsze pomiary
w [results](results/README.md) są punktami odniesienia; nie nadpisuj ich zwykłym
uruchomieniem. Czasy syntetycznych testów nie są prognozą czasu pracy całej biblioteki.

`benchmark_retained_merge` zachowuje po 12 obserwacji mocnych lub podtrzymujących
oraz 108 słabszych na klatkę przy progu publikacji 0,8 i retencji od 0,1.
Dodaj `--weak-labels 0`, aby sprawdzić wariant bez zbędnych obserwacji.
Do porównania wersji uruchom ten sam skrypt z tymi samymi parametrami i innym
`--project`; identyczne hashe wejścia i wszystkich trzech etapów potwierdzają
zgodność danych, markerów i dowodów. Cały pomiar `merge_cpu` obejmuje budowanie
markerów i scalanie przed zapisem do SQLite lub wywołaniem API Stash.

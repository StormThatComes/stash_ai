# Benchmarki offline

Uruchamiaj z katalogu głównego repozytorium, po instalacji `requirements-dev.txt`.
Skrypty używają syntetycznych danych, lokalnej tymczasowej bazy lub makiety Stash;
nie łączą się ze Stash i nie uruchamiają modeli GPU.

```powershell
python -m benchmarks.benchmark_performance --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_occurrences --frames 500 --labels 100 --repeats 5
python -m benchmarks.benchmark_cleanup_sql --markers 2500000 --legacy-time-budget 5
python -m benchmarks.benchmark_rethreshold --scenes 6 --frames 600 --labels 12 --repeats 5
```

| Skrypt | Co mierzy |
|---|---|
| `benchmark_performance` | budowanie markerów i scalanie, z hashem wyniku |
| `benchmark_merge_pipeline` | merge ze stanem SQLite, etapami i makietą API |
| `benchmark_occurrences` | wybór reprezentatywnych punktów Occurance |
| `benchmark_cleanup_sql` | odkrywanie kandydatów przez SQL; bez kosztu API i usuwania |
| `benchmark_rethreshold` | zmiana obu progów, czas/dekodowania Check i Apply, zapisy evidence i zgodność wyników; bez merge/API |

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

`benchmark_rethreshold` domyślnie mierzy wyniki zapisane przez wskazaną wersję
aplikacji, z gotowymi metadanymi walidacji, jeśli je obsługuje. Dodaj
`--legacy-metadata`, aby zmierzyć pierwsze uzupełnianie metadanych starego cache.
Opcja czyści tylko pochodne metadane w tymczasowej bazie benchmarku.
Raport osobno pokazuje dekodowania evidence i czas planowania w `Check`/`Apply`;
całkowity czas obejmuje też transakcje i zapisy progów.

Skrypty drukują JSON na stdout. `benchmark_merge_pipeline` i `benchmark_occurrences`
przyjmują też `--output PATH`. Do nowego pomiaru użyj nowej nazwy pliku:

```powershell
python -m benchmarks.benchmark_occurrences --output benchmarks/results/local_occurrences.json
```

Lokalne wyniki `local_*` są ignorowane przez Git. Zachowane wcześniejsze pomiary
w [results](results/README.md) są punktami odniesienia; nie nadpisuj ich zwykłym
uruchomieniem. Czasy syntetycznych testów nie są prognozą czasu pracy całej biblioteki.

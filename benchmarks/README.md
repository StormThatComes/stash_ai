# Benchmarki offline

Uruchamiaj z katalogu głównego repozytorium, po instalacji `requirements-dev.txt`.
Skrypty używają syntetycznych danych, lokalnej tymczasowej bazy lub makiety Stash;
nie łączą się ze Stash i nie uruchamiają modeli GPU.

```powershell
python -m benchmarks.benchmark_performance --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_occurrences --frames 500 --labels 100 --repeats 5
python -m benchmarks.benchmark_cleanup_sql --markers 2500000 --legacy-time-budget 5
```

| Skrypt | Co mierzy |
|---|---|
| `benchmark_performance` | budowanie markerów i scalanie, z hashem wyniku |
| `benchmark_merge_pipeline` | merge ze stanem SQLite, etapami i makietą API |
| `benchmark_occurrences` | wybór reprezentatywnych punktów Occurance |
| `benchmark_cleanup_sql` | odkrywanie kandydatów przez SQL; bez kosztu API i usuwania |

Każdy skrypt obsługuje też bezpośrednie uruchomienie przez ścieżkę, np.
`python benchmarks/benchmark_performance.py`. Bez `--project` benchmark wskazuje
katalog aplikacji, niezależnie od bieżącego katalogu terminala.

Do porównań z poprzednią kopią kodu:

```powershell
python -m benchmarks.benchmark_performance --project C:\poprzednia_paczka\stash_ai
python -m benchmarks.benchmark_merge_pipeline --project C:\poprzednia_paczka\stash_ai
```

Wskazany katalog musi zawierać pliki aplikacji `stash_ai.py` i `stash_*.py`.
Narzędzia pomiarowe oraz makieta Stash pochodzą z bieżącego repozytorium,
więc porównanie nie wymaga testów w starej instalacji. Porównuj identyczne dane
i parametry; przy zmianie zachowania aplikacji mogą się różnić również hashe.

Skrypty drukują JSON na stdout. `benchmark_merge_pipeline` i `benchmark_occurrences`
przyjmują też `--output PATH`. Do nowego pomiaru użyj nowej nazwy pliku:

```powershell
python -m benchmarks.benchmark_occurrences --output benchmarks/results/local_occurrences.json
```

Lokalne wyniki `local_*` są ignorowane przez Git. Zachowane wcześniejsze pomiary
w [results](results/README.md) są punktami odniesienia; nie nadpisuj ich zwykłym
uruchomieniem. Czasy syntetycznych testów nie są prognozą czasu pracy całej biblioteki.

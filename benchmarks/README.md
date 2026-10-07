# Benchmarki offline

Uruchamiaj z katalogu głównego repozytorium, po instalacji `requirements-dev.txt`.
W CI wszystkie dziesięć wejść przechodzi smoke test na małych danych.
Fixture używają bieżącego formatu analysis i pełnych metadanych.
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
python -m benchmarks.benchmark_tag_cascade --markers 100000 --batch-size 200 --repeats 3
python -m benchmarks.benchmark_scoped_reports --scenes 9000 --selected 32 --repeats 3
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
| `benchmark_tag_cascade` | kaskada popularnego tagu kontra wcześniejsze odpinanie relacji w partiach; całkowity czas, najdłuższa transakcja, liczba mutacji i hash zachowanego stanu |
| `benchmark_scoped_reports` | raport cache i całe Check/Apply dla małej selekcji w większej bazie, osobno writer/preview; czas, instrukcje SQL, zapisy, dekodowania i hashe wyników |

Wynik `results/scoped_reports_2026-10-07.json` porównuje PR 4 z PR 5 dla
32 wybranych scen i czterech modeli, przy 900 i 9000 scenach w bazie. Dla 9000
scen mediana raportu preview spadła z 437 do 7 ms, Check/Apply writer z 201 do
12 ms, a Check/Apply preview z 420 do 15 ms. Hashe wyników pozostały identyczne,
surowe evidence nie zmieniło się i żaden wariant nie dekodował BLOB-ów predykcji.
Liczba instrukcji SQL po poprawce pozostaje stała dla obu wielkości bazy.
To pomiar lokalnego syntetycznego workflow; nie obejmuje modeli ani API.

Każdy skrypt obsługuje też bezpośrednie uruchomienie przez ścieżkę, np.
`python benchmarks/benchmark_performance.py`. Benchmarki porównujące aplikację bez `--project` wskazują
katalog aplikacji, niezależnie od bieżącego katalogu terminala.

Do porównań z poprzednią kopią kodu:

```powershell
python -m benchmarks.benchmark_performance --project C:\poprzednia_paczka\stash_ai
python -m benchmarks.benchmark_merge_pipeline --project C:\poprzednia_paczka\stash_ai
python -m benchmarks.benchmark_rethreshold --project C:\poprzednia_paczka\stash_ai
```

Wskazany katalog musi zawierać kompletną aplikację: launcher `stash_ai.py`,
`update_stash_ai.py` i pakiet `stash_ai/`.
Porównywane rewizje muszą udostępniać aktualne API używane przez skrypty:
`union_intervals(..., borrow=True)`, filtrowanie tablic NumPy przez `select_scores`,
`MergeDiagnostics` i tabelę `result_metadata` w `StateStore`. Benchmarki nie
dopasowują wywołań do starszych API.
Narzędzia pomiarowe oraz makieta Stash pochodzą z bieżącego repozytorium,
więc porównanie nie wymaga testów we wskazanej instalacji. Porównuj identyczne dane
i parametry; przy zmianie zachowania aplikacji mogą się różnić również hashe.

`benchmark_tag_cascade` używa wyłącznie własnej tymczasowej bazy SQLite i nie
przyjmuje `--project`. Odtwarza indeksowane relacje markerów, scen i obrazów do
tagów. Pomija sieć, hooki oraz ładowanie obiektów Stash; wariant wcześniejszego
odpinania ma dzięki temu optymistyczny koszt. Kopiowanie fixture i weryfikacja
hasha są poza pomiarem. Baza używa WAL/NORMAL i natychmiastowych transakcji zapisu,
tak jak Stash v0.31.1. Wynik oraz decyzję o strategii cleanupu opisuje
[cleanup_performance](../docs/cleanup_performance.md).

`benchmark_merge_pipeline` pokazuje rozmiar JSON i skompresowanego BLOB-a każdego
snapshotu. Przy zmianie formatu zapisu porównuj `input_sha256`,
`logical_snapshot_sha256`, `stash_sha256` i liczniki operacji; `snapshot_sha256`
obejmuje surowe BLOB-y, więc może się zmienić przy identycznym wyniku logicznym.
Dekodowanie i hashowanie do tej weryfikacji odbywa się poza pomiarem czasu.

`benchmark_marker_matching` najpierw publikuje rozłączne przedziały, a następnie
mierzy ponowny sync na tych samych markerach. `--labels` rozkłada je na wiele
primary tagów, `--point-only` mierzy publikację markerów punktowych, a `--cleanup`
obejmuje również sprawdzanie obecności markerów przed cleanupem. Także ten skrypt
obsługuje `--project` i `--output`. Przy porównaniu wersji użyj tego samego skryptu
i parametrów; hashe oraz liczniki muszą być identyczne. Weryfikacja jest poza
pomiarem; wynik obejmuje lokalny zapis snapshotu, bez inferencji i opóźnień API.

`benchmark_rethreshold` domyślnie mierzy wyniki zapisane przez wskazaną wersję
aplikacji, z gotowymi metadanymi walidacji. Dodaj
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

## Koszt historii uruchomień

`python -m benchmarks.benchmark_run_history --frames 128 --labels 12 --repeats 3`
porównuje historię on/off dla świeżej inferencji, cache i wznowienia checkpointu.
Każda próbka działa w osobnym procesie. Raportuje wall/CPU, zmienione wiersze SQL
bazy stanu i historii, peak RSS procesu, rozmiary baz oraz oddzielny koszt retencji.
Hashe ukończonych surowych wyników muszą być identyczne; wariant cache nie może
uruchomić predykcji. Retencja usuwa 40 zakończonych rekordów fixture, chroniąc proofs.
Fixture wykorzystuje prawdziwą orkiestrację, VTT/PNG i checkpointy oraz syntetyczny
tagger CPU. Wynik nie jest pomiarem rzeczywistego backendu ML ani sieci;
peak RSS obejmuje również przygotowanie fixture.

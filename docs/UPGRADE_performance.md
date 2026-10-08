# Wydajność i pamięć

Bieżący pipeline jest zoptymalizowany pod sekwencyjną inferencję modeli i duży
lokalny cache merge.

## Domyślne limity

| Opcja | Domyślnie | Znaczenie |
|---|---:|---|
| `--ram-budget-gb` | 32 | cel pamięci całego procesu |
| `--ram-cache-gb` | 24 | wspólny cache wyników, indeksów i mediów |
| `--prepare-workers` | 4 | równoległe transformacje klatek na CPU |
| `--prefetch-batches` | 4 | liczba przygotowanych batchy przed inferencją |
| `--result-prefetch-batches` | 32 | liczba wyników CPU buforowanych podczas obróbki i zapisu, 1–128 |

Cache jest przydzielany na żądanie i dodatkowo ograniczony do 75% budżetu RAM.
Budżet nie jest twardym limitem procesu — poza cache pozostają modele, tensory,
aktywna scena, Python i SQLite.

## Jeden model na GPU

Modele są uruchamiane kolejno. Przygotowanie obrazów może korzystać z kilku
workerów CPU, ale wywołania inferencji nie działają równolegle między modelami.

To ogranicza szczyt VRAM i upraszcza odzyskiwanie po błędach.

## Wspólne wejście modeli

Podczas jednego przebiegu merged pobrane VTT/sprite'y mogą być współdzielone
przez kolejne modele. Aplikacja przechowuje hash wejścia i odrzuca sytuację, w
której dane zmienią się między modelami.

Nie ma trwałego cache obrazów na dysku.

## Cache merge

Po poprawnej publikacji `merge_checks` zapisuje fingerprint polityki i stanu
sceny. Identyczny kolejny przebieg może pominąć:
- import starych wyników;
- dekodowanie evidence;
- budowanie indeksów;
- union markerów;
- przygotowanie publikacji.

Zapisy wyników, snapshotów i cleanupu unieważniają odpowiednie sprawdzenia
transakcyjnie.

Zmiana kodu wpływającego na publikację lub konfiguracji może spowodować
jednorazowy re-merge. To nie jest równoznaczne z ponowną inferencją.

## SQLite

Wyniki są przechowywane jako skompresowane rekordy. Publikacja zapisuje plan
`pending`, a po udanym Stash commit promuje zapisany stan do `published`.

Dane pochodne i małe indeksy służą wydajności; źródłowe evidence pozostaje
podstawą do ponownego merge/cleanupu.

## Dry-run

Dla nowoczesnego formatu bazy dry-run dołącza źródło read-only i zapisuje tylko
różnice do tymczasowego overlay. Dzięki temu podgląd dużej bazy nie wymaga pełnej
kopii do pamięci.

## Sterowanie pamięcią

Zmniejsz wartości, jeśli system zaczyna swapować:

```powershell
python .\stash_ai.py --models all --ram-budget-gb 20 --ram-cache-gb 14
```

Przy bardzo dużej scenie pojedyncza aktywna struktura może chwilowo przekroczyć
typowy udział cache. Raport końcowy pokazuje peak cache, próbkowane RSS, trafienia,
misses, evictions i pressure trims.

## Pomiar

```powershell
python -m benchmarks.benchmark_performance --frames 400 --labels 28 --repeats 3
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
```

Benchmarki są syntetyczne i nie obejmują GPU ani rzeczywistego serwera Stash.
Do diagnozy konkretnej biblioteki użyj `--profile-merge`; patrz
[UPGRADE_diagnostics.md](UPGRADE_diagnostics.md).

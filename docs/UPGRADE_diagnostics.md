# Profilowanie merge

Do diagnozy wolnego lub nieoczekiwanie przeliczanego merge użyj
`--profile-merge`. Profil nie wymaga nowej inferencji, jeśli jest połączony z
`--merge-only`.

## Zbieranie profilu

```powershell
python .\stash_ai.py --merge-only --scene-id 123 --force-merge --profile-merge merge-profile.jsonl
```

Dla kilku scen:

```powershell
python .\stash_ai.py --merge-only `
  --scene-id 123 --scene-id 124 --scene-id 125 `
  --force-merge --profile-merge merge-profile.jsonl
```

Plik `.jsonl` musi być nowy; aplikacja nie nadpisuje istniejącego profilu.

`--force-merge` omija wyłącznie cache publikacji. Nie usuwa evidence i nie
uruchamia modeli.

`--verbose` dodaje szczegóły do konsoli, ale nie jest wymagane do profilu.

## Co zawiera JSONL

Profil zapisuje rekord startowy, rekordy scen i podsumowanie. Dla scen dostępne
są m.in.:

| Pole | Znaczenie |
|---|---|
| `wall_seconds` / `cpu_seconds` | czas rzeczywisty i CPU |
| `status` / `reason` | wynik i powód przeliczenia/pominięcia |
| `stages` | czas i liczba wywołań etapów |
| `api` | operacje GraphQL/SQL, czasy, błędy, rozmiary |
| `counts` | liczby wejść, wyników, cache hits i bajtów |
| `statistics` | planowane zmiany publikacji |
| `unmeasured_seconds` | czas poza zarejestrowanymi etapami |

Etapy mają czas exclusive oraz inclusive. Do szukania udziału danego etapu używaj
wartości exclusive; nie sumuj rodzica z jego dziećmi.

## Najważniejsze obszary

Typowe grupy etapów:
- obliczenia markerów/scen/Occurrence;
- lokalny stan: SQL, decode/encode, compression i cache;
- odczyt Stash oraz decode odpowiedzi;
- rozwiązywanie tagów i katalogu;
- przygotowanie, mutacje i commit publikacji.

Początkowy skan kolekcji jest raportowany globalnie i nie jest częścią czasu
końcowego merge pojedynczej sceny.

## Jak interpretować problem

Jeżeli kolejny identyczny przebieg nadal przelicza scenę, zacznij od `reason` i
liczników `merge_*`. Najczęstsze przyczyny to:
- zmiana polityki/kodu publikacji;
- zmiana tagów lub markerów w Stash;
- zapis nowych wyników;
- pending publication;
- zmiana zarządzanego katalogu tagów.

Jeżeli dominuje czas API, sprawdź liczbę operacji oraz czy działa zbiorczy
read-only SQL. Jeżeli dominuje lokalny stan, porównaj decode/encode/compression i
rozmiary snapshotów. Jeżeli dominuje compute, dopiero wtedy warto profilować
algorytmy fusion/union.

## Bezpieczeństwo profilu

Profil nie zapisuje klucza API, obrazów ani pełnych zapytań/parametrów GraphQL.
Zawiera identyfikatory scen, nazwy etapów i dane ilościowe potrzebne do analizy.

## Benchmark offline

Do reprodukowalnego pomiaru syntetycznego:

```powershell
python -m benchmarks.benchmark_merge_pipeline --frames 400 --labels 28 --repeats 3
```

Benchmark nie łączy się z produkcyjnym Stash i nie zastępuje profilu rzeczywistej
biblioteki.

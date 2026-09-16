# Diagnostyka merge trwającego 3–7 sekund na scenę

Dodano pomiary pozwalające rozdzielić obliczenia, serializację lokalnego stanu
i oczekiwanie na Stash. Ta aktualizacja nie zmienia kwalifikacji tagów,
markerów ani formatu bazy. Nie wymaga nowych inferencji ani zmiany YAML.

## Co potwierdził test offline

Syntetyczna scena: 400 próbek, 28 etykiet, cztery modele, 1812 wkładów
źródłowych i 453 wynikowe markery. Test obejmuje obliczenia merge oraz
rzeczywisty zapis snapshotów w tymczasowym SQLite. Stash jest atrapą bez
opóźnienia sieci; wynik nie określa szybkości Twojego serwera ani dysku.

Mediana całego testu z trzech prób: **2,14 s**. Mediany poszczególnych etapów:

| Etap | Czas |
|---|---:|
| `marker_fusion`: wspólne dane markerów i dodatków | 0,734 s |
| `marker_union`: unia zakresów i łączenie dowodów | 0,318 s |
| `state_json_encode`: serializacja snapshotów | 0,545 s |
| `state_compress`: kompresja | 0,307 s |
| `marker_build`: budowa markerów źródłowych | 0,055 s |
| `state_cache`: obsługa cache | 0,041 s |
| `state_write`: pozostała obsługa zapisu po odjęciu etapów zagnieżdżonych | 0,003 s |
| `state_sql_read`: odczyt lokalnego SQLite | 0,0002 s |

Mediany etapów nie muszą sumować się do mediany całości. Snapshot markerów
miał **44 153 763 bajty JSON**, choć po kompresji zajmował **1 690 459 bajtów**.
Dowody wykryć i kontekst powtarzają się we wkładach wielu markerów. Współdzielenie
obiektu w RAM nie zapobiega wielokrotnemu zapisaniu go przez JSON. Niewielki
plik SQLite może więc ukrywać koszt przetwarzania dziesiątek megabajtów tekstu
dla jednej sceny. W tej próbie sam SQLite nie był głównym problemem.

Poprzednia wersja miała medianę 2,28 s. Różnicy nie należy traktować jako
przyspieszenia: to trzy próby w środowisku pomiarowym, a zmiana dodaje
diagnostykę. Snapshoty obu wersji były identyczne bajt w bajt.
Pomiary zapisano w `diagnostic_benchmark_results.json`.

Drugi kandydat to publikacja: kod wykonuje osobne, sekwencyjne zapytanie dla
każdego tworzonego, aktualizowanego i usuwanego markera. Przykładowo 100 zmian
po 40 ms daje 4 s. To ilustracja kosztu, nie pomiar Twojego Stasha.
Nowy raport rozróżnia dużą liczbę szybkich zapytań od kilku wolnych zapytań.

## Jak zebrać pomiar

Rozpakuj całą paczkę. Zachowaj używaną bazę `tagger_sources.sqlite3` oraz swoje
ustawienia połączenia, ścieżek i reguł. Wybierz kilka scen, na których merge
jest wolny; zastąp przykładowe identyfikatory prawdziwymi:

```powershell
python .\stash_ai.py --merge-only --scene-id 123 --scene-id 124 --scene-id 125 --force-merge --profile-merge merge-profile.jsonl
```

- `--profile-merge` zapisuje szczegóły do nowego pliku `.jsonl`. Istniejącego
  pliku nie nadpisuje; przy kolejnym pomiarze użyj np. `merge-profile-2.jsonl`.
- `--force-merge` omija pominięcie niezmienionej sceny przez cache publikacji.
  Wymaga `--merge-only`; korzysta z zapisanych wyników, bez uruchamiania modeli
  i bez czyszczenia bazy. Nadal porównuje wynik ze stanem Stash: nie wymusza
  zbędnych zapisów niezmienionych markerów.
- Polecenie wykonuje zwykłą synchronizację wybranych scen. `--dry-run` pozwala
  obejrzeć koszt odczytu i obliczeń, lecz nie mierzy rzeczywistych zapisów API
  ani trwałego zapisu snapshotów. Do pomiaru publikacji użyj zwykłego przebiegu.
- `--verbose` dodatkowo wypisuje etapy każdej sceny na konsolę. Plik zawiera
  szczegóły także bez tej opcji.

Podczas normalnego merge można dodać sam `--profile-merge merge-profile.jsonl`.
Raport pokaże również pominięcia `reused`. Aktualizacja kodu może jednorazowo
unieważnić dotychczasowe sprawdzenia publikacji; surowe wyniki modeli pozostają.
Nie zmieniaj liczby workerów ani RAM podczas pierwszego pomiaru — potrzebna
jest obserwacja zwykłego przebiegu.

## Co zawiera raport

JSONL zawiera rekord startowy, po jednym rekordzie na scenę z końcowego
przebiegu merge i podsumowanie. Rekord sceny zapisuje się i jest opróżniany
z bufora od razu; przy Ctrl+C zapisuje się też częściowy pomiar. Pamięć
diagnostyki nie rośnie wraz z kolekcją: zachowuje sumy i trzy najwolniejsze
przykłady. Błąd zapisu raportu jest zgłaszany i wyłącza dalszy zapis diagnostyki.

| Pole | Znaczenie |
|---|---|
| `wall_seconds` / `cpu_seconds` | Czas rzeczywisty / czas CPU procesu |
| `status`, `reason` | Wynik sceny oraz powód przeliczenia lub pominięcia |
| `stages` | Czasy i liczba wywołań etapów |
| `api` | Nazwy operacji GraphQL, liczba wywołań, czasy, błędy i rozmiary odpowiedzi |
| `counts` | Liczby wejść, wykryć, markerów, trafienia cache oraz bajty JSON i danych skompresowanych |
| `statistics` | Planowane zmiany markerów i tagów, niezmienione markery i wyniki reguł |
| `unmeasured_seconds` | Czas sceny poza zarejestrowanymi etapami |

`statistics` opisuje plan synchronizacji; przy błędzie API nie jest
potwierdzeniem, że każda planowana zmiana została zapisana. Czasy CPU
obejmują wątki procesu, dlatego przy pracy równoległej mogą przekraczać czas
rzeczywisty.

Każdy etap ma `seconds` bez zagnieżdżonych etapów, `inclusive_seconds` razem
z nimi, `calls`, `max_seconds` i `errors`. Maksimum obejmuje etap wraz z jego
dziećmi. Do szukania udziału etapów używaj `seconds`; nie sumuj czasów
`inclusive_seconds` rodzica i dzieci. W `api` czasy obejmują pełną operację;
nakładają się na odpowiednie pomiary w `stages`.

Najbardziej użyteczne etapy:

- `frame_index`, `marker_seeds`, `marker_build`, `marker_fusion`, `marker_union`,
  `scene_fusion`, `occurrence_compute`: przygotowanie i obliczenia wyników.
- `state_sql_read`, `state_decompress`, `state_json_decode`, `state_json_encode`,
  `state_compress`, `state_cache`: poszczególne koszty lokalnego stanu.
  `state_read` i `state_write` zawierają pozostałą obsługę, po odjęciu
  zagnieżdżonych pomiarów; nie są czystym czasem dysku lub SQL.
- `scene_refresh`, `tag_resolution`, `marker_matching`, `publication_fingerprint`:
  odświeżenie sceny, rozwiązywanie tagów, plan zmian i przygotowanie fingerprintu.
- `stash_http`: zapytania HTTP wraz z pracą serwera i odbiorem odpowiedzi.
  `stash_json_decode` dekoduje zewnętrzną odpowiedź GraphQL;
  `stash_snapshot_decode` obejmuje dalszą obsługę wyniku szybkiego odczytu SQL.
  Rozdzielenie czasu serwera na jego SQL i pozostałą pracę wymagałoby pomiarów
  po stronie Stasha.
- `publication_prepare`, `publication_mutations`, `publication_commit`: etapy
  publikacji. Koszt zapytań i serializacji widać w etapach zagnieżdżonych oraz
  w ich czasach `inclusive_seconds`.

Początkowe pobranie scen i bootstrap są mierzone globalnie, m.in. przez
`initial_scan`, `scan_fingerprint` i `bootstrap`. Nie są doliczane do czasu
końcowego merge pojedynczej sceny. Podsumowanie obejmuje także przygotowanie,
odczyty zbiorcze i końcowe porządki. Przy zwykłym przebiegu z modelami obejmuje
również inferencję; do izolowania kosztu merge użyj `--merge-only`.

Raport nie zapisuje klucza API, adresu serwera, zapytań, parametrów GraphQL,
tytułów, nazw tagów, ocen modeli ani obrazów. Zawiera identyfikatory scen
i liczby potrzebne do diagnozy. Szczegółowe logi `--verbose` są odrębnym wyjściem.

## Kolejność dalszych optymalizacji

1. **Usunąć powielanie dowodów w snapshotach.** Zapisywać wspólny kontekst
   i dowody raz, a przy markerach referencje. Zachować informacje potrzebne
   cleanupowi i podglądowi pochodzenia. To wymaga zgodnej migracji odczytu
   starszych snapshotów i testów cofania wkładu modeli.
2. **Ograniczyć powtarzaną budowę i łączenie tych samych dowodów.** Wykorzystać
   wspólne dane z etapu fusion w unii oraz serializować końcowy stan tylko raz.
3. **Jeśli dominuje `api`, ograniczyć liczbę zapytań.** Dobierać grupowanie
   do tworzenia, aktualizacji i usuwania markerów, z zachowaniem kontroli
   błędów i wznawiania. Samo równoległe wysyłanie wielu zapisów może zwiększyć
   kolejkę do bazy Stasha. Aliasowanie GraphQL zmniejsza liczbę połączeń,
   lecz nie gwarantuje jednej transakcji ani mniejszej pracy serwera.
4. **Jeśli często występuje nieoczekiwane przeliczenie, zbadać `reason`.**
   Skuteczne pominięcie merge jest szybsze od każdej jego optymalizacji.

Więcej RAM nie usuwa powtarzanej serializacji. Zwiększanie równoległości
obliczeń CPU warto rozważyć po zmniejszeniu rozmiaru danych i po pomiarze
rzeczywistego profilu. Modele nadal korzystają z GPU pojedynczo.

## Weryfikacja

450 testów przeszło offline. Testy obejmują także zagnieżdżone pomiary,
błędy API, przerwanie, ograniczenie pamięci raportu, awarię jego zapisu,
dry-run, cache publikacji i niezmienność wyników po wymuszeniu merge.
Szczegóły: `validation_diagnostics.txt`.

```powershell
python -m pytest -q
python .\benchmark_merge_pipeline.py --frames 400 --labels 28 --repeats 3
```

Benchmark wymaga zależności testowych, korzysta z tymczasowej bazy i atrapy
Stasha. Nie łączy się z Twoim serwerem i nie uruchamia modeli.

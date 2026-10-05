# Stash AI

Automatyczne tagowanie scen i tworzenie markerów czasowych dla [Stash](https://github.com/stashapp/stash).

Stash AI używa czterech modeli: JoyTag (`jt`), WD EVA02-Large v3 (`wd`),
DanbooruTagQuery L/16 (`dtq`) i OppaiOracle V1.1 (`op`). Modele są
uruchamiane kolejno — w GPU znajduje się jeden model naraz.

Domyślny tryb `merged` publikuje wspólne tagi `ai:`, łączy zgodne wyniki modeli
i zachowuje osobne źródła oraz pewności w lokalnej bazie SQLite. Wejście wizualne
pochodzi wyłącznie z VTT i sprite'ów Stash; aplikacja nie otwiera plików wideo.

## Wymagania

- Python **3.12.14 lub nowszy** — target projektu jest zapisany w `.python-version`;
- Stash **0.31.1 lub nowszy** z wygenerowanymi sprite'ami/VTT;
- zależności z `requirements.txt`;
- CUDA jest zalecana; CPU działa, ale jest znacznie wolniejsze.

Wersja Stash jest sprawdzana przed odczytem scen i zmianami na serwerze, także
w `--dry-run`. Komendy lokalne, np. `--db-update`, `--db-check` i `--db-report`,
nie wymagają połączenia ze Stash. Znane przedziały zawsze publikują koniec;
occurrence i importowane markery z nieznanym końcem mają `end_seconds: null`.

Dla CPU zamień `onnxruntime-gpu` na zgodny `onnxruntime>=1.21,<2`.

## Szybki start

```powershell
pip install -r requirements.txt

$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<własny klucz ze Stash>"

python .\stash_ai.py --download-models --models all
python .\stash_ai.py --models all --scene-id 123 --dry-run --verbose
python .\stash_ai.py --models all --scene-id 123
```

Ustaw `STASH_API_KEY` jawnie w środowisku, jeśli Stash wymaga klucza.
Bez tej zmiennej klient nie wysyła nagłówka `ApiKey`. Historyczny klucz
został usunięty z bieżącego kodu; informacje o rotacji są w [SECURITY.md](SECURITY.md).

W `tagger_models.yml` każdy wybrany model musi mieć jawne `scene_threshold`,
`marker_threshold` i `batch_size`. Wybór modeli pochodzi z `--models` lub
wymaganego wtedy `enabled_models`; brakujące wartości nie korzystają ze starych
domyślnych ustawień JoyTag. Pola `scene_cache_floor` i `marker_cache_floor`
pozostają opcjonalne. Dopasowany cache jest używany przez resume; sama obecność
tagów lub markerów nie pomija nowej inferencji.

## Najczęstsze tryby

```powershell
# Zwykła inferencja; poprawny cache i checkpointy są używane automatycznie.
python .\stash_ai.py --models all

# Świeża inferencja tylko dla wybranego zakresu.
python .\stash_ai.py --models wd --scene-id 123 --restart

# Ponowne scalenie zapisanych wyników bez uruchamiania modeli.
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Równoległe read-only przygotowanie evidence; publikacja do Stash pozostaje sekwencyjna.
python .\stash_ai.py --merge-only --merge-workers 4
# --dry-run i --profile-merge celowo używają ścieżki sekwencyjnej.

# Zmiana progów z zachowanego cache, bez modeli.
python .\stash_ai.py --models all --merge-only --rethreshold-markers
python .\stash_ai.py --models all --merge-only --rethreshold-scenes --rethreshold-markers

# Cleanup wykluczeń.
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only

# Naprawa grup istniejących markerów.
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run

# Lokalny podgląd źródeł tagu.
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123

# Usuwanie nieużywanych definicji tagów.
python .\stash_ai.py --prune-empty-tags-only --dry-run

# Jednorazowe przejście starych wyników do ai: ze świeżą inferencją.
python -m stash_ai.tools.namespace_migration --models all --dry-run
python -m stash_ai.tools.namespace_migration --models all

# Zdrowie, aktualizacja schematu, spójność i kompaktowanie lokalnej bazy stanu.
python .\stash_ai.py --db-report
python .\stash_ai.py --db-check
python .\stash_ai.py --db-update --dry-run
python .\stash_ai.py --db-update
python .\stash_ai.py --db-compact --dry-run
python .\stash_ai.py --db-compact

# Lokalny, anonimowy raport diagnostyczny do ręcznego przekazania developerowi.
python .\stash_ai.py --anonymous-diagnostics .\stash-ai-diagnostics.json

# Użyteczność modeli i leave-one-model-out ablation bez inferencji.
# Domyślnie używa do 4 procesów CPU; --model-report-workers 1 wymusza tryb sekwencyjny.
python .\stash_ai.py --model-report .\model-utility.json
python .\stash_ai.py --model-report .\model-utility.json --model-report-workers 8
python .\stash_ai.py --model-report .\model-utility-scene-123.json --scene-id 123

# Tagi wymagające ręcznego przeglądu: powody, przykłady i liczniki per model.
python .\stash_ai.py --tag-review-report .\tag-review.json
python .\stash_ai.py --tag-review-report .\tag-review-scene-123.json --scene-id 123

# Kompletność cache względem YAML oraz plan uzupełnienia brakujących par scena–model.
python .\stash_ai.py --cache-report .\cache-coverage.json
python .\stash_ai.py --cache-report .\cache-coverage-all.json --models all
python .\stash_ai.py --cache-report .\cache-coverage-scene-123.json --models wd --scene-id 123

# Porzucenie niedokończonego planu merge/cleanup.
python .\stash_ai.py --abandon-pending
```

Zmiana progów zapisuje małe wiersze w tabeli `result_thresholds`; skompresowane
wyniki inferencji i evidence klatek pozostają bez zmian. Dotyczy to także
automatycznego użycia cache po zmianie progów w YAML. Odczyt wyników, merge,
cleanup i explain używają efektywnych progów z tej tabeli. Każda para model–wyjście ma jeden aktywny wynik i opcjonalny liczbowy próg w `result_thresholds.threshold`.
`--db-update` przenosi dawne inline overrides i usuwa wycofane `refinement`;
klatki, wyniki i podpisy dowodów zachowują swoje wartości.
`--dry-run` zapisuje progi wyłącznie w tymczasowej bazie.

Snapshoty merge zapisują współdzielone dowody `extra_support` raz, a contributions
odwołują się do nich przez referencje. Zmniejsza to koszt serializacji i kompresji.
Aktualny schemat to v20; starsze wersje programu odmówią jego otwarcia. Każdy nowy snapshot ma otoczkę formatu v5. `--db-update` konwertuje dawne
snapshoty, także pending i recovery, dopisuje wymagane `contributions: []` do pustych zapisów oraz uzupełnia kompletne opcje odtwarzania, również w zagnieżdżonych dowodach scen.
Dawny `blocklist` zostaje połączony z zapisanymi `tag_exclusions` wszystkich źródeł.
Brakujące dawne przełączniki współdzielenia i occurrences zachowują `False`.
Wersjonowanie analizy checkpointu dotyczy wyłącznie niedokończonej inferencji:
stare checkpointy są pomijane przy wznowieniu i dany model zaczyna scenę od początku.
Ukończone wyniki formatu 4 zachowują dowody i sygnatury; starsze aktywne wyniki wymagają ponownej inferencji. `--dry-run` zmienia wyłącznie tymczasową bazę.

Pełną listę opcji pokazuje:

```powershell
python .\stash_ai.py --help
```

## Jak działa stan i cache

Domyślna baza `tagger_sources.sqlite3` przechowuje surowe dowody modeli,
checkpointy, opublikowane snapshoty i informacje potrzebne do bezpiecznego
wznowienia. **Zachowuj ją w backupie.** Nie zawiera obrazów ani klucza API.

W trybie `merged` domyślne `resume`:
- pomija ukończoną inferencję, gdy model i wejście są zgodne;
- wznawia częściowy checkpoint;
- może przeliczyć merge bez GPU po zmianie reguł publikacji;
- przy zmianie progów wykorzystuje zachowane obserwacje, jeśli ich retencja jest
  wystarczająca.

`--restart` wymusza świeżą inferencję dla wybranych modeli/scen. Nie używaj go
do zwykłego przeliczenia merge, cleanupu ani zmiany progów.

### Retencja niezależna od progów publikacji

Opcjonalne `scene_cache_floor` i `marker_cache_floor` w sekcji każdego modelu
w `tagger_models.yml` zachowują słabsze wyniki do późniejszego przeliczenia
progów bez GPU. Nie obniżają progów publikacji tagów ani markerów:

```yaml
wd:
  scene_threshold: 0.80
  marker_threshold: 0.85
  scene_cache_floor: 0.20
  marker_cache_floor: 0.25
```

Wartości muszą być liczbami od 0 do 1. Pominięcie pola albo `null` zachowuje
dotychczasową automatyczną retencję. Faktyczny próg zapisu jest minimum ustawionej
wartości i progu wsparcia wymaganego przez bieżącą politykę publikacji, więc
ustawienie wyższej wartości nie usuwa potrzebnych obserwacji. Cache scen zachowuje
także dokładne agregaty dla słabszych etykiet.

Po świeżej inferencji można obniżyć progi publikacji w YAML i użyć
`--merge-only --rethreshold-scenes --rethreshold-markers`, jeśli zapisane progi
retencji obejmują nowe wymagane progi wsparcia. Zmiana samej retencji podczas
zwykłego przebiegu uruchamia świeżą inferencję i unieważnia częściowy checkpoint;
`--merge-only` nie uzupełnia wcześniej odrzuconych wyników. Niższa retencja zwiększa
rozmiar bazy i koszt przetwarzania cache. Ustawienia dotyczą trwałego cache inferencji.

Aktywny cache wymaga jawnych progów retencji i pełnego formatu analizy 4.
`--db-update` wycofuje starszą inferencję; uruchom ją ponownie przed merge i cleanup.
Nadpisanie progu publikacji nie uzupełnia wcześniej odrzuconych obserwacji.

Fazy `Check` i `Apply` rethreshold korzystają z małych rekordów `result_metadata` w SQLite,
zamiast dekodować klatki z `results.data`. Nowe wyniki zapisują je atomowo razem
z evidence; starsze wyniki uzupełniają metadane przy pierwszym sprawdzeniu.
Dotyczy to także historii wyników i starych nadpisań progów. Zmiana lub usunięcie
surowego wyniku unieważnia jego metadane przez trigger, również przy zapisie
starszą wersją programu. `Apply` zapisuje progi w `result_thresholds` i aktualizuje
`result_index` z metadanych: obecności wyników oraz sygnatury ostatniej inferencji
w historii. Przy gotowych metadanych żadna z obu faz nie dekoduje BLOB-ów klatek.
Dry-run uzupełnia wyłącznie tymczasową bazę; `--db-check`
sprawdza zgodność istniejących metadanych z evidence bez ich naprawiania.

Nowa inferencja zawsze zastępuje poprzednie wyniki danego modelu i rodzaju.
Historyczne serie są normalizowane przez `--db-update`; stare importowane
wyniki są wycofywane, a ich modele wymagają ponownej inferencji. Zwykłe przedziały opublikowane dawniej jako
punkty wymagają jednorazowego `python -m stash_ai.tools.point_migration` (także z `--dry-run`).
Komenda zapisuje długość ze snapshotów published, pending i recovery w natywnym `end_seconds`,
zachowując ID, tytuły i ręczne tagi. Punkty `OccurrenceTag` pozostają punktami.
Zwykły merge/cleanup wymaga natywnych zakresów i nie odtwarza długości w pamięci.
Oba migratory są osobnymi modułami w `stash_ai.tools`; dawne flagi głównego CLI
`--migrate-legacy-only` i `--migrate-point-markers-only` zostały usunięte.

### Jednorazowe przejście starych prefiksów do ai:

`python -m stash_ai.tools.namespace_migration --models all` wykonuje świeżą inferencję dla scen ze starymi
wynikami `jt:`, `wd:`, `dtq:` i `op:` oraz zastępuje zarządzane obiekty wynikami `ai:`.
Użyj dotychczasowej bazy `--state-db`, po wcześniejszym `--db-update`.
`--scene-id`, `--markers-only` i `--scene-tags-only` ograniczają zakres;
`--dry-run` wykonuje inferencję i pokazuje plan bez zmiany Stash lub bazy źródłowej.

Ręczne markery bez tagu ownership, ręczne tagi sceny i dodatki markerów są zachowywane.
Pasujące zarządzane markery zachowują ID; pozostałe stare wyniki są usuwane dopiero
po przygotowaniu nowej publikacji. Snapshot pending umożliwia ponowienie po błędzie API.
Sceny, które zakończyły przejście, są pomijane przy ponownym uruchomieniu tej komendy.
Jeśli CREATE_MISSING jest wyłączone, stary obiekt bez dostępnego tagu zastępczego pozostaje.

Zwykły merge obsługuje wyłącznie bieżący namespace `ai:`. Przed jego uruchomieniem
przenieś stare zarządzane prefiksy osobnym narzędziem; zwykły przebieg ich nie konwertuje.
Import historycznych ocen oraz jego trwały rejestr nie są już używane. Schemat v17
usuwa `legacy_imports` i importowane aktywne wyniki, zlecając świeżą inferencję.
Globalne definicje starych tagów pozostają dostępne dla ręcznych powiązań.

### Zdrowie, aktualizacja i kompaktowanie bazy stanu

`--db-update` aktualizuje całą istniejącą bazę wskazaną przez `--state-db`
do najnowszego obsługiwanego schematu (obecnie v20), we wszystkich scope'ach.
Zwykłe uruchomienie oraz `--dry-run` wymagają aktualnego schematu: starszą bazę
trzeba najpierw zaktualizować tą komendą. Nowa baza od razu otrzymuje v20.
Updater korzysta z tego samego locka writera; migracje są wyłącznie offline.
Tworzy aktualne tabele, indeksy i triggery, uzupełnia `result_index` dla baz
sprzed v2, uzupełnia historyczne `result_metadata` we wszystkich scope'ach
oraz usuwa wycofaną tabelę `refinement_progress`.
Nie łączy się ze Stash ani nie ładuje konfiguracji modeli. Zachowuje wyniki, checkpointy, pending publication/cleanup i recovery oraz
efektywny próg bieżącej inferencji. Wynik aktywny jest pojedynczym obiektem,
a polityka pojedynczą liczbą; dawne tablice odczytuje wyłącznie migracja offline.
Nie odtwarza utraconych detekcji: starsze niepełne wyniki wymagają ponownej inferencji.
Dowody przypięte przez publikacje pozostają w archiwum do czasu odświeżenia publikacji.

`--dry-run` wykonuje odczytowy preflight i pokazuje wersję źródłową oraz
docelową; nie uruchamia migracji ani nie tworzy locka. Polecenie odrzuca brakującą,
obcą lub nowszą niż obsługiwana bazę. Migracje są powtarzalne: mogą zatwierdzić
etapy pośrednie, ale najnowsza wersja jest ustawiana dopiero po ukończeniu
inicjalizacji. Przerwaną aktualizację można uruchomić ponownie.

`--db-check` wykonuje głęboki, wyłącznie odczytowy test spójności całej bazy
(we wszystkich scope'ach). Dla starszego schematu wykonuje tylko preflight wersji i integralności SQLite,
a przed głęboką walidacją wymaga `--db-update`. Oprócz pełnego SQLite `integrity_check` sprawdza m.in.
zgodność `results` z `result_index` (łącznie z zapisaną sygnaturą ostatniej
inferencji), dekodowalność trwałych BLOB-ów, dozwolone model/kind, relacje
checkpoint/frames/analysis, wymiary checkpointów oraz generacje cleanup/recovery.
Waliduje też zawartość wyników scen, klatek i zakresów markerów: wymagane pola,
tagi, czasy, prawdopodobieństwa oraz zapisane statystyki czasowe. Sprawdza klatki
checkpointów (również `scene_tags`) i wartości float32: odrzuca `NaN`,
nieskończoności oraz wyniki poza `[0, 1]`. Zachowuje obsługę starszych wyników bez
statystyk i metadanych retencji, a także importowanych markerów punktowych i
brakujących historycznych ocen (`score: null`).
Osierocone lub historyczne rekordy, które nie muszą oznaczać uszkodzenia, są
raportowane jako warnings. Pending publication/cleanup i refresh requests są
pokazywane jako stan roboczy, a nie automatycznie jako błędy.

Komenda nie łączy się ze Stash, nie ładuje modeli i niczego nie naprawia ani nie
modyfikuje. Podczas głębokiego skanu pokazuje pasek postępu z aktualną fazą
(np. SQLite integrity, results/index, state payloads, checkpoints). Zwraca exit
code `0`, gdy nie znaleziono błędów logicznych, oraz `1`, gdy wykryto
niespójność; same warnings nie powodują błędu.

`--db-report` nie łączy się ze Stash ani nie ładuje modeli. Pokazuje wersję
schematu, wynik SQLite `quick_check`, rozmiar pliku i sidecarów WAL/SHM, wolne
strony, zagregowane rekordy `results` według modelu/rodzaju oraz liczniki
checkpointów, refresh requests, pending/recovery i merge checks dla bieżącego
scope. Nie wypisuje nazw tagów, ID scen ani zapisanych score'ów.

`--db-compact` wykonuje SQLite `VACUUM` pod tym samym lockiem co zwykły writer.
Przed operacją ponownie sprawdza integralność, dostępne miejsce na dysku oraz brak
niedokończonego merge/cleanup. `--dry-run` wykonuje tylko preflight i pokazuje
szacunkowy odzysk z wolnych stron. Recovery evidence i checkpointy są zachowywane.

### Anonimowy raport dla developera

`--anonymous-diagnostics NEW.json` tworzy lokalny raport przeznaczony do
ręcznego przejrzenia i ewentualnego przekazania developerowi. Aplikacja **nigdy
nie wysyła go automatycznie** i odmawia nadpisania istniejącego pliku.

Raport obejmuje całą wskazaną bazę stanu, ale nie eksportuje scene IDs, wartości
`scope`, URL-i, ścieżek plików, nazw tagów, score'ów, timestampów markerów,
sygnatur ani hashy wejścia. Nieznane nazwy modeli/rodzajów są zastępowane
`other`, a niestandardowe tabele są raportowane tylko jako liczba.

Dla optymalizacji schematu raport zawiera m.in.:
- wersje Pythona/SQLite i rodzinę systemu operacyjnego;
- page size/count, freelist, rozmiar DB/WAL oraz liczbę scope'ów i scen;
- liczbę rekordów znanych tabel i rozkłady rozmiarów BLOB-ów;
- per model/rodzaj: percentyle rozmiaru compressed JSON, compression ratio,
  items/rekord, detekcje/klatkę i scene observations;
- zagregowany stan checkpointów, result index i refresh requests.

Raport jest pozbawiony bezpośrednich identyfikatorów i treści, ale dokładne
agregaty rozmiaru/liczebności nadal opisują konkretną instalację. Przed
udostępnieniem warto przejrzeć wygenerowany JSON.

### Model Utility / Ablation Report

`--model-report NEW.json` działa wyłącznie na zapisanym evidence i nie łączy się
ze Stash ani nie uruchamia modeli. Dla każdej sceny najpierw odtwarza normalny
wynik wszystkich zapisanych modeli, a następnie cztery kontrfaktyczne wyniki
`all minus one model`. Oznacza to, że po usunięciu modelu od początku
przeliczane są consensus seeds, temporal rules, shared scene tags, marker extras
i Occurrence — raport nie filtruje jedynie gotowych markerów.

Per model raport pokazuje m.in.:
- liczbę scen z zapisanym evidence i scen, w których pominięcie modelu zmienia wynik;
- finalne scene tags i marker components wspierane przez model;
- scene tags, markery, extras i Occurrence utracone po jego usunięciu;
- markery, które pozostają, ale zmieniają zakres lub extras;
- procent scen z evidence, w których model ma mierzalny marginalny wpływ.

Zmiana samego badge'a źródeł markera, np. `[jt+wd]` na `[wd]`, nie jest liczona
jako zmiana semantyczna. Dzięki temu raport mierzy wpływ na tagowanie, zakresy i
extras, a nie samą obecność provenance. `--scene-id` może ograniczyć analizę
do wybranych scen.

### Raport tagów wymagających przeglądu

`--tag-review-report NEW.json` analizuje zapisane tagi scen i główne etykiety
markerów, odtwarzając merge z efektywnymi progami zapisanymi w bazie. Używa
wybranej polityki temporalnej i wykluczeń. Nie ładuje modeli, nie pobiera mediów,
nie łączy się ze Stash i nie zmienia bazy źródłowej ani publikacji.

Raport wskazuje priorytety ręcznego przeglądu:

- `weak_or_borderline`: najlepszy wynik nie przekracza właściwego progu o więcej
  niż `--tag-review-margin` (domyślnie `0.05`). Dla scen zakwalifikowanych przez
  `strong_peak` używany jest próg silnego piku; pozostałe wyniki mają własny próg
  sceny lub markera. Słabe wyniki zaakceptowane przez reguły czasowe/konsensus też
  mogą zostać wskazane;
- `isolated_peak`: tag sceny kwalifikuje się przez silny pik, ale tylko jedna
  próbka przekracza próg sceny, przy co najmniej dwóch próbkach;
- `model_disagreement`: inny model zachował tę samą etykietę na identycznym
  wejściu, lecz nie kwalifikuje jej według własnych reguł sceny. Brak modelu lub
  etykiety w cache nie jest głosem negatywnym; wykluczone etykiety nie służą do
  porównań;
- `long_marker`: po scaleniu zakres ma co najmniej
  `--tag-review-long-marker-seconds` (domyślnie `300` sekund).

Tabela w terminalu i sekcja `models` JSON pokazują dla JoyTag, WD, DTQ i Oppai
liczbę różnych etykiet do przeglądu, liczbę par scena–tag oraz liczbę scen.
Para scena–tag per model jest liczona raz, również gdy ten tag ma kilka zakresów,
występuje jako tag sceny i core markera lub pochodzi z kilku zapisanych przebiegów.
Globalne liczniki dodatkowo deduplikują modele; sumowanie wierszy modeli nie daje
liczby różnych tagów w całej kolekcji. Liczniki powodów mogą się nakładać.
Model wspierający core markera głosem konsensusowym również jest uwzględniony,
nawet jeśli nie utworzył własnego zakresu. Jego wynik pochodzi ze zgodnych,
zachowanych obserwacji w zakresie wkładu, a przykład ma `support_only: true`.

Sekcja `tags` zawiera wiersze model–etykieta, liczniki powodów i do trzech
przykładowych scen z wynikami, progami i czasami. Limit przykładów zmienia
`--tag-review-examples 1..20`; liczniki obejmują wszystkie przeanalizowane sceny.
Wiersze są sortowane według liczby wskazanych scen, następnie liczby powodów.
`--scene-id` może ograniczyć analizę; istniejący plik raportu nie jest nadpisywany.

Sceny z pending refresh/publication lub nieużywalnym evidence są pomijane
i raportowane osobno z powodami, zamiast uczestniczyć w porównaniu modeli.
Raport opisuje wynik replay cache, a nie sprawdzony na żywo stan Stash.
Heurystyki nie oznaczają potwierdzonych błędów: poprawny tag może mieć tylko jeden
pik, długi zakres albo wsparcie jednego modelu. Raport nie uśrednia ocen modeli.

### Raport kompletności cache i plan uzupełnienia

`--cache-report NEW.json` sprawdza oba rodzaje evidence (sceny i markery) dla
modeli włączonych w YAML lub wybranych przez `--models`. Działa offline, bez
modeli, słowników, mediów i połączenia ze Stash. Nie zmienia źródłowej bazy
ani publikacji i nie nadpisuje istniejącego raportu.

Domyślny zakres obejmuje sceny znane **bieżącemu scope lokalnej bazy**, również
te obecne tylko w checkpointach, refresh requests, snapshotach lub recovery.
Sceny ze Stash, które nigdy nie trafiły do tej bazy, nie wchodzą do mianownika.
Powtarzalne `--scene-id` ustala dokładny zakres; ID bez żadnego lokalnego stanu
jest wtedy raportowane jako brakujące. Raport nie sprawdza aktualności wag,
słownika, VTT ani sprite'ów — ich zgodność potwierdza zwykły przebieg inferencji.

Tabela terminala i `models` w JSON pokazują per model liczbę par scena–model:
gotowych, możliwych do przeliczenia bez GPU, wymagających inferencji,
wymagających wyłącznie rethreshold oraz zablokowanych. `outputs` rozbija
liczniki osobno na scene/markers i rozłączne statusy:

| Status | Znaczenie |
|---|---|
| `ready` | Ukończona inferencja, zgodny indeks, efektywne progi i wystarczająca retencja |
| `rethreshold` | Cache obejmuje progi YAML; wystarczy przeliczenie bez GPU |
| `missing` | Brak wyniku danego rodzaju |
| `partial` | Brak wyniku, lecz istnieje checkpoint do weryfikacji przy wznowieniu |
| `refresh_required` | Oczekujące żądanie świeżej inferencji unieważnia stary wynik |
| `insufficient` | Brak evidence klatek, za wysoki floor lub wcześniej odfiltrowane odblokowane etykiety |
| `inconsistent` | Wynik i indeks są niespójne albo evidence jest nieczytelne; sprawdź `--db-check` |

Ukończony wynik bez żadnych detekcji jest poprawnym cache, a nie brakiem.
Retencja jest sprawdzana względem minimum floor z YAML i progu wsparcia
potrzebnego do publikacji. Sprawdzany jest jeden aktywny wynik każdego modelu i wyjścia.
Przy gotowych `result_metadata` raport nie dekoduje BLOB-ów predykcji; starsze
metadane uzupełnia wyłącznie w tymczasowej bazie. Głęboka kontrola payloadów
pozostaje zadaniem `--db-check`.

`issues` zawiera konkretne pary scena–model z powodami, wymaganymi progami,
blokadami oraz zapisanym postępem checkpointu. `plan` zawiera uporządkowane,
pogrupowane **tablice argumentów `argv`**, po maksymalnie 50 scen na polecenie:

- `resume_or_infer` uzupełnia tylko wskazane modele/sceny; zwykłe resume sprawdza
  zgodność checkpointu i wykorzystuje poprawne ukończone wyniki;
- `restart_inference` wymusza wymianę nieużywalnego ukończonego cache. Aktywne
  braki/refresh z checkpointem dostają zwykłe resume, aby nie kasować postępu;
- `rethreshold` stosuje oba progi z YAML bez inferencji;
- `merge_after_backfill` na końcu odświeża konsensus modeli i Occurrence.

Polecenia zachowują ścieżki `--state-db`, YAML modeli, polityki temporalnej
i wykluczeń. Raport **nie wykonuje planu**. Ręczne uruchomienie tych poleceń
może zmienić publikację w Stash; używaj tego samego `STASH_URL` i ustawień
środowiska. Przy wielu oczekujących refreshach publikacja może pozostać
zablokowana do ukończenia wszystkich wskazanych modeli.

Pending publication, globalny cleanup, refresh niewybranego modelu,
nieczytelne evidence lub wyłączone zastępowanie nieużywalnej historii blokują
wszystkie polecenia planu dla dotkniętej sceny, również uzupełnienie innych
modeli i końcowy merge. Każde z tych poleceń może próbować publikacji.
Najpierw dokończ pierwotny plan
albo wyjaśnij niespójność, a następnie wygeneruj raport ponownie. Gotowy evidence
może nadal mieć blokadę publikacji; liczniki tych dwóch cech mogą się nakładać.

Merge używa także zachowanego evidence niewybranych modeli. Raport sprawdza
jego możliwość publikacji według zapisanych efektywnych progów i wybranej
polityki, bez wymagania retencji z YAML dla tych modeli. Niepełny lub nieczytelny
cache takiego modelu blokuje scenę; `issues[].publication_dependencies` wskazuje
model, rodzaj wyniku, powód i sposób rozwiązania. Dołącz wskazany model przez
`--models` i wygeneruj raport ponownie, aby zaplanować jego naprawę. Liczniki
kompletności i cele inferencji nadal dotyczą wybranych modeli; raport nie
rozszerza automatycznie zakresu inferencji. `issues[].blocking_models` wskazuje
wybrane modele, których blokada uniemożliwia zaplanowanie pozostałych par sceny.

Exit code `0` oznacza poprawnie wygenerowany raport, również gdy zawiera braki.
Archiwalne evidence po `--abandon-pending` nie blokuje cache ani inferencji:
`publication_recovery_scene_ids` pokazuje te sceny, a końcowy merge planu
rekonsyliuje ich publikację, również gdy cache jest już kompletny, jeśli scena
nie ma innych blokad publikacji.

## Wynik w Stash

W trybie `merged`:

| Element | Przykład |
|---|---|
| Tag sceny / core markera | `ai:blue_hair` |
| Tytuł markera | `[AI] blue_hair [jt+wd+op]` |
| Tag własności | `ai:marker:stash-taggers` |
| Punkt Occurrence | dodatkowo `ai:marker:occurrence` |
| Grupa zwykłego markera | `m:blue_hair` |
| Grupa Occurrence | `m:blue_hair:occurance` |
| Korzeń grup markerów | `ai_markers` |
| Kategorie hierarchii | `ai:category:...` |

Pisownia `:occurance` w nazwach grup jest celowa i pozostaje zgodna z istniejącym
formatem.

Nowe wyniki zawsze używają wspólnej publikacji `ai:` i reguł temporalnych.
Przełączniki `--output-mode` i `--no-temporal` oraz klucz YAML `enabled`
zostały usunięte; usuń je ze skryptów i `temporal_policy.yml`. Poszczególne
reguły nadal konfiguruje polityka YAML, w tym `occurrence_markers`.
Historyczne namespace’y modeli przenosi osobny migrator; zwykłe publikowanie,
cleanup i domyślne pruning obsługują bieżące wyniki `ai:` oraz zarządzane grupy
`m:`. Definicje dawnych prefiksów można usuwać jawnie, np.
`--prune-empty-tags-only --prune-prefixes jt: wd: dtq: op:`. Ręczne dodatki
z innych namespace’ów pozostają zachowane.

Nowa inferencja zapisuje kompletny format analizy 5: sygnatury wejścia i
inferencji, jawne progi retencji oraz statystyki lub jawne `null` dla
niekwalifikujących się agregatów. `--db-update` wycofuje starsze aktywne wyniki,
które wymagają ponownej inferencji. Przypięte archiwum zachowuje dokładne dowody
publikacji, a geometria i ownership dawnych snapshotów pozostają dostępne dla
recovery i migracji punktów. Cleanup wymaga przebudowy wycofanych publikacji
przez świeżą inferencję i `--merge-only`.

Nowy fingerprint cleanupu zawsze zawiera zakres wyjść i nie zawiera stałego
pustego blocklist. Przerwany cleanup zachowuje generację i maski: adapter
historycznego podpisu pozwala dokończyć go wyłącznie z tym samym zakresem wyjść
i polityką wykluczeń. Nowe operacje zapisują tylko bieżący podpis.

## Konfiguracja

Hierarchie mają `version: 2` i pola `source` oraz `groups`. Ścieżki grup tworzą
bezpośrednio kategorie `ai:category:` pod korzeniem `ai:category:AI`.
Własne YAML-e v1 przekonwertuj lokalnie, bez ładowania modeli ani połączenia ze Stash:

```powershell
python -m stash_ai.commands.hierarchy_migration joytag_hierarchy.yml wd_hierarchy.yml dtq_hierarchy.yml oppai_hierarchy.yml
```

Dodaj `--dry-run`, aby sprawdzić migrację. Oryginały są zachowywane jako
`*.v1.yml`; migrator zachowuje przypisania grup, surowe etykiety i metadane źródła.

| Plik / opcja | Rola |
|---|---|
| `tagger_models.yml` | modele, katalogi, progi, batch size |
| `temporal_policy.yml` | reguły temporalne, consensus, extras i Occurrence |
| `scene_tag_exclusions.yml` | globalne wykluczenia tagów |
| `marker_core_exclusions.yml` | wykluczenia głównych tagów markerów |
| `*_hierarchy.yml` | hierarchie kategorii modeli |
| `--state-db` | lokalizacja trwałej bazy stanu |

Aktualne domyślne limity procesu:
- `--ram-budget-gb 32`;
- `--ram-cache-gb 24` — cache jest przydzielany na żądanie i ograniczony do
  maksymalnie 75% budżetu;
- `--prepare-workers 4`;
- `--merge-workers 4` — równoległy CPU replay evidence; zapisy Stash/SQLite pozostają sekwencyjne;
- `--prefetch-batches 4`.

## Aktualizacje

`update_stash_ai.py` aktualizuje zarządzane pliki aplikacji bez Gita i bez
uruchamiania `pip`. Wymaga tokena GitHub z dostępem `Contents: read` do
prywatnego repozytorium.

Przy pierwszym przejściu ze starego układu (moduły `stash_*.py` obok launchera)
uruchom starą wersję updatera raz, a potem nową wersję z jawną migracją:

```powershell
python .\update_stash_ai.py --migrate-flat-layout --dry-run
python .\update_stash_ai.py --migrate-flat-layout
```

Migrator usuwa wyłącznie znane dawne moduły i wycofane ścieżki, zachowując backupy.
Zmodyfikowany kod nadal wymaga `--force-code`. Kolejne zwykłe aktualizacje sprawdzają
kod zarządzany w zapisanej historii; nie skanują już nieśledzonych plików starego układu.
Przy `git pull` lub użyciu najnowszego updatera od początku nie ma etapu starego updatera.

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"

python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run
python .\update_stash_ai.py
```

Updater:
- pobiera pliki z jednego SHA `main`;
- instaluje launchery, pakiet `stash_ai/` oraz YAML-e; usuwa zastąpione moduły z katalogu głównego;
- zachowuje lokalnie zmienione YAML-e i tworzy propozycje `.incoming`;
- odmawia nadpisania zmienionego kodu bez `--force-code`;
- nie aktualizuje modeli, `.env`, SQLite ani zależności Pythona;
- używa blokady instalacji i dziennika odzyskiwania.

Po przerwaniu aktualizacji użyj:

```powershell
python .\update_stash_ai.py --recover
```

## Bezpieczeństwo

- używaj własnego, zrotowanego `STASH_API_KEY` z environment;
- nie commituj tokenów GitHub ani kluczy Stash;
- read-only GraphQL i media GET mogą być ponawiane po błędach przejściowych;
- mutacje GraphQL **nie są automatycznie retry'owane**, ponieważ serwer mógł
  wykonać zapis mimo utraconej odpowiedzi;
- `--dry-run` nie zapisuje do Stash ani trwałej bazy źródłowej.

## Testy i CI

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
```

CI testuje target z `.python-version` (obecnie **3.12.14**) oraz **3.14.7**.
Pełny zestaw testów, w tym integracyjny smoke test JoyTag na CPU, działa na Linux,
a updater jest dodatkowo sprawdzany na Windows na obu wersjach Pythona.
Testy nie wymagają produkcyjnego serwera Stash.

Osobny workflow `Codex Python availability` sprawdza na PR-ach, po zmianach
`main` i raz w tygodniu, czy dokładny target projektu jest już zainstalowany
w najnowszym publicznym obrazie `ghcr.io/openai/codex-universal:latest`.
Nie uruchamia instalacji Pythona ani setupu obrazu; brak tej wersji kończy job
błędem i pokazuje dostępne interpretery. Wersja na domyślnym `PATH` jest wypisywana
osobno — dostępność targetu nie oznacza, że jest on wybranym interpreterem.

Publiczny obraz jest [według OpenAI](https://github.com/openai/codex-universal)
tylko przybliżeniem środowiska Codex, a nie
identycznym obrazem ChatGPT Work. Ten job monitoruje obraz referencyjny, nie
gwarantuje wersji w konkretnym czacie. W rzeczywistym workspace można uruchomić
`bash .github/scripts/check-codex-python.sh`; gdy dostępny jest
`CODEX_PRIMARY_RUNTIME_PYTHON`, skrypt sprawdza dostarczony interpreter zamiast
wersji zainstalowanych przez użytkownika.

## Dokumentacja

- [Aktualny podręcznik](docs/README_multimodel.md)
- [Indeks dokumentacji](docs/README.md)
- [Cleanup SQL](docs/UPGRADE_cleanup_sql.md)
- [Wydajność i pamięć](docs/UPGRADE_performance.md)
- [Merge i szybka ścieżka](docs/UPGRADE_merge.md)
- [Wykluczenia i cleanup](docs/UPGRADE_exclusions.md)
- [Occurrence](docs/UPGRADE_occurrences.md)
- [Grupy markerów](docs/UPGRADE_marker_groups.md)
- [Porzucanie przerwanych operacji](docs/UPGRADE_abandon_pending.md)
- [Profilowanie merge](docs/UPGRADE_diagnostics.md)
- [Benchmarki](benchmarks/README.md)

Pliki w `docs/validation/` są historycznymi wynikami konkretnych walidacji i
benchmarków. Nie opisują bieżącej liczby testów ani aktualnego kontraktu aplikacji.

Nazwy konfiguracji i CLI są opisane w [instrukcji aktualizacji wykluczeń](docs/UPGRADE_exclusions.md#aktualizacja-starych-nazw). Używaj `excluded_scene_tags`, `--models`, `--scene-tag-exclusions` i `--cleanup-markers-only`; domyślny plik markerów to `marker_core_exclusions.yml`.


W schemacie v14 `results` zawiera najwyżej jedną aktywną inferencję na model i wyjście.
`--db-update` przenosi starsze segmenty do `inference_archive` tylko wtedy, gdy dokładna
sygnatura dowodu występuje w published, pending lub recovery. Nowa inferencja stosuje tę samą
zasadę; archiwum służy wyłącznie frozen replay i jest przycinane po zmianie publikacji.
Zwykły merge, raporty, progi i indeks ukończenia nie korzystają z historii inferencji.


W schemacie v19 aktywna inferencja wymaga analysis_format=5. --db-update wycofuje
starsze wyniki, zaznacza konieczność ponownej inferencji i zachowuje tylko dokładne
dowody przypięte przez published/pending/recovery. Snapshoty i owned_ids pozostają
bez zmian. Uruchom następnie inferencję dla wszystkich wskazanych modeli i wyjść;
--merge-only nie odtwarza brakującego cache. Raporty i occurrences używają pełnych
obserwacji scen oraz jawnych sygnatur. statistics=null nadal oznacza brak
kwalifikującego wsparcia, a nie zgodę na historyczną kwalifikację agregatu.

Schemat v17 wycofuje odtwarzanie historycznego algorytmu i wyjątków dla starego cache.
Dawne publikacje zachowują geometrię, ownership i stan pending/recovery, ale wymagają
ponownej inferencji i publikacji przed cleanup. Stare zwykłe punkty AI trzeba najpierw
przekształcić przez `python -m stash_ai.tools.point_migration`; punkty wykryć pozostają punktami.
Brak `end_seconds` w recovery oznacza dokładny punkt i nie pasuje do zakresu.

Schemat v18 zapisuje aktywny wynik jako jeden obiekt zamiast listy segmentów.
`result_metadata` w formacie 4 ma również jeden obiekt, a `result_thresholds.threshold`
jest opcjonalnym progiem liczbowym. Nowa inferencja zastępuje wynik i resetuje próg.
Migracja zachowuje tylko ostatnią aktywną inferencję i dokładne przypięte dowody archiwalne;
merge, rethreshold, raporty i indeks ukończenia nie odczytują dawnych list ani tablic progów.

Schemat v19 wycofuje aktywną inferencję w formatach 1–4. Uruchom `--db-update`,
a następnie inferencję; `--merge-only` wymaga nowych dowodów. Przypięte dowody
archiwalne oraz geometria, identyfikatory i stan published/pending/recovery
pozostają zachowane. Nowe publikacje wymagają pełnych dowodów czasowych;
historyczna geometria służy wyłącznie odzyskiwaniu stanu i migracji punktów.
Format analizy 5 identyfikuje kompletny kontrakt sceny, bez dodatkowego
`scene_frames_format`. Ręczne tagi dodatkowe pochodzą z obserwowanych markerów
Stash i są zachowywane podczas dopasowania, podziału i łączenia zakresów.

Schemat v20 usuwa stałą kolumnę `result_index.inference` i konwertuje historyczne
konteksty `[signature]` na jeden `signature` per źródło. Nowe snapshoty mają
format 5. Dawne konteksty wielu inferencji jednego źródła pozostają przypięte
jako retired i wymagają przebudowy publikacji. Efektywny próg publikacji jest
jednym polem `publication_threshold` w widoku rekordu; nie trafia do surowego
BLOB-a inferencji. Cleanup odtwarza zamrożony próg w tym samym widoku rekordu.

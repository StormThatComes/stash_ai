# Stash AI

Automatyczne tagowanie scen i tworzenie markerów czasowych dla [Stash](https://github.com/stashapp/stash).

Stash AI używa czterech modeli: JoyTag (`jt`), WD EVA02-Large v3 (`wd`),
DanbooruTagQuery L/16 (`dtq`) i OppaiOracle V1.1 (`op`). Modele są
uruchamiane kolejno — w GPU znajduje się jeden model naraz.

Domyślny tryb `merged` publikuje wspólne tagi `ai:`, łączy zgodne wyniki modeli
i zachowuje osobne źródła oraz pewności w lokalnej bazie SQLite. Wejście wizualne
pochodzi wyłącznie z VTT i sprite'ów Stash; aplikacja nie otwiera plików wideo.

## Wymagania

- Python **3.14.7** — to jedyna wspierana i testowana wersja;
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
# Pinned Python 3.14.7 environments: see constraints/README.md

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
Bazy sprzed v18 wymagają najpierw mostka aktualizacji opisanego poniżej;
obsługiwane schematy przechowują już pojedynczy rekord i liczbowy próg.
`--dry-run` zapisuje progi wyłącznie w tymczasowej bazie.

Snapshoty merge zapisują współdzielone dowody `extra_support` raz, a contributions
odwołują się do nich przez referencje. Zmniejsza to koszt serializacji i kompresji.
Aktualny schemat to v26, z obsługą aktualizacji od v18. Każdy nowy snapshot ma
otoczkę formatu v7. `--db-update` konwertuje snapshoty v4/v5/v6/v7, także pending i recovery,
w jednym przebiegu. Zachowuje geometrię i własność publikacji, normalizuje zamrożone opcje
odtwarzania oraz usuwa `extra_ids` i zbędne `algorithm: temporal`. Pełne payloady
archiwalnych inferencji sprzed formatu 5 są usuwane; ich publikacje otrzymują
`requires_rebase` i wymagają świeżej inferencji oraz `--merge-only`. Archiwum dowodów
formatu 5 pozostaje dostępne do dokładnego odtwarzania opublikowanych wyników.
Wersjonowanie analizy checkpointu dotyczy wyłącznie niedokończonej inferencji:
stare checkpointy są pomijane przy wznowieniu i dany model zaczyna scenę od początku.
Ukończone wyniki formatu 5 zachowują dowody i sygnatury; starsze aktywne wyniki wymagają ponownej inferencji. `--dry-run` zmienia wyłącznie tymczasową bazę.

`--explain-tag` wypisuje listę raportów scen z polem `format: 2`. Każde źródło
w `recorded_sources` ma pojedynczy obiekt `record` zawierający `meta`, `thresholds`
i `detections`, zamiast dawnej jednoelementowej listy `records`.

Pełną listę opcji pokazuje:

```powershell
python .\stash_ai.py --help
```

## Jak działa stan i cache

Przed pominięciem merge na podstawie cache aplikacja odświeża stan sceny
w Stash. Odczyty SQL są grupowane w paczki; zmiana markera przez inną
automatyzację po początkowym skanie uruchamia ponowne scalenie.

Tożsamość inferencji obejmuje SHA-256 zawartości plików modelu oraz normalizację
JoyTag (`JOYTAG_CLIP_MEAN` i `JOYTAG_CLIP_STD`). Zmiana wag jest wykrywana także
przy zachowanym rozmiarze i czasie modyfikacji pliku. Pliki są hashowane raz na
model w przebiegu; po pobraniu brakujących plików hashowane są tylko nowe zasoby.
Pierwszy przebieg po tej zmianie odświeży wcześniejsze inferencje i checkpointy.

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

### Nowa baza przy istniejących wynikach AI

`--db-init` pozwala odbudować stan po utracie bazy albo rozpocząć od nowa,
zachowując stare wyniki w Stash do czasu ukończenia inferencji danej sceny:

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3
```

Wymaga nowej lub pustej bazy i obu wyjść: tagów scen oraz markerów. Przejmuje
istniejące przypisania tagów `ai:<label>` i markery należące do aplikacji.
Po udanej inferencji wszystkich wybranych modeli usuwa nieaktualne wyniki AI;
brak VTT/sprite'a lub błąd modelu pozostawia dotychczasowe wyniki sceny.
Ręcznie przypisane tagi z prefiksem `ai:` również podlegają temu przejęciu,
z wyjątkiem kategorii i zarezerwowanych tagów markerów.

Przerwany przebieg wznów tą samą komendą bez `--db-init`, zachowując modele,
ścieżkę bazy i ewentualny zakres `--scene-id`. Ponowienie flagi też zachowuje
postęp. `--dry-run` korzysta wyłącznie z tymczasowej nakładki i nie zapisuje do
Stash ani źródłowej bazy. Szczegóły: [Odbudowa bazy](docs/UPGRADE_db_init.md).

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

Aktywny cache wymaga jawnych progów retencji i pełnego formatu analizy 5.
`--db-update` wycofuje starszą inferencję; uruchom ją ponownie przed merge i cleanup.
Nadpisanie progu publikacji nie uzupełnia wcześniej odrzuconych obserwacji.

Fazy `Check` i `Apply` rethreshold korzystają z małych rekordów `result_metadata` w SQLite,
zamiast dekodować klatki z `results.data`. Nowe wyniki zapisują je atomowo razem
z evidence; starsze wyniki uzupełniają metadane przy pierwszym sprawdzeniu.
Każdy model i rodzaj wyniku ma jeden aktywny rekord oraz opcjonalny liczbowy próg publikacji. Zmiana lub usunięcie
surowego wyniku unieważnia jego metadane przez trigger, również przy zapisie
starszą wersją programu. `Apply` zapisuje progi w `result_thresholds` i aktualizuje
`result_index` z metadanych: obecności wyników oraz sygnatury ostatniej inferencji
bieżącego rekordu. Przy gotowych metadanych żadna z obu faz nie dekoduje BLOB-ów klatek.
Dry-run uzupełnia wyłącznie tymczasową bazę; `--db-check`
sprawdza zgodność istniejących metadanych z evidence bez ich naprawiania.

Nowa inferencja zawsze zastępuje poprzednie wyniki danego modelu i rodzaju.
Aktualizacja z v18 wycofuje niepełne wyniki starszego formatu analizy i zleca ponowną inferencję.
Publikacje zachowują własność i geometrię, a potrzebne dowody pozostają przypięte w archiwum.
Zwykły merge i cleanup wymagają natywnych zakresów; punkty `OccurrenceTag` pozostają punktami.
Dawne zwykłe punkty AI, hierarchie YAML v1 i zarządzane prefiksy `jt:`, `wd:`, `dtq:` oraz
`op:` wymagają [migratorów w wersji pomostowej](#migratory-w-wersji-pomostowej).
Bieżący kod publikuje wyłącznie namespace `ai:` i nie konwertuje obiektów starych modeli.
Ręczne powiązania ze starymi definicjami tagów pozostają zachowane.

### Migratory w wersji pomostowej

Trzy zakończone migracje nie należą już do bieżącego pakietu. W razie potrzeby uruchom je
z osobnego checkoutu niezmiennego commita
[`9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
Bieżący pakiet korzysta ze schematu v26. Bazy sprzed v18 wymagają najpierw starszego
mostka opisanego w sekcji aktualizacji bazy. Zrób kopię bazy i konfiguracji oraz zainstaluj
zależności checkoutu w osobnym środowisku Python 3.14 zgodnie z instrukcją instalacji.

```powershell
git clone https://github.com/StormThatComes/stash_ai stash_ai_migration_bridge
git -C stash_ai_migration_bridge checkout --detach 9fcaa52502cd487455457547860b5db7d8710ad8
cd stash_ai_migration_bridge

# Konwersja własnych hierarchii v1; podaj wszystkie używane pliki.
python -m stash_ai.commands.hierarchy_migration C:/stash_ai/joytag_hierarchy.yml C:/stash_ai/wd_hierarchy.yml C:/stash_ai/dtq_hierarchy.yml C:/stash_ai/oppai_hierarchy.yml --dry-run

# Zwykłe punkty AI: długość z published/pending/recovery do end_seconds.
python -m stash_ai.tools.point_migration --state-db C:/stash_ai/tagger_sources.sqlite3 --dry-run

# Stare prefiksy: świeża inferencja i publikacja w ai:.
python -m stash_ai.tools.namespace_migration --models all --state-db C:/stash_ai/tagger_sources.sqlite3 --model-config C:/stash_ai/tagger_models.yml --temporal-policy C:/stash_ai/temporal_policy.yml --marker-core-exclusions C:/stash_ai/marker_core_exclusions.yml --scene-tag-exclusions C:/stash_ai/scene_tag_exclusions.yml --hierarchy C:/stash_ai/joytag_hierarchy.yml --wd-hierarchy C:/stash_ai/wd_hierarchy.yml --dtq-hierarchy C:/stash_ai/dtq_hierarchy.yml --oppai-hierarchy C:/stash_ai/oppai_hierarchy.yml --dry-run
```

Zastąp `C:/stash_ai` bezwzględną ścieżką dotychczasowej instalacji. Ustaw ten sam
`STASH_URL` i `STASH_API_KEY`; zachowaj konfigurację modeli, polityk i zakres wyjść.
Jeśli hierarchie są wyłączone, zastąp cztery opcje hierarchii przez `--no-hierarchy`.
Po sprawdzeniu planu powtórz wybrane komendy bez `--dry-run`. Konwersja YAML zachowuje
oryginały jako `*.v1.yml`, przypisania grup i metadane źródła. Migracja punktów nie ładuje
modeli, zachowuje ID, tytuły i ręczne tagi; occurrences pozostają punktami.
Migracja prefiksów zachowuje ręczne obiekty i dodatki markerów, a dopasowane markery
zachowują ID. Można ograniczyć ją przez `--scene-id`, `--markers-only` lub `--scene-tags-only`.
Przerwaną publikację prefiksów dokończ tym samym migratorem i konfiguracją przed powrotem
do bieżącej wersji. Jej pending snapshoty nadal służą do ponowienia po błędzie API.
Następnie wróć do bieżącej instalacji, wykonaj `--db-check` i potrzebną inferencję.

### Zdrowie, aktualizacja i kompaktowanie bazy stanu

`--db-update` aktualizuje bazę wskazaną przez `--state-db` ze schematu **v18 lub nowszego**
do aktualnego **v26**, we wszystkich scope'ach. Zwykłe uruchomienie oraz `--dry-run`
wymagają aktualnego schematu; nowa baza od razu otrzymuje v26.
Migracja dodaje stan odbudowy, ale nie przejmuje istniejących wyników Stash.
To uprawnienie powstaje wyłącznie po jawnym `--db-init` na nowej/pustej bazie.
Updater korzysta z tego samego locka writera. Pracuje offline, bez Stash i konfiguracji modeli.
Zachowuje liczbowe progi bieżących wyników, własność publikacji, pending publication/cleanup
i recovery. Starsze wyniki są kierowane do ponownej inferencji, a ich checkpointy usuwane.
Snapshoty v4/v5/v6/v7 są konwertowane w jednym przebiegu, z zachowaniem współdzielonych dowodów.
Wycofane konfiguracje algorytmów są usuwane; geometria i referencje do dowodów pozostają.
Aktualizacja nie odtwarza utraconych detekcji. Wyniki inferencji sprzed formatu 5 wymagają ponownej inferencji.
Metadane wyników są uzupełniane dopiero po zakończeniu konwersji. Klatki bieżących wyników
muszą być zapisane chronologicznie. Aktualizacja bazy sprzed v23 kieruje nieuporządkowane
rekordy do ponownej inferencji i oznacza zależne publikacje `requires_rebase`.
Nie sortuje zapisanych payloadów ani nie zmienia ich sygnatur. Uporządkowane
wyniki formatu 5 pozostają ważne. Odczyt i zapis odrzucają brakujące metadane
pochodzenia, progów oraz retention floors, zamiast pomijać głosy modelu.

Zwykły cleanup przyjmuje tylko aktualny fingerprint. Przerwany cleanup ze starym
fingerprintem wymaga jawnej konwersji offline, z **tymi samymi** plikami YAML
oraz zakresem wyjść (`--cleanup-only`, `--cleanup-markers-only` lub
`--cleanup-scene-tags-only`). Narzędzie weryfikuje wszystkie polityki per model,
zmienia wyłącznie klucz pending i zachowuje generację, maski oraz recovery:

```bash
python -m stash_ai.tools.cleanup_pending_migration --state-db /path/to/tagger_sources.sqlite3 --cleanup-only --marker-core-exclusions marker_core_exclusions.yml --scene-tag-exclusions scene_tag_exclusions.yml --dry-run
```

Po sprawdzeniu usuń `--dry-run`, a następnie ponów oryginalny cleanup.
Narzędzie nie łączy się ze Stash i nie ładuje modeli.

Dla bazy **sprzed v18** użyj najpierw kompatybilnego commita
[`c5bcd887bafe76b7ee2a9de944666799c9a0fc30`](https://github.com/StormThatComes/stash_ai/commit/c5bcd887bafe76b7ee2a9de944666799c9a0fc30)
w osobnym katalogu, z jego zależnościami. Zachowaj kopię bazy i wskaż tę samą bazę stanu:

```bash
git clone https://github.com/StormThatComes/stash_ai stash-ai-upgrade-bridge
git -C stash-ai-upgrade-bridge checkout c5bcd887bafe76b7ee2a9de944666799c9a0fc30
python stash-ai-upgrade-bridge/stash_ai.py --db-update --state-db /path/to/tagger_sources.sqlite3
```

Następnie uruchom `--db-update` z bieżącej instalacji. Kompatybilny commit konwertuje
historyczne serie i importy do v20, który bieżąca wersja obsługuje bezpośrednio.

`--db-update --dry-run` pokazuje wersję źródłową i docelową bez migracji, zmian pliku ani locka.
Baza sprzed v18 otrzymuje komunikat ze ścieżką aktualizacji; obca lub nowsza baza jest odrzucana.
Konwersja payloadów, wycofanie aktywnych wyników i ustawienie wersji są jedną transakcją.
Tworzenie brakujących tabel jest powtarzalne; przerwaną aktualizację można ponowić.

`--db-check` wykonuje głęboki, wyłącznie odczytowy test spójności całej bazy
(we wszystkich scope'ach). Dla starszego schematu wykonuje tylko preflight wersji i integralności SQLite,
a przed głęboką walidacją wymaga `--db-update`. Oprócz pełnego SQLite `integrity_check` sprawdza m.in.
zgodność `results` z `result_index` (łącznie z zapisaną sygnaturą ostatniej
inferencji), dekodowalność trwałych BLOB-ów, dozwolone model/kind, relacje
checkpoint/frames/analysis, wymiary checkpointów oraz generacje cleanup/recovery.
Waliduje też zawartość wyników scen, klatek i zakresów markerów: wymagane pola,
tagi, czasy, prawdopodobieństwa oraz zapisane statystyki czasowe. Sprawdza klatki
checkpointów (również `scene_tags`) i wartości float32: odrzuca `NaN`,
nieskończoności oraz wyniki poza `[0, 1]`. Aktywne wyniki muszą spełniać bieżący
kontrakt inferencji; starsze zakresy i częściowe statystyki są sprawdzane wyłącznie
w archiwum dowodów. Importy i oceny `score: null` są odrzucane. Wycofane tabele
`legacy_imports` i `refinement_progress` są zgłaszane jako błąd bez odczytu ich payloadów.
Niektóre brakujące powiązania snapshot/recovery są raportowane jako warnings.
Pending publication/cleanup i refresh requests są
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

Raport używa `format: 2`. Dla baz v18–v20 pokazuje informacje o bazie oraz
`database.upgrade_required: true`, bez dekodowania dowodów i szczegółowych statystyk.
Pełny raport wymaga wcześniejszego `--db-update`. Dawne tryby, pochodzenie i formaty
analizy są oznaczane jako `other`; nie ma osobnych pól dla historycznych zakresów
ani `scene_frames_format`.

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

### Historia uruchomień i raport profilowania

Inferencja, merge, cleanup i operacje zapisujące stan automatycznie zapisują
historię w osobnym pliku `<state-db>.runs.sqlite3`. `--run-history-db PATH`
wybiera inny plik, a `--no-run-history` wyłącza zapis. Baza historii jest
niezależna od cache: nie zmienia jego schematu, sygnatur ani zasad wznowienia.
Warto zachować ją razem z bazą stanu, jeżeli zależy Ci na porównaniach historycznych.

Inspekcje i raporty domyślnie nie zapisują historii; dodaj `--record-run`, aby
zmierzyć np. `--model-report`, `--cache-report` lub `--db-check`. `--dry-run`
oraz programowy `DRY_RUN` nigdy nie zapisują historii, także z `--record-run`.
Historia nie wymusza sekwencyjnego merge ani raportu modeli: procesy CPU
zwracają swoje agregaty, a rodzic zapisuje je obok własnych pomiarów.

```powershell
python .\stash_ai.py --models all
python .\stash_ai.py --merge-only
python .\stash_ai.py --model-report .\models.json --record-run
python .\stash_ai.py --run-report .\runs.json
python .\stash_ai.py --run-report .\wd-runs.json --run-model wd --run-command inference --run-status finished --run-limit 100
```

`--run-report NEW.json` czyta wyłącznie historię, drukuje podsumowanie i zapisuje
nowy JSON. Nie potrzebuje bazy cache, modeli ani połączenia ze Stash.
Filtry obejmują komendę, status i model; limit 1..10000 wybiera najnowsze
pasujące przebiegi (domyślnie 50). Istniejący raport nie jest nadpisywany.

Każdy przebieg zawiera status, czas rzeczywisty i CPU rodzica, skróty plików
konfiguracji, parametry workerów, modele i rewizje, dostępne hashe wag/słowników,
liczniki zmian i błędów oraz pomiary etapów/API. Nie zapisuje klucza API,
adresów mediów, payloadów ani treści błędów. Nie jest to raport anonimowy:
identyfikatory przebiegów, daty, konfiguracja i agregaty opisują Twoją instalację.
Zapis startu następuje przed wykonaniem akcji; Ctrl+C i błędy kończą rekord.
Status `running` oznacza brak zapisanego końca — proces może jeszcze pracować
albo zostać zabity bez możliwości zamknięcia historii. Błąd zapisu historii
nie przerywa właściwej komendy i nie przesłania jej błędu.

Raport pokazuje czasy scen, batchy inferencji, przygotowania klatek,
checkpointów, cache, rethreshold, merge, publikacji i operacji API, jeśli dany
etap został wykonany. Świeża inferencja, wznowienia i użycie cache są rozdzielone.
Metryki zawierają liczbę wywołań, sumę, średnią, min/max, przybliżone p50/p95
z ograniczonego histogramu oraz przepustowość, gdy znana jest liczba jednostek.
Średnia to suma czasów / liczba wywołań, również przy łączeniu przebiegów.
Porównania grupują zgodną konfigurację, środowisko i tożsamość modeli,
oddzielając statusy zakończenia oraz udane i nieudane operacje.

Czasy etapów zawierają zarówno czas własny, jak i czas z wywołaniami podrzędnymi.
Praca w wątkach/procesach może się nakładać; sumy etapów nie są czasem całego
przebiegu. `cpu_seconds` obejmuje rodzica, a `worker_cpu` mierzone zadania workerów.
Histogramy mają stały rozmiar; historia rośnie o jeden rekord na przebieg,
bez zapisywania wszystkich pojedynczych pomiarów. Raport opisuje wybrane
przebiegi, nie całą historię, gdy zastosowano limit lub filtry.

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
pustego blocklist. Przerwany cleanup zachowuje generację i maski. Historyczny
podpis wymaga jawnego narzędzia `stash_ai.tools.cleanup_pending_migration`,
które sprawdza ten sam zakres wyjść i politykę wykluczeń. Nowe operacje zapisują tylko bieżący podpis.

## Konfiguracja

Wszystkie modele korzystają z jednego pliku `tag_hierarchy.yml`, będącego unią
dotychczasowych czterech hierarchii: 21 154 unikalne tagi w 92 grupach.
Opcja `--hierarchy YAML` wskazuje wspólny plik również przy wyborze jednego modelu.
Usunięto `--wd-hierarchy`, `--dtq-hierarchy` i `--oppai-hierarchy`.

Hierarchia ma `version: 2` i pola `source` oraz `groups`. Ścieżki grup tworzą
bezpośrednio kategorie `ai:category:` pod korzeniem `ai:category:AI`.
Walidacja wymaga przypisania wszystkich tagów wybranych modeli; tagi pozostałych
modeli są dozwolone. Opcjonalne `source.tag_count` i `source.vocabulary_sha256`
opisują cały zbiór tagów wspólnego pliku, niezależnie od wybranych modeli.
Zmiana grupowania nie wymaga ponownej inferencji ani nowej bazy SQLite;
przypisania można zsynchronizować przez `--sync-hierarchy-only`.
Przy własnych zmianach w starych plikach przenieś je do wspólnego YAML-a;
aktualizator zachowuje stare konfiguracje lokalne, ale bieżący kod ich nie czyta.
Własne YAML-e v1 przekonwertuj za pomocą [wersji pomostowej](#migratory-w-wersji-pomostowej).
Bieżący walidator wymaga v2 i wskazuje commit migratora bez zmiany pliku.

| Plik / opcja | Rola |
|---|---|
| `tagger_models.yml` | modele, katalogi, progi, batch size |
| `temporal_policy.yml` | reguły temporalne, consensus, extras i Occurrence |
| `scene_tag_exclusions.yml` | globalne wykluczenia tagów |
| `marker_core_exclusions.yml` | wykluczenia głównych tagów markerów |
| `tag_hierarchy.yml` | wspólna hierarchia kategorii wszystkich modeli |
| `--state-db` | lokalizacja trwałej bazy stanu |
| `--db-init` | odbudowa nowej/pustej bazy z przejęciem istniejących wyników AI |

Publikacja nie ogranicza liczby tagów ani markerów na scenę. Wszystkie wyniki
spełniające progi, reguły temporalne i wykluczenia mogą zostać opublikowane;
dotyczy to każdego modelu, wspólnych wyników i punktów Occurrence.
`MAX_TAGS` i `MAX_MARKERS_PER_SCENE` zostały usunięte. Limit tagów na klatkę
`MARKER_MAX_TAGS_PER_FRAME` pozostaje osobnym ustawieniem (domyślnie 0).

Bazy v18–v25 wymagają `--db-update`. Migracja do v26 zapisuje snapshoty v7
bez dawnych limitów liczby wyników, zachowując geometrię, własność i plany
pending/recovery. Nie zmienia cache inferencji i nie zapisuje do Stash.
Aby uzupełnić wcześniej obcięte wyniki z zachowanego cache, wykonaj:

```powershell
python .\stash_ai.py --db-update
python .\stash_ai.py --merge-only
```

`--merge-only` stosuje aktualne reguły publikacji bez GPU; wymagania dotyczące
kompletności cache nadal obowiązują. `--dry-run` pozwala obejrzeć plan publikacji.

Aktualne domyślne ustawienia pamięci i równoległości:
- `--ram-budget-gb 32`;
- `--ram-cache-gb 24` — cache jest przydzielany na żądanie i ograniczony do
  maksymalnie 75% budżetu;
- `--prepare-workers 4`;
- `--merge-workers 4` — równoległy CPU replay evidence; zapisy Stash/SQLite pozostają sekwencyjne;
- `--prefetch-batches 4`.

`--db-compact` utrzymuje blokadę zapisu aż do końcowej kontroli bazy i
porównania liczników po VACUUM.

Równoległy merge dzieli budżet RAM i cache pomiędzy procesy pomocnicze oraz
proces nadrzędny zapisujący publikacje. Przy czterech procesach pomocniczych
każdy proces dostaje jedną piątą budżetu; po ich zamknięciu proces nadrzędny
odzyskuje pełny budżet. Procesy raportu modeli używają tych samych ustawień
kwalifikacji i limitów co raport sekwencyjny.

## Aktualizacje

`update_stash_ai.py` aktualizuje zarządzane pliki aplikacji bez Gita i bez
uruchamiania `pip`. Wymaga tokena GitHub z dostępem `Contents: read` do
prywatnego repozytorium.

Bieżący updater obsługuje układ pakietowy (`stash_ai/`). Stare instalacje z
modułami `stash_*.py` wymagają jednorazowego przejścia przez updater z commita
[`9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
Zatrzymaj aplikację i pobierz tę wersję jako osobny plik w katalogu instalacji:

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"
$bridge = "9fcaa52502cd487455457547860b5db7d8710ad8"
$headers = @{ Authorization = "Bearer $env:GH_TOKEN"; Accept = "application/vnd.github.raw+json" }
Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/StormThatComes/stash_ai/contents/update_stash_ai.py?ref=$bridge" -OutFile .\update_stash_ai_bridge.py
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout --dry-run
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout
```

Jeśli dawna aktualizacja została przerwana, najpierw uruchom
`python .\update_stash_ai_bridge.py --install-dir . --recover`.
Updater pomostowy instaluje bieżący pakiet, zachowuje backupy, lokalne YAML,
bazę i modele; zmodyfikowany kod wymaga `--force-code`. Następnie używaj
zwykłego `update_stash_ai.py`. Bieżący updater nie skanuje ani nie usuwa
nieśledzonych dawnych modułów. Przy `git pull` do układu pakietowego nie ma
etapu pomostowego updatera.

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

CI testuje Python 3.14.7. Pełny zestaw testów działa na Linux, a updater jest
dodatkowo sprawdzany na Windows. Testy używają atrap Stash i modeli; nie wymagają
produkcyjnego serwera.

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
- [Odbudowa nowej bazy](docs/UPGRADE_db_init.md)
- [Profilowanie merge](docs/UPGRADE_diagnostics.md)
- [Benchmarki](benchmarks/README.md)

Pliki w `docs/validation/` są historycznymi wynikami konkretnych walidacji i
benchmarków. Nie opisują bieżącej liczby testów ani aktualnego kontraktu aplikacji.

Nazwy konfiguracji i CLI są opisane w [instrukcji aktualizacji wykluczeń](docs/UPGRADE_exclusions.md#aktualizacja-starych-nazw). Używaj `excluded_scene_tags`, `--models`, `--scene-tag-exclusions` i `--cleanup-markers-only`; domyślny plik markerów to `marker_core_exclusions.yml`.


Aktywny wynik to jeden rekord w formacie analizy 5, z opcjonalnym liczbowym progiem
publikacji. Nowa inferencja zastępuje wynik i resetuje próg. Archiwum zachowuje tylko
konkretne dowody przypięte przez published/pending/recovery; zwykły merge, rethreshold,
raporty i indeks ukończenia nie odczytują historycznych segmentów.
`--db-update` zachowuje geometrię, ownership i recovery, a niepełne starsze wyniki
kieruje do ponownej inferencji. `--merge-only` wymaga nowych dowodów przed cleanup.
Brak `end_seconds` w recovery oznacza dokładny punkt i nie pasuje do zakresu.

### Deferred input verification (schema v26)

Run `python stash_ai.py --db-update` once for existing databases. This adds a small
input-verification table without changing raw inference records or publication
snapshots. Missing VTT/sprite paths defer publication and retain previous predictions.
After restoring the same inputs, `--resume` verifies and reuses matching results;
it does not force inference. Model utility reports exclude scenes awaiting input
verification or fresh inference and report the skipped counts.

Explicit missing scene IDs are reported independently. During resumed `--db-init`,
a successful Stash read confirming deletion retires that scene and lets remaining
scenes finish. Network/read errors do not authorize retirement.

Dokładny plan publikacji z cache: `--merge-only --dry-run --dry-run-plan review.jsonl`.
Opis zdarzeń i symbolicznych ID: [docs/dry_run_plan.md](docs/dry_run_plan.md).

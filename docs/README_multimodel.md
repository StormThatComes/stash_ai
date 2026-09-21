# Stash AI — automatyczne tagi, markery i wznawianie analizy

**Cleanup dużej kolekcji:** [nowa ścieżka SQL i instrukcja aktualizacji](UPGRADE_cleanup_sql.md).
**Wydajność, 32 GiB RAM:** [cache merge, przygotowanie CPU i pomiary](UPGRADE_performance.md).
**Dalsze przyspieszenie merge:** [indeks przedziałów, zbiorczy odczyt SQL i pomiary etapów](UPGRADE_merge.md).
**Lokalizacja tagów bez markerów:** [automatyczne punkty wykrycia i obsługa starych wyników](UPGRADE_occurrences.md).

**Grupy markerów:** [tagi `m:<core>` i `m:<core>:occurance` pod `ai_markers`](UPGRADE_marker_groups.md).
Nowe markery `ai:` dostają grupę automatycznie; istniejące można uzupełnić przez `--sync-marker-groups-only`.

**Zmiana reguł po przerwaniu:** [`--abandon-pending`](UPGRADE_abandon_pending.md) pozwala porzucić stary plan
cleanupu lub merge, zachowując wykonane zmiany i wyniki inferencji.
**Diagnostyka wolnego merge:** [czasy per scena i operacja API, rozmiary snapshotów, wyniki pomiarów](UPGRADE_diagnostics.md).

Skrypt obsługuje JoyTag, WD EVA02-Large v3, DanbooruTagQuery L/16 i OppaiOracle V1.1.
Można wybrać jeden model, dowolny zestaw lub wszystkie cztery. Modele pracują w osobnych
przebiegach: w pamięci GPU jest jeden model naraz. Domyślny tryb `merged` zapisuje
**jeden tag `ai:` dla identycznej etykiety** i łączy przedziały markerów.
Pochodzenie i pewność pozostają osobne dla każdego modelu.

Nowa wersja dodaje **jeden reprezentatywny punkt dla tagu sceny bez własnego
markera** oraz zapis czasu maksimum podczas inferencji. Zachowuje
**wspólną kwalifikację tagów scen z zapisanych wykryć** i
zachowuje krótkie, spójne epizody dodatków nawet przy małym pokryciu markera.
Wspólne indeksy ograniczają powtarzane obliczenia, cache wyników w RAM ma domyślny
budżet do 24 GiB przy celu 32 GiB dla procesu. Cztery wątki CPU przygotowują
klatki, a kolejka mieści cztery następne paczki. Kolejne uruchomienia pomijają
merge scen ze zgodnymi wynikami, ustawieniami i stanem wyjść w Stash.
Domyślny raport podsumowuje całą kolekcję, również przy około 9000 scen.
Zachowane są wspólne ciągi klatek, unia zakresów, pochodzenie, histereza,
blacklisty i wznawianie. Ustawienia reguł zawiera
`temporal_policy.yml`. Wszystkie funkcje korzystają wyłącznie z VTT i sprite’ów
udostępnianych przez Stash. Skrypt nie otwiera plików filmów
ani nie pobiera strumieni wideo. Nie dodano whitelisty, ręcznych ocen ani nowych modeli.

| Element | Wynik w trybie domyślnym |
|---|---|
| Tag sceny / główny tag markera | `ai:blue_hair` |
| Dodatkowy tag markera | również `ai:<etykieta>`, bez duplikatów |
| Tytuł markera | np. `[AI] blue_hair [jt+wd+op]` |
| Tag własności markera | `ai:marker:stash-taggers` |
| Hierarchia | wspólne `ai:category:…`, korzeń `ai:category:AI` |
| Pełne źródła, pewności i czasy | lokalny plik `tagger_sources.sqlite3` |

Krótki kod `op` oznacza OppaiOracle. `--models` wybiera modele do nowej inferencji;
wcześniej zapisany wkład pozostałych modeli nadal uczestniczy w scalaniu.
Nie trzeba za każdym razem uruchamiać wszystkich czterech.

Oryginalne prefiksy są dostępne przez `--output-mode separate`:

| Model | Opcja | Prefiks tagów | Tytuły markerów | Tag własności |
|---|---|---|---|---|
| JoyTag | `joytag` | `jt:` | `[JoyTag] …` | `jt:marker:joytag` |
| WD EVA02-Large v3 | `wd` | `wd:` | `[WD] …` | `wd:marker:wd` |
| DanbooruTagQuery L/16 | `dtq` | `dtq:` | `[DTQ] …` | `dtq:marker:dtq` |
| OppaiOracle V1.1 | `oppai` | `op:` | `[OppaiOracle] …` | `op:marker:oppai` |

W trybie `separate` każdy model nadal zarządza własnymi wynikami. Ten tryb nie
usuwa wyników `ai:` i nie aktualizuje bazy źródeł. Po powrocie do `merged`
nowe wyniki ze starymi prefiksami są dołączane do zapisanych źródeł. Dopiero
udana inferencja danego modelu w trybie `merged` zastępuje jego poprzedni wkład
(jeżeli odpowiednia opcja `REPLACE_SCENE_TAGS` / `REPLACE_MARKERS` jest włączona).

## Migracja istniejących tagów i markerów

Rozpakuj **całą** nową paczkę do katalogu skryptu. Zachowaj własne ustawienia
połączenia, ścieżek i progów. Wymagane są wszystkie moduły `stash_*.py`.
Do uporządkowania istniejących wyników nie potrzeba wag ani nowej inferencji:

```powershell
# Podgląd dla jednej sceny:
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Szczegółowe zmiany tagów i markerów dla tej sceny:
python .\stash_ai.py --merge-only --scene-id 123 --dry-run --verbose

# Zapis dla tej sceny:
python .\stash_ai.py --merge-only --scene-id 123

# Migracja wszystkich scen:
python .\stash_ai.py --merge-only

# Tylko markery lub tylko tagi scen:
python .\stash_ai.py --merge-only --markers-only --scene-id 123
python .\stash_ai.py --merge-only --scene-tags-only --scene-id 123
```

`--merge-only` importuje na wskazanych scenach wyniki ze wszystkich czterech znanych
prefiksów, niezależnie od `--models`. Zmienia powiązania tagów scen oraz markery
posiadające zgodny tytuł i tag własności. Ręczne markery pozostają zachowane.
Sam merge zachowuje obiekty starych tagów i kategorii. Dodanie
`--prune-empty-tags` usuwa po udanym scaleniu te, które są już nieużywane
i nie mają dzieci; zasady opisano niżej.

Przy pierwszym imporcie zapisane są źródłowe identyfikatory i zakresy. Stare
markery nie zawierają pewności modeli, więc wartość pozostaje `null`. Brak końca
oznacza punkt, a nie domyślną długość. Skrypt nie odtwarza nieznanych danych.
Gdy `CREATE_MISSING=False`, wynik ze starym prefiksem pozostaje, jeśli nie istnieje
jeszcze jego docelowy tag `ai:`. Blacklisty obowiązują również podczas migracji.

## Jak działa scalanie

Etykiety porównywane są dokładnie, po usunięciu prefiksu źródła: `jt:blue_hair`
i `wd:blue_hair` dają `ai:blue_hair`. Różnice pisowni, wielkości liter i synonimy
pozostają osobnymi tagami. Hierarchia nie scala znaczeń i nie jest whitelistą.

Dla sceny wystarczy kwalifikacja przez jeden model według jego własnych progów
albo wspólne potwierdzenie przez kilka modeli według reguł opisanych niżej.
Oceny modeli nie są uśredniane ani przedstawiane jako
jedna wspólna pewność. Markery powstają dwiema drogami:

1. Zachowywane są zakresy zakwalifikowane przez poszczególne modele.
2. Dla identycznych danych wejściowych budowany jest również wspólny ciąg
   wykryć tej samej etykiety. Minimum `MARKER_MIN_HITS` jest sprawdzane po
   zebraniu tych klatek. Dodawane są zakresy, których dotychczasowa unia nie
   pokrywa w całości.

Przykład przy minimum 2 trafień, zerowej tolerancji przerwy i silnych wykryciach:

| Model | 0–1 s | 1–2 s | 2–3 s | 3–4 s |
|---|---|---|---|---|
| WD | wykrycie | — | wykrycie | — |
| DTQ | — | wykrycie | — | wykrycie |

Powstaje marker **0–4 s**, ze źródłami `wd+dtq`. Każdy czas próbki liczy się
raz, niezależnie od liczby modeli. Wspólny zakres wymaga wkładu co najmniej
dwóch różnych modeli; zwykły marker jednego modelu nadal może powstać samodzielnie.

Na końcu wykonywana jest unia wszystkich zakwalifikowanych zakresów tej samej etykiety:

| Źródła dla jednej etykiety | Wynik |
|---|---|
| JT: 10–30 s, WD: 20–45 s, OP: 15–35 s | 10–45 s, źródła `jt+wd+op` |
| JT: 10–20 s, WD: 20–30 s | 10–30 s, źródła `jt+wd` |
| JT: 10–20 s, WD: 30–40 s | dwa markery: 10–20 s oraz 30–40 s |

Zakresy nakładające się lub stykające łączą się także przez łańcuch pośrednich
zakresów. Unia nie wypełnia dodatniej przerwy. Ustawienie
`MARKER_MERGE_GAP_SECONDS` działa zarówno podczas budowania markerów pojedynczego
modelu, jak i wspólnych ciągów. Nowe łączenie wymaga zgodnego skrótu VTT/sprite’ów
oraz identycznej pełnej osi czasu próbek. Starsze importy zawierające tylko
gotowe zakresy nadal uczestniczą w końcowej unii.

Dodatkowe tagi łączonych markerów tworzą wspólny zestaw `ai:`. Zachowywane są
również ręcznie dołączone tagi spoza przestrzeni `ai:`. Przy dzieleniu markera
po cleanup takie ręczne dodatki przechodzą do obu pozostałych części.
Tytuł pokazuje źródła głównego tagu danego zakresu.
Model dostarczający wyłącznie dodatek nie jest dopisywany w tytule jako źródło
tagu głównego; jego wkład jest osobno zapisany w danych tego dodatku.

## Baza źródeł i podgląd pochodzenia

**Zachowuj `tagger_sources.sqlite3` razem ze skryptem i uwzględnij go w kopiach
zapasowych.** Powstaje przy pierwszym zapisie w trybie `merged`; paczka nie
zawiera gotowej bazy ani wyników Twoich scen. Możesz wybrać stałą lokalizację
przez `--state-db D:\StashAI\tagger_sources.sqlite3` i używać jej we wszystkich
kolejnych wywołaniach. Baza rozdziela dane według adresu Stash, ID sceny, modelu
i typu wyjścia. Zmiana adresu Stash tworzy oddzielny zakres danych.

Zapisywane są agregaty i statystyki kandydatów na tagi scen oraz rzadkie
trafienia klatek od progu podtrzymania markera, domyślnie `0.85 × marker_threshold`,
przed blacklistami YAML i limitami liczby tagów. Statystyki scen powstają dla
wykryć od `0.85 × scene_threshold`: obejmują liczbę trafień, maksimum,
powtarzalność i pokrycie próbek. Towarzyszą im nazwa/repozytorium modelu,
rewizja, progi, agregacja, skrót słownika i czas inferencji. Baza nie zawiera
obrazów ani klucza API. Techniczny `MARKER_TAG_BLOCKLIST` filtruje klatki jeszcze
przed zapisaniem; domyślnie jest pusty.

Nie jest to cache wszystkich ocen wszystkich klatek. Zmiana progu modelu,
`marker_continue_ratio`, `scene_support_ratio`, agregacji scen lub wejściowych
miniatur wymaga nowej inferencji. Same edycje blacklist YAML, limitów, minimalnej
liczby trafień, wymaganej części przedziału lub reguł grupowania można zastosować
przez `--merge-only` do zachowanych danych. Nowe łączenie klatek i dodatków także
można zastosować w ten sposób do cache z poprzedniej wersji automatycznych reguł:
format wyników, schema bazy i identyfikacja inferencji pozostają bez zmian.
Nie trzeba ponownie pobierać sprite’ów ani ładować modeli. Cache bez informacji
o zgodności wejściowych klatek zachowuje wcześniejsze scalanie gotowych zakresów.
Statystyki scen opisują próg zapisany
podczas ich obliczania; zmiana tego progu wymaga nowego zwykłego przebiegu.
Aktualizacja jednego modelu nie zastępuje rekordów innych;
tryby częściowe nie zastępują drugiego typu wyjścia. Wyłączenie zastępowania
zachowuje także wcześniejsze przebiegi danego modelu i ich pewności.

```powershell
# Bez połączenia ze Stash i bez modeli:
python .\stash_ai.py --explain-tag ai:blue_hair --scene-id 123
```

Raport JSON zawiera źródła, osobne pewności, oryginalne zakresy lub trafienia
klatek, wynik unii i informację o oczekującej synchronizacji. Nowe wyniki dodają
powód kwalifikacji tagu sceny i statystyki dodatków markerów. Pokazuje również
zapisane wykrycia objęte blacklistą. Część wynikowa uwzględnia aktualne lokalne
reguły; nie jest odczytem bieżących markerów ze Stash.

Zakres powstały ze wspólnych klatek ma `derived: shared_frames`, łączną liczbę
różnych trafień `shared_hits` oraz `source_evidence` z ocenami, trafieniami
i faktycznymi podprzedziałami każdego modelu. Techniczne pola `source` i `score`
takiego wkładu odnoszą się do jednego wskazanego modelu; nie są wspólną oceną
całego zakresu. Pełne źródła pokazuje `source_evidence`. `as_marker_extra`
w raporcie `--explain-tag` pokazuje także występowanie etykiety jako dodatku,
jego kwalifikację, pokrycie i osobne źródła.

Surowe wyniki cache nie rosną od samego włączenia nowych reguł. Zapis
opublikowanych wyników może być większy o dowody wspólnych tagów scen,
zakresy wspólne i dowody krótkich epizodów dodatków. To metadane tekstowe,
bez obrazów; ich ilość zależy od liczby wykryć.

Sam tag `ai:` nie wystarcza do odtworzenia pewności i wkładu poszczególnych
modeli. Jeśli istnieją zarządzane markery `ai:`, a brakuje ich stanu źródłowego,
skrypt zatrzymuje synchronizację tej sceny i prosi o przywrócenie właściwej bazy.
Istniejące powiązania tagów `ai:`, których skrypt nie zarejestrował jako własne,
są zachowywane jako ręczne.

## Automatyczne reguły tagów i markerów

Domyślnie działają poniższe reguły, wspólne dla pełnych słowników wszystkich modeli:

| Reguła | Domyślne działanie |
|---|---|
| Wspólne klatki | Modele mogą podtrzymywać tę samą etykietę naprzemiennie; minimum trafień sprawdzane jest dla wspólnego ciągu. |
| Dodatki markera | Co najmniej 2 różne czasy próbek oraz pokrycie 20% całego przedziału **lub** spójny ciąg co najmniej 3 wysokich wykryć. |
| Histereza | Start od progu modelu, podtrzymanie od 85% tego progu, najwyżej 15 s po końcu ostatniej próbki rozpoczynającej lub wzmacniającej marker. |
| Zgodność modeli | Co najmniej 2 modele wykrywające identyczną etykietę na tej samej klatce VTT od swoich progów podtrzymania mogą wspólnie rozpocząć marker. |
| Tagi scen | Dotychczasowe reguły pojedynczego modelu oraz dodatkowe potwierdzenie przez kilka modeli z tego samego wejścia. |

**Dodatki** są oceniane na wszystkich próbkach wewnątrz przedziału, także przy
dopuszczonej krótkiej przerwie wykrywania tagu głównego. Liczą się wyniki dodatku
powyżej zwykłego progu markera danego modelu. Po unii przedziałów ponownie
sprawdzane jest pokrycie całego połączonego markera. Nakładające się czasy nie
liczą się podwójnie, a dwa modele oceniające próbkę w tym samym momencie nie dają
dwóch trafień. Importowane stare dodatki bez danych o klatkach są zachowywane.
Wykluczenie tagu z głównych nadal pozwala używać go jako dodatku.
Model nie musi wykryć tagu głównego, aby dostarczyć dodatek. Wymagana jest
zgodność całego wejścia i osi czasu z modelem budującym dany zakres.

**Krótki epizod dodatku** wymaga domyślnie co najmniej 3 kolejnych, różnych
czasów próbek z oceną od zwykłego progu markera danego modelu. Wykrycia mogą
pochodzić naprzemiennie z różnych zgodnych modeli. Brak wykrycia na pośredniej
próbce albo przerwa między przedziałami VTT przerywa ciąg; tolerancja przerw
tagu głównego nie wypełnia takiej luki w dodatku. Minimum `extra_min_hits`
nadal obowiązuje. Przykładowo: 3 kolejne wykrycia obejmujące 12 s w markerze
120 s mogą zachować dodatek mimo pokrycia 10%. Dwa odległe trafienia nie
wystarczą do tej ścieżki. Dowody ciągu są łączone przy unii bez podwójnego
liczenia tych samych próbek. Raport dodatku pokazuje `longest_run_hits` i
`qualification: short_episode` albo `coverage`.

**Histereza** przy progu `0.80` pozwala podtrzymywać marker wynikiem od `0.68`.
Słabsze pojedyncze źródło samo nie rozpoczyna zakresu. Nadal obowiązują
`MARKER_MIN_HITS`, istniejąca tolerancja przerw i limity liczby markerów.
Zupełny brak wykrycia nie jest automatycznie wypełniany. Końcowy wynik nadal jest
unią zakresów pojedynczych i wspólnych; dodatnich przerw sama unia nie skleja.

**Zgodność** wymaga identycznego zestawu VTT/sprite'ów, sprawdzonego skrótem
zawartości, i tego samego czasu próbki. Wiele zapisanych przebiegów jednego modelu
to nadal jeden głos. Nie są uśredniane pewności różnych modeli. Silny tag
unikalny dla jednego modelu nie wymaga zgody pozostałych. Głos modelu objęty
wykluczeniem nie wspiera tagu głównego. Źródła wsparcia trafiają do tytułu markera
i bazy wraz z oryginalnymi ocenami. Głosy wspólne są liczone wyłącznie na
klatkach wskazanych przez VTT.

**Tag sceny** nadal kwalifikuje się, jeśli przynajmniej jeden model spełni jedną regułę:

- Maksimum wynosi co najmniej `scene_threshold + (1 - scene_threshold) × 0.50`.
  To zachowuje krótkie, silne wykrycia.
- Dotychczasowy agregat `topk_mean` lub `max` osiąga próg, a co najmniej 2 klatki
  osiągają zwykły próg sceny.
- Co najmniej 4 klatki osiągają `0.85 × scene_threshold`, w tym co najmniej
  3 kolejne, i stanowią co najmniej 10% analizowanych klatek.

Dodatkowo, w trybie `merged`, identyczna etykieta może zostać przyjęta na
podstawie wspólnych zapisanych klatek:

- Co najmniej 2 różne czasy osiągają próg sceny odpowiedniego modelu, a wysokie
  wykrycia pochodzą od co najmniej 2 różnych modeli (`shared_repeated_high`).
- Co najmniej 4 różne czasy osiągają progi podtrzymania scen odpowiednich
  modeli, w tym 3 kolejne, pokrycie próbek wynosi co najmniej 10%, a wkład
  pochodzi od co najmniej 2 modeli (`shared_persistent_support`).

Wymagana liczba modeli wynika z `consensus_min_models`. Każdy model korzysta
ze swojego `scene_threshold`, a tagi są filtrowane wyłącznie przez wykluczenia
scen. Blacklista głównych tagów markerów nie blokuje tej kwalifikacji. Samo
istnienie markera nie awansuje jego tagu do tagów sceny. Wspólne tagi są
dodatkową drogą kwalifikacji; nie usuwają dotychczasowych kwalifikacji modeli.
Opcjonalny `MAX_TAGS` ogranicza dodatkową pulę wspólnych etykiet niezależnie od
dotychczasowych limitów poszczególnych przebiegów.

Łączone są tylko rekordy scen i klatek z tej samej inferencji, zgodnym wejściem
i pełną osią czasu. Skrypt porównuje również zapisane progi. Gdy próg zachowania
klatek był wyższy niż próg podtrzymania tagów scen, część słabszych wykryć może
być nieobecna. Wciąż można przyjąć tag, jeżeli **same zachowane dodatnie
wykrycia wystarczają do spełnienia reguły**. Brakujących ocen nie zastępuje się
zerami i nie używa do odrzucenia dotychczasowych tagów. Przy niewystarczających
danych pozostają poprzednie wyniki kwalifikacji; nie jest wymuszana nowa
inferencja kolekcji. Użyte źródła i osobne oceny są zapisywane pod
`derived: shared_scene_frames`, `source_evidence` i `joint_statistics`.

To reguły oparte na próbkach, nie pewność obecności przez określony procent
całego filmu. Nie dodano słownika synonimów: scalane są nadal identyczne etykiety.

```powershell
# Nowa analiza lub wznowienie z domyślnymi regułami:
python .\stash_ai.py --models all

# Zastosowanie reguł do dostępnych wyników, bez inferencji:
python .\stash_ai.py --merge-only --scene-id 123 --dry-run

# Dawna kwalifikacja i dodatki z pierwszej klatki:
python .\stash_ai.py --no-temporal
```

`shared_marker_frames: true` włącza wspólne ciągi klatek, a
`shared_marker_extras: true` zbieranie dodatków ze wszystkich zgodnych modeli.
Obie opcje są domyślnie włączone i można wyłączyć każdą osobno. Przy wyłączonym
zbieraniu ze wszystkich modeli nowy wspólny zakres korzysta z dodatków jego
własnych źródeł tagu głównego. Ustawienie obu opcji na `false` wyłącza te
dwa sposoby współpracy modeli, zachowując pozostałe reguły.

Nowe przełączniki są niezależne:

```yaml
short_marker_extras: true
extra_short_min_hits: 3
shared_scene_tags: true
```

Ustaw `short_marker_extras: false` i `shared_scene_tags: false`, aby zachować
kwalifikację z poprzedniej paczki przy korzystaniu z nowych optymalizacji.
Zmiana tych opcji nie unieważnia ukończonych inferencji ani checkpointów.

`--temporal-policy PATH` wybiera inny plik ustawień. `--no-temporal` nie zmienia
przestrzeni `ai:`, unii przedziałów ani blacklist i nie usuwa danych źródłowych.
Tryb `separate` korzysta z nowych reguł pojedynczego modelu; nie liczy głosów
między modelami i nie zapewnia trwałych wznowień.

## Wejście wyłącznie przez Stash VTT/sprite’y

Źródłem obrazów są wyłącznie istniejące sprite’y pobierane przez HTTP(S) ze Stash,
z czasami i prostokątami klatek wskazanymi przez VTT. Nie jest potrzebny dostęp
skryptu do katalogu z filmami ani mapowanie ścieżek. Zapytanie o scenę
nie żąda ścieżek plików wideo. Brak sprite’ów/VTT powoduje pominięcie sceny;
skrypt nie próbuje wtedy dekodować filmu inną metodą.

Usunięto etap dodatkowego próbkowania, flagi `--refine-markers` /
`--no-refine-markers` i ustawienia `refine_*`. Stare flagi są odrzucane przed
uruchomieniem pracy. Przy własnym pliku polityki usuń z niego klucze `refine_*`
lub użyj zaktualizowanego `temporal_policy.yml` z paczki.

Rozpakuj całą paczkę do katalogu skryptu. `stash_refine.py` jest teraz krótką
blokadą zgodności: zastępuje dawną implementację i odrzuca każdą próbę importu.
`tests/test_refine.py` zawiera wyłącznie testy wyłączenia tego etapu. Nadpisanie obu
plików usuwa poprzedni kod również przy aktualizacji w istniejącym katalogu.

Zapisane dawniej próbki wideo są pomijane. Stan ich niedokończonego przetwarzania
jest usuwany podczas migracji bazy. Wyniki VTT/sprite’ów, statystyki scen i ich
checkpointy pozostają dostępne. Nie wymaga to ponownej inferencji wyników
z poprzedniej wersji z regułami automatycznymi. Aby przebudować markery wyłącznie
z zapisanych danych VTT, uruchom:

```powershell
python .\stash_ai.py --merge-only --scene-id 123 --dry-run
python .\stash_ai.py --merge-only --scene-id 123
```

Możesz pominąć `--scene-id 123`, aby objąć wszystkie sceny. Przy `--merge-only`
modele nie są uruchamiane. Dawne próbki są usuwane z odczytywanego widoku danych;
nie ma jednorazowego odczytu i przepisywania wszystkich dużych rekordów cache.
Przy następnym zapisie wyników danego modelu te nieużywane pola znikają z rekordu.
Opublikowane markery i dziennik niedokończonej synchronizacji pozostają zachowane
do właściwego uzgodnienia ze Stash. Cleanup zależny od zmienionych dowodów może
wymagać wcześniejszego `--merge-only`, zgodnie z dotychczasową regułą.

Precyzję czasów nadal ogranicza odstęp między dostępnymi klatkami VTT.
Histereza, zgodność modeli i unia korzystają z tych danych; nie tworzą nowych
obserwacji obrazu pomiędzy klatkami.

## Wznawianie i wymuszenie nowej analizy

W domyślnym trybie `merged` wznowienie jest włączone automatycznie. Uruchom
ponownie tę samą komendę, wskazując tę samą bazę `--state-db` i te same modele:

```powershell
python .\stash_ai.py --models all

# Nowa analiza wybranej sceny; zachowuje wyniki innych scen i modeli:
python .\stash_ai.py --models oppai --scene-id 123 --restart

# Po przerwaniu powyższego przebiegu pomiń --restart:
python .\stash_ai.py --models oppai --scene-id 123
```

Dla każdej pary scena–model skrypt sprawdza podpis wejścia i wyników. Podpis
obejmuje zawartość VTT i użytych sprite'ów, czasy i prostokąty wybranych klatek,
model/rewizję, słownik z kolejnością etykiet, parametry inferencji oraz metadane
lokalnych plików modelu. Sprawdzenie obrazu pobiera jego bajty z lokalnego Stash;
nie wymaga dekodowania wszystkich klatek ani uruchamiania modelu. Zmieniona
zawartość pod tym samym URL unieważnia wcześniejszy podpis. Pliki wag porównywane
są według ścieżki, rozmiaru i czasu modyfikacji, bez odczytu gigabajtów wag.
Po nietypowej podmianie wag z zachowaniem tych metadanych użyj `--restart`.

Zgodny, ukończony wynik jest używany ponownie, także gdy model nie wykrył żadnych
tagów. Wagi modelu wczytywane są dopiero dla pierwszej sceny wymagającej pracy.
Przerwana scena wznawia się od ostatniej zapisanej pełnej partii. Checkpoint
zachowuje licznik klatek, stan agregacji top-k/max, statystyki scen i trafienia markerów; nie
publikuje częściowych wyników jako ukończonej sceny. Po awarii zapisu do Stash
wyniki inferencji pozostają gotowe do ponownej synchronizacji.

Postęp jest utrwalany co `--checkpoint-batches 8` partii oraz przy obsługiwanym
przerwaniu Ctrl+C lub błędzie. Twarde zabicie procesu bądź utrata zasilania
wymaga powtórzenia pracy od ostatniego zapisu. Można ustawić `--checkpoint-batches 1`
za cenę częstszych zapisów SQLite. `--restart` zapisuje żądanie odświeżenia
wszystkich wybranych zadań, więc po przerwaniu kolejny zwykły przebieg dokończy
również modele, do których restart nie zdążył dojść. Ponowne dodanie `--restart`
usunie ich częściowy postęp i ponownie wymusi przeliczenie.

Zmiany blacklist YAML, hierarchii, limitów i reguł budowania markerów korzystają
z zachowanych trafień. Zmiana progów, agregacji, modelu lub wejściowych klatek
uruchamia nowe obliczenia. `--markers-only` i `--scene-tags-only` nie zmieniają
drugiego wyjścia i potrafią użyć odpowiedniej części kompletnego wyniku.
`--merge-only` służy do publikacji zapisanych źródeł bez sprawdzania ich aktualności
względem obrazów i bez inferencji.

**Bazy z poprzednich wersji:** SQLite aktualizuje się automatycznie do wersji 4,
z zachowaniem źródeł. Ta poprawka zachowuje podpis inferencji i zgodność
checkpointów VTT z poprzednią wersją z regułami automatycznymi. Nie usuwaj bazy
ani tagów przed aktualizacją.

Jeszcze starsze wyniki, sprzed reguł automatycznych, nadal wymagają jednego
zwykłego przebiegu, aby uzupełnić słabsze trafienia, statystyki scen i podpis
identycznych wejść dla głosowania. Można używać ich przez `--merge-only`,
bez nowych obliczeń, lecz brakujących dowodów nie da się w ten sposób odtworzyć.
Tagi scen bez statystyk zachowują dawną regułę agregacji.

Tryb zgodności `--output-mode separate` zachowuje dotychczasową pełną inferencję
i osobne prefiksy. Nie korzysta z bazy wyników ani checkpointów;
`--resume`/`--restart` wymagają `merged`. Przygotowywanie następnej partii w RAM
jest dostępne w obu trybach.

## Dysk, RAM i ograniczenie odczytów

Nie powstaje dyskowy cache sprite'ów ani wyciętych klatek. Baza wyników otrzymuje
mały indeks statusów, żądania ponownej analizy i skompresowane checkpointy.
Checkpoint zawiera najwyżej `TOP_K_FRAMES × liczba_etykiet × 4` bajtów ocen przed
kompresją (dla `max` jeden wiersz), plus trafienia markerów i metadane. Dla
OppaiOracle i top-k=2 sama macierz to około 151 KiB przed kompresją. Dochodzą
liczniki obserwowanych etykiet scen, słabsze trafienia potrzebne do histerezy
oraz przedziały występowania dodatków.
Baza będzie większa o te dane; wzrost zależy od liczby trafień i długości scen.
Nie są zapisywane pełne macierze ocen wszystkich klatek ani żadne obrazy.
Dowody dodatków zachowują tylko tyle czasów próbek, ile potrzeba do sprawdzenia
`extra_min_hits`; raport `hits_at_least` jest dolnym ograniczeniem liczby trafień.

Po udanym zapisie wyniku jego checkpoint jest usuwany w tej samej transakcji.
Przez chwilę zapisywany wynik i checkpoint mogą współistnieć; rozmiar pliku
SQLite nie musi potem zmaleć, bo wolne strony są ponownie wykorzystywane.
Domyślne zastępowanie wyników nie dopisuje historii przy każdym uruchomieniu.
Przerwane zadania zachowują checkpointy do wznowienia lub restartu.

Domyślne `--prefetch-batches 4` przygotowuje na CPU do czterech następnych paczek,
podczas gdy model analizuje bieżącą. Zakres ustawienia to `0..16`; `0` wyłącza
wyprzedzanie inferencji. Jeden wątek jest właścicielem odczytu sprite’ów,
a `--prepare-workers 4` rozdziela transformacje obrazów między cztery wątki CPU.
Kolejność klatek i paczek pozostaje zachowana. Ustawienie `--prepare-workers 1`
wyłącza równoległe transformacje. Kolejka mieści N następnych paczek plus bieżącą, a po
błędzie lub przerwaniu zostaje zamknięta. Przygotowanie paczki nie oznacza jej
przeanalizowania: checkpoint zapisuje wyłącznie ukończone wyniki modelu.
Pozostaje dotychczasowy limit dwóch zdekodowanych arkuszy obrazów. W `merged`
wspólny cache RAM przechowuje również pobrane bajty i manifesty VTT, ponownie
używane przez kolejne modele. Duży plik lub zbiór przekraczający budżet może
wymagać ponownego pobrania; hash weryfikuje wtedy spójność danych. Następne
uruchomienie ponownie sprawdza wejście. Obrazy nie są zapisywane na dysku.

`--ram-budget-gb 32` ustawia cel pamięci procesu, a `--ram-cache-gb 24` maksymalny
budżet wspólnego cache wyników, indeksów i mediów w `merged`. Cache można
wyłączyć przez `0`; jego limit nie przekracza 75% celu pamięci procesu.
Pamięć jest przydzielana w miarę potrzeb. Limit uwzględnia szacowany rozmiar
obiektów po rozpakowaniu, a nie liczbę scen lub wielkość skompresowanych BLOB-ów.
Najdawniej używane wpisy są usuwane po przekroczeniu budżetu. Cache może
przechowywać dane wielu scen. Budżet nie jest limitem całego procesu: bieżące
obliczenia, modele, bufory SQLite i przygotowane obrazy potrzebują dodatkowego
RAM. Na Windows/Linux próbkowanie RSS usuwa wpisy cache pod presją pamięci
(próg 28 GiB dla celu 32 GiB). To cel sterujący cache, a nie twarda blokada
alokacji: pojedyncza duża scena, dekodowanie obrazu lub biblioteka modelu mogą
chwilowo przekroczyć cel. Domyślne ustawienia rezerwują 8 GiB poza cache.

Cache przechowuje odczytane wyniki, stan publikacji, informacje o imporcie i
statusy. Zapisy unieważniają odpowiednie wpisy; nie są zachowywane prowizoryczne
dane transakcji, która mogłaby zostać wycofana. Zbiorcze odczyty małego indeksu
źródeł i ukończenia zastępują wielokrotne zapytania o poszczególne pary
scena–model. Mała tabela `merge_checks` pamięta zgodność ostatniej publikacji;
triggery unieważniają ją przy zapisie wyników, snapshotów lub reguł cleanupu,
także przez poprzednią wersję skryptu. Format wyników i checkpointów pozostaje
zgodny. Zatwierdzenie publikacji promuje zapisany już BLOB `pending`, bez
ponownego kodowania tych samych danych.

Wspólny indeks klatek powstaje raz dla danego zestawu wejściowego. Markery
pojedynczych modeli pomijają własne obliczanie dodatków, gdy dodatki będą zebrane
ze wspólnego indeksu. Zapytanie o dodatki dla identycznego przedziału jest
ponownie używane przez kolejne tagi główne. Przywrócenie cache z RAM po
zamknięciu procesu nie jest potrzebne: wznowienia nadal opierają się na SQLite.

Sprawdzenie ukończenia korzysta z indeksu bez dekompresji zapisanych wyników.
Lista scen zawiera od razu potrzebne tagi i markery. Zgodny fingerprint stanu
Stash, ustawień i zapisanych wyników pozwala pominąć bootstrap, obliczenia merge
i publikację. Zmieniona scena przechodzi pełne uzgodnienie z ponownym odczytem
przed publikacją, zachowującym ręczne dodatki. Zmiana YAML lub hierarchii
unieważnia właściwe sprawdzenia. Pierwszy odczyt scen i ich markerów nadal jest
potrzebny; cache nie usuwa tego kosztu dla całej kolekcji. Dla wielu scen katalog tagów
jest wczytywany stronami i używany w RAM. Dopasowywanie markerów przeszukuje
markery z tym samym tagiem głównym. Tabela sprawdzeń przechowuje wyłącznie
małe hashe, bez drugiej trwałej kopii markerów Stash.

Jeśli Stash udostępnia `querySQL`, tagi i markery scen są teraz odczytywane
zbiorczo przez SELECT-y korzystające z indeksów. `--merge-only` czyta również
listę scen po rosnącym ID przez SQL, bez COUNT/OFFSET i bez ścieżek VTT/sprite.
Brak SQL lub błąd odczytu uruchamia ścieżkę GraphQL. Zapisy nadal wykonuje API.
Nowy indeks wystąpień dodatków ogranicza ponowne przeglądanie tych samych
klatek; cache zapytań danej osi jest zwalniany po obliczeniu sceny.

## Raportowanie dla dużych kolekcji

Domyślnie wyświetlane są odświeżane paski postępu oraz zbiorcze podsumowanie,
bez osobnej wiadomości dla każdej udanej sceny czy usuwanego pustego tagu.
Dotyczy to również `--dry-run`, cleanup i trybu `separate`. `--verbose`
włącza szczegółowe komunikaty, np. dla wybranego `--scene-id`.

Raport grupuje wyniki scen, markerów, wspólnych wykryć i wykorzystania RAM.
W `merged` dodaje `Timings (exclusive)` z czasami obliczeń, planowania, lokalnej
bazy i odczytów/zapisów Stash. `--verbose` pokazuje też czasy pojedynczych scen.
Znaczenie czasów i liczników `merge_*` opisuje `UPGRADE_merge.md`.
`shared_scene_tags` liczy dodatkowo zakwalifikowane etykiety, `shared_marker_runs`
wspólne zakresy, `shared_marker_extensions` te z nich, które rozszerzają
dotychczasową unię. `short_extras` liczy dodatki przyjęte dzięki krótkiej serii,
a `cross_model_extras` dodatki mające źródło spoza źródeł tagu głównego.
To liczniki kwalifikacji bieżącego przebiegu; liczniki `created`, `updated` i
`deleted` osobno opisują zaplanowane lub wykonane zmiany Stash.

`scene_cache_partial` liczy sceny, dla których wspólna kwalifikacja korzystała
z klatek zachowywanych od wyższego progu niż potrzebny do pełnej oceny słabszych
wykryć. `scene_cache_missing` oznacza brak wystarczającego zestawu zgodnych
rekordów klatek, gdy w bazie są wyniki scen co najmniej wymaganej liczby modeli.
Żaden z tych liczników nie uruchamia dodatkowej inferencji.

Powtarzające się błędy są grupowane według etapu, modelu i typu wyjątku.
Pierwszy błąd grupy jest widoczny od razu, a podsumowanie zawiera liczbę
wystąpień i maksymalnie dwa przykłady. Zapamiętywanych jest najwyżej sześć grup;
pozostałe błędy trafiają do osobnego licznika. `--verbose` pokazuje wszystkie
błędy. Liczniki nie wymagają dodatkowych zapytań Stash ani zapisywania każdego
odrzuconego tagu w bazie. `--explain-tag` pozostaje szczegółowym raportem
wybranej etykiety, również jako dodatku lub wspólnego tagu sceny.

```powershell
# Zastosowanie reguł do całej kolekcji z podsumowaniem:
python .\stash_ai.py --merge-only

# Budżet 24 GiB cache i cztery przygotowywane paczki:
python .\stash_ai.py --models all --ram-cache-gb 24 --prefetch-batches 4

# Szczegółowy podgląd jednej sceny:
python .\stash_ai.py --merge-only --scene-id 123 --dry-run --verbose
```

## Usuwanie pustych tagów i kategorii

```powershell
# Podgląd sprzątania wszystkich przestrzeni tego skryptu:
python .\stash_ai.py --prune-empty-tags-only --dry-run

# Wykonanie:
python .\stash_ai.py --prune-empty-tags-only

# Tylko pozostałości OppaiOracle:
python .\stash_ai.py --prune-empty-tags-only --prune-prefixes op: --dry-run

# Sprzątanie po udanym scaleniu lub tagowaniu:
python .\stash_ai.py --merge-only --prune-empty-tags
python .\stash_ai.py --models all --prune-empty-tags
```

Domyślny zakres to `jt:`, `wd:`, `dtq:`, `op:`, `ai:` i zarządzane grupy `m:`.
Ręczne tagi `m:` oraz korzeń `ai_markers` pozostają. Usuwany tag musi nie mieć dzieci ani żadnego użycia w dostępnych
licznikach Stash: scenach, markerach (głównych lub dodatkowych tagach), obrazach,
galeriach, wykonawcach, studiach i grupach/filmach obsługiwanych przez serwer.
Skrypt wykrywa możliwości API; brak niezbędnych liczników zatrzymuje sprzątanie.

Najpierw pobierany jest kompletny katalog, a przed każdym rzeczywistym usunięciem
następuje ponowny odczyt kandydata. Po usunięciu liścia sprawdzani są jego rodzice;
puste gałęzie mogą zniknąć aż do korzenia. Każdy usuwany rodzic również musi
mieścić się w wybranych prefiksach i spełniać kryteria. Podgląd symuluje tę
rekurencję bez zapisów.

Sprzątanie pokazuje dwa paski postępu: `Read tag catalog` podczas odczytu
tagów oraz `Prune empty tags` podczas ich sprawdzania i usuwania (w podglądzie:
`Preview empty tags`). Widać tempo, szacowany pozostały czas oraz liczniki
usuniętych/zaplanowanych, zachowanych po ponownym sprawdzeniu i oczekujących
tagów. Suma kandydatów rośnie, gdy do kolejki trafiają nowo opróżnieni rodzice.
Liczba tagów katalogu przychodzi z pierwszą stroną, bez dodatkowego zapytania API;
do jej otrzymania wyświetlany jest licznik bez procentów. Komunikaty o tagach
pojawiają się nad paskiem.

Sprzątanie ma zasięg **globalny**: nieużywane definicje tagów nie należą do jednej
sceny. `--scene-id` ogranicza tagowanie/merge, lecz nie późniejsze sprzątanie.
Samodzielny tryb `--prune-empty-tags-only` nie przyjmuje `--scene-id` i nie wymaga
plików modeli. Połączone sprzątanie działa tylko po przebiegu bez błędów; przy
błędach można je uruchomić oddzielnie po dokończeniu synchronizacji.

Usuwanie pustych definicji nie kasuje zapisanych źródeł w SQLite. Jeżeli późniejszy
merge znów potrzebuje takiego tagu, może go odtworzyć. Nie uruchamiaj równolegle
innego taggera ani synchronizacji hierarchii podczas sprzątania: końcowy odczyt
i usunięcie są osobnymi operacjami API Stash. Blokada SQLite dotyczy tylko procesów
używających tego samego pliku stanu.

## Nowe nazwy i zgodność konfiguracji

Główny plik i miejsce ustawień to teraz `stash_ai.py`. Stary
`joytag_stash_markers.py` pozostaje krótkim launcherem, więc dotychczasowe
wywołania i zadania harmonogramu nadal działają po podmianie całej paczki.
Ustawienia edytuj w nowym głównym pliku i w `tagger_models.yml`.

| Poprzednia nazwa | Aktualna nazwa |
|---|---|
| `REPLACE_EXISTING_JT_TAGS` | `REPLACE_SCENE_TAGS` |
| `REPLACE_EXISTING_JT_MARKERS` | `REPLACE_MARKERS` |
| `ONLY_UNTAGGED` | `SKIP_EXISTING_OUTPUTS` — domyślnie `False` |
| `THRESHOLD` | `JOYTAG_DEFAULT_SCENE_THRESHOLD` |
| `INFERENCE_BATCH_SIZE` | `JOYTAG_DEFAULT_BATCH_SIZE` |
| `MARKER_THRESHOLD` | `DEFAULT_MARKER_THRESHOLD` |
| `HIERARCHY_FILE` | `JOYTAG_HIERARCHY_FILE` |
| `joytag_marker_core_exclusions.yml` | `marker_core_exclusions.yml` |

Stare nieużywane wspólne stałe prefiksów usunięto; identyfikatory źródeł są
zdefiniowane przy modelach. Nazwy i ścieżki dotyczące samego modelu JoyTag
pozostają, podobnie jak jego własny `joytag_hierarchy.yml`. Domyślna lokalizacja
bazy źródeł i modelowe identyfikatory własności markerów nie zmieniły się.

Nowy plik wykluczeń głównych tagów zachowuje wszystkie 32 dotychczasowe wpisy.
Jeżeli `marker_core_exclusions.yml` nie istnieje, skrypt obsłuży poprzednią nazwę.
Przy obecności obu plików wybierana jest nowa; własne późniejsze zmiany przenieś
do niej albo wskaż właściwy plik przez `--marker-core-exclusions`.

## Instalacja i wybór modeli

Rozpakuj całą paczkę do katalogu skryptu. Pliki `stash_taggers.py`, `stash_merge.py`,
`stash_progress.py`, `stash_prune.py`, `stash_exclusions.py`, `stash_cleanup.py`, `stash_cleanup_sql.py`, `stash_temporal.py`, `stash_fusion.py`,
**`stash_scene.py`, `stash_runtime.py`, `stash_performance.py`, `stash_preview.py`,
`stash_timeline.py`, `stash_merge_sql.py`** i YAML powinny pozostać
obok `stash_ai.py`. Nadpisz także blokadę dawnego modułu `stash_refine.py`.
Użyj istniejącego środowiska Python 3.10 lub nowszego. Nowe funkcje korzystają
z biblioteki standardowej i dotychczasowych zależności.

```powershell
python -m pip install -r requirements.txt

# Pobranie wszystkich czterech modeli bez połączenia ze Stash:
python .\stash_ai.py --models all --download-models

# Tylko JoyTag:
python .\stash_ai.py --models joytag --download-models

# Pobranie OppaiOracle V1.1 (około 993 MB wag):
python .\stash_ai.py --models oppai --download-models

# Sam OppaiOracle; najpierw podgląd:
python .\stash_ai.py --models oppai --scene-id 123 --dry-run
python .\stash_ai.py --models oppai --scene-id 123

# Podgląd dla jednej sceny; zamień 123 na jej ID:
python .\stash_ai.py --models all --scene-id 123 --dry-run

# Wykonanie dla tej sceny:
python .\stash_ai.py --models all --scene-id 123

# Wszystkie sceny:
python .\stash_ai.py --models all
```

Przykłady wyboru: `--models joytag`, `--models wd`, `--models dtq`,
`--models oppai`, `--models wd oppai`. Działa też `--models joytag,wd,dtq,oppai`
i alias `--model`. `--models all` obejmuje teraz wszystkie cztery modele.
Bez argumentu wybór pochodzi z `enabled_models` w `tagger_models.yml`;
domyślnie pozostaje `[joytag]`. Zmień na `[joytag, wd, dtq, oppai]`, aby zawsze
uruchamiać wszystkie cztery. Stary YAML bez sekcji `oppai` jest obsługiwany;
przy wyborze tego modelu używane są jego wartości domyślne.

Wszystkie cztery modele, również JoyTag, pobierają brakujące pliki automatycznie,
także przy zwykłym uruchomieniu. Wagi mają łącznie około 3,90 GB; pobieranie
pozostawia również kopię w cache Hugging Face. Paczka zawiera kod i konfigurację, bez wag.
`--offline` wymaga lokalnych plików. Rewizje modeli są przypięte w adapterach.
Istniejące pliki nie są nadpisywane. `download_missing: false` w YAML także
wyłącza pobieranie, w tym w trybie `--download-models`.

W `tagger_models.yml` ustawisz ścieżki, progi i wielkości paczek. Ścieżki
względne są liczone względem katalogu YAML. Zmienne `JOYTAG_MODEL_PATH`,
`JOYTAG_CODE_PATH`, `WD_MODEL_PATH`, `DTQ_MODEL_PATH` i `OPPAI_MODEL_PATH` mają pierwszeństwo.
Dotychczasowe domyślne ścieżki JoyTag oraz konfiguracja połączenia Stash są
zachowane. Połączenie można nadpisać przez `STASH_URL` i `STASH_API_KEY`.
`--model-config PATH` wybiera inny plik konfiguracji modeli.

Przesłany `config.env` dotyczy osobnego serwera AI; ten skrypt nie odczytuje
go automatycznie. Jego ustawienia są w sekcji CONFIG i `tagger_models.yml`.

## JoyTag z Hugging Face

JoyTag pobiera z `fancyfeast/joytag` pliki `top_tags.txt`, `config.json` oraz
`model.safetensors` (około 366 MB wag). Pobieranie działa zarówno przez
`--download-models`, jak i przy zwykłym tagowaniu. Sama walidacja lub synchronizacja
hierarchii potrzebuje wyłącznie słownika `top_tags.txt`.

Zachowany jest dotychczasowy loader PyTorch. `joytag.code_dir` lub `JOYTAG_CODE_PATH`
musi wskazywać istniejący katalog źródeł JoyTag z `Models.py` i jego modułami.
Źródła są w [repozytorium autora](https://github.com/fpgaminer/joytag);
wagi i słownik są pobierane z Hugging Face. Samo `--download-models` nie wymaga
loadera ani ładowania PyTorch. Do inferencji kod źródłowy jest nadal potrzebny.

Opcjonalnie zmień katalog wag, dodając `model_dir: models/joytag` w sekcji
`joytag` pliku `tagger_models.yml`; ustaw też `code_dir` na swój katalog źródeł.
Bez tych zmian działają dotychczasowe ścieżki skryptu i nadpisania środowiskowe.
Istniejący `model.pt` jest obsługiwany, jeśli nie ma obok `model.safetensors`;
skrypt nie pobiera wtedy drugiego formatu wag. Gdy oba istnieją, oryginalny
loader wybiera Safetensors.

| Model | Wagi używane przez ten skrypt |
|---|---|
| JoyTag | `model.safetensors`; także istniejący lokalny `model.pt` |
| WD EVA02-Large v3 | `model.onnx` |
| DanbooruTagQuery L/16 | `model.onnx` |
| OppaiOracle V1.1 | `model.onnx` |

Pliki msgpack nie są potrzebne. `--download-models` pobiera również metadane
wymagane przez wybrane modele.

## OppaiOracle V1.1

Opcja modelu to `oppai`, a kod źródła w tytule markera to krótki `op`.
Na przykład surowy tag `watermark` zostanie zapisany jako `ai:watermark`
(w trybie `separate`: `op:watermark`). W `by_model` obu blacklist
używaj klucza `oppai`, a we wpisach samych surowych nazw bez prefiksu.

Adapter używa wariantu **V1.1 ONNX, 448×448** z `Grio43/OppaiOracle/V1.1_onnx`.
Potrzebuje plików `model.onnx`, `vocabulary.json`, `selected_tags.csv`,
`config.json` i `preprocessing.json` w jednym `model_dir`. Domyślny katalog to
`models/oppai-oracle-v1.1` obok YAML. Wersje Safetensors i msgpack nie są używane.
Rewizja wszystkich pobieranych plików jest przypięta w `stash_taggers.py`.

Model zwraca 19 294 wyniki. Skrypt zachowuje **19 292 rzeczywiste etykiety**,
usuwając wyłącznie techniczne `<PAD>` i `<UNK>`. Zachowuje ratingi, emotikony
i rzadkie tagi. Nie ma dodatkowej whitelisty, filtra kategorii ani limitu top-k.
Przed inferencją sprawdzana jest zgodność słownika, kolejności CSV, liczby wyjść
i konfiguracji przygotowania obrazu.

Przygotowanie klatek odpowiada kodowi autora: RGB, EXIF, kompozycja przezroczystości
na szarym tle, zmniejszanie bilinear bez powiększania małych obrazów, dopełnienie
do 448×448 kolorem `(114,114,114)` i normalizacja `(x/255 - 0.5)/0.5`.
Drugie wejście ONNX, `padding_mask`, jest generowane dla każdej klatki;
`True` oznacza dopełnienie. Partia zawiera obrazy oraz ich odpowiadające maski.
Sigmoid jest już częścią grafu i nie jest stosowany drugi raz.

Progi startowe to `scene_threshold: 0.65` i `marker_threshold: 0.80`.
Edytuj je niezależnie w sekcji `oppai`. Są dobrane jako punkt wyjścia;
wyniki pewności nie są skalibrowane identycznie jak w WD lub DTQ.
`pr_thresholds.json` autora jest materiałem pomocniczym i nie jest wymagany:
skrypt nie odczytuje z niego automatycznie progów ani listy dozwolonych tagów.

Adapter stosuje poziom optymalizacji ONNX `BASIC`, a w razie błędu ładowania
próbuje wyłączenia optymalizacji. To obejście autora dla problemów z węzłem
`MemcpyToHost` na ścieżce maski bool. Obsługa CUDA/CPU pozostaje wspólna z innymi
modelami; `--device cuda` wymaga działającego GPU.

`--oppai-hierarchy PATH`, `--no-hierarchy`, oba tryby częściowe, cleanup,
`--dry-run`, wybór scen i `SKIP_EXISTING_OUTPUTS` działają również dla OppaiOracle.
Źródła są zachowane także po połączeniu wyników we wspólne tagi `ai:`.

## GPU i progi

`device: auto` preferuje CUDA. `--device cuda` wymaga działającego GPU i
zgłasza błąd zamiast przejść na CPU. `--device cpu` wymusza CPU. Log podaje
rzeczywiście wybrane urządzenie.

Zależności używają `onnxruntime-gpu>=1.21,<1.27` z rodziny CUDA 12/cuDNN 9.
Loader próbuje wczytać biblioteki CUDA z PyTorch i przez `preload_dlls()`.
Zachowaj działającą parę torch/torchvision i dopasuj główne wersje CUDA/cuDNN
do ONNX Runtime. [Tabela zgodności](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Instaluj tylko jeden pakiet: `onnxruntime-gpu` albo `onnxruntime`. Jeśli masz
pakiet CPU, usuń go przed instalacją wariantu GPU. Dla instalacji wyłącznie
CPU zastąp pozycję GPU w requirements przez `onnxruntime>=1.21,<2`.

| Model | Próg sceny | Próg markera | Batch |
|---|---:|---:|---:|
| JoyTag | 0.40 | 0.65 | 16 |
| WD | 0.35 | 0.65 | 4 |
| DTQ | 0.20 | 0.65 | 4 |
| OppaiOracle V1.1 | 0.65 | 0.80 | 4 |

To wartości startowe, do dopasowania do materiału. Przy braku pamięci GPU
zmniejsz `batch_size`. Pewności różnych modeli nie są wspólną skalą jakości.

Nie ma whitelisty ani listy preferowanych tagów. Pełne słowniki i wszystkie
kategorie są dostępne, w tym NSFW, oceny treści, postacie i serie, jeśli model
je przewiduje. Nie używamy listy ignorowanych tagów z przykładowego CLI DTQ.

Domyślne `MAX_TAGS`, `MARKER_MAX_TAGS_PER_FRAME` i `MAX_MARKERS_PER_SCENE`
wynoszą `0` — brak limitów. Progi pewności i `MARKER_MIN_HITS=2` nadal
obowiązują. Opcjonalne limity można ustawić ponownie w skrypcie; dotyczą
każdego przebiegu modelu osobno. Dodatkowe wspólne zakresy mają osobną pulę
ograniczoną również przez `MAX_MARKERS_PER_SCENE`; zachowane zakresy modeli nie
są z niej usuwane. Dlatego ten limit nie jest limitem wszystkich końcowych
markerów sceny. Wspólne kandydatury są wybierane według liczby silnych i wszystkich
trafień, a następnie czasu i etykiety, bez porównywania ocen różnych modeli.
Wykluczenia stosowane są przed limitami.
Przy migracji istniejące tagi i zakresy nie są przycinane limitami liczby wyników.
Agregat scen nadal jest liczony jako `topk_mean` z dwóch najlepszych klatek dla
tagu; dostępne jest również `max`. Domyślna kwalifikacja korzysta dodatkowo ze
statystyk opisanych wyżej; `--no-temporal` przywraca sam próg agregatu.

## Blacklisty i cleanup

`marker_core_exclusions.yml` zachowuje Twoje 32 wpisy. Blokuje
wyłącznie główny tag markera, wspólnie dla czterech modeli. Wykluczony tag
może pozostać dodatkowym tagiem markera i tagiem sceny.

`scene_tag_exclusions.yml` jest teraz **globalną blacklistą wyników**: blokuje
tagi całej sceny, główne i pomocnicze tagi markerów oraz tworzenie tych etykiet
podczas synchronizacji hierarchii, także z `--create-all-tags`. Nazwa pliku
i klucz `excluded_scene_tags` pozostają zgodne ze starszą konfiguracją.
Można też używać klucza `excluded_tags` oraz opcji `--tag-exclusions`;
w jednym YAML należy użyć tylko jednego z tych dwóch kluczy.
Domyślnie plik jest pusty. W obu YAML główna lista jest wspólna, a `by_model` dodaje
wykluczenia tylko dla `joytag`, `wd`, `dtq` lub `oppai`. Podawaj dokładne surowe
nazwy bez prefiksów. Nazwa nieobecna w słowniku danego modelu jest pomijana.

```powershell
python .\stash_ai.py --models all --cleanup-markers-only --dry-run
python .\stash_ai.py --models all --cleanup-markers-only
python .\stash_ai.py --models all --cleanup-scene-tags-only
python .\stash_ai.py --models all --cleanup-only
python .\stash_ai.py --models wd --cleanup-only --scene-id 123
```

Cleanup nie potrzebuje modeli, słowników, miniatur ani inferencji. Jeżeli
wykluczony model wspierał słabsze wykrycie innego, przeliczane jest również to
wsparcie, wyłącznie w obrębie już opublikowanych zakresów. Nowe, jeszcze
nieopublikowane wykrycia nie są przy tej okazji dodawane. Jeżeli źródła od
publikacji zostały zastąpione, najpierw wymagane jest `--merge-only`.
Dotyczy to również wspólnego ciągu: po odebraniu jednego źródła pozostałe modele
muszą nadal spełniać reguły kwalifikacji. Źródła dodatków i źródła tagu głównego
są traktowane osobno. Po skróceniu zakresu przeliczane są dowody dodatków oraz
oceny i trafienia wspólnych źródeł w jego zachowanych granicach. Cleanup korzysta
z reguł zapisanych przy publikacji; nie włącza nowych opcji we wcześniejszych
markerach. Do zastosowania aktualnych reguł służy `--merge-only`.
Wspólne tagi scen również są przeliczane po wykluczeniu użytego źródła.
Cleanup wymaga zachowania oryginalnych, opublikowanych rekordów zarówno scen,
jak i klatek; nowa inferencja któregoś z wymaganych źródeł wymaga najpierw
`--merge-only`. Nie są dobierane nowe, wcześniej nieopublikowane źródła.
Alias
`--sync-marker-core-only` nadal działa. Zwykły sync stosuje odpowiednią
blacklistę automatycznie na przetworzonych scenach, również przy wyłączonym
zastępowaniu starych wyników. `REPLACE_SCENE_TAGS` i `REPLACE_MARKERS`
sterują zastępowaniem poprzednich wyników danego modelu.

Globalna blacklista usuwa powiązania tagów ze sceną i dodatki markerów;
marker z zabronionym głównym tagiem jest usuwany, bez awansowania dodatku na główny.
Dotyczy to wskazanych nazw w przestrzeniach `ai:`, `jt:`, `wd:`, `dtq:` i `op:`,
także gdy taki tag został dopięty ręcznie. Ręczne tagi bez tych prefiksów pozostają.
Sama lista `marker_core_exclusions.yml` nadal dotyczy wyłącznie głównych tagów
markerów zarządzanych przez skrypt.
`--models all --cleanup-only` bez `--scene-id` domyślnie wybiera odczyt
samych identyfikatorów przez parametryzowane `SELECT` w API `querySQL` Stash.
Nie ma globalnych zliczeń markerów ani stron `OFFSET`; wykorzystywane są
istniejące indeksy oraz kursory ID/rowid. Wspólne globalne wykluczenia wyszukują
tylko tagi główne markerów, a usunięcie definicji tagu sprząta jego inne relacje.
Wspólne core-only nadal wymagają oznaczeń właściciela i zachowują definicję.

Zmiany zależne od konkretnego modelu kierowane są do scen wyszukanych po ID.
Tryby ograniczone do scen lub markerów mogą zbiorczo odpinać tagi przez `REMOVE`.
Mutacje nadal wykonuje oficjalne API, z ponowną kontrolą przed zapisem.
Brak SQL w trybie `auto` przełącza cleanup na przebieg po scenach.
`--cleanup-strategy sql` wymaga szybkiej ścieżki i zatrzymuje pracę, jeśli nie
jest dostępna. `--cleanup-strategy scenes` wymusza przebieg po scenach.
Dawna strategia `legacy` została usunięta.
Szczegóły zakresów, wznowienia, rozmiarów partii i pomiaru na 2,5 mln markerów
opisuje [UPGRADE_cleanup_sql.md](UPGRADE_cleanup_sql.md).

W pozostałych trybach sync/cleanup automatyczne sprzątanie definicji usuwa
tylko wykluczone, nieużywane tagi bez dzieci, ponownie sprawdzając ich liczniki.
Nie usuwa innych pustych tagów.
Opcjonalne `--cleanup-only --prune-empty-tags` sprząta także pozostałe puste
definicje we wskazanych przestrzeniach nazw.
W trybie `merged` cleanup usuwa wkład wykluczonego modelu:
jeżeli tę samą etykietę potwierdza inne źródło, tag pozostaje. Unia markerów
może się skrócić albo rozdzielić, jeżeli usunięte źródło łączyło dwa zakresy.
`--models wd --cleanup-only` stosuje nowe wykluczenia tylko do wkładu WD;
`--models all --cleanup-only` obejmuje wszystkie źródła.
Żeby etykieta całkowicie zniknęła z wyników wszystkich modeli, wpisz ją na
głównej liście globalnego YAML i uruchom cleanup dla `--models all`.
Tryby `--cleanup-markers-only` i `--cleanup-scene-tags-only` ograniczają zakres
zmian zgodnie z nazwą; nie zastępują pełnego cleanup.

Pełne sprzątanie zapisuje w SQLite małe reguły usunięć i numer generacji.
Opublikowany stan jest filtrowany dopiero przy odczycie; istniejące wyniki
i skompresowane snapshoty 9000 scen nie są masowo przepisywane.
Migracja formatu stanu z wersji 4 do 5 jest automatyczna. Szybkie globalne
tryby częściowe używają masek zależnych od rodzaju wyjścia i podnoszą format
do 6; po tej zmianie używaj nowej wersji skryptu.
Przerwany globalny cleanup wznów tą samą komendą i z tymi samymi blacklistami.
Skrypt ponownie wyszukuje pozostałe obiekty, więc nie odtwarza listy usuniętych
już identyfikatorów. Do zakończenia operacji blokuje inne zapisy korzystające
z tej bazy stanu. Wcześniej przerwany merge należy najpierw dokończyć przez
`--merge-only`. `--dry-run` symuluje zmiany w tymczasowej bazie SQLite, czyta istniejące wyniki stanu 5/6
bez kopiowania całej bazy i nie modyfikuje Stash
ani pliku bazy, również gdy podglądasz wznowienie.

Cleanup używa ostatnio opublikowanych wyników i nie przywraca ręcznie usuniętych
tagów ani markerów. Obejmuje też pozostające stare prefiksy. Nie usuwa bazowych
wykryć z SQLite: po zdjęciu blokady `--merge-only` może ponownie zbudować wynik
z zachowanych danych, bez inferencji. Zwykły merge stosuje aktualne blacklisty
wszystkich zapisanych źródeł. W trybie `separate` odtworzenie usuniętych wyników
wymaga nowej inferencji.

## Hierarchia i istniejące funkcje

Domyślny tryb `merged` składa cztery pełne hierarchie YAML w jedną przestrzeń
`ai:category:`. Kategorie i rodzice identycznych etykiet są łączeni; ogólne
`Other / Unclassified`, `General` i `Uncategorized` są pomijane dla tagu, jeśli
ma on bardziej szczegółową grupę. Wczytywane są wszystkie cztery lokalne YAML,
także gdy inferencję wykonuje tylko jeden model; nie wymaga to pobierania jego
nieaktywnych odpowiedników. `--no-hierarchy` pomija ten etap.

Oryginalny
plik JoyTag pozostaje bez zmian; trzy nowe pliki mają ten sam format i układ grup.

| Model | Domyślny plik | Pełny słownik | Prefiks w trybie `separate` |
|---|---|---:|---|
| JoyTag | `joytag_hierarchy.yml` | 5 813 | `jt:category:` |
| WD | `wd_hierarchy.yml` | 10 861 | `wd:category:` |
| DTQ | `dtq_hierarchy.yml` | 11 424 | `dtq:category:` |
| OppaiOracle | `oppai_hierarchy.yml` | 19 292 | `op:category:` |

Przykładowe gałęzie to `Appearance`, `Clothing`, `Poses and actions`, `Sexual content`,
`Image and style`, `Names and references` oraz `Objects`. Tag `blue_hair` trafia
pod `ai:category:Appearance / Hair / Color`. W trybie `separate` pozostają
dotychczasowe prefiksy modeli.

Każdy tag wspólny z JoyTag zachowuje dokładnie jego grupy. Pozostałe przypisania
korzystają z kategorii WD/DTQ, rozpoznawalnych nazw i reguł słownych. To propozycje
organizacji do edycji, a nie relacje wyuczone przez model. OppaiOracle ma jedną
kategorię CSV dla wszystkich wyjść, również nazw własnych, więc do ich organizacji
wykorzystano także wspólne słowniki. Niejednoznaczne wpisy są w
`Other / Unclassified`; ta grupa nie wyklucza ich z tagowania.

Hierarchie zawierają każdy rzeczywisty tag modelu, bez whitelisty i bez zmiany progów.
Możesz przenosić tagi między grupami albo powtarzać je w kilku grupach, aby nadać
kilku rodziców. Zachowaj dokładną pisownię surowych tagów oraz pełne pokrycie
słownika. Prefiksy i nazwy korzeni powinny pozostać stabilne po pierwszym użyciu.
Metadane zawierają liczność i skrót SHA-256 słownika; walidacja wykrywa brakujące
lub obce etykiety przed zapisami do Stash.

`--hierarchy PATH`, `--wd-hierarchy PATH`, `--dtq-hierarchy PATH` i
`--oppai-hierarchy PATH` wybierają własny YAML w tym samym formacie.
`--no-hierarchy` wyłącza synchronizację rodziców kategorii dla wybranych modeli.
Grupy markerów nadal są bezpośrednimi dziećmi `ai_markers` i wymagają obsługi hierarchii w Stash.

```powershell
python .\stash_ai.py --models all --validate-hierarchy
python .\stash_ai.py --models all --sync-hierarchy-only --dry-run
python .\stash_ai.py --models all --sync-hierarchy-only
python .\stash_ai.py --models all --sync-hierarchy-only --create-all-tags
python .\stash_ai.py --models all --no-hierarchy
python .\stash_ai.py --models all --markers-only
python .\stash_ai.py --models all --scene-tags-only
```

Tryby hierarchii potrzebują słowników, ale nie wag ani inferencji. Ręczni
rodzice tagów pozostają zachowani. `--create-all-tags` importuje pełne
słowniki wybranych modeli. Zwykły przebieg tworzy tagi dopiero, gdy są potrzebne.
Przy przejściu ze wcześniejszych ogólnych kategorii WD/DTQ/OppaiOracle wykonaj
podgląd `--sync-hierarchy-only --dry-run`, a potem synchronizację.
`parent_sync: replace_managed` zastępuje tylko rodziców zarządzanych przez
hierarchię danej przestrzeni; stare obiekty kategorii pozostają w Stash.
Kategorie są rodzicami, nie dodatkowymi przewidywanymi tagami scen.

Opcjonalne `SKIP_EXISTING_OUTPUTS=True` pomija istniejące wyniki bez sprawdzania
aktualności wejścia. Domyślnie jest wyłączone; do bezpiecznego wznowienia służy
domyślne `--resume`. Ta starsza reguła pomijania działa osobno dla każdego modelu
i włączonych typów wyjścia.
W trybie `merged` sprawdza bazę źródeł: udany przebieg bez trafień również
liczy się jako przetworzony. `--merge-only` zawsze publikuje zapisane wyniki.
Tryby cleanup ignorują tę opcję. Czasy VTT, tolerancja przerw, minimalna liczba
trafień i wykrywanie obsługi końca markera nadal obowiązują. Nowe reguły zmieniają
kwalifikację wykryć i dodatków; `--no-temporal` przywraca poprzedni mechanizm.

Sync odczytuje scenę ponownie przed zmianami. Marker jest zarządzany tylko przy
zgodnym tytule i tagu własności. W trybie `merged` udana inferencja zapisuje
wynik w SQLite, a po zakończeniu przebiegów następuje jedna synchronizacja
wspólnego wyniku dla sceny. Awaria modelu pozostawia jego poprzedni wkład;
pozostałe modele mogą kontynuować. Końcowy kod wyjścia jest niezerowy po błędach.

Nowe markery są zapisywane przed usunięciem starych. Stash nie zapewnia tu
jednej transakcji; przerwany zapis może być częściowy. Baza zapisuje zamiar
synchronizacji przed zmianami scen/markerów. Uruchom ponownie `--merge-only`,
żeby dokończyć zapis bez ponownej inferencji; cleanup wykrywa oczekującą
synchronizację i wymaga jej wcześniejszego dokończenia.

Uruchamiaj jeden proces naraz. Blokada bazy źródeł zapobiega dwóm jednoczesnym
zapisom przez ten sam plik stanu; nie blokuje innych skryptów ani edycji Stash.
`--dry-run` nie zapisuje Stash ani trwałej bazy źródeł. Dla formatów 5/6 czyta
źródło tylko do odczytu, a zmienione rekordy zapisuje w tymczasowej bazie SQLite,
usuwanej przy zamknięciu. Ograniczony cache stron pozwala użyć dysku tymczasowego
przy długim podglądzie. Starsze formaty wymagają kopii do tej bazy na czas migracji.
Zmiana źródłowej bazy przez inny proces przerywa podgląd; uruchom go ponownie.
Zwykły podgląd inferencji może pobrać pliki modeli, `--merge-only --dry-run` ich
nie potrzebuje. Licznik `processed` dla inferencji liczy pary model–scena,
`reused` liczy ponownie użyte wyniki, a `scenes_synced` końcowe scalenia. Gdy Stash obsługuje tylko punkty markerów,
pełne przedziały pozostają dostępne w bazie i raporcie źródeł.

## Weryfikacja i źródła

Przechodzi **406 testów** w tej wersji. Obejmują izolację modeli, idempotencję, awarie, tryb podglądu, obie
blacklisty, tryby częściowe, hierarchię, agregację i zachowanie czasów markerów.
Nowe testy sprawdzają dokładną zgodność wznowienia z nieprzerwanym przebiegiem,
Ctrl+C i awarie API, wymuszony restart wielu modeli, zmiany wejścia pod tym samym
URL, progi, częściowe wyjścia, brak zapisu przy podglądzie i ograniczenie kolejki
przygotowywanych klatek. Testy sprzątania obejmują wszystkie liczniki użycia,
rekurencję rodziców, paginację, zmianę tagu przed usunięciem, brak wymaganych
możliwości API i uruchamianie bez modeli. API Stash i inferencja są w tych testach
zastąpione atrapami. Nie wykonywano zapisu do Twojego Stash ani testu GPU.
Kod adapterów modeli nie zmienił się w tej wersji.

41 nowych testów obejmuje wspólne tagi scen, osobne progi i wykluczenia,
częściowy cache bez wymyślania brakujących ocen, zgodność inferencji oraz
cleanup i zachowanie ręcznych usunięć. Sprawdzane są krótkie dodatki, ciągi
przechodzące przez granicę unii, cache w bajtach, unieważnianie po zapisie
i rollback oraz prefetch 0/2/4/8 z dokładnym wznowieniem.

Test zbiorczego podglądu przetwarza **9000 syntetycznych pustych scen** i
sprawdza raport najwyżej 15 wierszy oraz brak trwałych zmian. Drugi test
grupuje 9000 powtarzalnych błędów przy stałym limicie przykładów. Dla 30
etykiet i dwóch modeli o takich samych zakresach test potwierdza jedno
obliczenie dodatków oraz identyczny wynik przy wyłączonym cache. Sprawdzenie
100 par scena–model po wczytaniu małego indeksu nie wykonuje kolejnych zapytań
SQLite. To testy poprawności i kosztu operacji; nie są pomiarem czasu
inferencji Twojej kolekcji ani dokładności tagowania na Twoich filmach.

42 testy reguł automatycznych obejmują histerezę i jej limit, dodatki z całego
przedziału, pokrycie po unii bez podwójnego liczenia modeli, wykluczenia głosów
i cleanup, trzy drogi kwalifikacji scen, walidację ustawień oraz zgodność
wznowionych statystyk z ciągłą analizą.

32 testy wspólnego przetwarzania sprawdzają naprzemienne wykrycia, zgodność
wejścia i czasu, osobne progi, liczenie różnych czasów, ograniczenie histerezy,
zachowanie dotychczasowych zakresów i limity. Obejmują dodatki z modeli bez
tagu głównego, ich pochodzenie, pokrycie po unii, niezależne przełączniki,
cleanup także po zmianie źródła rozpoczynającego zakres i brak przyjmowania
nieopublikowanych głosów. Testy całego przebiegu potwierdzają działanie
`--merge-only` bez inferencji i pobierania obrazów, niezmienność zapisanych
wyników źródłowych, idempotencję oraz brak zmian przy `--dry-run`.

16 testów wycofanego etapu wideo sprawdza odrzucenie starych flag i YAML,
blokadę importu starego modułu, odczyt wyłącznie VTT/PNG ze Stash, pomijanie
ścieżek filmów w obu trybach, wznowienie wyników i checkpointów bazy v3,
migrację bez zmian przy `--dry-run`, raport źródeł i obsługę dawnych zapisanych
polityk. Testy blokują uruchamianie programów zewnętrznych oraz próby otwierania
lub sprawdzania plików filmów. Ponowne dopisanie identycznego wyniku do historii jest
porównywane w formacie JSON, aby różnica tuple/list nie tworzyła duplikatów.

Wcześniejsze testy sprawdzają również pobieranie JoyTag z przypiętej
rewizji, użycie plików offline, zachowanie lokalnych wag oraz przygotowanie plików
przed oryginalnym loaderem. Testy pobierania podstawiają lokalne pliki zamiast połączenia z HF;
nie uruchamiają inferencji JoyTag. Pełne YAML zweryfikowano też względem rzeczywistych
słowników z przypiętych rewizji. Sprawdzono domyślne ścieżki, zgodność grup
wspólnych z JoyTag i ponawianie migracji bez zmian ręcznych rodziców.

Rzeczywisty model OppaiOracle V1.1 ONNX
uruchomiono na CPU w ONNX Runtime 1.26.0: partia dwóch klatek zwróciła macierz
`(2, 19292)`, a porównanie z osobnymi wywołaniami dla tych samych klatek dało
maksymalną różnicę `0.0`. Przygotowanie obrazu i maski było dokładnie zgodne
z funkcją autora dla 10 przypadków, w tym przezroczystości RGBA/LA/palety,
orientacji EXIF, małych klatek i skrajnych proporcji.

W toku prac oba rzeczywiste modele ONNX uruchomiono na CPU: WD zwrócił
10 861 ocen tagów, DTQ 11 424. Wyniki pojedynczych klatek i paczek były zgodne.
Przygotowanie obrazu DTQ zgadzało się dokładnie z kodem autora dla pięciu
proporcji obrazu. Nie wykonywano zapisów do Twojego Stash ani testu Twojej
konfiguracji CUDA. Testy nie mierzą trafności tagów NSFW na Twoich scenach.

```powershell
python -m pip install pytest
python -m pytest -q
```

- [JoyTag: pliki modelu](https://huggingface.co/fancyfeast/joytag/tree/6b7f16331a6ccf0fdce37d5a9564715f6e772b22)
  i [loader autora](https://github.com/fpgaminer/joytag/blob/main/Models.py):
  pobierane są Safetensors, konfiguracja i słownik; zachowany jest lokalny loader PyTorch.
- [WD: model](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
  i [przykład autora](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py):
  BGR 0–255, białe dopełnienie przed bicubic; ONNX zwraca prawdopodobieństwa.
- [DTQ: model](https://huggingface.co/realphongha/danbooru-tag-query),
  [transformacje](https://github.com/realphongha/DanbooruTagQuery/blob/master/src/transforms.py),
  [inferencja](https://github.com/realphongha/DanbooruTagQuery/blob/master/src/onnx_utils.py):
  RGB, bilinear przed czarnym dopełnieniem, ImageNet; sigmoid logitów dokładnie raz.
- [OppaiOracle V1.1 ONNX](https://huggingface.co/Grio43/OppaiOracle/tree/96992fa30568c386e9fe7c8a1a68f798a3202c09/V1.1_onnx),
  [specyfikacja przygotowania](https://huggingface.co/Grio43/OppaiOracle/blob/96992fa30568c386e9fe7c8a1a68f798a3202c09/V1.1_onnx/preprocessing.json)
  i [kod autora](https://huggingface.co/spaces/Grio43/OppaiOracle/blob/main/app.py).

- [Stash: schemat tagów](https://github.com/stashapp/stash/blob/develop/graphql/schema/types/tag.graphql)
  i [liczenie markerów po tagu](https://github.com/stashapp/stash/blob/develop/pkg/sqlite/scene_marker.go):
  sprzątanie sprawdza liczniki użycia oraz dzieci; licznik markerów obejmuje tag
  główny i dodatkowe.

# Stash AI — aktualny podręcznik

Ten dokument opisuje bieżące zachowanie aplikacji. Szybki start znajduje się w
[README](../README.md), a szczegóły wybranych obszarów są podlinkowane na końcu.

## 1. Model działania

Stash AI analizuje VTT i sprite'y generowane przez Stash. Nie otwiera ani nie
dekoduje plików wideo.

Obsługiwane modele:

| Model | Klucz CLI | Prefiks w trybie `separate` |
|---|---|---|
| JoyTag | `joytag` | `jt:` |
| WD EVA02-Large v3 | `wd` | `wd:` |
| DanbooruTagQuery L/16 | `dtq` | `dtq:` |
| OppaiOracle V1.1 | `oppai` | `op:` |

Modele wykonują inferencję kolejno. W pamięci GPU znajduje się jeden model naraz;
przygotowanie klatek może działać równolegle na CPU.

Domyślny `--output-mode merged` publikuje wspólne tagi `ai:` i scala zgodne
wyniki modeli. `--output-mode separate` zachowuje prefiksy źródłowe i nie używa
mechanizmu wspólnego resume/merge.

## 2. Dane wejściowe i baza stanu

Domyślna baza `tagger_sources.sqlite3` jest częścią trwałego stanu aplikacji.
Przechowuje m.in.:

- surowe dowody i agregaty inferencji;
- checkpointy częściowych przebiegów;
- snapshoty opublikowanego stanu;
- indeksy potrzebne do resume i szybkiego merge;
- dane cleanupu i recovery.

Baza nie zawiera obrazów ani klucza API. Powinna być backupowana razem z
konfiguracją.

Aplikacja obsługuje migrację starszych wspieranych formatów bazy automatycznie.
Po użyciu nowszej funkcji recovery/cleanup nie uruchamiaj starszej kopii programu
na tej samej bazie.

## 3. Resume, restart i checkpointy

W trybie `merged` resume jest domyślnie włączone.

```powershell
python .\stash_ai.py --models all
```

Dla każdej pary scena/model aplikacja sprawdza tożsamość modelu, konfigurację i
wejście. Gdy zapisany wynik jest zgodny, inferencja jest pomijana. Częściowy
checkpoint wznawia się od ostatniej bezpiecznie zapisanej partii.

`--restart` wymusza świeżą inferencję wybranego zakresu:

```powershell
python .\stash_ai.py --models wd --scene-id 123 --restart
```

Nie używaj `--restart` do:
- ponownego merge zapisanych wyników;
- cleanupu;
- zmiany progów z cache;
- naprawy hierarchii lub grup markerów.

Do tych zadań istnieją tryby bez inferencji.

Checkpointy są zapisywane domyślnie co 8 batchy oraz przy kontrolowanym
przerwaniu/błędzie. `--checkpoint-batches` zmienia częstotliwość.

## 4. Merge i cache publikacji

Merge łączy zapisane dowody modeli z bieżącym stanem Stash. Może działać bez
ładowania modeli:

```powershell
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run
```

Po udanej publikacji zapisywany jest mały `merge_check`. Jeśli kod/reguły,
katalog tagów i stan sceny są zgodne, kolejny przebieg może pominąć dekodowanie
dowodów i planowanie publikacji.

Zmiana kodu lub polityki publikacji może jednorazowo unieważnić ten cache i
spowodować re-merge. Nie oznacza to automatycznie ponownej inferencji.

`--force-merge` wymusza przeliczenie merge zapisanych wyników i wymaga
`--merge-only`:

```powershell
python .\stash_ai.py --merge-only --scene-id 123 --force-merge
```

## 5. Replace vs historia wyników

Domyślnie:

```text
REPLACE_SCENE_TAGS = True
REPLACE_MARKERS = True
```

Przy `True` nowy poprawny wynik modelu zastępuje poprzedni wynik tego samego
rodzaju. Przy `False` aplikacja zachowuje wcześniejsze przebiegi jako historyczne
dowody i merge może je uwzględniać.

Po ponownym ustawieniu `True`, jeżeli bieżący wynik inferencji można
jednoznacznie ponownie wykorzystać, aplikacja zwija zachowaną historię danego
rodzaju bez uruchamiania modelu. Zmiana polityki publikacji nadal powoduje
odpowiednie uzgodnienie Stash.

## 6. Progi i rethreshold bez inferencji

Progi modeli są w `tagger_models.yml`.

### Markery

```powershell
python .\stash_ai.py --models all --merge-only --rethreshold-markers
```

Operacja:
- nie uruchamia modeli;
- używa zachowanych wyników klatek;
- odrzuca zmianę, jeśli retencja jest niewystarczająca;
- nie zgaduje brakujących danych.

### Tagi scen

```powershell
python .\stash_ai.py --models all --merge-only --rethreshold-scenes
```

Można połączyć oba tryby:

```powershell
python .\stash_ai.py --models all --merge-only --rethreshold-scenes --rethreshold-markers
```

Zmiana progu sceny wymaga zapisanych obserwacji pozwalających ponownie obliczyć
kwalifikację. Starszy lub niepełny cache może wymagać świeżej inferencji danej
pary scena/model; `--merge-only` jej nie uruchomi, tylko zgłosi problem.

Zwykły przebieg z resume również potrafi wykorzystać cache po bezpiecznej zmianie
progów, jeśli wszystkie wymagane obserwacje są dostępne.

## 7. Reguły temporalne

`temporal_policy.yml` steruje automatycznym postprocessingiem:

- `shared_marker_frames` — wspólna kwalifikacja zgodnych klatek modeli;
- `shared_marker_extras` — dodatki z innych modeli;
- `short_marker_extras` — krótkie spójne epizody dodatków;
- `shared_scene_tags` — dodatkowa kwalifikacja tagów scen;
- `occurrence_markers` — reprezentatywne punkty dla tagów scen bez markera;
- histereza kontynuacji markerów;
- wymagane liczby trafień i pokrycie dodatków.

`--no-temporal` wyłącza te reguły i przywraca prostsze zachowanie bazowe.

Wyniki modeli nie są uśredniane do jednego wspólnego prawdopodobieństwa. Źródła
i pewności pozostają rozdzielone.

## 8. Tagi scen, markery i Occurrence

W trybie `merged` wspólna etykieta jest publikowana jako `ai:<label>`.

Markery zakresowe zachowują start/koniec wynikający z próbek VTT. Marker
Occurrence to pojedynczy reprezentatywny punkt dla zaakceptowanego tagu sceny,
który nie ma własnej lokalizacji zakresowej.

Occurrence:
- jest kontrolowane przez `occurrence_markers`;
- używa wyłącznie zapisanych czasów/obserwacji;
- nie zgaduje czasu, jeśli cache go nie zawiera;
- respektuje wykluczenia i limit liczby markerów;
- ma techniczny tag `ai:marker:occurrence`.

Szczegóły: [UPGRADE_occurrences.md](UPGRADE_occurrences.md).

## 9. Wykluczenia

Dwa główne pliki:

- `scene_tag_exclusions.yml` — globalne wykluczenia scen, markerów, dodatków i
  tworzenia hierarchii;
- `marker_core_exclusions.yml` — wykluczenia głównych tagów markerów.

Wykluczenia są stosowane podczas normalnego merge oraz w trybach cleanupu.

```powershell
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only
```

Dostępne są też:
- `--cleanup-markers-only`;
- `--cleanup-scene-tags-only`.

Cleanup nie wymaga modeli ani ponownego pobierania sprite'ów.

## 10. Cleanup SQL

`--cleanup-strategy auto` próbuje użyć read-only `querySQL`, a gdy to
niemożliwe, przechodzi na odczyt scen. `sql` i `scenes` wymuszają konkretną
ścieżkę.

SQL służy wyłącznie do odczytu ID i stanu potrzebnego do cleanupu. Wszystkie
zmiany w Stash wykonują jego API/hooki.

```powershell
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql
```

Rozmiary partii:
- `--cleanup-page-size 2000`;
- `--cleanup-write-batch-size 200`.

Mutacje GraphQL nie są automatycznie retry'owane. Po utracie odpowiedzi wznowienie
tej samej operacji ponownie odczytuje aktualny stan i kontynuuje z pozostałymi
obiektami.

Szczegóły: [UPGRADE_cleanup_sql.md](UPGRADE_cleanup_sql.md).

## 11. Pending operations i recovery

Publikacja zapisuje lokalny plan przed wykonaniem mutacji Stash. Dzięki temu
przerwany merge/cleanup można bezpiecznie wznowić.

Jeśli chcesz użyć zmienionej polityki zamiast dokończenia starego planu:

```powershell
python .\stash_ai.py --abandon-pending
```

Można też porzucić stary plan i od razu wykonać nową operację:

```powershell
python .\stash_ai.py --merge-only --abandon-pending
python .\stash_ai.py --models all --cleanup-only --abandon-pending
```

Porzucenie nie odtwarza zmian, które Stash już wykonał. Zachowuje wyniki
inferencji i informacje potrzebne do późniejszego uzgodnienia własności.

Szczegóły: [UPGRADE_abandon_pending.md](UPGRADE_abandon_pending.md).

## 12. Hierarchie

Każdy model ma własny plik `*_hierarchy.yml`. W trybie merged kategorie są
publikowane pod wspólnymi tagami `ai:category:...`.

```powershell
python .\stash_ai.py --sync-hierarchy-only --dry-run
python .\stash_ai.py --validate-hierarchy
```

`--create-all-tags` działa tylko z `--sync-hierarchy-only`.

`--no-hierarchy` wyłącza synchronizację kategorii modeli, ale nie wyłącza grup
markerów `m:`.

## 13. Grupy markerów

Każdy zarządzany marker `ai:` może otrzymać dodatkowy tag grupy:

- `m:<core>` dla normalnego markera;
- `m:<core>:occurance` dla Occurrence.

Korzeń to `ai_markers`. Grupy nie są obiektami Stash Group — są tagami
organizacyjnymi markerów.

Naprawa istniejących markerów:

```powershell
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run
python .\stash_ai.py --sync-marker-groups-only
```

Szczegóły: [UPGRADE_marker_groups.md](UPGRADE_marker_groups.md).

## 14. Dry-run

`--dry-run` nie zapisuje zmian do Stash ani trwałej bazy stanu. Dla nowoczesnej
bazy źródło jest otwierane read-only, a planowane zmiany trafiają do tymczasowego
overlay SQLite.

Dry-run może nadal:
- odczytywać Stash;
- pobrać brakujące zasoby modeli w trybach, które ich wymagają;
- wykonywać kosztowne obliczenia merge.

Do testowania konfiguracji przed dużym przebiegiem używaj małego zestawu
`--scene-id`.

## 15. Sieć i retry

Klient ma osobne zasady dla readów i write'ów:

- media GET korzystają z retry na błędy przejściowe;
- read-only GraphQL retry'uje timeout/connection errors oraz HTTP
  429/500/502/503/504;
- prawdziwe mutacje GraphQL są wykonywane tylko raz.

To rozróżnienie jest celowe: po timeoutcie mutacji nie można bezpiecznie założyć,
że serwer niczego nie zapisał.

Redirecty mediów są dozwolone tylko w obrębie tego samego originu Stash, aby nie
wysłać `ApiKey` do innego hosta. GraphQL redirect jest traktowany jako błąd
konfiguracji `STASH_URL`.

## 16. Pamięć i przygotowanie klatek

Domyślne wartości:

| Opcja | Wartość |
|---|---:|
| `--ram-budget-gb` | 32 |
| `--ram-cache-gb` | 24 |
| `--prepare-workers` | 4 |
| `--prefetch-batches` | 4 |

Cache jest przydzielany na żądanie i ograniczony do maksymalnie 75% budżetu RAM.
Budżet nie jest twardym limitem procesu — model, aktywne tensory, Python i SQLite
również zużywają pamięć.

Wspólna pamięć wejścia pozwala kolejnym modelom używać tych samych pobranych
VTT/sprite'ów podczas jednego przebiegu, o ile dane nadal mieszczą się w cache.

Szczegóły: [UPGRADE_performance.md](UPGRADE_performance.md).

## 17. Diagnostyka merge

Do wymuszonego pomiaru zapisanych wyników bez inferencji:

```powershell
python .\stash_ai.py --merge-only --scene-id 123 --force-merge --profile-merge merge-profile.jsonl
```

`--profile-merge` zapisuje JSONL z czasami etapów, API i licznikami. Plik musi
być nowy. `--verbose` dodaje szczegółowe informacje do konsoli.

Jeżeli kolejny identyczny przebieg nie pokazuje oczekiwanego `merge_reused`,
sprawdź pole `reason` i liczniki `merge_*`.

Szczegóły: [UPGRADE_diagnostics.md](UPGRADE_diagnostics.md).

## 18. Aktualizacje aplikacji

`update_stash_ai.py`:
- pobiera zarządzane pliki root z jednego SHA `main`;
- używa fine-grained `GH_TOKEN` / `GITHUB_TOKEN`;
- nie uruchamia `pip`;
- nie aktualizuje modeli, bazy ani `.env`;
- chroni lokalnie zmienione pliki;
- ma blokadę instalacji i recovery.

```powershell
python .\update_stash_ai.py --check
python .\update_stash_ai.py --dry-run
python .\update_stash_ai.py
python .\update_stash_ai.py --recover
```

Lokalnie zmienione YAML-e są zachowywane, a nowa propozycja trafia do
`.incoming`. `--overwrite-configs` pozwala jawnie przyjąć wersję z repo.
Zmodyfikowany kod wymaga `--force-code`.

## 19. Bezpieczeństwo kluczy

Ustaw własny `STASH_API_KEY` w environment. Aktualny kod nadal zawiera
historyczny fallback, który należy traktować jako skompromitowany; szczegóły
są w [SECURITY.md](../SECURITY.md).

Token GitHub updatera powinien mieć wyłącznie dostęp do tego repozytorium i
`Contents: read`.

## 20. Dokumenty szczegółowe

- [Cleanup SQL](UPGRADE_cleanup_sql.md)
- [Wydajność i pamięć](UPGRADE_performance.md)
- [Merge i szybka ścieżka](UPGRADE_merge.md)
- [Wykluczenia i cleanup](UPGRADE_exclusions.md)
- [Occurrence](UPGRADE_occurrences.md)
- [Grupy markerów](UPGRADE_marker_groups.md)
- [Porzucanie pending](UPGRADE_abandon_pending.md)
- [Profilowanie merge](UPGRADE_diagnostics.md)

Nazwy `UPGRADE_*.md` są zachowane dla stabilności linków. Dokumenty opisują
bieżące zachowanie, a nie kolejność historycznych wdrożeń.

# Grupy markerów

W trybie `merged` markery o tym samym głównym tagu dostają dodatkowy tag grupy.
Grupowanie nie scala przedziałów czasowych i nie tworzy obiektów Stash `Group`.
Do przeglądania grupy użyj filtra dodatkowych tagów markerów w Stash.

| Główny tag | Typ markera | Dodatkowy tag grupy | Bezpośredni rodzic tagu grupy |
|---|---|---|---|
| `ai:abc` | zwykły, również bez końca przedziału | `m:abc` | `ai_markers` |
| `ai:abc` | punkt wykrycia (Occurance) | `m:abc:occurance` | `ai_markers` |
| `ai:def` | zwykły | `m:def` | `ai_markers` |

Pisownia `:occurance` jest celowa. Dotychczasowy techniczny tag typu
`ai:marker:occurrence` zachowuje swoją nazwę. Typ wynika z danych markera,
a przy uzupełnianiu istniejących markerów z tego technicznego tagu; brak
`end_seconds` sam w sobie nie oznacza Occurance.

Każdy opublikowany marker ma jedną zarządzaną grupę wyliczoną z jego końcowego
core i typu. Dodatkowe etykiety markerów nie tworzą grup. Grupa jest wspólna
dla wszystkich scen, ale grupy zwykłych markerów i punktów wykrycia są niezależne.
Tagi `m:` nie są dopisywane do scen ani do dowodów z modeli i nie zmieniają
tagu własności `ai:marker:stash-taggers`.

## Nowe i istniejące markery

Zwykłe uruchomienie i `--merge-only` automatycznie nadają grupy podczas
publikacji. Ponowne uruchomienie nie duplikuje tagów ani markerów. Zmiana core
lub typu usuwa poprzednie zarządzane przypisanie; ręczne dodatki pozostają.

Aby uzupełnić grupy bez inferencji, odtwarzania źródeł i zmiany czasów:

```powershell
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run --verbose
python .\stash_ai.py --sync-marker-groups-only --scene-id 123

# Wszystkie sceny
python .\stash_ai.py --sync-marker-groups-only
```

Ten tryb zachowuje ID, tytuł, core, początek i koniec markera. Obejmuje tylko
markery `ai:` należące do skryptu; nie modyfikuje ręcznych markerów ani wyników
`jt:`, `wd:`, `dtq:`, `op:`. Nie potrzebuje wag, słowników, polityk temporalnych
ani plików hierarchii. Używa tej samej blokady `--state-db` co merge i cleanup.
Jeżeli poprzednia publikacja została przerwana, najpierw należy wznowić
`--merge-only` na tej samej bazie. Przerwane uzupełnienie grup można powtórzyć.

`--no-hierarchy` wyłącza tylko hierarchie kategorii modeli. Grupy nadal są
dziećmi `ai_markers`. Tryb `separate` nie nadaje grup `m:`.

## Cleanup

Markery nadal podlegają istniejącym blacklistom i cleanupowi. Po udanej
publikacji markerów, cleanupie markerów lub uzupełnieniu grup skrypt usuwa
zarządzane tagi grup, które są globalnie nieużywanymi liśćmi. Dotyczy to także
obu grup po wykluczeniu core oraz starej grupy po zmianie typu.

Sprawdzane są wszystkie liczniki użycia udostępniane przez Stash, także poza
aktualną sceną i poza markerami, oraz dzieci tagu. Tag używany na innej scenie,
obrazie lub innym obiekcie pozostaje. `ai_markers` zawsze zostaje. Sprzątanie
pustych definicji jest globalne również przy `--scene-id`; przypisania markerów
na innych scenach pozostają bez zmian.

Obsługiwane są strategie cleanupu `sql`, `legacy` i `scenes`. `--dry-run`
uwzględnia planowane zmiany przypisań grup przy obliczaniu pustych tagów,
nie zapisując zmian w Stash ani w bazie źródeł. Błąd publikacji lub uzupełnienia
grup wstrzymuje końcowe sprzątanie grup. Cleanup tylko tagów scen nie zmienia
grup markerów.

Samodzielny `--prune-empty-tags-only` również uwzględnia zarządzane `m:`
w domyślnych prefiksach. Można ograniczyć go do grup:

```powershell
python .\stash_ai.py --prune-empty-tags-only --prune-prefixes m: --dry-run --verbose
```

## Własność i kolizje nazw

Nowe grupy mają opis zaczynający się od `[Stash AI marker group v1]`, zawierający
core i typ, oraz `ignore_auto_tag=true`. Nowo utworzony `ai_markers` również
ma wyłączony auto-tagging. Istniejący korzeń jest używany bez przejmowania
własności. Ręcznie dodani inni rodzice zarządzanej grupy są zachowywani.

Sam prefiks `m:` nie upoważnia do zmiany lub usunięcia tagu. Gdy wymagana nazwa
jest zajęta przez tag bez oznaczenia własności, skrypt zgłasza kolizję dla sceny.
Przed ponowieniem można zmienić nazwę istniejącego ręcznego tagu.
Usunięcie oznaczenia z opisu wyłącza automatyczne zarządzanie tym tagiem.

Konwencja ma jedno ograniczenie: zwykły core `abc:occurance` koliduje nazwą
grupy z punktem wykrycia dla core `abc`. Taki konflikt także powoduje błąd
zamiast połączenia dwóch różnych grup.

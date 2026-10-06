# Odbudowa nowej bazy: `--db-init`

Opcja pozwala rozpocząć od nowej bazy SQLite, mimo że sceny w Stash mają już
wspólne tagi i markery AI. Przydaje się po utracie bazy albo gdy świadomie
chcesz przeliczyć kolekcję od początku z aktualnych VTT i sprite'ów.

## Uruchomienie i zakres

Wskaż nowy plik zamiast dotychczasowej bazy:

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3
```

Możesz dodać `--scene-id 123`, powtarzając opcję dla kolejnych scen.
Zestaw modeli i selektor scen zostają zapisane w bazie. Pierwszy udany skan
zamyka listę scen odbudowy; flaga nie daje stałego prawa do przejmowania
wyników nowych scen.

Opcja wymaga nowej bazy albo aktualnej bazy zawierającej wyłącznie pusty
schemat. Odrzuca bazę z wynikami, checkpointami, snapshotami, recovery lub
innym stanem, również dla innego adresu Stash. Nie czyści ani nie nadpisuje
takiej bazy. Zmiana adresu Stash, modeli lub zakresu odbudowy wymaga osobnej
bazy. Stary plik możesz zachować do powrotu do poprzedniego stanu.

Zalecane jest `--models all`. Wybranie np. tylko `wd` zastępuje wcześniejsze
wyniki AI nowymi wynikami WD, także gdy wcześniej powstały z kilku modeli.
Baza otrzymuje rzeczywiste dowody świeżej inferencji, a nie wyniki odtworzone
z nazw istniejących tagów. Nowe lub dokładniejsze VTT są analizowane jako
aktualne wejście; zmiana wejścia podczas wznowienia unieważnia niezgodny cache.

## Przejęcie wyników i publikacja

| Element | Zachowanie podczas odbudowy |
|---|---|
| Przypisania `ai:<label>` do sceny | Przejmowane; nieaktualne przypisania są usuwane po ukończeniu inferencji. Dotyczy także ręcznie przypisanych tagów z tym prefiksem. |
| Kategorie `ai:category:*` i zarezerwowane tagi markerów | Nie są przejmowane jako etykiety scen. |
| Markery `[AI] ` z tagiem `ai:marker:stash-taggers` | Synchronizowane według nowych wyników. |
| Tagi spoza `ai:` i markery bez własności aplikacji | Podlegają dotychczasowym regułom publikacji i globalnych wykluczeń. |
| Błąd modelu, brak VTT/sprite'a lub pusty VTT | Scena pozostaje niedokończona; dotychczasowe wyniki nie są zastępowane. |
| Udana inferencja bez wykryć | Jest ukończonym wynikiem; stare przejęte wyniki AI mogą zostać usunięte. |

Scena jest publikowana dopiero po ukończeniu obu wyjść wszystkich wybranych
modeli. Inne gotowe sceny mogą zostać opublikowane, nawet gdy jedna ma błąd.
Własność tagów jest zapisywana przed zmianami w Stash. Publikacja korzysta ze
zwykłego trwałego planu: błąd API może pozostawić część zmian w Stash, ale baza
zachowuje plan i własność do bezpiecznego ponowienia. Oznaczenie sceny jako
odbudowanej zatwierdzane jest razem ze snapshotami obu wyjść.

Zwykłe markery AI bez natywnego `end_seconds` nadal wymagają migracji opisanej
w README. `--db-init` nie zastępuje migracji starych przestrzeni nazw ani
geometrii markerów.

## Wznowienie i podgląd

Po przerwaniu wystarczy ta sama komenda bez flagi:

```powershell
python .\stash_ai.py --models all --state-db .\tagger_sources_fresh.sqlite3
```

Zachowaj pierwotny zestaw modeli i ewentualne `--scene-id`. Ponowienie
`--db-init` też nie resetuje postępu. Ukończone inferencje są używane ponownie,
a zgodne checkpointy kontynuowane. Jeśli inferencja jest już kompletna i
pozostał wyłącznie niedokończony zapis do Stash, możesz wznowić publikację
przez `--merge-only` bez `--db-init`.

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3 --dry-run
python .\stash_ai.py --db-report --state-db .\tagger_sources_fresh.sqlite3
```

Podgląd działa w tymczasowej nakładce: nie tworzy wskazanego pliku bazy i nie
zmienia istniejącego, nie zapisuje do Stash. Może pobrać modele i wykonać
inferencję. Podsumowanie przebiegu i `--db-report` pokazują liczbę scen
odbudowanych i oczekujących; `--db-check` sprawdza zapisany stan odbudowy.

Pierwsze uruchomienie flagi wymaga normalnej inferencji obu wyjść. Nie łącz
jej z `--merge-only`, wyjściami częściowymi, cleanupem, raportami, `--restart`
ani `--abandon-pending`. Niedokończona odbudowa blokuje również zmianę
zakresu, wyjścia częściowe, restart, porzucenie planu i rethreshold w zwykłym
przebiegu inference/merge.

Schemat SQLite ma wersję v24. Istniejące bazy v18–v23 aktualizuj przez
`--db-update`; migracja zachowuje stan i nie uruchamia przejęcia wyników AI.

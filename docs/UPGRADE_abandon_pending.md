# Porzucanie przerwanych operacji

`--abandon-pending` pozwala porzucić niewykonaną część cleanupu lub merge i użyć
zmienionych YAML-i. Nie trzeba najpierw kończyć starego cleanupu.

Najpierw zatrzymaj działający proces, np. przez `Ctrl+C`. Opcja nie zatrzymuje
innego procesu ani wywołania, które zostało już wysłane do Stash. Nadal obowiązuje
blokada jednego procesu zapisującego do danej bazy stanu.

## Użycie

```powershell
# Po zatrzymaniu procesu i edycji YAML: podgląd nowego cleanupu
python .\stash_ai.py --models all --cleanup-only --abandon-pending --dry-run

# Porzucenie starego planu i wykonanie cleanupu według aktualnych YAML-i
python .\stash_ai.py --models all --cleanup-only --abandon-pending

# Porzucenie i ponowne przeliczenie merge bez inferencji
python .\stash_ai.py --merge-only --abandon-pending

# Samo porzucenie; kończy działanie bez kontaktu ze Stash i bez modeli
python .\stash_ai.py --abandon-pending

# Podgląd samego porzucenia
python .\stash_ai.py --abandon-pending --dry-run
```

Użyj tej samej bazy `--state-db` i adresu `STASH_URL` co podczas przerwanej
operacji. Porzucenie obejmuje wszystkie niedokończone plany dla tego adresu
w tej bazie. Z `--merge-only` lub cleanupem wybór modeli, rodzaju wyjścia i
`--scene-id` ogranicza **nową operację**, nie zakres porzucenia. Dane odzyskiwania
dla pozostałych scen pozostają dostępne do późniejszego użycia.

Obsługiwane są `--cleanup-only`, `--cleanup-markers-only`,
`--cleanup-scene-tags-only` i wszystkie strategie cleanupu. Opcja wymaga trybu
`merged`; nie łączy się z inferencyjnym `--restart`, trybami hierarchii,
pobieraniem modeli, synchronizacją samych grup ani samodzielnym pruningiem.
Brak niedokończonego planu jest poprawnym, bezczynnym wynikiem.

## Co zostaje zachowane

- Zmiany już wykonane w Stash pozostają: skrypt nie odtwarza usuniętych obiektów
  w ramach samego porzucenia ani cleanupu.
- Surowe wyniki modeli, checkpointy inferencji i dane importu pozostają w bazie.
- Informacje o własności przypisań i planowanych markerach są zachowywane jako
  dane odzyskiwania. Tagi zapisane tuż przed utratą odpowiedzi API nie stają się
  przez porzucenie tagami ręcznymi.
- `--dry-run` zmienia tylko tymczasowy stan podglądu. Nie porzuca rzeczywistego
  planu i nie zapisuje zmian do Stash ani źródłowej bazy.

Przykład: cleanup usuwa `abc`, przerywasz go i zdejmujesz `abc` z blacklisty.
Nowy cleanup z `--abandon-pending` nie kontynuuje starych usunięć `abc`.
Pozostałe markery zostają, a już usunięte nie wracają. Jeśli chcesz odtworzyć
wyniki AI, uruchom `--merge-only` z aktualnymi regułami i zachowaną bazą źródeł.
Nie oznacza to odtworzenia wcześniejszych ID, ręcznych relacji ani metadanych.

Bez tej opcji nadal obowiązuje dotychczasowe wznowienie niedokończonego
globalnego cleanupu z tymi samymi regułami.

## Spójność i zgodność

Porzucenie jest jedną transakcją lokalną. Przenosi niedokończone publikacje do
danych odzyskiwania, oznacza maski przerwanego cleanupu jako niepotwierdzone
i unieważnia cache publikacji. Nie wykonuje starego planu w Stash.

Przed cleanupem sceny skrypt odczytuje aktualne markery i tagi oraz dopasowuje
je do zachowanych publikacji. Zachowuje istniejące przedziały i dodatki,
uwzględniając częściowe usunięcia. Zapisuje tak ustalony stan również wtedy,
gdy nowe reguły nie wymagają zmian w Stash. Szybka globalna ścieżka SQL
nadal nie musi dekodować źródeł wszystkich scen; dopasowanie jest wykonywane
dopiero przy przetwarzaniu konkretnej sceny.

Jeżeli marker należący do skryptu został zmieniony poza znanymi publikacjami
i nie pasuje do żadnej zachowanej publikacji, cleanup danej sceny zgłosi potrzebę
`--merge-only` z **aktualnymi** YAML-ami. Zachowuje dane odzyskiwania; nie wymaga
dokończenia starego planu ani nie zgaduje źródeł markera.

Obsługiwane są również przerwane operacje z baz wersji 5 i 6. Pierwsze rzeczywiste
porzucenie planu podnosi format bazy do **7**, aby starszy skrypt nie zignorował
danych odzyskiwania. Do tej bazy należy następnie używać nowej wersji programu.
Zwykłe uruchomienie bez porzucenia nie podnosi formatu do 7.

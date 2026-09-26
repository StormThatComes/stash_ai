# Porzucanie przerwanych operacji

`--abandon-pending` pozwala porzucić niewykonaną część pending merge/cleanupu i
kontynuować z aktualnymi regułami. Nie cofa mutacji, które Stash już wykonał.

Najpierw zatrzymaj poprzedni proces. Opcja nie zatrzymuje innego działającego
procesu i nadal obowiązuje blokada jednego writera dla bazy stanu.

## Użycie

```powershell
# Samo porzucenie; bez kontaktu ze Stash.
python .\stash_ai.py --abandon-pending

# Podgląd samego porzucenia.
python .\stash_ai.py --abandon-pending --dry-run

# Porzuć stary merge i przelicz zapisane evidence według obecnych reguł.
python .\stash_ai.py --merge-only --abandon-pending

# Porzuć stary cleanup i wykonaj nowy według obecnych YAML-i.
python .\stash_ai.py --models all --cleanup-only --abandon-pending
```

Użyj tej samej `--state-db` i tego samego `STASH_URL` co w przerwanej
operacji.

## Co zostaje zachowane

Porzucenie zachowuje:
- surowe wyniki modeli;
- checkpointy inferencji;
- importowane legacy evidence;
- informacje odzyskiwania potrzebne do rozpoznania własności;
- wszystkie zmiany już wykonane w Stash.

Nie przywraca:
- skasowanych obiektów;
- starych ID;
- ręcznych relacji/metadanych;
- mutacji wykonanych przez Stash przed utratą odpowiedzi.

`--dry-run` nie zmienia trwałej bazy ani Stash.

## Kiedy używać

Bez `--abandon-pending` przerwany globalny cleanup należy wznowić z tym samym
zakresem i polityką.

Użyj porzucenia, gdy:
- zmieniłeś blacklistę przed dokończeniem cleanupu;
- zmieniły się reguły publikacji;
- nie chcesz kończyć starego planu;
- chcesz ponownie uzgodnić stan na podstawie bieżących YAML-i.

## Recovery

Porzucenie archiwizuje potrzebne informacje lokalnie i unieważnia cache
publikacji. Kolejny cleanup/merge odczytuje aktualny stan Stash i próbuje go
dopasować do zachowanych publikacji.

Jeżeli zarządzany marker został zmieniony poza znanym stanem i aplikacja nie
potrafi bezpiecznie ustalić jego pochodzenia, operacja nie zgaduje. Może wymagać
`--merge-only` z bieżącymi regułami.

Nie uruchamiaj starszej wersji programu na bazie, która została zaktualizowana
przez nowszy mechanizm recovery.

# Merge: szybka ścieżka i odczyt Stash

Merge przelicza opublikowany stan z zachowanych evidence. Może działać całkowicie
bez modeli przez `--merge-only`.

## Szybka ścieżka

Po udanej publikacji scena otrzymuje `merge_check`. Jeśli:
- polityka publikacji jest zgodna;
- katalog zarządzanych tagów jest zgodny;
- fingerprint tagów/markerów sceny jest zgodny;
- nie ma pending publication;

kolejny przebieg może oznaczyć scenę jako `merge_reused` i pominąć kosztowne
obliczenia.

Zmiana wyników, YAML, kodu publikacji, hierarchii lub stanu Stash powoduje
ponowne uzgodnienie.

## Odczyt zbiorczy SQL

Jeśli Stash udostępnia zgodne `querySQL`, aplikacja używa stałych read-only
SELECT-ów do pobierania tagów i markerów bez kosztu wielu resolverów GraphQL.

Dla `--merge-only` sceny mogą być stronicowane po rosnącym ID bez COUNT/OFFSET.
Podczas zwykłej inferencji GraphQL nadal dostarcza ścieżki VTT/sprite, a stan
tagów/markerów może zostać dołączony zbiorczo.

Jeśli read-only SQL nie jest dostępny lub przestaje działać, merge może wrócić
do GraphQL. Zapisy zawsze używają API Stash.

## Indeksy temporalne

Powtarzane zapytania o wystąpienia w tym samym przedziale korzystają z indeksów
osi czasu i cache wspólnych wyników. Dla nietypowych, nakładających się starych
osi aplikacja zachowuje bezpieczną ścieżkę referencyjną.

## Publikacja

Plan jest zapisany lokalnie przed mutacjami. Po udanej synchronizacji pending
snapshot jest zatwierdzany razem z aktualnym merge checkiem.

Read-only GraphQL może być retry'owany po błędach przejściowych. Mutacje nie są
retry'owane automatycznie.

## Przydatne komendy

```powershell
# Normalny merge bez modeli
python .\stash_ai.py --merge-only

# Jedna scena
python .\stash_ai.py --merge-only --scene-id 123 --verbose

# Wymuś przeliczenie mimo zgodnego cache
python .\stash_ai.py --merge-only --scene-id 123 --force-merge

# Profil JSONL
python .\stash_ai.py --merge-only --scene-id 123 --force-merge --profile-merge merge-profile.jsonl
```

W podsumowaniu zwracaj uwagę na:
- `merge_reused`;
- `merge_graphql_fallback`;
- powody typu policy/stash/cache change;
- czasy odczytu Stash i etapów merge.

Szczegółowy format profilu opisuje
[UPGRADE_diagnostics.md](UPGRADE_diagnostics.md).

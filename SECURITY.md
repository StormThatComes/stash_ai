# Security notes

## Stash API key

`STASH_API_KEY` powinien pochodzić z environment lub lokalnego mechanizmu
sekretów i nigdy nie powinien być commitowany do repozytorium.

Aktualny `stash_ai.py` nadal zawiera historyczny fallback klucza API.
Traktuj ten fallback jako **skompromitowany**. Nie używaj go do produkcyjnego
Stash; wygeneruj nowy klucz i ustaw go jawnie:

```powershell
$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<nowy-klucz>"
```

Samo usunięcie fallbacku z bieżącej wersji kodu nie unieważnia sekretu obecnego
w historii Git. Klucz należy zrotować po stronie Stash.

Klient pomija nagłówek `ApiKey`, gdy przekazana wartość jest pusta. Komunikaty
błędów maskują jawnie przekazany klucz oraz typowe parametry tokenów w URL.

## GitHub updater token

`GH_TOKEN` / `GITHUB_TOKEN` powinien być fine-grained PAT ograniczonym do
tego repozytorium i uprawnienia `Contents: read`. Updater nie zapisuje tokena
do plików projektu i odrzuca przekierowania HTTP, aby nie przekazać credentiali
innemu originowi.

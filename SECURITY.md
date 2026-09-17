# Security notes

## Stash API key

Do not commit a Stash API key. `stash_ai.py` currently has a hardcoded default
for `STASH_API_KEY`. Treat that value as compromised and rotate it in Stash
before the next run.

Replace the assignment in `stash_ai.py` with:

```python
STASH_URL = os.getenv("STASH_URL", "http://localhost:9999")
STASH_API_KEY = os.getenv("STASH_API_KEY", "").strip()
```

Then export the new key locally (PowerShell):

```powershell
$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<new-key>"
```

Or copy `.env.example` to `.env` (gitignored) and load it in your shell.
The client already skips the `ApiKey` header when the variable is empty.

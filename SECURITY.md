# Security notes

## Stash API key

`STASH_API_KEY` should be provided through the environment or a
local secret manager and **must never be committed to the repository**.

Historical versions contained a fallback API key. The current runtime
reads it exclusively from the environment and defaults to an empty value.
Treat the historical key as **compromised**: generate a new key
and configure it explicitly:

```powershell
$env:STASH_URL = "http://localhost:9999"
$env:STASH_API_KEY = "<new-key>"
```

Removing the fallback from current code does not revoke a secret
that remains in Git history. Rotate the key in Stash itself.

The client omits the `ApiKey` header when the configured value is
empty. Error messages redact explicitly supplied keys and common
URL token query parameters.

## GitHub updater token

`GH_TOKEN` / `GITHUB_TOKEN` should be a fine-grained personal access
token restricted to this repository and `Contents: read` permission.
The updater does not write the token to project files and rejects
HTTP redirects to avoid forwarding credentials to another origin.

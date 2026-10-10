# Abandoning interrupted operations

`--abandon-pending` discards the unfinished part of a pending merge
or cleanup so work can continue under current policies. It does **not**
undo mutations already performed by Stash.

Stop the previous process first. The option does not terminate another
running process, and the state database still enforces its single-writer lock.

## Usage

```powershell
# Abandon only; does not contact Stash.
python .\stash_ai.py --abandon-pending

# Preview abandonment.
python .\stash_ai.py --abandon-pending --dry-run

# Abandon old merge, then replay stored evidence with current rules.
python .\stash_ai.py --merge-only --abandon-pending

# Abandon old cleanup and start again using current YAML.
python .\stash_ai.py --models all --cleanup-only --abandon-pending
```

Use the same `--state-db` and `STASH_URL` as the interrupted operation.

## What is preserved

Abandonment retains:
- raw model results;
- inference checkpoints;
- imported legacy evidence;
- recovery records needed to identify ownership;
- every change already applied in Stash.

It does not restore deleted objects, old IDs, manual relationships/
metadata, or mutations committed in Stash before its response was lost.

`--dry-run` changes neither persistent state nor Stash.

## When to use it

Without `--abandon-pending`, an interrupted global cleanup must
resume with the **same scope and policy**.

Abandon when exclusions changed before cleanup finished, publication
rules changed, the old plan should not be completed, or you want to
reconcile published state using current YAML.

## Recovery

Abandonment archives required information locally and invalidates
publication cache. The next cleanup/merge reads live Stash state
and attempts to reconcile it with saved publications.

If a managed marker has been modified outside known state and
the application cannot establish its provenance safely, it will
not guess. A `--merge-only` run under current rules may be required.

Do not use an older application release on a database already
updated by the newer recovery mechanism.

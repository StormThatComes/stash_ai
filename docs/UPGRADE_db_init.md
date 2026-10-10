# Rebuilding a fresh database: `--db-init`

This option initializes a new SQLite database even if Stash
scenes already contain shared AI tags and markers. Use it
after losing your database or to deliberately reprocess
your library against current VTT and sprites.

## Starting and scoping the rebuild

Point to a new file rather than the existing database:

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3
```

You may repeat `--scene-id 123` to select individual scenes.
The model set and scene selector are saved in the database.
The first successful scan freezes the rebuild scene list;
the flag does not grant permanent ownership of future scenes.

`--db-init` requires a new database or a current database
containing only an empty schema. It rejects databases with
results, checkpoints, snapshots, recovery or any other
existing state — even if it belongs to another Stash URL.
It never clears or overwrites such a database. Changing
Stash URL, selected models or rebuild scope requires a
separate database; retain the old file for rollback.

`--models all` is recommended. Selecting only `wd`, for
example, replaces older AI output with WD-only predictions,
even when the old output combined several models. The
new database holds actual fresh inference evidence, not
predictions inferred from existing tag names. New or more
precise VTT is treated as current input; changed inputs
during resume invalidate incompatible cache.

## Adopting results and publishing

| Item | Behavior during rebuild |
|---|---|
| Scene assignments of `ai:<label>` | Adopted; outdated assignments are removed after inference completes, including manual tags with this prefix |
| `ai:category:*` categories and reserved marker tags | Not adopted as scene labels |
| `[AI] ` markers with `ai:marker:stash-taggers` ownership tag | Reconciled against new predictions |
| Tags outside `ai:` and markers not owned by the application | Normal publication and global exclusion rules apply |
| Model error, missing VTT/sprite or empty VTT | Scene remains incomplete; existing output stays intact |
| Successful inference with zero detections | Valid complete result; old adopted AI results can be removed |

A scene is published only after both output types complete
for all selected models. Other completed scenes may publish
even if one scene fails. Tag ownership is saved **before**
modifying Stash. Publication uses the normal durable plan:
an API failure may leave partial changes in Stash, but the
database retains enough plan/ownership data for safe retry.
A scene is marked rebuilt in the same transaction that commits
its two output snapshots.

Legacy ordinary AI markers without native `end_seconds`
still require the migration described in the main README.
`--db-init` does not replace legacy namespace or marker
geometry migration.

## Resuming and previewing

After interruption, rerun the command without the flag:

```powershell
python .\stash_ai.py --models all --state-db .\tagger_sources_fresh.sqlite3
```

Keep the original model selection and any `--scene-id` scope.
Repeating `--db-init` also preserves progress.
Completed results are reused and compatible checkpoints resumed.
If inference is complete but Stash publication is still pending,
`--merge-only` can resume it without `--db-init`.

```powershell
python .\stash_ai.py --db-init --models all --state-db .\tagger_sources_fresh.sqlite3 --dry-run
python .\stash_ai.py --db-report --state-db .\tagger_sources_fresh.sqlite3
```

Dry-run uses a temporary overlay: it neither creates the specified
database file nor changes any existing file or Stash data.
It may still download models and perform inference.
The run summary and `--db-report` show completed/pending
rebuild scene counts; `--db-check` validates rebuild state.

The first run with `--db-init` requires normal inference
for both outputs. Do not combine it with `--merge-only`,
partial output modes, cleanup, reports, `--restart`, or
`--abandon-pending`. An incomplete rebuild also blocks
scope changes, partial output, restart, plan abandonment
and rethreshold in ordinary inference/merge runs.

The SQLite schema is **v27**. Upgrade existing v18–v26
databases using `--db-update`; upgrading preserves state
and does not adopt existing Stash AI results.

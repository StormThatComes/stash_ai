# Legacy installation migrations

[Documentation index](../README.md)

Use this page only for an older installation. Current operations are described in
[updating](updating.md) and [database maintenance](database.md). Back up state and configuration first.

## Migration tools from the transition release

Three completed migrations are no longer included in the current package.
If needed, run them from a separate checkout of immutable commit
[`9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
The current package uses schema v27. Databases older than v18 first need
the earlier upgrade bridge described in [database maintenance](database.md).
Back up the database and configuration, and install the checkout's
dependencies in a separate Python 3.14 environment.

```powershell
git clone https://github.com/StormThatComes/stash_ai stash_ai_migration_bridge
git -C stash_ai_migration_bridge checkout --detach 9fcaa52502cd487455457547860b5db7d8710ad8
cd stash_ai_migration_bridge

# Convert custom v1 hierarchies; specify all hierarchy files in use.
python -m stash_ai.commands.hierarchy_migration C:/stash_ai/joytag_hierarchy.yml C:/stash_ai/wd_hierarchy.yml C:/stash_ai/dtq_hierarchy.yml C:/stash_ai/oppai_hierarchy.yml --dry-run

# Legacy AI points: derive end_seconds from published/pending/recovery state.
python -m stash_ai.tools.point_migration --state-db C:/stash_ai/tagger_sources.sqlite3 --dry-run

# Legacy prefixes: rerun inference and publish in the shared ai: namespace.
python -m stash_ai.tools.namespace_migration --models all --state-db C:/stash_ai/tagger_sources.sqlite3 --model-config C:/stash_ai/tagger_models.yml --temporal-policy C:/stash_ai/temporal_policy.yml --marker-core-exclusions C:/stash_ai/marker_core_exclusions.yml --scene-tag-exclusions C:/stash_ai/scene_tag_exclusions.yml --hierarchy C:/stash_ai/joytag_hierarchy.yml --wd-hierarchy C:/stash_ai/wd_hierarchy.yml --dtq-hierarchy C:/stash_ai/dtq_hierarchy.yml --oppai-hierarchy C:/stash_ai/oppai_hierarchy.yml --dry-run
```

Replace `C:/stash_ai` with the absolute path of your existing installation.
Use the same `STASH_URL` and `STASH_API_KEY`, and preserve model selection,
policies, and output scope. If hierarchies are disabled, replace the four
hierarchy options with `--no-hierarchy`. Once the plan is verified, repeat
the relevant commands without `--dry-run`. YAML conversion saves originals
as `*.v1.yml` and retains group assignments and source metadata.
Point migration does not load models, and retains IDs, titles, and manual
tags; occurrences remain points. Prefix migration preserves manually
managed objects and marker extras; matching markers keep their IDs.
Its scope can be limited with `--scene-id`, `--markers-only`, or
`--scene-tags-only`. Finish an interrupted prefix publication with the same
migrator and configuration before returning to the current version.
Pending snapshots remain available for API failure recovery.
Then return to the current installation and perform `--db-check` plus
any required inference.

## Flat-layout updater bridge

The current updater supports the package layout (`stash_ai/`).
Older installations using flat `stash_*.py` modules require a
one-time migration via the updater from
[commit `9fcaa52502cd487455457547860b5db7d8710ad8`](https://github.com/StormThatComes/stash_ai/commit/9fcaa52502cd487455457547860b5db7d8710ad8).
Stop the application and download that version as a separate file
inside the installation directory:

```powershell
$env:GH_TOKEN = "<fine-grained PAT>"
$bridge = "9fcaa52502cd487455457547860b5db7d8710ad8"
$headers = @{ Authorization = "Bearer $env:GH_TOKEN"; Accept = "application/vnd.github.raw+json" }
Invoke-RestMethod -Headers $headers -Uri "https://api.github.com/repos/StormThatComes/stash_ai/contents/update_stash_ai.py?ref=$bridge" -OutFile .\update_stash_ai_bridge.py
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout --dry-run
python .\update_stash_ai_bridge.py --install-dir . --migrate-flat-layout
```

If a legacy update was interrupted, first run
`python .\update_stash_ai_bridge.py --install-dir . --recover`.
The bridge installs the current package while preserving backups,
local YAML files, database and models; use `--force-code` for
locally modified code. Continue with the regular updater afterwards.
The current updater does not scan for or delete untracked legacy
modules. A `git pull` directly into the package layout does not
require the updater bridge.

## Retired publication options

New results always use shared `ai:` publication and temporal rules.
`--output-mode`, `--no-temporal`, and the YAML `enabled` key have been
removed; remove them from scripts and `temporal_policy.yml`. Individual
rules, including `occurrence_markers`, are still configurable via the
temporal-policy YAML. Historical per-model namespaces require the
separate migrator. Normal publication, cleanup and default pruning
manage current `ai:` results and `m:` groups. Legacy tag definitions
can be explicitly pruned with, for example,
`--prune-empty-tags-only --prune-prefixes jt: wd: dtq: op:`.
Manual extras from other namespaces are preserved.

For renamed exclusion options, see [configuration names](../tagging/exclusions.md#renamed-configuration-and-cli-options).

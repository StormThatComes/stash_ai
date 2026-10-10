# Exclusions and cleanup

Exclusions are applied during ordinary merge and dedicated
GPU-free cleanup modes.

## Files

`scene_tag_exclusions.yml` defines global and per-model scene-tag
rules. A global exclusion also applies to marker cores/extras
and hierarchy creation.

`marker_core_exclusions.yml` excludes primary marker tags
without globally prohibiting the label in other roles.

The code-level `MARKER_TAG_BLOCKLIST` option has been retired.
Use these YAML files. Core-only rules allow the label as
a marker extra or scene tag, while
`scene_tag_exclusions.yml` covers scenes and primary/extra
marker tags. There is no longer a separate all-marker-role
exclusion that also preserves scene-tag usage.

Cache saved with an empty blocklist remains usable.
Non-empty or unknown historical blocklists require
`--restart` for affected scene/model pairs before normal
marker merge or rethreshold. Cleanup can still retract
previously published evidence without inference, respecting
historical rules and boundaries.

## Renamed configuration and CLI options

Each setting now has one supported name:

| Legacy name | Current name |
|---|---|
| YAML `excluded_tags` | `excluded_scene_tags` |
| Automatically selected `joytag_marker_core_exclusions.yml` | `marker_core_exclusions.yml` |
| `--model` | `--models` |
| `--tag-exclusions` | `--scene-tag-exclusions` |
| `--sync-marker-core-only` | `--cleanup-markers-only` |

Rename the YAML list key without changing its values or
`by_model` section. Move your custom marker-exclusions
file to the new name, or pass it explicitly with
`--marker-core-exclusions PATH`. If both files exist,
keep your custom policy rather than overwriting it
with the application's empty default. Update scripts
and schedules. The CLI requires full option names;
retired aliases/YAML keys now fail validation.

## Cleanup

```powershell
python .\stash_ai.py --models all --cleanup-only --dry-run
python .\stash_ai.py --models all --cleanup-only
```

Partial scopes:

```powershell
python .\stash_ai.py --models all --cleanup-markers-only
python .\stash_ai.py --models all --cleanup-scene-tags-only
```

Cleanup runs neither models nor sprite downloads.

## Global exclusions

A global rule can remove forbidden `ai:/jt:/wd:/dtq:/op:`
tags from scenes and markers, and during full cleanup
may delete the tag definition itself. Stash deletes its
other relationships when that definition is removed.

This is deliberately stronger than core-only exclusion.
Before adding a global blacklist entry, make sure
the tag definition is not needed for manual use.

## Core-only exclusions

`marker_core_exclusions.yml` removes only managed markers
whose core label is forbidden. The tag definition remains,
and the label may still be used as an extra.

Manual markers are not adopted merely because their
tag names are similar.

## Persistence and replay

Raw model evidence remains in `tagger_sources.sqlite3`.
Cleanup stores compact metadata needed to read earlier
publications correctly without rewriting all raw results.

After removing an exclusion, ordinary `--merge-only` can
republish a result from retained evidence unless it was
manually lost in a way that cannot be reconstructed.

## Interrupted cleanup

Either resume interrupted cleanup with **identical rules**,
or run `--abandon-pending` before using a new policy.
Do not silently change policies while cleanup is pending.

## Empty tag definitions

Optionally:

```powershell
python .\stash_ai.py --models all --cleanup-only --prune-empty-tags
python .\stash_ai.py --prune-empty-tags-only --dry-run
```

Pruning removes only verified unused leaves and respects
managed marker groups.

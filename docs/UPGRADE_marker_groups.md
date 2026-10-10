# Marker groups

In `merged` mode, markers sharing the same primary tag
receive an additional group tag. This does not merge
time intervals or create Stash `Group` objects.
Browse them using Stash's additional-marker-tag filter.

| Primary tag | Marker type | Additional group tag | Direct parent |
|---|---|---|---|
| `ai:abc` | Ordinary (including intervals with no end) | `m:abc` | `ai_markers` |
| `ai:abc` | Occurrence point | `m:abc:occurance` | `ai_markers` |
| `ai:def` | Ordinary | `m:def` | `ai_markers` |

The spelling `:occurance` is intentional. The technical
type tag remains `ai:marker:occurrence`. Marker type is
determined from its data and, for existing markers, the
technical tag; a missing `end_seconds` does **not**
by itself mean Occurrence.

Each published marker receives exactly one managed group
derived from its final core and type. Marker extras do
not create groups. Groups are shared across scenes, but
ordinary and Occurrence groups are distinct. `m:` tags
are not assigned to scenes or model evidence, and do not
change the `ai:marker:stash-taggers` ownership tag.

## New and existing markers

Normal runs and `--merge-only` assign groups at publication.
Repeating a run does not duplicate tags or markers.
Changing marker core/type removes the previous managed
group association while retaining manual extras.

To backfill group assignments without inference, source
replay or changing marker times:

```powershell
python .\stash_ai.py --sync-marker-groups-only --scene-id 123 --dry-run --verbose
python .\stash_ai.py --sync-marker-groups-only --scene-id 123

# All scenes
python .\stash_ai.py --sync-marker-groups-only
```

This mode preserves marker ID, title, core, start and end.
It affects only script-owned `ai:` markers, not manual
markers or `jt:`/`wd:`/`dtq:`/`op:` results.
It needs neither model weights/vocabularies nor temporal
policies or hierarchy files. It uses the same
`--state-db` lock as merge/cleanup.

If a previous publication was interrupted, resume
`--merge-only` with the same database first.
An interrupted group synchronization can be retried.

`--no-hierarchy` disables model-category hierarchies
only. Marker groups remain children of `ai_markers`.
Historical source-prefixed markers gain groups after
import and merge.

## Cleanup

Existing exclusions and cleanup still apply to markers.
After successful marker publication, marker cleanup or
group synchronization, the script removes managed group
tags that are globally unused leaf tags. This includes
both groups after excluding a core and the previous
group after a marker-type change.

All Stash usage counts are verified, including objects
outside the current scene or outside markers, as well
as child tags. A tag used by another scene, image or
object is preserved. `ai_markers` is always kept.
Pruning empty definitions is global even with
`--scene-id`; marker assignments on other scenes
remain unchanged.

Cleanup strategies `sql` and `scenes` are supported;
`auto` uses SQL when available, otherwise scene reads.
`--dry-run` incorporates planned group-assignment
changes when calculating empty tags, without writing to
Stash or the source database. Publication/group-sync
failures prevent final group pruning. Scene-tag-only
cleanup does not change marker groups.

For SQL cleanup, `Find empty marker groups` first
checks all groups using indexed `NOT EXISTS`.
Groups already associated with markers or scenes are
skipped without loading full usage counts. Reads use
`--cleanup-page-size` (default 2000 tags per query).
Global discovery also identifies old orphaned groups.
During dry-run, groups losing associations in the
planned changes undergo further validation because SQL
still sees their current, not-yet-removed relationships.

Next, `Check tag usage` verifies only remaining
candidates. It still enforces full usage counts,
including images/other objects, ownership, no children
and a fresh check before deletion. If there are no
candidates, this stage is skipped. Non-SQL strategies
retain the full group-read path. Summary fields include
`marker_group_tags_checked_sql` and
`marker_group_tags_candidates`.

The `Check tag usage` counter advances after each
verified batch (up to 100 tags). While waiting for
Stash, a `waiting for Stash` message refreshes elapsed
time every second. The existing `Prune empty tags`
bar then follows (or `Preview empty tags` in dry-run).
This works without `--verbose` and during `--dry-run`.
Refreshes do not send extra requests or shorten Stash
usage-count query time.

Standalone `--prune-empty-tags-only` also includes
managed `m:` among its default prefixes.
To limit pruning to groups:

```powershell
python .\stash_ai.py --prune-empty-tags-only --prune-prefixes m: --dry-run --verbose
```

## Ownership and naming collisions

New groups have a description beginning with
`[Stash AI marker group v1]` and including the core
and type, plus `ignore_auto_tag=true`. Newly created
`ai_markers` also has automatic tagging disabled.
An existing root is used without claiming ownership.
Other manually assigned parents of a managed group
are preserved.

The `m:` prefix alone does **not** authorize tag
modification/deletion. If the required name belongs
to a tag without the ownership marker, the script
reports a collision for the scene. Rename the
manual tag before retrying. Removing the marker
from a description opts that tag out of automatic
management.

One convention limitation remains: ordinary core
`abc:occurance` collides with the Occurrence-group
name for core `abc`. The script raises an error
rather than merging unrelated groups.

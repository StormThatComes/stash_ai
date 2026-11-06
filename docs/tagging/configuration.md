# Tagging configuration

[Documentation index](../README.md) · [Tagging workflow](README.md)

- [Temporal rules](#temporal-rules)
- [Tag hierarchy](#tag-hierarchy)

Edit the repository YAML files for persistent policy; explicit CLI options override
the corresponding settings. Use `python stash_ai.py --help` for all command options.

| File / option | Purpose |
|---|---|
| `tagger_models.yml` | Models, directories, thresholds and batch sizes |
| `temporal_policy.yml` | Temporal rules, consensus, extras and Occurrence |
| `scene_tag_exclusions.yml` | Global scene-tag exclusions |
| `marker_core_exclusions.yml` | Marker primary-tag exclusions |
| `tag_hierarchy.yml` | Shared category hierarchy for all models |
| `--state-db` | Persistent state database path |
| `--db-init` | Initialize a new/empty database and adopt existing AI results |

Selected models must explicitly define `scene_threshold`, `marker_threshold` and
`batch_size`. Select them with `--models`, or set `enabled_models` in
`tagger_models.yml` when omitting that option. Missing values do not fall back to
legacy JoyTag defaults. See [state and cache](state-and-cache.md) for retention floors.

Publication has **no maximum number of scene tags or markers**.
All results qualifying under thresholds, temporal policies and
exclusions may be published, for individual models, combined results,
and Occurrence points. `MAX_TAGS` and `MAX_MARKERS_PER_SCENE` were
removed. The independent per-frame tag limit `MARKER_MAX_TAGS_PER_FRAME`
remains (default `0`).

Databases on schema v18–v26 require `--db-update`. Upgrading to v27
writes snapshot format v7 without the legacy output-count limits,
while retaining geometry, ownership and pending/recovery plans.
It neither modifies inference cache nor writes to Stash.
To regenerate previously truncated published results from cached evidence:

```powershell
python .\stash_ai.py --db-update
python .\stash_ai.py --merge-only
```

`--merge-only` applies the current publication rules without a GPU;
cache-completeness requirements still apply. Use `--dry-run` to preview
publication.

## Temporal rules

`temporal_policy.yml` controls automated post-processing:

- `shared_marker_frames` — combined qualification of compatible
  frames across models;
- `shared_marker_extras` — extras contributed by other models;
- `short_marker_extras` — brief coherent episodes of extras;
- `shared_scene_tags` — additional scene-tag qualification;
- `occurrence_markers` — representative points for scene
  tags without their own marker;
- marker-continuation hysteresis;
- minimum matches and extra coverage requirements.

The temporal engine always runs. `--no-temporal` and the YAML
`enabled` key are removed; individual rules remain configurable.
The historical algorithm is only retained for frozen
cleanup-snapshot replay.

After changing `marker_continue_ratio`, `--merge-only` verifies
that retained marker evidence covers the new continuation
threshold. If the retained floor is higher, inference is
required before publishing that scene, including dry-run.
The effective `marker_threshold` is checked even when
previously rethresholded. Legacy frames without retention
metadata and imported intervals retain historical checks.

`MARKER_TAG_BLOCKLIST` has been retired. Put exclusions in
`marker_core_exclusions.yml` or `scene_tag_exclusions.yml`;
these affect publication but preserve raw cache detections.
Existing checkpoints/cache recorded with an empty blocklist
remain compatible.

The `marker_tag_blocklist` field remains in historical
result metadata; new results store an empty list.
Older cache with a non-empty blocklist requires fresh
inference, because previously rejected detections
are absent from saved frames. `--merge-only` and marker
rethresholding report an error before publishing the
affected scene, including `--markers-only` and `--dry-run`.
Legacy cache without the blocklist metadata also needs
a one-time refresh before normal merge/rethreshold.

Cleanup can still retract votes from previously published
consensus markers without inference: it checks persisted
proofs and restricts changes to published intervals/extras.
Historical publication exclusions remain respected;
cleanup does not add new votes or newly unblocked tags.
Recompute affected scene/model pairs before normal merge:

```powershell
python .\stash_ai.py --models wd --scene-id 123 --restart
```

`--force-merge` cannot restore missing detections.
Imported `separate`-mode intervals retain their historical
behavior. Scene-tag-only operations do not require marker
blocklist metadata.

After changing `scene_support_ratio`, `--merge-only`
recalculates scene statistics from saved frames even
if `scene_threshold` is unchanged. Raw observations
remain in the database, so the old ratio can be restored.
If the newly required support floor is below the
cache retention floor, or required frames are missing,
the scene needs inference before publication.
Legacy aggregates without retention metadata retain
their prior behavior; fresh inference populates
the evidence needed for future rule changes.

**Model scores are not averaged into one probability.**
Per-model sources and confidence values remain distinct.

## Tag hierarchy

The hierarchy uses `version: 2`, `source` and `groups`. Group paths form
`ai:category:` tags under `ai:category:AI`.

All models share `tag_hierarchy.yml`, the union of the
previous four hierarchies: 21,154 unique tags in 92 groups
with unchanged assignments. Categories are published under
shared `ai:category:...` tags. One raw-label entry
defines its parents for all models.

`--hierarchy YAML` specifies the shared file for any selected
model set. `--wd-hierarchy`, `--dtq-hierarchy` and
`--oppai-hierarchy` have been removed. Validation checks
the full YAML and selected models' vocabulary coverage.
Entries from unselected models are allowed but do not
cause their tags to be created by
`--sync-hierarchy-only --create-all-tags` when running
with a subset of models.

Optional `source.tag_count` and `source.vocabulary_sha256`
metadata describe distinct tags in the **entire file**,
not an individual model's vocabulary. Update both fields
or remove both after adding/removing labels. Changing
only grouping does not change the vocabulary fingerprint.

Move custom assignments from `joytag_hierarchy.yml`,
`wd_hierarchy.yml`, `dtq_hierarchy.yml`, and
`oppai_hierarchy.yml` into the shared file.
The updater preserves old local YAML, but the application
no longer reads it. Hierarchy changes do not invalidate
inference cache or require database migration.

Synchronization replaces hierarchy-managed parents
with assignments from current YAML, preserving parents
outside that hierarchy, including user-created categories.
Removing an assignment does not delete the category tag
object itself from Stash.

`parent_sync` has been removed from the schema.
Delete it from custom or locally preserved YAML
before running the current version.

```powershell
python .\stash_ai.py --sync-hierarchy-only --dry-run
python .\stash_ai.py --validate-hierarchy
python .\stash_ai.py --models wd --validate-hierarchy --hierarchy C:/stash_ai/my_tag_hierarchy.yml
```

`--create-all-tags` is only supported with
`--sync-hierarchy-only`.

Exclusions still apply independently to model predictions
and category creation from their vocabularies.
The shared hierarchy skips a label excluded by **all**
models; excluding it in only one model does not remove
the mapping for the others.

`--no-hierarchy` disables model-category synchronization,
but not `m:` marker groups.

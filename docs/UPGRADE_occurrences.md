# Occurrence points for scene tags

Occurrence is enabled by default through
`occurrence_markers: true` in `temporal_policy.yml`.
Set that rule to `false` to disable Occurrence points
while retaining the other temporal rules.

## What it creates

For an accepted scene tag with no corresponding
interval marker, Stash AI may create one
representative point:

- core tag: `ai:<label>`;
- technical tag: `ai:marker:occurrence`;
- no end time;
- position from a saved VTT/sprite sample.

This is **not** a list of all appearances and does
not attempt to determine the exact event onset.

An existing manual marker with the same core provides
a location and prevents the representative point
from being created; the rule does not modify manual markers.

## Usage

Occurrence is evaluated as part of normal merge:

```powershell
python .\stash_ai.py --merge-only
python .\stash_ai.py --merge-only --scene-id 123 --dry-run --verbose
python .\stash_ai.py --merge-only --markers-only
```

Changing `occurrence_markers` affects publication
but does not by itself require fresh inference.

## Choosing a timestamp

A tag must first qualify as a scene tag. Its location
is chosen from retained maxima or frame observations.

Candidate comparison considers, among other things:
- how many models support the sample;
- how much retained evidence is available;
- relative margin above the scene threshold;
- earlier time as a tie-breaker.

Model scores are **not** averaged into one probability.

If saved evidence has no time suitable for locating
the tag, the application will not guess.
Inspect available evidence:

```powershell
python .\stash_ai.py --explain-tag ai:tag_name --scene-id 123
```

If a location is needed but missing from cache,
rerun inference on that scene/model:

```powershell
python .\stash_ai.py --models wd --scene-id 123 --restart
```

## Exclusions and eligibility

Occurrence honors `scene_tag_exclusions.yml` and
`marker_core_exclusions.yml`. There is no per-scene
marker-count limit. Intervals belonging to other tags
do not suppress Occurrence points or change their
evidence provenance.

An ordinary interval marker takes precedence over
a representative point for the same label.

Cleanup may delete or preserve the point based on
remaining published evidence. It never relocates
an existing point; only a later merge chooses a new location.

## Precision

Timestamp precision is limited by VTT/sprite sampling.
An Occurrence point denotes representative detection,
not the exact timestamp of an event.

For a synthetic benchmark of the selection logic:

```powershell
python -m benchmarks.benchmark_occurrences
```

This benchmark does not measure Stash publication.

# Tagging reports

[Documentation index](../README.md) · [Tagging workflow](README.md)

## Model utility / ablation report

`--model-report NEW.json` works entirely from saved evidence, without
Stash connectivity or model inference. It first reconstructs the normal
result across all recorded models for each scene, then computes four
counterfactual `all minus one model` results. Omitting a model
recalculates consensus seeds, temporal rules, shared scene tags,
marker extras and Occurrence from scratch; it does not merely filter
the resulting markers.

Per-model metrics include:
- scenes with retained evidence and scenes whose result changes when
  that model is omitted;
- final scene tags and marker components supported by the model;
- scene tags, markers, extras and Occurrence points lost without it;
- surviving markers with changed intervals or extras;
- the share of evidence-bearing scenes with a measurable marginal impact.

A change only in the marker's provenance badge, e.g. `[jt+wd]` to
`[wd]`, is not counted as a semantic change. The report measures
the effect on tagging, geometry and extras rather than merely
model attribution. Use `--scene-id` to restrict the sample.

### Tag review report

`--tag-review-report NEW.json` examines stored scene tags and primary
marker labels by replaying merge with effective database thresholds.
It uses the selected temporal policy and exclusion rules. It does not
load models, retrieve media, connect to Stash, modify source state,
or publish changes.

Scenes whose `input_checks` state is `missing_media`, `read_error`
or `deleted` are skipped with `unverified_inputs`, even when an
unselected model caused the problem on that scene. The report keeps
the stored evidence and includes the reason in its examples.

Review priorities are determined by the following heuristics:

- `weak_or_borderline`: best score is no more than
  `--tag-review-margin` (default `0.05`) above the relevant threshold.
  For scene tags qualifying via `strong_peak`, the strong-peak
  threshold applies; otherwise use that scene/marker's own threshold.
  Weak results accepted through temporal or consensus rules can
  also be flagged.
- `isolated_peak`: a scene tag qualified through a strong peak
  but only one sample exceeds the scene threshold, with at least
  two samples present.
- `model_disagreement`: another model retained the same label for
  identical input but did not qualify it under its own scene rules.
  Missing models/labels are **not** negative votes, and excluded
  labels are ignored for comparison.
- `long_marker`: a merged interval is at least
  `--tag-review-long-marker-seconds` (default `300`) seconds long.

The terminal table and JSON `models` section show JoyTag, WD, DTQ,
and Oppai counts for distinct flagged labels, scene/tag pairs,
and scenes. Each model's scene/tag pair counts once, even when
several intervals exist, the tag is also a scene tag or marker core,
or it came from multiple recorded runs. Global counts deduplicate
across models; summing model rows does not give total distinct tags.
Reason counts may overlap. A model contributing to marker-core
consensus also counts even if it created no independent interval:
the supporting score is derived from compatible retained observations
within the contribution, and its example has `support_only: true`.

The JSON `tags` section contains model/label rows, reason counts,
and up to three example scenes with scores, thresholds and times.
Configure 1–20 examples using `--tag-review-examples`; counts always
include all analyzed scenes. Rows sort by affected-scene count,
then reason count. `--scene-id` narrows the analysis. Existing report
files are never overwritten.

Scenes awaiting refresh/publication or with unusable evidence are
skipped with explicit reasons instead of participating in comparisons.
The report shows **replayed cache output**, not live Stash state.
Heuristics are not proof of an error: a valid tag can have one peak,
a long interval or support from a single model. Model scores are
not averaged.

### Cache coverage report and backfill plan

`--cache-report NEW.json` checks scene and marker evidence for models
enabled in YAML or explicitly selected with `--models`. It runs
offline without loading models, vocabularies or media and without
connecting to Stash. It changes neither source state nor published
results and refuses to overwrite an existing report.

By default, it covers scenes known to the **current local database scope**,
including scenes only referenced by checkpoints, refresh requests,
snapshots, recovery or `input_checks`. Stash scenes never encountered
by this database are not counted. Repeatable `--scene-id` arguments
set an exact scope; IDs without any local state then count as missing.
The report does not validate current weights, vocabulary, VTT or sprites;
a normal inference run performs those checks.

The terminal table and JSON `models` list per-model scene/model pairs
that are ready, rethresholdable without GPU, requiring inference,
requiring rethreshold only, or blocked. `outputs` splits the counts
by scene/marker outputs and mutually exclusive statuses:

| Status | Meaning |
|---|---|
| `ready` | Completed inference, consistent index, effective thresholds, sufficient retention |
| `rethreshold` | Cache covers the YAML thresholds; only GPU-free rethreshold is required |
| `missing` | No result for this output kind |
| `partial` | No result, but a checkpoint exists for resume-time validation |
| `refresh_required` | Pending fresh-inference request invalidates old results |
| `insufficient` | No frame evidence, retention floor too high, or required labels previously filtered out |
| `inconsistent` | Result/index mismatch or unreadable evidence; investigate with `--db-check` |

Completed inference with zero detections is valid cache, not missing
evidence. `evidence_ready_pairs` describes coverage of persisted
results. Known deferred input checks block `fully_ready_scenes` and the
whole plan for that scene, even if they concern an unselected model.
`issues.input_checks` identifies the model and reason. Restore the media
or resolve scene deletion, then verify input during a normal run;
the blockage alone does not require `--restart`.

Retention is compared against the lower of the YAML floor and the
support threshold needed for publication. Exactly one active result
per model/output is checked. With populated `result_metadata`, the
report avoids decoding prediction BLOBs; missing legacy metadata is
filled only in a temporary database. Use `--db-check` for deep
payload validation.

`issues` identifies scene/model pairs, reasons, required thresholds,
blockers and checkpoint progress. `plan` provides ordered, grouped
**argument arrays (`argv`)** with at most 50 scenes per command:

- `resume_or_infer`: fill selected models/scenes; normal resume
  validates checkpoints and reuses matching completed results;
- `restart_inference`: forcibly replace unusable completed cache.
  Active missing/refresh states with checkpoints use normal resume
  to retain progress;
- `rethreshold`: apply both YAML thresholds without inference;
- `merge_after_backfill`: refresh model consensus and Occurrence last.

Commands preserve `--state-db` and YAML paths for models, temporal
policy and exclusions. The report **does not execute** its plan.
Manually running these commands can modify Stash; keep the same
`STASH_URL` and environment. Publication may remain blocked until
all models with pending refreshes have finished.

Pending publication, global cleanup, refresh for an unselected model,
unreadable evidence, or disabled replacement of unusable history
blocks all planned commands for an affected scene, including
backfilling other models and the final merge. Every command might
attempt publication. Finish the original operation or resolve the
inconsistency, then generate the report again. Evidence readiness
and publication blockage can overlap in aggregate counts.

Merge also uses retained evidence from **unselected models**.
The report checks their publication eligibility using stored effective
thresholds and the selected policy, without requiring those models'
current YAML retention settings. Incomplete or unreadable evidence
from such a model blocks the scene. `issues[].publication_dependencies`
identifies its model, output kind, reason and remedy. Include it in
`--models` and regenerate the report to plan repairs. Completeness
counts and inference targets still concern only selected models;
the report does not silently expand the inference scope.
`issues[].blocking_models` identifies selected models whose blockage
prevents planning other pairs for the same scene.

Exit code `0` means the report was successfully generated, even
when gaps remain. Archived evidence after `--abandon-pending` does
not block cache or inference. `publication_recovery_scene_ids`
lists affected scenes; the plan's final merge reconciles their
publication even with complete cache, provided no other
publication blocker exists.

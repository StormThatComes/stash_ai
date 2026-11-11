# Recognition evaluation

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Measured runtime and pilot scope](#measured-runtime-and-pilot-scope)
- [Model smoke coverage](#model-smoke-coverage)

## Measured runtime and pilot scope

`python -m benchmarks.benchmark_recognition_adapters` explicitly runs two isolated
CPU workers on the same synthetic 128px texture/supplied landmarks, with warmup
and five embeddings. This developer benchmark is not a public recognition stage.
A local run on 2026-10-12 (Python 3.14.7, OpenCV 4.14.0.94, ORT 1.26.0) measured:

| Adapter | Owned model assets | Load | Median embedding | Sampled peak RSS |
| --- | ---: | ---: | ---: | ---: |
| SFace | 37.1 MiB | 0.137 s | 11.7 ms | 190.4 MiB |
| AuraFace | 248.8 MiB | 0.652 s | 238.6 ms | 581.5 MiB |

RSS is sampled every 10ms, so this is an observed lower bound, not an exact peak
allocation guarantee. Artifact bytes include YuNet and the selected recognizer;
the external shared HF cache is not included. Absolute timing depends on host and
runtime threading. Neither fixture contains a labeled human face: detected faces
were zero, and false merges/misses/accuracy are deliberately unmeasured here.
The measured cost and lack of a user-domain quality pilot support keeping SFace
as the initial baseline, without claiming either model is universally superior.

For an explicit developer pilot, call
`stash_ai.recognition.pilot.evaluate_observations(database, run_id, configurations, labels)`
with both model-specific `FaceConfig` objects and a bounded mapping of selected
retained observation IDs to human ground-truth person labels. Use representative
different/same people, blur, pose and thumbnail sizes; choose thresholds separately
per model on a calibration subset, then evaluate a separate held-out subset.
Both adapters receive exactly the same retained crop and re-based source landmarks.
Each selected scene must use observations from one retained scan revision;
mixing versions is rejected before model preparation to avoid false co-presence.
The helper runs the real grouping policy in disposable in-memory SQLite and
reports embedding/accepted coverage, withheld/ambiguous observations, pair false
merges and misses (including withheld), pair threshold errors, time and asset size.
It makes no Stash/media calls or changes to published membership. Labels,
configuration, derived results and scope stay in the existing face SQLite's run
progress/history; no evaluation state sidecars or manual normal-use workflow are
needed. There is no claimed real-library accuracy until such labeled data exists.

### Offline identity evaluation on retained VTT observations

`recognition tune` searches threshold combinations using production rebuilds in
private RAM, starting every trial from the same read-only retained snapshot.
It reads existing VTT embeddings; it does not open media, load inference models,
publish performers, change assignments in the source database, or modify your
configuration. Trials run sequentially and use the configured rebuild workers.

Create a JSON grid such as:

```json
{
  "recognition.match_threshold": [0.65, 0.70, 0.75],
  "clustering.cohesion_threshold": [0.60, 0.65]
}
```

```sh
python stash_ai.py recognition tune --all-scenes --grid thresholds.json --output tuning.json
python stash_ai.py recognition tune --all-scenes --grid thresholds.json --labels truth.json --yaml-output proposed.yml --output tuning-labelled.json
```

`--max-trials` defaults to 64 (including baseline, maximum 1024). Oversized grids
are rejected before replay; invalid parameter combinations are reported and
skipped. Supported parameters are numerical recognition and clustering match,
cohesion, ambiguity and support gates; model, detection and input contracts cannot
be tuned this way. Each trial reports assignment coverage, identity count and
newly joined/split observation pairs relative to baseline and retained state.
These are behavioral changes, not accuracy estimates.

Labels are optional and use the same observation-to-human-truth JSON as
`evaluate`. Without labels, tuning produces comparisons and candidate patches,
with no winner or YAML export. With sufficient labels, whole people are split
deterministically between training and holdout; both halves need at least two
people and a same-person pair. The training winner maximizes pair recall then
coverage, subject to `--max-false-merge-pairs` (default zero). Only that winner is
checked on the holdout, including a final check for cross-partition false merges.
A failed holdout does not trigger another candidate search. YAML is exported
only if validation passes and the candidate improves training over baseline.
Accuracy claims remain limited to the supplied labels; automatic assignments
are never used as truth. Output files must be new; no YAML is silently applied.

`recognition report-merge --simulate` additionally runs baseline and each unique
recommended patch through a full joint rebuild of **all latest compatible scans
in the identities' scope**. This checks competitors beyond the requested pair,
identity births, attachments, reconciliation and publication bindings in RAM.
It reports whether all members of the requested pair from the selected latest
scans end in one identity, including withheld members and global split/join/coverage
changes. Members outside that latest selection are counted separately.
The frozen catalogue analysis and replay use the same database snapshot.
Missing, partial or incompatible latest scans stop the simulation instead of
silently excluding them. Results describe this selection and snapshot; they do
not guarantee future runs or prove that newly merged faces are the same person.

```sh
python stash_ai.py recognition report-merge --performer-id 123 --performer-id 456 --simulate --output merge-simulation.json
```

`recognition evaluate` runs the production identity rebuild in disposable RAM.
It does not load models, fetch VTT/video media, acquire the database writer lock,
change retained assignments or publish to Stash. Supply explicit human labels
using observation IDs from an identity/fragmentation report:

```json
{"observation-uuid-1": "person-A", "observation-uuid-2": "person-A", "observation-uuid-3": "person-B"}
```

```sh
python stash_ai.py recognition evaluate --all-scenes --labels truth.json --face-config facial_recognition.yml --compare-config tuned.yml --batch-size 0 --batch-size 1 --batch-size 20 --output evaluation.json
```

Use repeated `--scene-id` instead of `--all-scenes` for a pilot. The latest
selected scans must be complete and match the detection contract. Compared YAML
files must use the same recognizer, detector, detection settings and cue limit;
matching settings may differ. Each configuration starts from the same selected
raw observations with no historical identity labels. A batched trial introduces
only its current scenes and the previously processed batches, in numeric scene
ID order. Checkpoints are disabled. Existing quality measurements are retained,
but image crops are never read. Unlabelled selected observations remain rivals.

The JSON includes fragmentation, incorrect merges, coverage and partition
comparisons, plus the original production decisions for labelled observations.
Metrics measure only the supplied truth labels, not library-wide accuracy.
Default batch sizes compare joint replay (`0`) with individual scenes (`1`).
Output files must be new. Changing the model or scan gates requires scanning
before evaluation; this command cannot recover faces rejected during detection.

Normal runs now record `identity_decisions` in run history, with counts by status
and reason and three original evidence examples per group. Identity exports and
evaluation share the same evidence projection; neither invents a missing blocker.

### Evaluation with history, import order and retuning

`recognition evaluate` also accepts repeatable `--scene-order numeric|reverse|shuffle`
and `--history cold|retained`, plus `--shuffle-seed`, `--repeat-runs` and
`--retune-from-baseline`. The Cartesian product of configurations, batch sizes,
orders and history modes runs in independent RAM databases. Each repeated pass
keeps only its own preceding state. A retuned comparison first processes its
selection with the baseline configuration, then applies the comparison policy.

Retained history copies external scenes and their existing decisions, including
supersession and source authority. Every version of a selected scene is excluded
until its batch arrives. Crops, portraits, publication state and rebuild caches
are not copied. The source stays read-only. History labels are prior automatic
decisions, not evaluation truth. Labelled selected observations alone contribute
to metrics and reason counts; unlabelled evidence remains a competitor.

For example, compare import order, history and warm stability with:

```sh
python stash_ai.py recognition evaluate --scene-id 1 --scene-id 2 --labels truth.json --batch-size 1 --scene-order numeric --scene-order reverse --scene-order shuffle --shuffle-seed 27 --history cold --history retained --repeat-runs 2 --output scenarios.json
```

Report format 2 records the effective scene order, history counts, pass number,
retuning mode, and elapsed replay time for every trial. Timing excludes source
copying, media and inference. Joint replay canonicalizes its own input order;
order sensitivity is tested through batched admission. Existing default trial
names (`baseline/joint`, `baseline/batch-1`) remain unchanged.

### Anchor evidence experiments

Separate `matching.face.anchors.quality.min_face_pixels`, `min_anchor_sharpness`,
`min_auxiliary_face_pixels` and `min_auxiliary_sharpness` apply to retained quality
measurements. Null preserves the existing detection floor. Set the detection
floors low enough to retain auxiliary faces; lowering a detection floor requires
a VTT rescan because previously rejected observations have no embedding. Anchor
floors must be at least the auxiliary floors, and both must be at least the scan
floor. Invalid alignment remains rejected for both roles. SQL admission, Python
admission and compact replay measurements use the same policy.

Three independent switches default to false so labelled evaluation can measure
quality changes separately:

- `anchor_outlier_quarantine`: derive a frozen query core only when at least
  `max(3, view_min_support, min_merge_support)` independent retained timestamps
  support a coherent appearance. Isolated samples below cohesion with every
  supported sample are excluded from that query core. The remaining original
  samples must still pass complete-link cohesion. Original anchors stay in the
  strict bank as rivals and merge witnesses; no membership is deleted and no
  cleaner identity is asserted. Recovery is attachment-only, records excluded
  IDs, and cannot seed a promoted appearance or an identity merge.
- `adaptive_view_witnesses`: grow retained witnesses from eight with the square
  root of independent support, up to `max_view_witnesses` (default 32). Existing
  medoid/diversity selection preserves rare endpoints and supported reserve
  looks. All additional vectors count against `max_reference_mb`.
- `deduplicate_detections`: same-cue detections need box IoU at least 0.90,
  every landmark displacement at most 3% of the smaller box diagonal, and
  embedding cosine at least 0.995 to share physical-face evidence. Similar
  embeddings at different locations remain competitors. Matching, pending
  claims and assigned/contextual co-presence use the same predicate. Repeated
  timestamps still supply only one independent support vote.

Run each toggle separately with `recognition evaluate`, retained history, scene
order variations and false-merge metrics before combining them. These changes
claim no accuracy gain on unlabelled media and introduce no extra video inputs.

## Model smoke coverage

Normal core tests do not download models. Recognition CPU CI jobs install the complete application CPU profile and
load pinned detector/recognizers and FaceOcc on Linux and Windows. A public-domain
portrait with a synthetic lower-face covering tests the real visibility mask;
this is a behavior smoke, not domain accuracy calibration. GPU provider routing,
strict fallback handling and device overrides are tested without GPU hardware;
these hosted jobs do not claim actual CUDA inference. This is a runtime,
alignment and embedding-shape smoke test, not an accuracy benchmark.

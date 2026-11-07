# SapiensID body and hybrid matching

[Documentation index](README.md)

- [Configure collection](#configure-collection)
- [Person detection and keypoints](#person-detection-and-keypoints)
- [Merge rules and publication](#merge-rules-and-publication)
- [Recognition command compatibility](#recognition-command-compatibility)
- [Storage and resource limits](#storage-and-resource-limits)
- [Validation limits](#validation-limits)
- [Independent person collection](#independent-person-collection)
- [Scene person identities](#scene-person-identities)
- [Global body consolidation](#global-body-consolidation)

Hybrid recognition keeps the configured face recognizer and its namespace.
SapiensID is an additional identity signal, stored against observations rather
than mutable performer IDs. The shipped v2 template collects body observations
using RF-DETR-XL boxes and PMPose points, including cues with no face.
Both shipped profiles enable scene-local body identities and cross-scene body merging.

## Configure collection

Install `requirements-cpu.txt` or `requirements-gpu.txt`; both include the
inference dependencies for every supported body/person/pose backend. On first use, `execution.models.download_missing: true`
allows pinned assets for the selected models to download. Relative asset paths
resolve from the YAML location. The [v2 template](../facial_recognition.yml)
lists every supported setting and its range or choices where applicable.

```yaml
version: 2
pose:
  detector:
    backend: rfdetr_xl
  estimator:
    backend: vitpose_plus_large
body:
  enabled: true
  encoder:
    variant: wb12m
  collection:
    enabled: true
```

A version 1 file keeps its original defaults, including the previous opt-in
collection policy. Configuration examples use v2; existing diagnostic reports
may still use internal v1 field names. To migrate an existing file, see [Recognition YAML v2](facial-recognition.md#recognition-yaml-v2).
The updater preserves local edits and may provide a `facial_recognition.yml.incoming` file.

Keep `face.recognizer.backend` set to your face recognizer. Downloads use the source
archive of `mk-minchul/sapiensid` at
`a8d8011117db23f20c7ade3bc38e8075810ac886`; an existing clean Git checkout at
that revision is also accepted. The required wb12m model.pth hash is
`b91db9d4ea13542b1bff76deaa0d678b9586c73506c7ae0116ab907f688a1a60`;
wb4m uses `8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02`.
The adapter verifies source contents and asset hashes before loading weights.

`body.encoder.variant` selects `wb12m` (default) or `wb4m`. Both use the same
ViT-Base architecture, 384x384 input and 4096-dimensional output. The 4M/12M
labels describe the training datasets. The authors report better wb12m results
on all five benchmarks in
[Table 12 of the paper](https://arxiv.org/html/2504.04708v1#A2.SS7), which motivates
the accuracy-oriented default. This does not establish the gain on your own scenes.

Model paths are download destinations, not prerequisites. Override them to
use another disk or reuse an existing installation. Empty paths in older YAMLs
use `sapiensid_v1/source`, `sapiensid_v1/<variant>` and
`sapiensid_v1/yolo26m-pose.pt` beneath `execution.models.directory`; an empty pose SHA256
selects the pinned default checkpoint. The legacy YOLOv8n hash with an empty path
selects its own `yolov8n-pose.pt` filename. A custom pose/model.yaml hash requires
an existing matching local file and is never silently replaced with a default.

YOLO26m-pose comes from the official
[Ultralytics v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Its 49,036,866-byte checkpoint is pinned by `pose.detector.yolo.sha256`
in the [configuration template](../facial_recognition.yml). Install the
updated CPU/GPU profile (`ultralytics>=8.4.189,<9`): inference explicitly
selects the one-to-many head with NMS, so `pose.detector.nms_threshold` remains effective.
Both models supply COCO17 points; SapiensID still receives 17 pose points plus
the five face landmarks, with the same 384x384 preparation and audit overlays.

## Person detection and keypoints

See [pose backends](recognition/pose-backends.md) for ViTPose++, PMPose, SAM refinement,
Sapiens Pose, D-FINE, RF-DETR-XL and YOLO recipes.

## Merge rules and publication

The planner uses the production quality/diversity-selected strict face references.
It compares face and person cosines separately; it never averages different
encoders into one score. Every original cross-identity witness must satisfy its
respective floor, and both identities need independent scene/timestamp support.
Body references are selected independently from admitted strict members. Missing
body data on a face reference does not veto a sufficiently supported body bank. Copresent faces must also
pass the existing stronger reflection/cooccurrence face threshold.

Every compatible strict identity remains a possible competitor, including weak
identities and face-compatible rivals without body evidence. Components must be
complete: A–B and B–C do not prove A–C. The ambiguity margin is checked against
all outsiders before a component can merge. References remain frozen for the
entire pass. Missing strict references are not reconstructed from attachments.

Automatic application runs after ordinary face rebuilding. It only applies
components whose entire retained membership is inside the selected scans; select
all involved scenes together to consolidate them. Partial selections can report
an eligible pair while correctly leaving it unapplied. The oldest identity wins.
Source samples become attachment-only and do not become new strict face anchors.
Hybrid query support is counted separately for strict face proofs and for each
frozen body appearance/core. Face-only hits cannot supply the body route's
minimum support. A newly born face identity is reassigned only if all of its
members retain independent support for the same target. Repeated physical crops
have one scoring weight as well as one independent vote; all vectors still keep
cohesion-veto power.
Original cores and numeric proof remain in alias evidence; candidate records
point to that proof without duplicating it per observation. On later passes,
the proof retains every body witness used by the complete mode check, even when
the representative cap is smaller. An unsupported mode or uncovered confirmed
mode blocks merging; it never falls back to a smaller representative-only proof.
Inherited cores on both endpoints must agree with each other as well as with
the current endpoints. Older proofs recover omitted vetoes from retained source
attachments without granting those attachments reference authority.

Original cores can still veto contradictory merges but cannot nominate merge
candidates or add support votes. Their frozen appearances can support query-only
recovery; recovered observations remain attachments. Missing or superseded original evidence blocks further
hybrid merges instead of hiding a contradictory endpoint.

Aliases are local SQLite aliases, following the current publication design.
Selected scene links are reconciled through the normal ownership-checked journal
and publication queue; old Stash profiles, names, images and manual links survive.
No new destructive performer-merge operation is introduced. `sync-identities`
handles publication debt; `cleanup` still protects known historical profiles.
Turning this feature off prevents new supplemental inference/merges, but does not
undo existing local aliases or restore earlier scene links.

A semantic warm stamp covers scoped scans, vectors, memberships, candidates,
aliases, origins, the model contract and matching policy. Changed evidence or
competitors invalidate it; publication-only status/binding changes do not force
another face rebuild. No checkpoint is treated as ground truth.

## Recognition command compatibility

| Action | Hybrid behavior |
|---|---|
| `run` | Collect missing evidence, rebuild faces, apply guarded hybrid plan, reconcile selected scene links. |
| `report` | Read-only frozen plan with pair gates, competing identities and missing evidence. |
| `report-identity` | Recorded hybrid explanations, attachment roles, retained body PNGs and association metadata. Annotated copies show face points and COCO17 pose points on retained crops, using audit colors. No models/media needed. |
| `report-merge` | Add measured hybrid proposals; `--simulate` verifies final membership through production replay. Merge goals and the solver retain all competing identities and set-separation constraints. |
| `audit-alignment` | Export the exact reconstructed model input, whitespace mask, 22 normalized points and annotated source/input PNGs. Loads pose and face detection, not SapiensID identity weights. |
| `evaluate` | Copy only admitted retained evidence into isolated trials; report coverage, matching settings and recorded decisions. Future selected scenes cannot supply body votes. |
| `tune` | Support the four `sapiensid.matching.*` parameters. Labelled holdout validation remains required for a recommendation. |
| `portraits` | Use canonical face memberships and existing face portrait scoring; no person encoder, no body portraits. |
| `sync-identities` | Reconcile the normal publication queue without models/media; preserve historical profiles. |
| `cleanup` | Keep existing ownership/orphan protections; supplemental evidence cannot authorize profile deletion. |

Reports and replay work without torch, CUDA or model files. They never create
missing embeddings. Their coverage distinguishes ready, unavailable and missing
current-contract evidence. Frozen-report recommendations do not guarantee a full
rebuild; use replay verification before applying a configuration. Offline replay
refuses changed person-input contracts. Full configuration exports resolve local
model paths so moving the YAML does not silently change their meaning.

Person alignment exports include `-person-source.png` with the measured person
rectangle and keypoints, and `-person-input-annotated.png` with the rectangle and
points transformed into the actual model input. Yellow marks the person bounds;
magenta points 0–16 are pose landmarks. Points 17–21 are the five face-detector
landmarks (cyan eyes, red nose, green mouth corners, matching the face audit).
Numbers index the 22-point array in the JSON. Source pose points use COCO order;
input points use the SapiensID left/right canonicalization. Missing or rejected
points are omitted, and ambiguous face/person associations produce no overlay.
The existing `-person-input.png` and `-person-mask.png` remain unannotated.
The manifest lists all four images and records the colors, drawn indices and
input rectangle in `person_input.annotations`. Drawing does not change model
inputs, embeddings or the person evidence contract.

## Storage and resource limits

See [recognition database compatibility](recognition/storage.md#schema-and-backups)
for schema migration and read-only behavior. Supplemental rows
contain an input contract, availability status, metadata, a normalized 4096D
float32 feature and an optional body crop. Matching thresholds, execution budgets
and local paths do not change the vector contract. Pose/preparation semantics do.
Writes invalidate existing observation dependency certificates. Face namespaces
and face scan identities remain unchanged.

Batch size is 1–16. CPU preparation builds bounded tensors; one owning thread
transfers and executes a batch. The scan pipeline can batch prepared person
inputs across VTT cues while preserving complete-cue checkpoint boundaries.
Collection does not construct another face detector. Retained PNGs
exceeding `storage.max_crop_bytes` are omitted with an explicit metadata reason;
features remain usable. Disabling person-crop retention saves space without
changing embeddings or face portraits.

Candidate nomination uses unnormalized centroids: their dot product equals the
mean of all cross-sample cosines. Products are computed in 64×64 blocks, without
an N×N matrix or approximate/top-k pruning. Only nominated pairs receive exact
original-witness gates. A second bounded pass checks every outsider, without an
N² Python rival list. Arithmetic remains quadratic in identity count.

`reference_bytes` bounds numerical reference/centroid working storage, including
conservative copy allowances; it is not a total process-RSS limit. `plan_bytes`
bounds serialized pair proofs, retained core history and alias evidence per
transaction; `max_candidate_pairs` bounds graph candidates.
Python object and SQLite overhead are additional. Exceeding a budget stops the
hybrid pass explicitly, without partially applying its plan or silently dropping
competitors. Collection, nomination, competitor checks and application expose
progress stages.

Run `python -m benchmarks.benchmark_hybrid_lookup --identities 256 --samples 8`
for a deterministic CPU nomination comparison. The checked-in result compares
32,640 pairs with identical candidate sets. It measures nomination only, excluding
exact candidate gates, SQLite, inference and media; it is not end-to-end speed or
accuracy evidence.

## Validation limits

Tests cover CPU tensor preparation/batch parity, association and malformed inputs,
cue failure/resume, missing evidence, contradictory faces, cooccurrence, ambiguous
rivals, incomplete components, budget failure, cold/warm rebuild, offline replay,
reports, tuning, merge goals and mocked Stash publication/maintenance.

No real checkpoint inference, CUDA throughput or labelled-corpus accuracy has
been validated in this implementation session. This adapter combines SCRFD/YuNet
face landmarks with YOLO pose; equivalence to upstream DFA alignment is not
established. Audit real inputs and calibrate against independent labels before
broad use. Upstream SapiensID source/weights use CC BY-NC 4.0; Ultralytics has
separate licensing.

## Independent person collection

Set `body.enabled: true` and `body.collection.enabled: true` to
collect every detected person in each sampled VTT frame, including frames
without faces. This replaces face-driven collection with a shared pass: one
pose prediction per frame and one embedding per eligible detected person.
A linked face receives the same feature through a compatibility projection.
Rejected faces never supply facial landmarks to an independent body query.
Ambiguous face/person ownership is withheld instead of guessed.

The independent records retain person bounds, pose points, optional face ID,
quality measurements, rejection reasons and optional PNGs. The quality controls
are `min_person_pixels`, `min_sharpness`, `min_keypoints`, `max_overlap`.
Overlap is measured bounding-box overlap, not a semantic occlusion estimate.
Crop-border contact is recorded for later portrait ranking. Body evidence does
not create identities in this first stage.

Completed person cues,
including empty cues, are checkpointed with the face scan. Existing scans can
be enriched by reopening their source frames without rerunning face inference.
Changing person-input or body-quality settings selects a new body contract.
Retained face vectors and their scan contracts remain unchanged.

## Scene person identities

In v2, `matching.body.identities.enabled` defaults to the value of `body.collection.enabled`.
Scene rebuilding creates body-only identities without requiring a visible face.
High-quality body anchors form complete-link cores; weak observations may attach
to a frozen accepted bank, but never become new anchors through assignment.
Direct face/person associations keep the existing face identity. Missing face
observations can query its independently supported, frozen body bank.
Each face owner's body bank is partitioned into frozen complete-link appearances.
An attachment needs independent support in the matching appearance; other
appearances of that owner do not dilute its score, and every competing owner
still participates in ambiguity checks. On a local split, the largest physical
overlap keeps the historical identity and performer; the oldest label wins a merge.

Exact body pixel fingerprints include RGB bytes and crop dimensions. This changes
the collection contract: the next run recollects the selected VTT body evidence.
Compatible old grayscale banks remain competitors, but cannot supply new merges
until rebuilt under the current contract. Physical origins preserve identity labels.

New cores require `min_support` independent cues, with at most one vote per
`(scan, cue)`, regardless of the time between cues. Exact pixel digests and
`min_appearance_distance` (64-bit perceptual hash distance) still prevent repeated
images from multiplying support, including across scans. `attach_min_support`
counts independent cues in the frozen body bank of a face identity. Remove the
obsolete `sapiensid.identities.min_seconds` key from existing configuration files;
loading it reports an explicit migration error. Rebuilding matching reuses retained
body vectors and does not require recollection. Provisional
cores remain local and are not published. `anchor_min_keypoints`,
`anchor_min_sharpness` and `anchor_min_person_pixels` are separate from collection
gates. The matching, cohesion and ambiguity thresholds are uncalibrated starting values.

`matching.body.anchors.min_person_pixels` defaults to `null` (disabled).
Set an integer from 1 to 8192 to require a minimum shorter side of the original
person crop for body anchors. For example, collection at 64 px with anchors at
96 px retains 64–95 px bodies as queries/auxiliary evidence; they cannot create
an identity, add anchor support or train a reference bank. A crop exactly 96 px
wide qualifies if its height is at least 96 px and the other anchor gates pass.
This also filters the body reference bank of face identities, including the
legacy face-driven SapiensID route when independent identities are disabled.
The associated face can still be a face anchor. Original merged body witnesses
retain veto power even if the new size gate removes their reference authority.

The gate uses retained source `quality.width`/`height`, falling back to legacy
`person.bounds` when dimensions are absent. Model input resolution and portrait
resizing never count. Missing or invalid source dimensions cannot qualify when
the gate is enabled. Reports include `anchor_pixel_gate` with the measurement,
threshold and exclusion reason. Changing this threshold rebuilds matching from
retained evidence without inference or recollection; it cannot recover bodies
previously rejected by collection. `null` preserves the prior admission behavior.

No temporal track is inferred from sparse VTT cues. Distinct simultaneous person
boxes cannot support one another. Memory and comparison budgets fail explicitly
before replacing any selected assignments. Confirmed body identities use the
normal owned Stash profile and scene-publication journal; they need no portrait
in order to publish. The database retains derived body decisions and stable origins.

## Global body consolidation

In v2, `matching.body.global.enabled` defaults to the value of `matching.body.identities.enabled`.
Set the global switch to `false` to keep body matching within each scene.
The global match, cohesion, ambiguity and support thresholds are separate from
local assignment. Candidate banks contain accepted body anchors only. Every
original core must agree; connected A–B–C edges do not establish A–C. Weak or
under-supported identities remain competitors. Missing face evidence does not
veto a match; contradictory admitted face evidence does. Face and SapiensID
vectors are never averaged or compared across encoder spaces.

Application requires all scenes of every participating identity in the selected
scope. Source anchors become attachments, while frozen original body and face
witnesses retain veto power. Missing original evidence blocks further merges.
Local aliases preserve old Stash profiles and manual metadata. Face-bearing
identities take precedence as the target, then creation order and ID break ties.
Publication validates the current collection contract, body decision policy and
complete cue/decision coverage. Missing body decisions are an incomplete rebuild,
not an empty scene. The same validation applies to journal-inventory recovery;
old face decisions cannot be adopted under a different matching policy. A
validated global reference repair records its current publication policy without
promoting the repaired attachments into new recognition references.
Body-owned aliases are revalidated when all involved scenes are selected; a
partial selection cannot reactivate or blindly publish a retired source core.
Turning global consolidation off and rerunning the full involved scope restores
the current scene partition for body-owned aliases only. Face-owned aliases are
not removed. A body core wholly confirmed by an existing face bank records a
local promotion alias even when global body consolidation is disabled.

Global body matching first compares unnormalized mean vectors in bounded tiles.
The reference cap limits nomination and rival lookup, not the complete core
cohesion check or persisted veto witnesses. If a replacement scan is incomplete
and its scene previously supplied identity memberships, reports block merge
decisions with `incomplete_reference_scans`. A SapiensID run must finish that scene before
rebuilding identities in the same scope and model namespace; the failed rescan
cannot silently remove a historical competitor.
Only nominated identity pairs and explicitly requested diagnostic pairs receive
full witness checks. A conservative rounding envelope preserves threshold-edge
candidates; exact retained-vector scores still decide every acceptance gate.
All identities remain eligible competitors, including under-supported and rejected
merge candidates. Outsider maxima are found in tiled passes and near ties are
checked exactly. Original-core, co-presence and face-contradiction vetoes remain.

Reference selection happens before embedding loading. The planner keeps bank
metadata and a bounded vector cache in RAM; float64 means are stored as contiguous blocks in a disposable
binary file in the system temporary directory. Allow roughly 33 KiB of temporary
disk space per appearance (about 318 MiB for 10,000 appearances). The index is removed
on success, failure or interruption. No source database writes or inference are
needed for reports. Repeated core checks reuse scores within the frozen pass.

`execution.rebuild.body.max_candidate_pairs` counts nominated identity pairs in the
hybrid/global planners. `max_witness_comparisons` (default 1,000,000) separately
limits exact vector comparisons per body scene or global pass, including original
core and face checks. `reference_bytes` and `plan_bytes` still bound working
references and retained proofs. Budget exhaustion raises an error before applying
a global merge plan; no rivals or original witnesses are silently dropped.

During a production rebuild, a byte-bounded derived cache reuses independent
scene results and numeric evidence across selections. Its entries share the
recognition database transaction. New or changed competitors still participate
in the global decision. The total entry budget is the smaller of 64 MiB and
one eighth of `execution.rebuild.cache_bytes`. Setting that byte budget to zero
disables durable reuse. The face-order, strict-scene, body-scene and hybrid-pair
cache counters appear in `rebuild_profile.counts` when profiling is enabled.
`hybrid_unchanged_pair_proofs_recomputed` counts newly computed hybrid proofs
whose two endpoint versions were retained from a previous rebuild. The
`hybrid_nomination`, `hybrid_copresence` and `hybrid_proof_scoring` entries in
`rebuild_profile.stages` time the corresponding work on the coordinator.

`execution.rebuild.nomination_device` defaults to `cpu`. Set it to `auto` to
try CUDA for hybrid centroid nomination on catalogues of at least 1,024
identities with a GPU-enabled model configuration. The planner checks memory
headroom and measures upload, tile computation and result transfer against CPU
work. It uses CPU for borderline nomination scores and after a CUDA failure.
Exact witness checks and competitor ranking continue on CPU.

Global reports declare `pair_coverage: candidates_and_requested`: `pairs` is a
sparse list, not an exhaustive list of every rejected pair. Pair reports and the
solver explicitly request their target pairs even below the current threshold;
`nearest_competitors` supplies their exact strongest external competitors.
The `search` block separates identity-pair, centroid, candidate, exact-witness,
embedding-read and cache-hit counts, with reserved reference bytes and index size.

`python -m benchmarks.benchmark_body_global --identities 10000 --samples 16
--matches 16 --threads 2` exercises the frozen planner without inference, source
catalogue SQL or publication. Recorded synthetic results in `benchmarks/results/`
verify the intended partition. Mean-vector nomination remains quadratic in the
identity count, but no all-pairs witness expansion or retained N-by-N score graph
is built. This benchmark does not certify end-to-end scanning time or recognition
accuracy on a real 10,000-scene library; dense candidate graphs and large inherited
witness banks can still reach their explicit budgets.


### Body portraits and offline tools

`portraits.selection.kind: auto` follows `portraits.selection.auto_precedence`, defaulting to
`[face+pose, face, pose]`: faces with usable associated SapiensID evidence under
the current contract, remaining faces, then accepted person crops. A lower tier
is considered only when the earlier tiers have no usable portrait, including
retained-pixel fallbacks; quality scores rank within each tier. All three entries
must appear exactly once. `face` restricts
selection to faces; `person` uses only retained person crops without loading face,
pose, visibility or identity models. `portraits.selection.anchor_only: true` requires a body
anchor for person crops, including after global aliases downgrade source samples.
Body ranking uses overlap, clipped borders, sharpness, visible keypoints and source
resolution. It applies `portraits.crop.aspect_ratio` to the retained crop, keeping
measured head landmarks with clearance where possible and preferring the top
when head points are unavailable. `null` preserves the original crop shape.
The source evidence remains unchanged; portrait metadata records the applied
crop in retained-person coordinates. It never invents a face.
`retain_person_crops: false` leaves no offline body portrait candidate.

The `face+pose` tier ranks and validates the face observation but frames its
associated person using the same head-aware aspect-ratio rules. It uses the
verified VTT thumbnail or that association's retained person crop, never a
retained face crop as a substitute. With `aspect_ratio: null` it keeps the whole
measured person box; face-only `margin` is not applied. Existing face-framed
paired portraits are regenerated by `recognition portraits` without an identity
rebuild; cached bytes cannot cross face/person framing or changed pose geometry.

Person portraits use the same ownership tokens, intent-before-write, digest checks,
retry reconciliation and manual-image preservation as face portraits. Auto mode
can replace an acknowledged generated image when a higher-priority candidate becomes available.
`recognition portraits` includes body-only performers and respects scene/performer
filters without rebuilding identities.

`report-identity` exports retained body crops, measured pose annotations, optional
face links, current roles and recorded decisions alongside face samples.
`audit-alignment` exports faceless model inputs and masks with all five unavailable
face landmarks left at `(-1,-1)`; it shares the frame's pose pass and loads no
identity encoder. Fragmentation and pair reports include body coverage and global
blockers. Pair recommendations from the independent-person route require production
replay, rather than accepting a frozen pair score alone.

Replay, labelled offline evaluation, `tune` and merge-goal solving accept independent
body observation IDs. Face-associated evidence already represented by an accepted
face is not counted again as another merge-goal sample. Body anchors contribute
core support; auxiliary bodies remain soft evidence. Solver trials can vary the
local and global similarity, cohesion, margin and support gates, as well as
`attach_min_support`, `min_appearance_distance`, `anchor_min_keypoints`,
`anchor_min_sharpness` and a numeric `anchor_min_person_pixels`. These parameters
participate in heuristic proposals, bidirectional search, refinement and TPE
sampling with their integer or numeric domains. Quality proposals use retained
keypoint, sharpness and source-size measurements;
each candidate is validated by full production replay without inference or source
database writes. Body goal support also uses cues, without a minimum time gap.
The automatic solver leaves a `null` size gate disabled; set a number to enable
its search dimension, including for legacy face-associated body references.
An explicit `tune` grid can compare both numeric thresholds and `null`, for example
`{"matching.body.anchors.min_person_pixels": [null, 64, 96, 128]}`.
Input/collection and portrait settings remain outside the solver; input/collection
changes still require recollection. Automatic memberships are not truth labels.

Evaluate the matching thresholds with representative masked, out-of-frame,
similar-clothing and changing-clothing examples from your library. Synthetic lifecycle tests
verify inference reuse, missing-face handling, replay and publication; they do not
measure SapiensID recognition accuracy.

Body anchors in both recognition routes now share `identities.anchor_min_keypoints`,
`anchor_min_sharpness` and `anchor_min_person_pixels`. Missing/non-finite measurements
cannot supply anchor authority; weak body observations may still be attachment queries.
The face-associated collector measures the same source crop quality as independent
collection, even when independent body identities are disabled. Changing these
thresholds invalidates matching decisions, not retained embeddings.

### Regional body anchor keypoints

`matching.body.anchors.keypoint_rules` adds regional requirements **alongside**
the global `anchor_min_keypoints` floor. The default `{all: [], any: []}` preserves
existing admission. Rules apply to both independent body anchors and face-associated
body references, including when `identities.enabled` is false. Face anchor authority
itself is unchanged; a weak body sample can still be an attachment query.

```yaml
version: 2
matching:
  body:
    anchors:
      min_keypoints: 6
      keypoint_rules:
        all:
        - group: shoulders
          min: 1
        - group: hips
          min: 1
        - points:
          - left_elbow
          - right_elbow
          - left_wrist
          - right_wrist
          min: 2
        any:
        - group: left_leg
          min: 2
        - group: right_leg
          min: 2
```

Every `all` rule and at least one `any` rule must pass. An empty list adds no
requirement. The example requires a shoulder, a hip, two elbow/wrist points and
two points on at least one leg; it is illustrative, not a calibrated default.
Each rule accepts exactly one `group` or a nonempty `points` list, plus `min`.
The minimum is an integer from zero to the number of distinct selected points;
zero adds no requirement (and makes an `any` alternative pass). Unknown names,
duplicate custom points, extra keys and impossible minima fail configuration
validation before recognition starts.

| Group | COCO17 points | Maximum |
| --- | --- | ---: |
| `eyes` | left/right eye | 2 |
| `head_without_eyes` | nose, left/right ear | 3 |
| `head` | nose, both eyes, both ears | 5 |
| `shoulders` | left/right shoulder | 2 |
| `hips` | left/right hip | 2 |
| `torso` | shoulders and hips | 4 |
| `left_torso`, `right_torso` | shoulder and hip on that side | 2 each |
| `left_arm`, `right_arm` | shoulder, elbow, wrist on that side | 3 each |
| `arms` | both arms | 6 |
| `left_leg`, `right_leg` | hip, knee, ankle on that side | 3 each |
| `legs` | both legs | 6 |
| `body` | all points except nose, eyes and ears | 12 |

Custom point names are `nose` and `left_`/`right_` followed by `eye`, `ear`,
`shoulder`, `elbow`, `wrist`, `hip`, `knee` or `ankle`. These are pose landmarks,
not the additional five face-detector landmarks. All supported pose backends use
this COCO17 vocabulary; dense Sapiens points do not add extra rule points.
Groups may overlap, but each physical point counts only once in the global total.
No head/eye requirement is added by default, so back views remain eligible.

Left/right means the detector's anatomical label, even after SapiensID swaps
paired points into image-side order. Counts include only points surviving the
confidence, crop-boundary and encoder head-outlier filters. No missing point is
invented. Existing coordinates suffice for offline rule changes; this does not
change the embedding/collection contract. Legacy rows with only a scalar count
cannot satisfy positive regional minima: reports mark their regional measurement
unknown, and they need recollection to provide the missing geometry (or an
explicitly relaxed rule). Matching and body reference caches include these rules.

Identity report threshold JSON/PNG companions show each selection and its measured
count/minimum, plus separate `all`/`any` results. A failed alternative does not fail
a passing `any` group. `anchor_gate.keypoint_rules` also exposes selected/present
point names; failed regional gates use `body_anchor_keypoint_rules` or
`body_anchor_keypoint_rules_missing` when geometry is unavailable.

`report-merge --solve` keeps the configured regional rules for the baseline and
throughout numeric search. Diagnostics still show which landmarks failed a rule.
The separate `recognition tune` command accepts explicitly requested indexed
minima such as `sapiensid.identities.anchor_keypoint_rules.all.0.min`.

With `--compare-anchor-rules`, the solver reserves up to two trials within
`--max-trials`. It takes the best completed trial using the configured rule and
replays the same scalar settings with each distinct whole policy below.

| Profile | Required visible pose landmarks |
| --- | --- |
| `shoulder_and_hip` | At least one shoulder and one hip. Admits views from behind. |
| `face_or_torso` | Nose and both eyes, or at least three of the four shoulders/hips. Admits close-ups and back views. |

The configured policy runs first even when it is custom or empty. If it already
matches a preset, that preset is skipped. With a budget of one trial only the
configured policy is replayed. Reports identify each comparison and its control
trial. A winning alternative is exported with both complete `all` and `any` lists.

Pose points with sufficient confidence at x=0 or y=0 are valid inside the person
box. Right/bottom edges remain exclusive and low-confidence points remain missing.
The person input contract is revised for every model variant (including the legacy
pose model), so recognition recollects affected person evidence instead of reusing
vectors prepared with discarded boundary points. Compatible older independent
body banks remain conservative rivals until rebuilt.

Normal recognition rebuilds and query recovery build pixel aliases only for the
selected server scope and model namespace. Evidence from another server/model
cannot exhaust the current run's alias working budget.

Building disk-backed appearance modes now shares one
`execution.max_witness_comparisons` budget across all owners in the catalogue.
The limit is checked before the next exact dot product and progress reports
completed comparisons. An exhausted budget fails the rebuild atomically;
incomplete mode banks cannot authorize merges.

Local body grouping can certify a sole candidate group using its fixed pivot and
maximum distance to every retained witness. Cauchy–Schwarz gives a conservative
lower bound for every similarity, including a float32 rounding allowance. Only a
bound above both match and cohesion gates skips the full cross product. Competing
groups, score reporting and inconclusive bounds still use exact witness scores.
This also accelerates construction of a single frozen face-owner appearance.
The 1415-distinct-vector regression uses fewer than three comparisons per sample
instead of exceeding one million, with the same final assignments.


Independent body support uses body crop fingerprints, including face-associated
projections; face crop hashes and face aliases do not supply body votes. Perceptual
deduplication reduces support without removing original witnesses from veto checks.
Anchor keypoint counts reflect the normalized pose points actually used by the
encoder. Collection rejection remains effective on the compatibility projection.

Scene body groups are rechecked against all frozen competing cores before they
receive support or stable labels. A dense appearance cannot conceal a later core
by raising a mixed group's mean. After global confirmation, weak scene observations
may attach in a separate frozen pass; these attachments never train reference banks.
Face-owned global banks expose individual appearances for nomination and competition,
while complete original cores still control merge acceptance.

Disabling a body route retires its aliases only when all involved scenes are selected.
Original face authority is restored before replay. Retained anchor evidence from an
incompatible body contract blocks global merges and identifies scenes to recollect.

Face vetoes and simultaneous face claims share the scene comparison budget with body
scores. Identical features may reuse a bounded cached score; distinct comparisons
remain charged, and streamed checks observe cancellation. Completed body cues repair
missing face projections from retained features before loading a model or decoding.
Absent/ambiguous associations and rejected bodies stay unavailable. Optional crop
backfill may still decode frames when retained images are requested.

Normal publication revalidates scene decisions and the publication queue revision
before writes and acknowledgement. Changed assignments retain synchronization debt.
A guard or ownership failure before HTTP dispatch is retryable; an uncertain outcome
after dispatch still requires reconciliation. Synchronization inventory includes
current accepted body-only identities, without counting provisional body cores.

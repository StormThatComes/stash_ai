# Supplemental SapiensID evidence

Hybrid recognition keeps the configured face recognizer and its namespace.
SapiensID is an additional identity signal, stored against observations rather
than mutable performer IDs. It is opt-in (`sapiensid.enabled: false` by default).
Defaults are uncalibrated starting values, not accuracy claims.

## Enable collection

Install `requirements-sapiensid.txt` alongside the normal runtime, preserving the
CPU/CUDA PyTorch build appropriate for your installation. With
`models.download_missing: true` (the default), the first inference that needs
SapiensID automatically downloads its pinned upstream source, wb12m checkpoint,
`model.yaml` and YOLO26m-pose checkpoint into the configured destinations.
You do not need to create those folders, clone the upstream repository, install
Git, or manually download weights. Relative paths resolve from the configuration
file. Python packages remain managed through the requirements file.

The shipped [facial_recognition.yml](../facial_recognition.yml) lists every
supported setting, including all SapiensID subsections, with comments and
explicit default values for SapiensID. The example below only shows the settings
needed to enable collection and its main matching/resource controls; omitted
settings inherit their defaults. SapiensID uses `models.device` and the shared
`execution.scan_cpu_workers` / `execution.scan_queue_bytes` controls.

If the updater preserves your locally modified YAML, review
`facial_recognition.yml.incoming` and copy the new settings you need. Partial
local configurations remain supported; omitted settings inherit their defaults.

```yaml
models:
  download_missing: true
sapiensid:
  enabled: true
  model:
    variant: wb12m
    source_directory: recognition_models/sapiensid_v1/source
    checkpoint_directory: "" # models.directory/sapiensid_v1/<variant>
    model_yaml_sha256: 2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9
  pose:
    path: recognition_models/sapiensid_v1/yolo26m-pose.pt
    sha256: 2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4
  matching:
    face_floor: 0.4
    person_threshold: 0.85
    ambiguity_margin: 0.1
    min_support: 3
  execution:
    batch_size: 1
    reference_bytes: 134217728
    plan_bytes: 16777216
    max_candidate_pairs: 100000
    max_witness_comparisons: 1000000
  storage:
    retain_person_crops: true
```

Keep `models.recognizer` set to your face recognizer. Downloads use the source
archive of `mk-minchul/sapiensid` at
`a8d8011117db23f20c7ade3bc38e8075810ac886`; an existing clean Git checkout at
that revision is also accepted. The required wb12m model.pth hash is
`b91db9d4ea13542b1bff76deaa0d678b9586c73506c7ae0116ab907f688a1a60`;
wb4m uses `8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02`.
The adapter verifies source contents and asset hashes before loading weights.

`sapiensid.model.variant` selects `wb12m` (default) or `wb4m`. Both use the same
ViT-Base architecture, 384x384 input and 4096-dimensional output. The 4M/12M
labels describe the training datasets. The authors report better wb12m results
on all five benchmarks in
[Table 12 of the paper](https://arxiv.org/html/2504.04708v1#A2.SS7), which motivates
the accuracy-oriented default. This does not establish the gain on your own scenes.

The paths above are download destinations, not prerequisites. Override them to
use another disk or reuse an existing installation. Empty paths in older YAMLs
use `sapiensid_v1/source`, `sapiensid_v1/<variant>` and
`sapiensid_v1/yolo26m-pose.pt` beneath `models.directory`; an empty pose SHA256
selects the pinned default checkpoint. The legacy YOLOv8n hash with an empty path
selects its own `yolov8n-pose.pt` filename. A custom pose/model.yaml hash requires
an existing matching local file and is never silently replaced with a default.

YOLO26m-pose comes from the official
[Ultralytics v8.4.0 release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
Its 49,036,866-byte checkpoint is pinned by the SHA256 shown above. Install the
updated `requirements-sapiensid.txt` (`ultralytics>=8.4.189,<9`): inference explicitly
selects the one-to-many head with NMS, so `pose.nms_threshold` remains effective.
Both models supply COCO17 points; SapiensID still receives 17 pose points plus
the five face landmarks, with the same 384x384 preparation and audit overlays.

### ViTPose++ Large keypoints

For a separate top-down pose estimator, install the updated
`requirements-sapiensid.txt` and select:

```yaml
sapiensid:
  enabled: true
  pose:
    backend: vitpose_plus_large
    batch_size: 4
    vitpose_directory: ""
```

The dependency file includes SciPy, required by the Transformers processor for
affine crops and heatmap decoding. If an existing installation reports
`NameError: name 'inv' is not defined`, reinstall the updated
`requirements-sapiensid.txt` in the active CPU/GPU environment and restart the
process. Missing SciPy is now reported before ViTPose assets are downloaded.

YOLO still detects person boxes using the existing `pose.path`, `sha256`,
`score_threshold`, `nms_threshold` and `max_people_per_frame` settings. ViTPose++
Large replaces its keypoints for **every** accepted person, including people
without a detected face. The COCO expert (`dataset_index=0`) returns 17 points
in image coordinates. Shared boundary/confidence filtering, face association,
SapiensID's 22-point input and audit overlays remain in use. `audit-detection`
and `audit-alignment` select the same backend as recognition. The default
`backend: yolo` preserves existing configurations and evidence contracts.

The first use downloads the three pinned files from
[`usyd-community/vitpose-plus-large`](https://huggingface.co/usyd-community/vitpose-plus-large/tree/e211df377e49c89dba508ab3e83ddef13c0832b4)
at revision `e211df377e49c89dba508ab3e83ddef13c0832b4`: `config.json`,
`preprocessor_config.json` and `model.safetensors` (1,737,705,492 bytes).
SHA256 checks cover downloads and existing files. The default directory is
`models.directory/vitpose_plus_large/<revision>`; `vitpose_directory` overrides
it relative to the YAML. With `models.download_missing: false`, all three
matching files must already exist. No remote Python code or pickle weights
are loaded for ViTPose. Only selecting this backend loads its assets.

Inference runs on `models.device` in FP32, with at most `pose.batch_size` people
per batch (1..16; independent of the SapiensID embedding batch). Lower this
setting if GPU memory is tight. Cost grows with the number of people in a cue.
ViTPose heatmap peaks are clipped to [0,1] for the existing
`keypoint_score_threshold`; they are not calibrated probabilities and should
be tuned separately from YOLO scores. Missing face points in rear views are
allowed; increasing the number of accepted points does not prove better accuracy.

Switching the backend changes the person/body evidence contract. Run recognition
again on the selected scenes to collect ViTPose-based embeddings before using
the solver; retained face detections can be reused. Paths and pose batch sizes
do not invalidate embeddings. Model revision, weights and preprocessing do.
Accuracy on rear-view VTT thumbnails needs comparison on the operator's scenes.

Offline regression coverage includes the real Transformers processor and a tiny
random-weight model. To additionally exercise the pinned pretrained Large model
on CPU, set `STASH_AI_VITPOSE_SMOKE=1` and run
`python -m pytest -q tests/test_recognition_vitpose_smoke.py` (downloads ~1.8 GB,
including YOLO; no SapiensID identity checkpoint is needed).

### Sapiens Pose keypoints

`sapiensid.pose.backend: sapiens` selects Meta's **Sapiens v1 Pose** as another
top-down pose estimator. It is separate from the SapiensID identity encoder and
from Sapiens2. Install the existing `requirements-sapiensid.txt` alongside your
CPU/GPU requirements; no MMPose/MMCV installation is needed.

```yaml
models:
  download_missing: true
sapiensid:
  enabled: true
  pose:
    backend: sapiens
    sapiens_variant: "0.3b"
    sapiens_batch_size: 1
    sapiens_directory: ""
```

| `sapiens_variant` | Public Meta model | Download size |
| --- | --- | --- |
| `0.3b` (default) | [Sapiens Pose 0.3B](https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript) | 1.32 GB |
| `0.6b` | [Sapiens Pose 0.6B](https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript) | 2.65 GB |
| `1b` | [Sapiens Pose 1B](https://huggingface.co/facebook/sapiens-pose-1b-torchscript) | 4.68 GB |

Only the selected variant is downloaded, at a pinned revision, with size and
SHA256 verification before loading. The public Goliath TorchScript exports do
not require a Hugging Face token. Files are cached under
`models.directory/sapiens_pose/<variant>/<revision>/`. `sapiens_directory`
overrides the directory relative to the YAML and must contain the selected
model's original `.pt2` filename. `models.download_missing: false` requires an
existing verified file. A corrupt or different checkpoint is rejected rather
than overwritten. The upstream weights have a separate CC BY-NC 4.0 license;
see [third-party notices](../THIRD_PARTY_NOTICES.md).

YOLO supplies person boxes with its pose head disabled. Sapiens alone supplies
keypoints for those boxes, including people without a detected face. The model
predicts 308 Goliath landmarks; the adapter selects and reorders the matching
17 COCO body points (including wrists at Goliath indices 62 and 41). The other
landmarks are not retained. SapiensID still receives these 17 points plus the
five landmarks from the face detector; existing association, missing-point
filtering and audit overlays apply.

Crops use 1.25 padding, the 768×1024 target aspect ratio, RGB training
normalization, and UDP affine sampling/DARK subpixel heatmap decoding. Inference
uses FP32 on `models.device`, without flip augmentation or compilation.
`sapiens_batch_size` bounds people per Sapiens inference, including across cues;
`pose.batch_size` continues to bound YOLO frame batches. Start with one person
per Sapiens batch: the 1K input and larger variants require substantially more
memory and compute than YOLO. GPU throughput and VRAM requirements depend on
the variant and other loaded models; no fixed capacity is assumed.

Heatmap maxima are clipped to [0,1] and are not calibrated probabilities.
Tune `pose.keypoint_score_threshold` independently for this backend. Changing
backend or variant changes the person/body evidence contract; run recognition
again on the selected scenes before solving. Changing Sapiens paths or batch
limits leaves evidence reusable. Existing YOLO/ViTPose contracts are preserved.

The regular tests cover geometry, channel mapping, batching, downloads, missing
faces, and a small real TorchScript model. The optional full pretrained CPU test
also runs YOLO detection, face association, 22-point SapiensID preparation and
audit export:

```sh
STASH_AI_SAPIENS_POSE_SMOKE=1 python -m pytest -q tests/test_recognition_sapiens_pose_smoke.py
```

This downloads the 0.3B model and YOLO, without the SapiensID identity encoder.
`STASH_AI_SAPIENS_POSE_DIRECTORY` can point the smoke test at cached weights.
Integration tests do not establish an accuracy improvement on rear-view VTT
thumbnails; compare those scenes with `recognition audit-detection`.

### Shared geometry and existing YOLO configurations

After a unique face/person association, landmarks outside the actual person
crop are marked missing (`-1,-1`), including face-detector points and coordinates
that round onto the excluded right/bottom edge in float32. They are not clamped
to the border. This applies to scan preparation, batched inference and alignment
audits, so a crop-boundary disagreement does not abort the scene. Malformed or
non-finite geometry still fails validation; ambiguous associations remain withheld.

To switch an existing installation, update **both** `pose.path` and `pose.sha256`
to the values above, or set both to empty strings to use the default model folder.
An explicit old path/hash remains YOLOv8n; it is not silently migrated. Its pinned
checkpoint is still downloaded when missing. Existing files are never overwritten
to switch models. Configurations that omit pose settings now select YOLO26m.

Changing the pose checkpoint selects a separate evidence contract. The next
`run` collects new person embeddings while reusing retained face detections.
Reports, merge planning, tune and replay select that same contract and do not mix
YOLOv8n and YOLO26m evidence. Explicitly selecting the old pinned YOLOv8n preserves
its existing evidence contract. This change does not undo historical merges or
establish accuracy on your scenes; recalibrate hybrid matching on labelled examples.

An optional real-weight CPU smoke checks download integrity, NMS selection,
COCO17 association and the exact 22-point audit input without fetching identity
weights: `STASH_AI_POSE_SMOKE=1 python -m pytest -q tests/test_recognition_pose_smoke.py`.

For an existing wb4m installation, set `variant: wb12m` and
`checkpoint_directory: ""` to download wb12m into its separate default folder.
To retain wb4m instead, explicitly set `variant: wb4m`; an omitted variant now
means wb12m. Explicit checkpoint paths are always respected: pointing wb12m
at existing wb4m weights fails verification with an actionable error, without
overwriting those files. With an empty checkpoint path, changing only the variant
automatically selects the corresponding folder.

The two variants have identical model.yaml files, so their weight hashes are
part of evidence selection. Changing variant collects separate person embeddings
on the next `run`; it reuses the face scan and does not mix embeddings across
variants. Existing wb4m evidence remains valid when wb4m is explicitly selected.
Reports, merge planning and replay only use the selected variant's evidence;
without it they report missing evidence. Changing models does not undo historical
identity merges. Matching thresholds remain uncalibrated defaults and should be
checked against labelled examples after switching variants.

The first full setup transfers about 52 MB of source, 1.53 GB of compressed model
weights and 49 MB of pose weights. Either identity variant expands to about 1.65 GB.
HTTP range requests select only the configured variant from the official
upstream Google Drive archive. If a proxy ignores ranges, the downloader streams
past the preceding archive bytes instead. Downloads show byte progress, use
temporary files and publish only after size/SHA256 verification. Interrupted or
invalid downloads are removed; a retry starts that missing file again.
Completed files are reused. Existing invalid files or modified source produce
an error rather than being overwritten.

For an offline installation, prepare those assets once, then set
`models.download_missing: false`. Missing assets then fail with their path and
the disabled setting. Reading configuration, reports and replay do not download
SapiensID assets; `audit-alignment` needs pose only, without identity weights.

`run` collects missing current-contract features even when its face scan is
reused. It reopens the VTT source and associates retained face boxes with body
pose; it does not rerun the face encoder. All retained detections, including
rejected faces, participate in association ambiguity checks. Only faces already
accepted by scanning receive supplemental features. Body-only recognition and
rescuing rejected faces are outside this implementation.

Missing/ambiguous associations become explicit unavailable records. They are
cached for the same input contract. Inference failures propagate, preserve
previously completed cues and never become false negative identity evidence.
Each cue commits atomically. Changing pose settings creates a new contract and
allows unavailable associations to be measured again.

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

Writer initialization migrates SQLite versions 1–5 to version 6 transactionally.
Read-only reports can still open version 5 without migrating it. Supplemental rows
contain an input contract, availability status, metadata, a normalized 4096D
float32 feature and an optional body crop. Matching thresholds, execution budgets
and local paths do not change the vector contract. Pose/preparation semantics do.
Writes invalidate existing observation dependency certificates. Face namespaces
and face scan identities remain unchanged.

Batch size is 1–16. CPU preparation builds bounded tensors; one owning thread
transfers and executes a batch. Batches stay within a frame, avoiding additional
frame queues. Collection does not construct another face detector. Retained PNGs
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

Set `sapiensid.enabled: true` and `sapiensid.collection.enabled: true` to
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

SQLite version 9 migrates versions 1–8 transactionally. Completed person cues,
including empty cues, are checkpointed with the face scan. Existing scans can
be enriched by reopening their source frames without rerunning face inference.
Changing person-input or body-quality settings selects a new body contract.
Retained face vectors and their scan contracts remain unchanged.

## Scene person identities

With independent collection enabled, opt in to `sapiensid.identities.enabled`.
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

`sapiensid.identities.anchor_min_person_pixels` defaults to `null` (disabled).
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
in order to publish. SQLite v10 adds derived body decisions and stable origins.

## Global body consolidation

Enable `sapiensid.global_matching.enabled` after calibrating scene identities.
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
metadata and a bounded vector cache in RAM; float64 means are stored in a disposable
SQLite index in the system temporary directory. Allow roughly 33 KiB of temporary
disk space per appearance (about 318 MiB for 10,000 appearances). The index is removed
on success, failure or interruption. No source database writes or inference are
needed for reports. Repeated core checks reuse scores within the frozen pass.

`sapiensid.execution.max_candidate_pairs` counts nominated identity pairs in the
hybrid/global planners. `max_witness_comparisons` (default 1,000,000) separately
limits exact vector comparisons per body scene or global pass, including original
core and face checks. `reference_bytes` and `plan_bytes` still bound working
references and retained proofs. Budget exhaustion raises an error before applying
a global merge plan; no rivals or original witnesses are silently dropped.

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

`portraits.kind: auto` prefers an eligible face without portrait quality shortfalls,
then an accepted person crop, then the existing face fallback. `face` restricts
selection to faces; `person` uses only retained person crops without loading face,
pose, visibility or identity models. `portraits.anchor_only: true` requires a body
anchor for person crops, including after global aliases downgrade source samples.
Body ranking uses overlap, clipped borders, sharpness, visible keypoints and source
resolution. It applies `portraits.aspect_ratio` to the retained crop, keeping
measured head landmarks with clearance where possible and preferring the top
when head points are unavailable. `null` preserves the original crop shape.
The source evidence remains unchanged; portrait metadata records the applied
crop in retained-person coordinates. It never invents a face.
`retain_person_crops: false` leaves no offline body portrait candidate.

Person portraits use the same ownership tokens, intent-before-write, digest checks,
retry reconciliation and manual-image preservation as face portraits. Auto mode
can replace an acknowledged generated body image when a good face becomes available.
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
`{"sapiensid.identities.anchor_min_person_pixels": [null, 64, 96, 128]}`.
Input/collection and portrait settings remain outside the solver; input/collection
changes still require recollection. Automatic memberships are not truth labels.

The opt-in defaults are starting values, not calibrated accuracy guarantees. Test
with representative masked, out-of-frame, similar-clothing and changing-clothing
examples before enabling global matching for a library. Synthetic lifecycle tests
verify inference reuse, missing-face handling, replay and publication; they do not
measure SapiensID recognition accuracy.

Body anchors in both recognition routes now share `identities.anchor_min_keypoints`,
`anchor_min_sharpness` and `anchor_min_person_pixels`. Missing/non-finite measurements
cannot supply anchor authority; weak body observations may still be attachment queries.
The face-associated collector measures the same source crop quality as independent
collection, even when independent body identities are disabled. Changing these
thresholds invalidates matching decisions, not retained embeddings.

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

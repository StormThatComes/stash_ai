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
Original cores and numeric proof remain in alias evidence; candidate records
point to that proof without duplicating it per observation. On later passes,
original cores can still veto contradictory merges but cannot nominate merge
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

New cores require `min_support` independent observations. `min_seconds`, exact
pixel digests and `min_appearance_distance` (64-bit perceptual hash distance)
prevent adjacent/repeated thumbnails from multiplying support. Provisional
cores remain local and are not published. `anchor_min_keypoints` and
`anchor_min_sharpness` are separate from collection gates. The matching,
cohesion and ambiguity thresholds are uncalibrated starting values.

No temporal track is inferred from sparse VTT cues. Distinct simultaneous person
boxes cannot support one another. Memory and comparison budgets fail explicitly
before replacing any selected assignments. Confirmed body identities use the
normal owned Stash profile and scene-publication journal; they need no portrait
in order to publish. SQLite v10 adds derived body decisions and stable origins.

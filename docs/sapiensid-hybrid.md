# Supplemental SapiensID evidence

Hybrid recognition keeps the configured face recognizer and its namespace.
SapiensID is an additional identity signal, stored against observations rather
than mutable performer IDs. It is opt-in (`sapiensid.enabled: false` by default).
Defaults are uncalibrated starting values, not accuracy claims.

## Enable collection

Install `requirements-sapiensid.txt` alongside the normal runtime, preserving the
CPU/CUDA PyTorch build appropriate for your installation. With
`models.download_missing: true` (the default), the first inference that needs
SapiensID automatically downloads its pinned upstream source, wb4m checkpoint,
`model.yaml` and YOLOv8n-pose checkpoint into the configured destinations.
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
    source_directory: recognition_models/sapiensid_v1/source
    checkpoint_directory: recognition_models/sapiensid_v1/wb4m
    model_yaml_sha256: 2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9
  pose:
    path: recognition_models/sapiensid_v1/yolov8n-pose.pt
    sha256: c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212
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
that revision is also accepted. The required model.pth hash is
`8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02`.
The adapter verifies source contents and asset hashes before loading weights.

The paths above are download destinations, not prerequisites. Override them to
use another disk or reuse an existing installation. Empty paths in older YAMLs
use `sapiensid_v1/source`, `sapiensid_v1/wb4m` and
`sapiensid_v1/yolov8n-pose.pt` beneath `models.directory`; an empty pose SHA256
selects the pinned default checkpoint. A custom pose/model.yaml hash requires
an existing matching local file and is never silently replaced with a default.

The first full setup transfers about 52 MB of source, 1.53 GB of compressed wb4m
weights and 6.8 MB of pose weights. The wb4m weights expand to about 1.65 GB.
HTTP range requests select wb4m from the official upstream Google Drive archive
without downloading wb12m. If a proxy ignores ranges, the downloader streams
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
| `report-identity` | Recorded hybrid explanations, attachment roles, retained body PNGs and association metadata. No models/media needed. |
| `report-merge` | Add measured hybrid proposals; `--simulate` verifies final membership through production replay. Merge goals and the solver retain all competing identities and set-separation constraints. |
| `audit-alignment` | Export the exact reconstructed model input, whitespace mask and 22 normalized points. Loads pose and face detection, not SapiensID identity weights. |
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

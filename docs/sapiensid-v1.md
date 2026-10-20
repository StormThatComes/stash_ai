# SapiensID v1 recognition backend

Select `models.recognizer: sapiensid_v1` to use the official **wb4m** PyTorch
model with 4096-dimensional, unit-normalized person embeddings. The existing
CPU/GPU torch and torchvision installation is shared with JoyTag. No JAX,
Equinox, ONNX conversion, training stack, or background task is required.

This integration uses **face-anchored person crops**. It detects faces with the
configured SCRFD/YuNet model and persons/17 pose landmarks with YOLOv8n-pose,
then requires a unique association in both directions: the body box must contain
the face and a measured nose/eye must lie in the face box. Missing or ambiguous
associations are retained as rejected observations without an embedding. The
adapter never chooses the first/largest person in a multi-person frame.

Consequently a person **without a detected, eligible face is not recognized**
by this implementation. Existing `detection.*` gates still apply to the face.
This preserves face observations, cooccurrence rules, alignment audits, and
face portraits while adding information from the associated body to matching.
It does not claim body-only presence detection.

## Installation

1. Keep your existing `requirements-cpu.txt` or `requirements-gpu.txt` environment.
   Install `python -m pip install -r requirements-sapiensid.txt` into it. Extras
   include Ultralytics, timm, OmegaConf, easydict, pandas and matplotlib. The
   upstream training requirements are not needed.
2. Clone https://github.com/mk-minchul/sapiensid and check out the exact revision
   `a8d8011117db23f20c7ade3bc38e8075810ac886`. Keep the checkout clean. Source is
   loaded only for the model architecture, not the upstream YOLO/DFA aligner.
3. Obtain the official checkpoint archive linked from
   https://github.com/mk-minchul/sapiensid/blob/main/tasks/sapiensID/README.md.
   Extract **checkpoints/sapiensid_wb4m/model.pth** and **model.yaml** to a local directory.
   The model must be 1,646,175,550 bytes with SHA256
   `8df1e2d7a64fbc87969326862aeafccc83eb5ee0462a4f007d7f09603782cf02`.
4. Obtain a trusted local **yolov8n-pose.pt** checkpoint from Ultralytics.
   Verify its provenance and calculate its SHA256 independently. The runtime
   requires a supplied hash and opens a verified temporary copy. Ultralytics
   may deserialize pickle internally; only use trusted publisher weights.
5. The released `model.yaml` SHA256 is
   `2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9`.
   Verify it and configure the paths below. Paths are
   relative to `facial_recognition.yml`; absolute paths are also supported.

```yaml
models:
  detector: scrfd_10g
  recognizer: sapiensid_v1
  device: cuda
sapiensid:
  model:
    source_directory: ../sapiensid
    checkpoint_directory: recognition_models/sapiensid_v1/wb4m
    model_yaml_sha256: "2aa66245bdf81631cf243aff26dc2bca30788ea77958981d1947bd750cbf93b9"
  pose:
    path: recognition_models/sapiensid_v1/yolov8n-pose.pt
    sha256: "<64-character lowercase SHA256 of yolov8n-pose.pt>"
    score_threshold: 0.50
    keypoint_score_threshold: 0.50
    nms_threshold: 0.70
    max_people_per_frame: 100
```

Then use the normal command, for example:

```bash
python stash_ai.py recognition run --scene-id 5
```

`models.download_missing` continues to control the standard face/quality assets.
It does not fetch SapiensID source or these local checkpoints. Missing/incompatible
assets fail model initialization instead of falling back to a face recognizer.

## Input and provenance

Each associated person is cropped from the current VTT cue (or decoded video
frame). RGB tensor input is padded symmetrically with white to a square,
resized to **384×384** with bilinear interpolation (`antialias=None`), and
normalized with mean/std `0.5/0.5`. The same padding transform is applied to
**17 measured YOLO body points + 5 configured face-detector landmarks**;
unavailable body points remain `(-1,-1)`. Left/right pose canonicalization and
the whitespace foreground mask match upstream. The model's KP19 preprocessor
selects its own 19 points; the adapter never fabricates body points from a face.

This deliberately uses SCRFD/YuNet face points instead of the released DFA
aligner. It has its own namespace and is not presented as numerically identical
to the complete upstream detector/aligner pipeline. Namespace provenance covers
the model checkpoint, source revision, model YAML hash, pose hash, detector,
preprocessing contract, and person/keypoint confidence gates. Paths do not
change identity semantics. Existing AuraFace/SFace/LVFace/AdaFace namespaces
are unchanged.

SQLite keeps the real 15-value face geometry for existing consumers and stores
the measured body box, pose points, score, and association contract in quality
metadata. Rebuild reuses retained 4096D embeddings; no body inference is needed
for rethresholding/rebuild. VTT portraits still use faces; video portrait
identity verification uses the same person/face backend on the decoded frame.
The retained-face `recognition pilot` rejects SapiensID explicitly because those
small face crops cannot reproduce its original person input.

## Thresholds and resource use

SapiensID cosine thresholds require separate validation on your own same- and
different-person pairs, especially changes of clothing. Existing YAML values
are operator policy, not model-calibrated probabilities. Evaluate fragmentation
and false merges before a broad run. Model selection creates a separate identity
namespace and follows the existing scoped Stash publication rules.

A float32 feature occupies 16 KiB (8× a 512D feature). Existing reference RAM
budgets remain enforced; adjust `clustering.max_reference_bytes` explicitly
when necessary. Model/pose inference stays synchronous on the invocation's
owning thread, while existing rebuild workers continue to operate on retained
vectors. CUDA requests fail if unavailable; `auto` can select CPU.

## Tests and licensing

Deterministic tests cover normalization/preprocessing, verified weight loading,
22-point geometry, crop-coordinate transforms, ambiguous/missing associations,
quality gates, namespaces, scan/reuse and 4096D retained-vector validation.
The released wb4m checkpoint was also loaded strictly and exercised on CPU
with the exact upstream architecture and supplied person geometry: finite
4096D float32 output, L2 norm 1.0000000005. This model smoke used a synthetic
image, not an accuracy dataset. CUDA and recognition accuracy are not verified.
To repeat with your own normal config and pinned Git checkout:
`STASH_AI_SAPIENSID_CONFIG=/path/to/face.yml python -m pytest -q tests/test_recognition_sapiensid_loading.py -k real_local`.
Synthetic tests do not establish recognition accuracy or GPU memory usage.

Upstream SapiensID source/weights are CC BY-NC 4.0. Ultralytics has separate
AGPL-3.0/enterprise terms. This PR references local source and weights rather
than redistributing them. The wb4m size/hash is independently recorded at
https://huggingface.co/peabody124/sapiensid-eqx; that conversion is not used at
runtime. Respect the terms for each model used in your environment.


## Shared policy, anchors and auxiliary observations

SapiensID uses the same cold replay and assignment pipeline as face encoders.
An **anchor** is still determined by the measured **face** pose proxy and
`clustering.max_anchor_pose_deviation`. It can seed an identity and enter the
reference set. An eligible higher-pose observation can attach as **auxiliary**,
subject to `auxiliary_matching`, `auxiliary_min_support`, cosine/cohesion,
ambiguity, cooccurrence and view policies. Auxiliary membership does not turn
that observation into an anchor reference or merge vote.

`max_anchor_pose_deviation: null` disables this split: all eligible observations
are anchor-eligible. The body pose keypoints do not replace the face pose proxy.
These are deliberate shared policies, not disabled face-only knobs. If detection
rejects a face before embedding (for example `max_pose_deviation` or visibility),
its strong body input cannot rescue it through the auxiliary pass.

| YAML section/keys | Meaning with SapiensID |
| --- | --- |
| `detection.*` | Gates and detector settings for the **face**: confidence, NMS, face count, resolution, sharpness, alignment/completeness, face pose and FaceOcc. All remain in effect. |
| `sapiensid.pose.*` | Separate person/pose checkpoint, person confidence, keypoint confidence, person NMS and person count. Face NMS/count do not configure YOLO. |
| `sapiensid.model.*` | Local upstream architecture, wb4m checkpoint directory and model-YAML integrity. Used when identity inference is needed. |
| `recognition.*` | Cosine matching, ambiguity, independent support, cross-scene and cooccurrence policy on **4096D SapiensID** vectors. Values need SapiensID calibration. |
| `clustering.*` | Anchor/auxiliary split, view prototypes, cohesion, sample count and RAM limits use the common pipeline. Optional features remain conditional on their enable flags/support. |
| `portraits.*` | Rank/frame **face** portraits. Video identity verification uses the associated person input; retained VTT portraits reuse recorded identity evidence. |
| `models.device` | Shared CPU/CUDA/auto request for configured face detector and SapiensID/YOLO. |
| `models.directory`, `models.download_missing` | Standard pinned face/quality assets; do not relocate/download explicitly configured local SapiensID/YOLO files. |
| `execution.*`, `storage.*`, `names.*` | Existing stage-specific execution, storage and metadata controls remain applicable. Rebuild controls affect rebuild, not model inference. |

The new model-specific keys are grouped as `sapiensid.model` and `sapiensid.pose`.
The earlier flat keys from the unmerged draft are rejected as unknown keys;
update them using the example above. Existing face configurations continue to
load without a SapiensID section. Shared threshold sections stay shared so an
operator can see the policy actually applied after selecting an encoder.

## Alignment audit and identity report

`recognition audit-alignment` selects the SapiensID geometry adapter, runs the
configured face detector and local YOLO pose, and applies the same detection,
FaceOcc and unique face/person association gates as scanning. **It does not load
wb4m identity weights or produce embeddings.** In addition to the labelled
112x112 face alignment diagnostic, each passing sample has:

- `-person-input.png`: the actual 384x384 RGB input, reconstructed for viewing
  from the exact normalized model tensor;
- `-person-landmarks.png`: the 17 body + 5 face points on that input;
- `-person-mask.png`: the model's foreground mask;
- manifest metadata containing the person box/pose, normalized 22 points,
  foreground fraction and explicit person-input contract.

Inference and audit call the same tensor preparation function. The ArcFace
preview is a face-quality diagnostic, not a SapiensID model input.

`recognition report-identity` remains offline and exports **all retained face
crops** belonging to the selected identity. It includes recorded anchor/auxiliary
assignment reasons, current face-pose anchor eligibility, embedding dimensions,
stored model provenance and the retained person association/pose metadata.
It does not rerun the current detection gates, load models, or recreate body PNGs
from face crops. The original body pixels and exact historical body input are
not retained in SQLite; use the alignment audit on original cues to inspect
those inputs. YAML changes to person geometry gates/checkpoints create a different
namespace, so reporting old identities requires the matching model configuration.

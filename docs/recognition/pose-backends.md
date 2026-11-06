# Person detectors and pose backends

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [ViTPose++ Large keypoints](#vitpose-large-keypoints)
- [PMPose-B keypoints, presence and visibility](#pmpose-b-keypoints-presence-and-visibility)
- [SAM-pose2seg masks, second PMPose pass and tight-box recovery](#sam-pose2seg-masks-second-pmpose-pass-and-tight-box-recovery)
- [Duplicate PMPose detections before SAM](#duplicate-pmpose-detections-before-sam)
- [Sapiens Pose keypoints](#sapiens-pose-keypoints)
- [D-FINE person detectors for Sapiens / ViTPose / PMPose](#d-fine-person-detectors-for-sapiens--vitpose--pmpose)
- [RF-DETR-XL person detector](#rf-detr-xl-person-detector)
- [Shared geometry and existing YOLO configurations](#shared-geometry-and-existing-yolo-configurations)

See [SapiensID setup](../sapiensid-hybrid.md) for installation and body matching.
Recipes below use **v2** configuration. Merge the shown fields into the
corresponding sections of your existing YAML; unshown fields keep that file's values.
For an older file, use [configuration migration](../facial-recognition.md#recognition-yaml-v2) first.

## ViTPose++ Large keypoints

For a separate top-down pose estimator, install the updated CPU/GPU profile
and select:

```yaml
version: 2
body:
  enabled: true
pose:
  estimator:
    backend: vitpose_plus_large
    vitpose_plus_large:
      directory: ''
      batch_size: 4
  detector:
    frame_batch_size: 4
```

The dependency file includes SciPy, required by the Transformers processor for
affine crops and heatmap decoding. If an existing installation reports
`NameError: name 'inv' is not defined`, reinstall the updated
CPU/GPU profile in the active environment and restart the
process. Missing SciPy is now reported before ViTPose assets are downloaded.

The shipped v2 configuration uses RF-DETR-XL boxes. To use YOLO boxes, select
`pose.detector.backend: yolo` and configure weights under `pose.detector.yolo`.
Detection thresholds and limits live under `pose.detector`. ViTPose++
Large replaces its keypoints for **every** accepted person, including people
without a detected face. The COCO expert (`dataset_index=0`) returns 17 points
in image coordinates. Shared boundary/confidence filtering, face association,
SapiensID's 22-point input and audit overlays remain in use. `audit-detection`
and `audit-alignment` select the same backend as recognition. Existing v1 files
retain their YOLO defaults until explicitly changed or migrated.

The first use downloads the three pinned files from
[`usyd-community/vitpose-plus-large`](https://huggingface.co/usyd-community/vitpose-plus-large/tree/e211df377e49c89dba508ab3e83ddef13c0832b4)
at revision `e211df377e49c89dba508ab3e83ddef13c0832b4`: `config.json`,
`preprocessor_config.json` and `model.safetensors` (1,737,705,492 bytes).
SHA256 checks cover downloads and existing files. The default directory is
`execution.models.directory/vitpose_plus_large/<revision>`; `vitpose_directory` overrides
it relative to the YAML. With `execution.models.download_missing: false`, all three
matching files must already exist. No remote Python code or pickle weights
are loaded for ViTPose. Only selecting this backend loads its assets.

Inference runs on `execution.gpu.device` in FP32, with at most `pose.estimator.vitpose_plus_large.batch_size` people
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

## PMPose-B keypoints, presence and visibility

`pose.estimator.backend: pmpose` selects **PMPose-B 1.0.0** as a standalone
pose estimator. It replaces ViTPose/Sapiens keypoints and accepts person boxes
from YOLO, D-FINE-L, D-FINE-X or RF-DETR-XL:

```yaml
version: 2
body:
  enabled: true
pose:
  estimator:
    backend: pmpose
    pmpose:
      batch_size: 1
      presence_threshold: 0.5
      visibility_threshold: 0.0
    keypoint_score_threshold: 0.5
  detector:
    backend: dfine_l
```

By default this integration uses **boxes only**, without segmentation masks,
SAM or the iterative BBoxMaskPose/BMPv2 pipeline. It does not inherit reported results for
the full masked pipeline. Evaluate overlapping people and profile faces on
your own VTT cues before replacing a working backend. Changing pose models
alone does not guarantee that a nearby person's ankle cannot be selected.

PMPose predicts three different per-point quantities:

| Setting / audit field | Meaning |
| --- | --- |
| `keypoint_score_threshold` / `keypoint_scores` | Predicted localization quality (expected OKS), not a heatmap peak or visibility probability |
| `pmpose_presence_threshold` / `keypoint_presence` | Probability that the landmark lies within the model crop |
| `pmpose_visibility_threshold` / `keypoint_visibility` | Predicted visibility of the landmark |

The point must exceed the configured thresholds and remain inside its person
box (the original detector box unless mask refinement admits an expansion).
A zero presence/visibility threshold disables that filter.
Visibility filtering defaults to disabled, preserving useful occluded-body
estimates. To omit hidden points, including an eye on the far side of a profile,
try `pmpose_visibility_threshold: 0.5` and inspect the audit; the prediction can
still be wrong. This filter applies to all 17 points, so it can also remove
occluded joints useful for body association. These thresholds need calibration
on your scenes. Scores are never substituted for missing visibility outputs.

Both recognition and audits use the same filtered COCO17 points. Detection
audit JSON retains all three score arrays in COCO order and labels
`keypoint_score_kind: expected_oks`; PNGs draw only retained points. The detector's
person score remains separate. Face association, SapiensID's 22-point input and
the independent body collection path keep their existing geometry rules.

The first use downloads `PMPose-b-1.0.0.pth` (617,696,116 bytes) from
[`vrg-prague/BBoxMaskPose`](https://huggingface.co/vrg-prague/BBoxMaskPose/tree/cfc89052cf16655a52889f1f4e1b866e74a6c19c/PMPose).
SHA256 `23b0c1896a3ba5fd802a1cef635a3dad598e85960de57a29815a5bb3dd3ce5f7`
is verified before every load, using the same open file for hashing and restricted
`weights_only=True` deserialization. Training metadata is decoded into inert
containers; no upstream Python or arbitrary pickle fallback is used. An
incompatible or corrupt local checkpoint fails instead of being overwritten.
Only the pinned B variant is supported in this initial integration.

The default directory is `execution.models.directory/pmpose_b/<revision>`.
`pmpose_directory` overrides it relative to the configuration YAML. With
`execution.models.download_missing: false`, place the matching checkpoint there first.
The inference adapter uses the existing PyTorch/NumPy/OpenCV dependencies;
MMCV, MMEngine and MMPose are not runtime requirements. Install the usual
CPU/GPU profile for your selected detector and identity backend.

Inference follows the pinned upstream architecture in FP32, on `execution.gpu.device`,
with 192x256 RGB crops, padding 1.25, horizontal flip averaging and DARK/UDP
decoding of the first 17 of the checkpoint's 23 outputs. At most
`pmpose_batch_size` people are processed per forward pass (1..16), including
batches spanning several frames; the flipped pass runs separately. This limit
is independent of the detector's `batch_size` and embedding batches.

Selecting PMPose or changing its score/presence/visibility thresholds changes
person and body evidence contracts; recollect affected evidence before solving.
Paths and batch sizes do not change contracts. Existing YOLO, ViTPose and Sapiens
contracts and the default backend remain unchanged.

An opt-in pretrained CPU smoke test is available with
`STASH_AI_PMPOSE_SMOKE=1 python -m pytest -q tests/test_recognition_pmpose_smoke.py`.
It downloads the ~618 MB model and a pinned sample image, uses fixed person boxes
and checks real pose inference against the upstream reference fixture plus audit
geometry. The fixture uses MMPretrain 1.2.0 and the pinned upstream PMPose head
and decoder with the same bbox-only input and flip averaging. Set
`STASH_AI_PMPOSE_DIRECTORY` to reuse an existing verified checkpoint directory.

## SAM-pose2seg masks, second PMPose pass and tight-box recovery

Both CPU/GPU profiles include the dependencies for this optional path:

```yaml
version: 2
body:
  enabled: true
pose:
  estimator:
    backend: pmpose
  detector:
    backend: dfine_x
  refinement:
    enabled: true
    mode: all
    expand_boxes: true
```

`SAM-pose2seg` is the pose-guided segmenter from BMPv2, built on SAM2.1. It is
not a new pose estimator and does not replace D-FINE. This adapter performs one
initial PMPose pass, one segmentation per eligible person, then one masked
PMPose pass. It does not run the full BMP detector loop or 3D estimation.

SAM receives the full RGB cue and up to three reliable, visible head/torso
points from the initial pose. Elbows, wrists, knees and ankles are deliberately
excluded from prompts because limb ownership is the problem being corrected.
Presence, localization quality and visibility have independent prompt gates;
points coinciding with another person's reliable core are excluded. The
segmenter receives no box prompt and no image crop, so it can recover an arm
outside a tight D-FINE box. People need at least two usable prompts by default.
SAM's image features are computed once per eligible cue and released even on
errors. Accepted masks are queued in batches bounded by `pmpose_batch_size`.

The second PMPose pass uses upstream `MaskBackground(alpha=0.25)` preprocessing:
instance pixels are unchanged, background RGB is reduced to 75%, then affine
sampling and normalization are applied. The same flip averaging and COCO17
mapping are used as in the first pass. This is mask conditioning, not a hard
cutout or a post-hoc deletion of keypoints outside the mask.

With `expand_boxes: true`, the person box becomes the union of its original
bounds and the accepted mask bounds, clipped by the source image. It never
shrinks, which preserves space for occluded body parts. Expanded bounds feed
keypoint filtering, face association, body/face-associated collection, SapiensID
crops and audits consistently. The original detector score stays separate from
SAM's mask score. There is no second box NMS that could silently delete a person.

Admission rejects low-quality/empty masks, missing prompt support, masks covering
another person's reliable head/torso points, or excessive expansion. Defaults
limit union area to 2x the original area and extension on each side to 0.5x the
original width/height. Tiny disconnected speckles are removed; substantial
disconnected hands/feet are retained. The second pose must retain at least the
configured minimum number of prompt landmarks inside the mask, with adequate
quality/presence and displacement within 15% of the original box diagonal
(minimum 3 px). If any admission gate fails, **both the first pose and original
box are retained**. Model errors, malformed outputs, missing dependencies or
invalid assets raise errors; they are not treated as successful refinement.

`mode: all` also attempts recovery for isolated, tightly cropped people.
`mode: overlap` saves segmentation work by requiring intersection divided by the
smaller detector-box area to reach `overlap_threshold` (default 0.20). It cannot
repair an isolated person. Turning off `expand_boxes` still allows a masked
second pose but keeps the original point/crop boundaries. All tuning parameters
and their short descriptions are in `facial_recognition.yml`.

Detection audit JSON includes `pose_refinement` for every person: status/reason,
original and final bounds, prompt indices/points, mask score and mask area when
admitted. Gray outlines in PNGs show the old box when it was expanded; the
normal person color shows the final box/pose. No masks are retained in the DB.
Pipeline profiling separates `pmpose_mask_refinement` (total, including nested
work) and `pmpose_masked_second_pass`.

On first enabled use, the adapter downloads the 357,549,748-byte
`SAM-pose2seg_hiera_b+.pt` from the pinned VRG Prague revision
`cfc89052cf16655a52889f1f4e1b866e74a6c19c`, with SHA256
`c89ee52fdc323f61ec5c52075d49bebb7b4e76f39b4e1e0c87b331bf0dcd3b98`.
It also downloads a small, per-file SHA256-pinned inference source subset and
license from BBoxMaskPose commit `49a070a6f8396147323b1c4959474077dbe2ce8e`.
The source manifest is bundled in `sam_pose2seg_registry.py`, so updater-based
installations include it without a separate JSON file or manual setup.
No OpenMMLab/training imports, Git installation, CUDA extensions or automatic
pip installation are needed. The existing CPU/GPU torch and torchvision profile
is reused. Source bytes are verified before imports; weights are hashed and
loaded from the same descriptor with `weights_only=True` and strict state keys.
The directory defaults to `execution.models.directory/sam_pose2seg/<revision>`; set
`refinement.directory` to relocate it. `execution.models.download_missing: false` requires
all assets already present. Existing corrupt assets are never overwritten.

Enabling refinement or changing its semantic settings invalidates person/body
and SapiensID identity evidence contracts; rerun recognition on affected scenes
before solving. Inactive refinement preserves existing contracts; relocating
assets and changing batch size do not change vectors' contracts. Refinement
settings are collection settings, not offline solver parameters.

The opt-in real CPU test runs both pretrained models:

```sh
STASH_AI_SAM_POSE2SEG_SMOKE=1 python -m pytest -q tests/test_recognition_sam_pose2seg_smoke.py
```

It downloads about 975 MB including PMPose; `STASH_AI_PMPOSE_DIRECTORY` and
`STASH_AI_SAM_POSE2SEG_DIRECTORY` can reuse cached assets. Offline tests cover
cross-person mask rejection, disconnected limbs, clipped box expansion,
second-pass drift, batching, failure cleanup and evidence invalidation. The
real-model smoke verifies execution, not accuracy on overlapping VTT people.
Benchmark representative audit cues before enabling this across a large library;
wrong initial poses can still produce wrong masks, and small thumbnails may not
contain enough detail to recover the limb.

## Duplicate PMPose detections before SAM

When two high-confidence person boxes describe the same subject, optional
`pose.deduplication.enabled: true` compares their **first-pass PMPose
points before SAM prompt selection**. It also works with refinement disabled.
It keeps the current box NMS threshold and makes no additional model calls.
The option is off by default and requires `pose.estimator.backend: pmpose`.

```yaml
version: 2
pose:
  estimator:
    backend: pmpose
  deduplication:
    enabled: true
```

Only in-box head, shoulder and hip points exceeding the independent
`min_keypoint_score`, `min_presence` and `min_visibility` gates (all `0.5` by
default) can support suppression. Anatomical indices must match; limbs and
nearest arbitrary joints are not evidence of a duplicate. Boxes must overlap,
but there is no high box-IoU gate. Matching requires at least
`min_shared_points: 4` agreeing points from **two of head/shoulders/hips**, with
`min_match_fraction: 0.8` agreement among all shared reliable core points.
Distances use the smaller original box diagonal and must be at most
`max_point_distance: 0.05` times that diagonal.

A head/shoulder/hip group with at least two shared points and median distance
above that limit vetoes suppression. Any shared reliable point farther than
twice the limit also vetoes it. Matching points must span at least the larger
of 10% of the smaller diagonal and twice the distance tolerance, so collapsed
or near-coincident skeleton points cannot manufacture support. No rule assumes
upright people, a head above hips, or a particular camera rotation.

Candidates are ranked by the sum of `score * presence * visibility` over
eligible core points, divided by nine: a coverage-weighted quality heuristic,
not a calibrated probability. Reliable point count and original candidate
index break ties. Only retained candidates may suppress another candidate;
suppression never propagates through a discarded candidate. Survivors keep
their original detector order, boxes and detector scores. Complementary body
fragments without shared core evidence are retained; this stage does not union
boxes or invent a combined skeleton. In SAM `mode: overlap`, eligibility is
evaluated on the surviving people.

The normal single-frame and batched paths, face associations and independent
body collection all consume the same surviving rows. SAM masks use the new
survivor indices; removed candidates cannot block prompts as foreign people.
Detection audits (including `--sweep` and `--mask-view off`) draw removed
candidate boxes as gray dashes and append reasons, box IoU and matched/shared
point counts. D indices refer to candidates after box NMS/score filtering;
P indices refer to surviving people. JSON `pose_deduplication` includes quality
scores, original bounds, normalized point distances, regions and D-to-P mapping.
All suppression explanations and up to 128 overlapping kept-pair explanations
are retained per frame; omitted/non-overlapping pairs have aggregate counts.
More than 50,000 greedy comparisons fails explicitly rather than silently
changing selection. Profiling records `pmpose_pose_deduplication` and
`pmpose_duplicate_people_removed`.

These gates are conservative heuristics, not a guarantee against two people
whose estimated cores coincide. Compare enabled/disabled results on crowded
and unusual-view VTT cues before using the setting broadly. Enabling it or
changing active settings invalidates person/body/identity evidence contracts:
rerun recognition on affected scenes. With it disabled, previous contracts
and detection behavior remain unchanged. These are collection settings, not
offline identity-solver parameters.

## Sapiens Pose keypoints

`pose.estimator.backend: sapiens` selects Meta's **Sapiens v1 Pose** as another
top-down pose estimator. It is separate from the SapiensID identity encoder and
from Sapiens2. Its dependencies are included in both CPU/GPU profiles;
no MMPose/MMCV installation is needed.

```yaml
version: 2
execution:
  models:
    download_missing: true
body:
  enabled: true
pose:
  estimator:
    backend: sapiens
    sapiens:
      variant: 0.3b
      batch_size: 1
      directory: ''
```

| `sapiens_variant` | Public Meta model | Download size |
| --- | --- | --- |
| `0.3b` (default) | [Sapiens Pose 0.3B](https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript) | 1.32 GB |
| `0.6b` | [Sapiens Pose 0.6B](https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript) | 2.65 GB |
| `1b` | [Sapiens Pose 1B](https://huggingface.co/facebook/sapiens-pose-1b-torchscript) | 4.68 GB |

Only the selected variant is downloaded, at a pinned revision, with size and
SHA256 verification before loading. The public Goliath TorchScript exports do
not require a Hugging Face token. Files are cached under
`execution.models.directory/sapiens_pose/<variant>/<revision>/`. `sapiens_directory`
overrides the directory relative to the YAML and must contain the selected
model's original `.pt2` filename. `execution.models.download_missing: false` requires an
existing verified file. A corrupt or different checkpoint is rejected rather
than overwritten. The upstream weights have a separate CC BY-NC 4.0 license;
see [third-party notices](../../THIRD_PARTY_NOTICES.md).

With the v1 default, YOLO supplies person boxes with its pose head disabled. Sapiens alone supplies
keypoints for those boxes, including people without a detected face. The model
predicts 308 Goliath landmarks; the adapter selects and reorders the matching
17 COCO body points (including wrists at Goliath indices 62 and 41). The other
landmarks are not retained. SapiensID still receives these 17 points plus the
five landmarks from the face detector; existing association, missing-point
filtering and audit overlays apply.

Crops use 1.25 padding, the 768×1024 target aspect ratio, RGB training
normalization, and UDP affine sampling/DARK subpixel heatmap decoding. Inference
uses FP32 on `execution.gpu.device`, without flip augmentation or compilation.
`sapiens_batch_size` bounds people per Sapiens inference, including across cues;
`pose.detector.frame_batch_size` continues to bound person-detector frame batches. Start with one person
per Sapiens batch: the 1K input and larger variants require substantially more
memory and compute than YOLO. GPU throughput and VRAM requirements depend on
the variant and other loaded models; no fixed capacity is assumed.

Heatmap maxima are clipped to [0,1] and are not calibrated probabilities.
Tune `pose.estimator.keypoint_score_threshold` independently for this backend. Changing
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

## D-FINE person detectors for Sapiens / ViTPose / PMPose

`pose.detector.backend` selects person boxes independently of `pose.estimator.backend`, which
selects keypoints. Version 1 defaults to `detector: yolo`.
To replace YOLO entirely while retaining Sapiens Pose:

```yaml
version: 2
body:
  enabled: true
pose:
  estimator:
    backend: sapiens  # or vitpose_plus_large
    vitpose_plus_large:
      batch_size: 4
  detector:
    backend: dfine_l  # dfine_l or dfine_x
    frame_batch_size: 4  # frames per detector batch; reduce for less VRAM
    dfine:
      directory: ''  # optional local folder containing all three files
```

| Selector | Pinned Transformers checkpoint | Weight download |
| --- | --- | --- |
| `dfine_l` | [D-FINE-L Objects365 -> COCO E25](https://huggingface.co/ustc-community/dfine-large-obj2coco-e25/tree/c89de47153e338c07cddc81fc21a776b97cbf75e) | 126 MB |
| `dfine_x` | [D-FINE-X Objects365 -> COCO](https://huggingface.co/ustc-community/dfine-xlarge-obj2coco/tree/15f18d917eaddcedf9e3ffb082adcfb97a0b2d4d) | 252 MB |

Install the updated CPU/GPU profile (Transformers >=4.53).
First use downloads `config.json`, `preprocessor_config.json` and
`model.safetensors` into `execution.models.directory/<detector>/<revision>/`.
Every file is checked against a pinned size and SHA256, including cached files.
`dfine_directory` overrides this directory relative to the YAML; with
`execution.models.download_missing: false`, all three matching files must already exist.
Only the selected variant is loaded, in FP32 on `execution.gpu.device`; there is no
remote Python code, pickle loading or separate backbone download.

D-FINE uses RGB frames resized to 640x640 and returns boxes in original image
coordinates. Only COCO person scores are considered. `pose.detector.score_threshold`
filters people, `nms_threshold` suppresses duplicate person boxes, and
`max_people_per_frame` caps the remaining results. NMS is explicitly applied
after D-FINE's end-to-end predictions to keep this application's configured
duplicate-suppression behavior. Other object classes do not consume the limit.
Scores need calibration on your scenes; they are not interchangeable with YOLO.

Finite proposals with non-positive width/height are discarded before NMS and
the result limit; a warning reports their count and first frame/query example.
NaN/Inf outputs are handled separately: only affected frames are retried once,
individually in FP32 on the configured device, preserving healthy batch results.
If the retry still contains NaN/Inf, inference fails with the tensor name, count,
frame/query/component and example value; it is never saved as an empty detection.
Frame indices are zero-based within the detector's input call, not scene cue IDs.
Incorrect tensor shapes fail immediately with expected/actual dimensions.
Autocast is disabled inside D-FINE inference. This recovery policy updates the
D-FINE evidence contract (v2); rerun recognition for affected scenes.

In this mode no YOLO model is downloaded, loaded or executed; `pose.detector.yolo.path` and
`pose.detector.yolo.sha256` are unused. `backend: yolo` with a D-FINE detector is rejected,
because D-FINE supplies no keypoints. Sapiens/ViTPose/PMPose supplies the 17 body points
used by recognition, independent body collection and both detection/alignment
audits. The existing crop-boundary filtering remains in place.

Switching detector/variant changes the person/body evidence contract; rerun
recognition on the selected scenes before solving/merging. Existing YOLO
contracts remain valid. Local paths and batch sizes do not change evidence.
Compare bounding-box coverage on your VTT frames with `audit-detection`; these
integration tests do not establish a quality advantage over YOLO.

The opt-in pretrained CPU smoke covers both variants (about 378 MB total):
`STASH_AI_DFINE_SMOKE=1 python -m pytest -q tests/test_recognition_dfine_smoke.py`.
Set `STASH_AI_DFINE_MODEL_DIRECTORY` to reuse the parent model cache directory.

## RF-DETR-XL person detector

`pose.detector.backend: rfdetr_xl` selects Roboflow's **detection XL** model
(126.4M parameters, 700x700 input), paired with `backend: pmpose`,
`vitpose_plus_large` or `sapiens`. It supplies person boxes to the existing pose
pipeline, including optional PMPose deduplication and SAM refinement.

RF-DETR dependencies are included in `requirements-cpu.txt` and
`requirements-gpu.txt`; no additional installation profile is needed.

```yaml
version: 2
body:
  enabled: true
pose:
  estimator:
    backend: pmpose
    vitpose_plus_large:
      batch_size: 1
  detector:
    backend: rfdetr_xl
    frame_batch_size: 1  # frames per detector batch (1..16); increase after checking VRAM
    rfdetr_xl:
      directory: ''  # optional folder containing rf-detr-xlarge.pth
```

The integration pins `rfdetr==1.11.2` and `rfdetr-plus==1.1.0` in both runtime
profiles, without requesting training/export extras or replacing the selected
torch build. The application updater refreshes the manifests but does not run
pip. Other detectors do not import either package. The v2 template defaults to
RF-DETR-XL; version 1 configurations keep their existing defaults.

First use downloads the official **505,646,472-byte** checkpoint to
`execution.models.directory/rfdetr_xl/<SHA256>/rf-detr-xlarge.pth`. The Google Storage
object generation and SHA256 are pinned in `rfdetr_registry.py`; cached and
downloaded files are checked. `rfdetr_directory` is relative to the YAML unless
absolute. With `execution.models.download_missing: false`, the matching local checkpoint
is required. An existing invalid file fails verification and is not overwritten.
No independent backbone download occurs. We construct the detection architecture
locally, then load the hash-verified file using `weights_only=True` and strict
state-dict matching. Only the generated empty keypoint-schema buffer is excluded
from persistence; no learned weights are omitted or randomly reinitialized.
The two expected upstream notices about positional encodings and patch size 20
skipping standalone DINOv2 weights appear only in verbose diagnostics (DEBUG
logging without a console report). They occur while building the XL architecture,
before the complete RF-DETR checkpoint is loaded. Other warnings and errors,
including weight-loading failures, remain visible.

Inference uses FP32 on `execution.gpu.device`, including inside an outer autocast
context. RGB pixels are scaled by 255, resized bilinearly to 700x700 without
antialiasing, then normalized with the upstream ImageNet mean/std. The pinned
checkpoint has 91 raw COCO slots: **person is category 1**, not index 0.
Person scores are selected before limiting results, so other classes and
alternative labels for a query cannot consume the people budget. Boxes return
in original cue coordinates and are clipped to the frame. Non-positive boxes
are discarded; malformed shapes and NaN/Inf fail explicitly instead of becoming
empty retained evidence.

`pose.detector.score_threshold`, `pose.detector.nms_threshold` and `pose.detector.max_people_per_frame`
apply as with the existing detectors. Although RF-DETR is an end-to-end detector,
this adapter applies the application's configured person-box NMS. Scores need
calibration on representative VTT cues. The shared single/batch paths, face
association, body collection and detection audit/sweeps consume the same boxes.
Changing detectors invalidates person/body/identity evidence; rerun recognition
before solving or merging. Local directories and batch size do not change those
contracts, and existing YOLO/D-FINE contracts remain valid.

RF-DETR-XL weights and `rfdetr-plus` use Roboflow's **Platform Model License 1.0**,
not the core package's Apache 2.0 license. Upstream documents a Roboflow account
requirement; see the [official model documentation](https://rfdetr.roboflow.com/latest/learn/run/detection/)
and [PML terms](https://github.com/roboflow/rf-detr_plus/blob/main/LICENSE).
This adapter runs inference locally and does not upload cue images.

The opt-in actual-checkpoint CPU test covers strict loading, people detection
and single/batched original coordinates:
`STASH_AI_RFDETR_SMOKE=1 python -m pytest -q tests/test_recognition_rfdetr_smoke.py`.
Set `STASH_AI_RFDETR_DIRECTORY` to an existing weight folder to reuse its checkpoint.
This smoke test does not establish better accuracy than D-FINE on intertwined people;
compare the detectors on the same cues using `audit-detection` sweeps.

## Shared geometry and existing YOLO configurations

Face/person association preserves the contained-face path: allow 5% of the
shorter face-box side as a boundary tolerance and require at least one usable
pose nose/eye inside the face box. Faces that protrude beyond that tolerance
can now use a landmark-supported partial-box path:

- At least 50% of the face-box area must overlap the person box.
- At least two distinct corresponding head landmarks must be usable: COCO
  nose/left eye/right eye map to the five-point face detector's indices 2/1/0.
- Every usable compared pair must be within 15% of the shorter face-box side.
  One contradictory measured landmark vetoes the partial-box path. Missing
  points, points outside their actual crop/face box and collapsed predictions
  cannot provide the required support.
- Reciprocal uniqueness still applies across all detections, including rejected
  faces: no nearest-person fallback, confidence tie-breaking or eye swapping.

This automatic policy applies to every detector/pose backend. It uses already
filtered pose points and does not relax confidence, presence or visibility
thresholds. The original person crop and face geometry are preserved.

The policy is included in person/body evidence fingerprints and model
provenance. After updating, `recognition run` recollects SapiensID evidence for
the selected scenes, including previously cached unavailable matches and
completed body cues, while reusing existing face detections and embeddings.
Subsequent runs reuse the new evidence. Older body banks remain conservative
competitors until recollected; they cannot train current cores. Offline
rebuild/solver work does not collect the new evidence, so run recognition first.

After a unique face/person association, landmarks outside the actual person
crop are marked missing (`-1,-1`), including face-detector points and coordinates
that round onto the excluded right/bottom edge in float32. They are not clamped
to the border. This applies to scan preparation, batched inference and alignment
audits, so a crop-boundary disagreement does not abort the scene. Malformed or
non-finite geometry still fails validation; ambiguous associations remain withheld.

To switch an existing installation, update **both** `pose.detector.yolo.path` and `pose.detector.yolo.sha256`
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
`execution.models.download_missing: false`. Missing assets then fail with their path and
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

# Detection audits

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Desktop detection tuner](#desktop-detection-tuner)
- [Full-cue face and pose detection audit](#full-cue-face-and-pose-detection-audit)
- [Single-cue parameter comparisons](#single-cue-parameter-comparisons)
- [Read-only five-landmark alignment audit](#read-only-five-landmark-alignment-audit)

## Desktop detection tuner

Run `python stash_ai.py recognition tune-detection --face-config facial_recognition.yml` on
a Windows or Linux desktop. Tkinter and Pillow ImageTk are required for this command.
Use `python -m tkinter` to check the interpreter. On Debian the Tk package is
`python3-tk`. The tuner connects to Stash using the usual `STASH_URL` and `STASH_API_KEY`.

Enter a **Scene ID** and choose **Load scene**, or supply an initial `--scene-id` on the
command line. The slider, Previous and Next browse raw VTT crops at their original
zero-based indices. Select a detector setting, enter its value and press **Set value**.
Only **Run detection** loads models and computes an annotated image. Each detection
starts from fresh cue context, independently of navigation history. Edits are local to the
window. The recognition database and Stash identities are never modified.

To preview ViTPose++ with the bundled YAML, set `pose.estimator.backend` to
`vitpose_plus_large` and choose **Run detection**. The tuner preserves each panel's
mask/refinement settings, including the optional generic second pose pass.
Body and anchor sharpness floors apply during recognition collection and matching.

After a run, a second **Phase** slider browses intermediate results for that cue.
**Previous phase**, **Next phase**, the arrow keys on that slider, and **Final result**
use the recorded result; they never call a model. The view initially shows the final
result. Depending on the enabled stages, the history includes:

- Person boxes after detector NMS and initial keypoints before pose deduplication.
- Contact recovery, deduplication, and the pose before mask/refinement processing.
- Prompt eligibility/recovery, then the selected raw and cleaned mask for each
  decoder attempt, with positive prompts, negative prompts, and rejection reasons.
- Cross-person mask conflicts and the mask selected for pose processing.
- Raw keypoints from the masked pose pass, followed separately by accepted/fused
  keypoints. A rejected second pass remains visible alongside the preserved first pose.
- Ownership/maskless fallback, output keypoint filters, and the final annotated result.

Only executed stages appear. A mask-only estimator has no synthetic second pose pass.
Raw points are displayed before the final score/presence/visibility filters. White
rings identify positive prompts; red crosses identify negative prompts. `D` labels
identify detector candidates; after deduplication `P0 (D2)`, for example, maps the first
retained pose to detector candidate 2. Labels are local to a cue/configuration.
Mask stages show the selected candidate for each attempt, including rejected attempts;
they do not retain every alternative mask or model-internal feature map.

Hover over a keypoint in any recorded phase or the final result to see its person,
COCO17 name/index, source coordinates, score, presence, visibility, and available
flip-agreement/provenance information. Values belong to that exact phase. PMPose
scores are expected OKS; generic estimators' internal confidence proxies are explicitly
marked **not measured**, and missing fields show **N/A**.

In **After output keypoint filters**, removed points with known source coordinates
remain visible as gray hollow markers. Hovering them explains every failed output
cutoff by its YAML path and measured value, or identifies the output-box boundary
that excluded them. The final preview retains these markers when its geometry and
measurements match that recorded output phase. Markers are diagnostic overlays;
the filtered recognition keypoints remain unavailable. Final face landmarks also
support hover, with face detection score kept separate from unavailable per-landmark
scores. Hover respects zoom/panning and never runs inference.

The phase bar and each A/B image caption show **Step** (elapsed wall time since the
previous recorded phase) and **Total** (cumulative time for this cue), in milliseconds
or seconds. **Original cue** starts at zero; **Final result** includes the remaining
detection work, including final association/retries. Measurements stop before audit
annotation/layout rendering and exclude model loading, preceding-cue context, snapshot
copying/packing, and browsing the preview. Reusing a cached result preserves its original
timings; a new detection measures again.

These are intervals between actual checkpoints, not isolated GPU kernel timings.
Preparation, waiting, and shared/batched work are charged once, to the first checkpoint
after that work; for example face detection precedes the first person-box checkpoint.
No extra CUDA synchronization is introduced. If snapshots are omitted by the history
limit, the final **Remaining (includes omitted phases)** time still covers that work.

History is transient and belongs to the current cue and configuration. Masks are
packed on CPU and stages are rendered on demand, with a limit of 256 snapshots and
a conservative 64 MiB history budget per distinct result. If the limit is reached,
the phase caption reports omitted stages; detection and the final result remain available.
Ordinary recognition and CLI audit exports do not retain this timeline.

Enable **Compare A/B** to show two copies of the same cue. Select **B** under
**Configuration / phases**, use **Copy A to B** or enter a second file path and choose
**Load YAML**. You can also start with both files:

```sh
python stash_ai.py recognition tune-detection --scene-id 123 --face-config baseline.yml --compare-config candidate.yml
```

The two panels share zoom and scrolling. **Differences** lists the edited detector
settings. The phase slider controls the panel selected under **Configuration / phases**.
A and B keep independent phase positions so different retry counts cannot silently
align unrelated stages; select a phase in A, switch to B, then select its comparison
phase. **Run detection** computes the missing results sequentially on one model
thread. Identical configurations share their result. Changing B preserves A for the
current cue. Navigating during detection immediately clears the overlays, and late
results cannot replace the new preview. An error is shown in the affected panel,
so you can fix a setting and retry that panel. When temporal prompt recovery is enabled,
the preceding cue supplies context; the phase history covers only the selected cue.

## Full-cue face and pose detection audit

Use `recognition audit-detection` to tune the detectors visually on **every VTT
cue of one scene**, including cues without faces, without people, or without
any detections:

```powershell
python .\stash_ai.py recognition audit-detection --scene-id 123 --face-config facial_recognition.yml --output-dir recognition_detection_audit/scene-123-run01
```

The new output directory contains `scene-123_cue-000000.png`, one PNG per cue
in original VTT order (zero-based), and `manifest.json`. Each PNG preserves the
whole cue at its original scale and shows **all returned face and person
boxes**, five face landmarks and measured COCO17 pose points. Pose boxes,
points and labels use **purple, green, blue**, then **red for every additional
person**, in person detection order. Face boxes are **yellow** when associated
with a pose and **red** when no pose is matched (`person_index: null`), including
cues with no detected people. Face points and labels remain **yellow**.
Colors reset for each cue and do not track identity across cues.

When `pose.refinement.enabled: true`, the audit also shows the
SAM-pose2seg masks already produced during detection. The default is a **25%
fill in the person's pose color plus a contour**. Masks and white rings around
SAM's positive head/torso prompts are drawn below pose points and face annotations.
The legend shows SAM validation, the pose outcome and quality-mask availability.
A solid contour shows the exact crop-local quality mask used by recognition.
A dashed contour shows an excluded candidate, including a shared-mask conflict.
A validated mask can survive second-pass pose drift inside the original crop.

Audit-only display switches (also apply to `--sweep`):

- `--mask-view overlay`: default fill and contour. Displays the quality mask when available,
  then the cleaned or raw SAM candidate.
- `--mask-view contour`: contours without fill, for unobstructed source details.
- `--mask-view panels`: overlay plus one row per person below the overview,
  showing the **raw SAM mask** on the left and **cleaned mask** on the right.
  Each panel shows only that person's mask, prompt rings and final pose. Headers
  state whether the cleaned mask was actually used by the second PMPose pass.
  Unavailable stages show an explicit placeholder. These are full-cue mask
  overlays, not the resized/normalized tensor fed to PMPose.
- `--mask-view off`: previous box/point-only view; no mask capture or legend.
- `--mask-opacity 0.4`: change fill opacity in `0..1`; contours stay opaque.
- `--hide-rejected-masks`: hide rejected mask overlays, prompt rings and detail
  panels while keeping their reasons in the legend and JSON.

For example, inspect one cue with separate person masks:

```powershell
python .\stash_ai.py recognition audit-detection --scene-id 123 --cue-index 42 --mask-view panels --face-config facial_recognition.yml --output-dir recognition_detection_audit/scene-123-cue42-masks
```

Display switches never enable refinement or override its YAML gates: `no_overlap`
and `no_reliable_prompts` mean SAM did not run. The audit reuses the original
inference masks without extra model calls, preserves source pixels/coordinates,
and releases packed mask data after each cue/trial, including failed exports.
The audit writes its mask visualizations to PNG. During normal recognition,
`storage.retain_body_masks` controls storage of validated body masks for Inspect
Scene. The audit manifest records options in `annotations.sam_masks` and stage availability,
second-pass use and panel offsets in each frame's `mask_visualization`.
Panel offsets are frame-local; for sweeps add the frame's `source_offset`.
Capture is limited to 16 MiB of packed masks per frame and extended PNGs to
40 million pixels; an explicit error suggests a smaller display mode on overflow.

Optional `pose.deduplication.enabled: true` suppresses duplicate PMPose
candidates using shared reliable head/torso points before SAM and face matching.
It requires PMPose, is off by default, and also works without SAM. The audit
draws removed candidate boxes as gray dashes and adds a footer with D-to-P
mapping, first-pass core quality, detector score, box IoU, point agreement and
keep/suppress reasons. `D0` means candidate 0 after detector score/box-NMS
filtering; `P0` means output person 0 after deduplication. Normal pose colors
and face matching refer to surviving P indices. This footer also appears with
`--mask-view off`; JSON stores the details in each frame's `pose_deduplication`.
Its legend offset is frame-local, so add the frame `source_offset` in sweeps.
Use a sweep row `{parameter: pose.deduplication.enabled, value: true}`
against a disabled baseline to compare the same cue with box NMS unchanged.
See [PMPose deduplication](pose-backends.md#duplicate-pmpose-detections-before-sam)
for the gates, limitations and evidence recollection requirements.

Face/person matching keeps the existing contained-face path (5% of the shorter
face side as boundary tolerance, with a valid pose nose/eye inside the face).
A partially protruding face can also match when at least **50% of its box area**
overlaps the person box and at least **two corresponding nose/eye landmarks**
agree within **15% of the shorter face side**. All available compared landmarks
must agree; missing, out-of-crop or collapsed points cannot provide extra support.
The anatomical COCO nose/left-eye/right-eye order maps to face indices 2/1/0.
Multiple eligible people or competing faces still prevent an association.
The gates are automatic, shared across pose backends, and recorded in
`annotations.face_person_association`; no YAML changes are required.

Face boxes and their landmarks move together to free space near the measured
face. Each valid landmark has a faint hollow marker at its original position
and a thin, translucent dashed line to its translated position. Pose points
are drawn above these leaders, so overlapping eyes/noses remain readable.
Placement avoids original face boxes, measured pose points and other face
annotations. If there is no room inside the cue, the PNG gains a dark margin
on the right for the face annotations; source pixels stay unscaled at `(0, 0)`.
Partially visible face boxes are clipped to the source before translation.
`F0` and `P0` labels show detection indices and scores; face labels additionally
show `-> P0` for the matched person, or `-> unmatched`. Point labels retain
their landmark indices. Unmatched faces add a second line explaining why:

- `no detected pose`: no person survived detector filtering.
- `face outside pose box`: person boxes have insufficient overlap for either matching path.
- `no valid nose/eye points`: eligible overlapping poses have no usable nose/eye points.
- `nose/eyes outside face`: usable head points of eligible overlapping poses miss the face box.
- `nose/eyes missing or outside face`: different overlapping poses fail for those two reasons.
- `partial face: need 2 head points`: fewer than two distinct corresponding landmarks support a partial box.
- `face/pose head points disagree`: a corresponding nose/eye pair exceeds the distance tolerance.
- `pose candidates fail different gates`: overlapping poses fail for different reasons; see JSON `pair_rejections`.
- `multiple poses: P0, P1`: more than one person satisfies the pair gates.
- `competing faces: F1`: another face also nominates the sole person candidate.

Candidate/competitor lists show at most three IDs and the remaining count.
Labels reserve their full height and width, including the extra line; crowded
or edge cases use the same margin fallback. Missing points are reported as
unavailable, without guessing which confidence/visibility filter removed them.
These diagnostics use the existing matching gates and do not relax them.
Other identity/person audits retain their existing
rendering. A cue with no detections is still exported with unchanged pixels.
The manifest records timestamps, original sprite rectangles, cue dimensions,
geometry, scores, face/person associations, counters, source fingerprint and
the applied detector settings. Manifest **format 3** keeps `box`/`landmarks`
and person `bounds`/`keypoints` in original cue coordinates. Each face adds
`display_offset: [dx, dy]`, `display_bounds` (xyxy) and `display_landmarks` in
PNG coordinates; invalid/missing source landmarks remain omitted and use
`[-1, -1]` in display geometry. `source_dimensions` describes the original cue,
while `output_dimensions` includes any added margin. The palette and layout
conventions are in `annotations`; each detection retains its RGB
`annotation_color`, and faces also record their actual RGB `box_color`.
`annotations.unmatched_face_box_color` defines the red unmatched-face outline.
Each face's `association` records its reason (null when matched), candidate
person indices and competing face indices. `annotations.association_reasons`
maps reason codes to the PNG labels; full index lists remain in the manifest.
When no pair passes, `association.pair_rejections` lists each person's rejection.
The translation is only a visualization change and has
no effect on detector thresholds, face/person matching or recognition.

The command uses `face.detector.backend` and `sapiensid.pose` from the current YAML,
including confidence, NMS, result-count and keypoint thresholds. **Both
detectors always run**, even if `body.enabled` or independent collection
is disabled. Both CPU/GPU profiles include the required pose runtime
dependencies, including RF-DETR-XL (`pose.detector.backend: rfdetr_xl`).
Configured pose weights are still required and verified weights download on
first use; see
[RF-DETR-XL setup and model terms](pose-backends.md#rf-detr-xl-person-detector).
Weights follow the existing `execution.models.download_missing` policy. Face and body
recognition encoders are not loaded. `--device cpu` is an optional override.

For complete coverage, this action **ignores `execution.scan.max_cues`** and reports
that override in the manifest. It applies no later face/person quality,
FaceOcc visibility, anchor or identity filters: rejected candidates must remain
visible for inspection. Detections suppressed by the detector's score/NMS/count
thresholds cannot be drawn; missing or rejected pose points are omitted.
To compare configurations, rerun with another YAML and a **new output folder**.
`--progress-style off` disables cue progress and ETA.

The action reads fresh VTT/sprite inputs, performs no video sampling, and makes
no recognition database or Stash changes. Persistent HTTP caching is disabled
for this export. PNGs are processed one cue at a time and are private local
files (0700 directory / 0600 files on POSIX). Existing output directories are
refused. Interrupted/failed exports can leave partial PNGs; `manifest.json` is
written only after all cues succeed. The suggested `recognition_detection_audit/`
parent is gitignored.

## Single-cue parameter comparisons

After inspecting a whole-scene detection audit, take the zero-based cue index
from its PNG name: `scene-123_cue-000042.png` means scene `123`, cue `42`.
The index identifies an original VTT thumbnail, not a decoded video frame or
a timestamp. To rerun only that cue with the current configuration:

```powershell
python .\stash_ai.py recognition audit-detection --scene-id 123 --cue-index 42 --face-config facial_recognition.yml --output-dir recognition_detection_audit/scene-123-cue42-rerun
```

For a visual comparison, create `detection_sweep.yml` with a table of parameter
names (dotted YAML paths) and single values to test, in the desired order:

```yaml
- {parameter: face.detector.score_threshold, value: 0.7}
- {parameter: face.detector.score_threshold, value: 0.8}
- {parameter: pose.estimator.keypoint_score_threshold, value: 0.3}
- {parameter: pose.estimator.pmpose.presence_threshold, value: 0.4}
- {parameter: pose.estimator.pmpose.presence_threshold, value: 0.6}
```

The PMPose settings require the corresponding backend to affect detections.
CSV is also accepted (`.csv` extension), with the exact header `parameter,value`;
cell values use YAML scalar types, e.g. numbers, `true`/`false`, or quoted strings:

```csv
parameter,value
face.detector.score_threshold,0.7
face.detector.score_threshold,0.8
pose.estimator.keypoint_score_threshold,0.3
```

```powershell
python .\stash_ai.py recognition audit-detection --scene-id 123 --cue-index 42 --sweep detection_sweep.yml --face-config facial_recognition.yml --output-dir recognition_detection_audit/scene-123-cue42-sweep01
```

The new directory contains **one** `scene-123_cue-000042_sweep.png` and
`manifest.json`. The PNG stacks the baseline first, then every table row
vertically, without resizing the original cue. Every panel has a header with
scene/cue/time, parameter, tested and baseline values, and face/person/match
counts. Existing colors, shifted face callouts, pose points and unmatched-face
reasons are preserved, including any right gutter needed by a panel.

**Each row changes exactly one parameter relative to the current configuration.**
Changes do not accumulate and there is no Cartesian product. The YAML file is
not edited. Each trial runs fresh face and pose inference on an independent
copy of the same unannotated VTT crop; detections are not reused. Detector
instances are loaded and closed sequentially, so model initialization can
dominate runtime. Only the selected cue is inferred; source initialization
still validates the scene's VTT/sprites and records their fingerprint.

Supported parameters: `face.detector.backend`, `face.detector.score_threshold`,
`face.detector.nms_threshold`, `face.detector.max_faces_per_frame`, and existing scalar
leaves under `sapiensid.pose` (including `refinement`). Backend-specific options
only affect their backend; use the appropriate base YAML. Face/person quality,
anchor and identity filters are not applied in this audit, so their parameters
are rejected instead of producing a misleading comparison. Unknown paths,
invalid values and incompatible configurations are rejected before network or
model loading. A table has 1–64 rows; the final canvas is limited to 40 million
pixels. Use fewer rows for large cues.

Sweep manifest **format 4** records baseline configuration, ordered trials,
old/new parameter values, detections and the original source fingerprint.
`scene.frames` contains one record per panel, with `variant_index: 0` for the
baseline. Source geometry remains cue-local; rendered face coordinates and
each frame's `output_dimensions` remain panel-local. Add `source_offset` to
rendered coordinates to locate them in the combined PNG. `panel_offset` marks
the top of its header and `scene.output_dimensions` describes the full PNG.
This mode has the same read-only recognition/Stash behavior as the whole-scene
audit and requires a new output directory.

## Read-only five-landmark alignment audit

Use the opt-in `recognition audit-alignment` action to inspect detector landmarks
and the shared ArcFace 112 × 112 crop for the detector selected in
`face.detector.backend` (YuNet, SCRFD or RetinaFace). It never loads face recognizer weights,
produces embeddings, modifies SQLite, or publishes to Stash.

```powershell
python stash_ai.py recognition audit-alignment --scene-id 1 --scene-id 2 --face-config facial_recognition.yml --output-dir recognition_alignment_audit/run01
```

**By default all detected faces that pass the configured detection gates are
saved**; there is no audit face-count limit. `--max-samples-per-scene 64`
optionally bounds PNG exports to 64 accepted faces per scene (with evenly
spaced VTT cue selection), without a hard upper bound on the argument.
`--cue-index 135` is a repeatable zero-based VTT cue filter, and has no
128-index limit. With no sample cap and no cue filter, all selected VTT cues
are eligible. `execution.scan.max_cues` still limits source cue selection, while
`face.detector.max_faces_per_frame` remains a detector-level limit.

The audit applies the same hard detection gates as `recognition run`:
detector score, minimum source face pixels, `face.quality.min_sharpness`,
alignment validity, complete-face and pose restrictions. When
`face.quality.min_visible_face_fraction > 0`, it additionally loads **FaceOcc**
and checks the visibility threshold, just as recognition scanning does.
`face.quality.faceocc_logit_threshold` sets the strict per-pixel cutoff on raw
FaceOcc output (default `0.0`). Lowering it admits more pixels into the mask
used for both visibility and sharpness. The minimum visible fraction sets the
admission floor after measuring that mask.
With every face visibility floor disabled (`0`/`null`), no scan FaceOcc asset is
loaded; all scan/query/anchor sharpness floors must also be disabled. Rejected faces
are **counted with their rejection reason but not exported as PNGs**.
The manifest (format 2) includes `detection_rejections`, counters for
detected/rejected/passing faces, exact applied `detection_policy`, saved
face quality including optional visibility, and each image's geometry.
This audit validates detection acceptance, **not** embedding, clustering,
or the accuracy of identity matching.

`face.quality.min_sharpness` measures facial foreground rather than the complete
face box. When FaceOcc runs for a visibility policy, it measures only visible
face pixels inside the face ellipse; the sharpness gate is applied after this
segmentation. Any positive scan/query/anchor sharpness floor requires at least
one positive face visibility floor; invalid YAML raises an error before work
starts and identifies the settings to change. Without FaceOcc, ellipse quality
is diagnostic only because these floors must be disabled. Both use the same
112px aligned recognition input scale. FaceOcc remains 256px; its mask is mapped
into encoder coordinates without resampling the FaceOcc RGB into an encoder crop.
The directional reblur score ranges from 0 to 100.
Complete 15x15 filter footprints touching a mask boundary
or synthetic padding are excluded, so a sharp occluder/background cannot rescue
a blurred face when a correct semantic mask is available. No additional model
inference is introduced. Higher `min_sharpness` values reject more blur; `0`
disables the gate. Anchor/query sharpness floors use this encoder-input value;
portrait ranking retains its separate 256px `portrait_sharpness`. Active portrait sharpness weighting with a positive
quality target likewise requires visibility weighting, or anchor-only visibility
measurement. Portrait quality targets remain preferences, not admission gates.

FaceOcc receives an upright crop from the five-landmark whole-face template,
including sideways and upside-down faces. `face.quality.visibility_region` is
`mesh` by default. A pinned MediaPipe mesh-only ONNX session uses those same
pixels to estimate 478 xyz landmarks; the projected envelope of the 468 facial
surface points supplies the measurement contour (iris points are excluded).
It includes the forehead, jaw, cheeks and profile nose. Projected pixels count
once, so a profile does not acquire an extra denominator for its far side.
This convex surface envelope is still an estimate, not exact skin segmentation.

The contour is independent of FaceOcc logits. Low face presence, inconsistent
anchors, a clipped contour or implausible area select the five-landmark ellipse
fallback. Geometry checks normalize by eye-to-mouth distance, not compressed
profile eye spacing. Both choices require the configured real-source coverage;
otherwise the measurement is unavailable. Model download/runtime failures are
reported as model errors, not silently disguised as geometric fallback.
`ellipse` explicitly selects the previous geometry without loading the mesh.
The mesh runs on the visibility model owner, including prepared scan and portrait
batches; blendshape inference is not needed for visibility.

Visibility counts FaceOcc-positive pixels inside the expected contour divided
by its **full** area, including externally occluded and off-frame portions.
The contour is never fitted to the visible mask. Foreground sharpness uses the
same logits with a separate support region: a six-pixel dilation of an accepted
mesh contour in the 256px crop, intersected with real-source pixels. This retains
nearby skin without admitting arbitrary distant positive components. Ellipse
fallback keeps its original support. Adjacent faces touching this narrow support
remain a limitation; this is a geometric bound, not instance segmentation.
Masks retain holes and do not fill occluders. Existing sharpness edge exclusion
and mapping into the exact 112px encoder input remain in place.

With partial boxes enabled, `face.quality.min_face_source_fraction` sets how much
of the selected region must lie in the VTT cue (default 0.85). Source coverage is
checked after contour selection. Quality records retain `region_method`,
`region_requested` and `region_fallback_reason` inside `visibility`; Inspect Scene
explains fallback alongside the visibility gate. The updated visibility and scan
contracts reprocess selected scenes, including previously occluded observations,
without deleting the database. Detector/encoder identity spaces are unchanged.

The mesh needs accurate upright crops and can fail on extreme profiles, heavy
occlusion or very small faces. Its presence score is not a calibrated contour
accuracy score. The pinned-model smoke compares a public-domain portrait with
an artificial lower-face occluder and quarter-turn rotations; representative VTT
frames are still needed to calibrate thresholds and assess difficult profiles.

Quality records include `sharpness_method` (`faceocc_mask`, `face_ellipse`, or
`unavailable`), `sharpness_pixels`, and the previous rectangular measurement as
`crop_sharpness` for comparison. These scores differ from the old 112px crop
variance; all existing thresholds require recalibration on representative frames.
See [sharpness measurement and migration](sharpness.md) for the algorithm,
noise handling, shared score units and cache refresh.
The scan/quality contracts change so the next run recomputes face measurements.
Body foreground sharpness has its own `body.collection.min_sharpness` threshold
and is measured after SapiensID square padding and resize to 384x384; face and body scores are not
interchangeable.

The created folder contains a `manifest.json` and three PNGs per passing
face: `*-source.png` (VTT image with face box and five landmarks),
`*-aligned.png` (112 × 112 pre-normalization crop), and
`*-source-mask.png` (white = real source pixels, black = synthetic padding).
Each scene records its `vtt_source_fingerprint` and deterministic
`expected_recognition_scan_id`. Compare the latter with `scan_id` from
`recognition report` to verify that source, recognition-model namespace,
scope and scan policy match the embedding run. Alignment source/face-region
coverage uses the same inverse-warp geometry as `quality.alignable`.

**Privacy and disk space:** local screenshots are sensitive biometric data.
Explicit scene selection and a *new, non-existing* output folder are required;
POSIX permissions are 0700 for directories and 0600 for files. Exporting all
faces can create **many PNG files and substantial disk usage**. Never commit
or share the folder without inspecting it; the suggested
`recognition_alignment_audit/` parent is gitignored. No automatic upload
occurs and normal recognition is unchanged.

## Rolled and translated face landmarks

Set `face.detector.landmarks_from_facebox: true` to re-estimate the five landmarks
on **every detected face**, including upright faces with an already valid alignment.
This optional mode works with SCRFD, RetinaFace and YuNet, independently of body
collection and `alignment_retry_error`. The selected detector runs again on the
original image pixels inside the face box plus 25% margin on each side, clipped
to the cue boundary. SCRFD fits the crop within 320px on its fixed 640px input
canvas to avoid excessive face magnification; RetinaFace and YuNet use their
normal preprocessing. Crop-local
points are restored to cue coordinates; the full-frame face box and score are
preserved. The crop result must match the same head and pass the existing
confidence, ownership and alignment checks, including alignment with the original
box. Full-frame landmarks are discarded: they neither select crop orientations
nor set the crop's alignment threshold (normalized fit error at most 0.20).
They remain available only in the audit's `original` diagnostic record.

The default `false` preserves the existing conditional retry behavior described
below. `true` adds one crop inference per face; when existing rotation recovery
is enabled, every face uses all four crop orientations. It
does not enable rotation recovery by itself or introduce a separate landmark
model. A missing, invalid or ambiguous crop result retains the box and score but
marks all five landmarks unavailable (`[-1, -1]`). There is no full-frame
landmark fallback. Tune Detection and Inspect Scene show the box without points
and label the missing landmarks; recognition rejects that face as
`face_landmarks_invalid` before embedding or FaceOcc inference. This mode can
change context and effective face resolution but is not a
guarantee of improved accuracy on extreme profiles.

The setting is available under `face.detector` in Tune Detection. Press **Detect**
again to compare it on the same cue. Recognition uses the same detector path;
rerun the affected scenes and refresh Inspect Scene to view the new measurements.
The crop-only policy invalidates older `true` face inference and scan identities,
including cached full-frame fallback points. Changing this setting also invalidates
face inference and scan identities, while
`false` retains compatibility with existing results from before the option.

With `face.detector.pose_guided_rotation: true`, SCRFD and RetinaFace retry already detected
rolled faces and spatially inconsistent landmarks before face embeddings or body
collection consume them. The existing option still requires SCRFD or RetinaFace and enabled
body collection. It does not depend on `alignment_retry_error`: a translation of
all five landmarks leaves the similarity-fit residual unchanged. The crop is
rotated using the measured facial axis, including the eye-to-mouth profile axis.
If landmarks are outside the box tolerance, their measured axis is untrusted:
all four quarter turns are tried, even if that axis appears well-defined. An
unavailable axis also tries all four quarter turns. Retry batches remain bounded
by `execution.detector_batch_size` and eight inputs.

Replacement requires a consistent face box, valid source alignment, comparable
detection confidence and no competing detected head. Multiple eligible heads or
inconsistent results across orientations leave the original detection intact.
This is a new detector measurement, never a fixed offset applied to landmarks.
Upright spatially consistent faces add no inference unless `landmarks_from_facebox`
or the separate local `alignment_retry_error` trigger is active. When that trigger is nonzero it also
retries out-of-box landmarks, even when the alignment residual is small.

For a pose without a face, a missing or compressed far eye now leaves orientation
unknown instead of assuming 0 degrees. Recovery checks quarter-turn candidates
against the original pose and all competing detections. Borrowed/conflicted head
points cannot authorize the crop; a failed or ambiguous recovery adds no face.

Landmarks outside the face box by more than 5% of its shorter side now fail the
mandatory alignment gate even with `face.quality.max_pose_deviation: null`. The
tolerance admits minor edge noise; it does not impose frontal pose or screen-up
ordering. The legacy stricter `landmarks_inside` pose check is still optional.
Face inference, quality and selected-scene scan versions are updated. Rerun
recognition for affected scenes to replace old detections; refresh Inspect Scene
after the run. Model weights and identity namespaces are unchanged.

Detection audits and Tune Detection also retain numeric `face_landmark_repairs`
in the frame record. Each SCRFD/RetinaFace batch frame (and YuNet with crop mode
enabled) records source dimensions and whether rotation/local retries and
`landmarks_from_facebox` were enabled. Each face records its original geometry and
alignment measurements, estimated orientation, crop (`crop_xyxy`), attempted
counterclockwise `turn_degrees`, and final `result`. An attempt retains both
`crop_detection` (coordinates in the rotated crop) and `source_detection`
(coordinates restored to the original cue). These are independent of the
annotation's `display_offset`.

Candidate `rejection` is null for an eligible result, or explains the first
failed check: `retry_not_upright`, `different_face_box`, `competing_face`,
`lower_detection_score`, `invalid_alignment`, `invalid_original_box_alignment`,
`alignment_error`, or `alignment_not_improved`. A successful crop-only point
replacement has face status `refined_from_facebox`; its final `result` retains
the original box and score. The optional candidate `quality` is null when an earlier
check rejected it before alignment was measured. An empty `candidates` list means
that the retry returned no detections. Attempt status distinguishes
`ambiguous_candidates` from `no_eligible_candidate`; face status also identifies
`repaired`, `conflicting_orientations`, `empty_crop`, or `not_requested`.
This evidence makes failed retries inspectable without sharing PNGs. It contains
no images or model tensors and is collected only during audit/tuner rendering.
An unrepaired detection remains visible in the diagnostic overlay; the mandatory
alignment gate still prevents invalid landmarks from producing recognition evidence.

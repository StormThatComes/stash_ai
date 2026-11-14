# Portraits and performer names

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Refresh existing portraits](#refresh-existing-portraits)
- [Portrait selection](#portrait-selection)
- [Automatic portraits and operator names](#automatic-portraits-and-operator-names)

## Refresh existing portraits

To refresh portraits after editing the portrait YAML settings, reuse the existing
identity memberships without scanning scenes, rebuilding identities, changing
scene links, or applying performer names:

```powershell
python stash_ai.py recognition portraits
python stash_ai.py recognition portraits --performer-id 123
python stash_ai.py recognition portraits --performer-id 123 --performer-id 456
python stash_ai.py recognition portraits --performer-id 123 --scene-id 6306
python stash_ai.py recognition portraits --scene-id 6306 --scene-id 6307
```

With no filters (or with `--all-scenes`), this action selects all published
identities with retained, complete evidence in the current Stash instance and
configured model namespace. It reads the local face SQLite instead of listing
the entire Stash library. Each performer is processed once. Repeated
`--performer-id` selects numeric Stash performer IDs, rather than internal
recognition UUIDs, and uses all their eligible retained scenes. `--scene-id`
limits both the identities and the evidence used for their portraits; combining
it with `--performer-id` intersects the filters. Unknown or ineligible performer
IDs do not expand the selection. Empty selections are recorded as successful
no-ops with `portrait_people_selected=0`.

`recognition portraits` replaces `recognition run --portraits-only`; the old
switch is no longer accepted. `--performer-id` belongs only to the portraits
action. Normal `recognition run` requires `--scene-id`, `--scene-range` or `--all-scenes`.
Both actions support `--face-config`, `--faces-db`, `--device`,
`--progress-style`, and `--retry-unresolved-images`.

Portrait scoring, framing, VTT/video selection, manual-image protection and
`portraits.enabled: false` use the same rules as a normal run. Cached VTT
reranking does not load detection/recognition models; original thumbnails may
still be fetched for measurement or encoding. Model assets are prepared lazily
if portrait inference needs them. Video mode can decode and verify new portrait
frames, and newly requested quality measurements can run their models. This action
keeps identity memberships unchanged: use a normal scoped run to apply detection
or grouping changes. Pending performer merges are left for a normal run too.

`--retry-unresolved-images` can be used with `recognition portraits` under the
same fresh ownership/image checks. Run history records `invocation.command: portraits`,
any explicitly selected `invocation.performer_ids`, the selected scene IDs,
portrait progress and the existing portrait counters.
The console shows a performer progress bar and nested portrait-candidate work.

## Portrait selection

Auto portraits use strict, configurable fallback tiers:

```yaml
version: 2
portraits:
  selection:
    kind: auto
    auto_precedence: [face+pose, face, pose]
```

`face+pose` selects a face observation with usable, associated SapiensID person
evidence for the current model/pose contract, then crops that associated **person**.
It does not combine different observations. `face` selects remaining eligible
faces, and `pose` uses accepted retained person crops. List all three exactly once,
in the desired order (for example `[pose, face+pose, face]`). Older YAML files use
the default above. Explicit `kind: face` and `kind: person` ignore this order.

Each tier exhausts its usable candidates, including retained-pixel fallbacks,
before the next tier is tried. Quality scores rank candidates within a tier;
quality shortfalls alone cannot let a lower tier win. Identity checks,
`anchor_only`, crop/encoding limits and manual-image protection still apply.
With SapiensID disabled or no current paired evidence, `face+pose` is empty.
Decoded video faces belong to `face`: a VTT hint's pose does not prove a pose in
different video pixels. The `face+pose` tier therefore uses the measured person
bounds in the verified VTT thumbnail, even with `source: video`, without new pose
inference. If that thumbnail is unavailable, only the associated retained person
crop can supply this tier; a retained face crop cannot stand in for a person.
Without usable person pixels, selection continues to another candidate or tier.

Like `pose`, `face+pose` applies `portraits.crop.aspect_ratio` within the person bounds,
preserving measured head points with clearance where possible. `null` keeps the
entire measured person crop. A fixed ratio can trim other parts of the body.
`portraits.crop.margin` applies only to face crops. Face identity checks, anchor gates
and face quality ranking still choose the `face+pose` observation; framing does
not change its membership or require body-only identity collection.

Changing the order invalidates portrait caches only. Apply it to existing images
with `python stash_ai.py recognition portraits`; no identity rebuild is needed.
Portrait metadata records the winning `selection_category`.
The framing upgrade also invalidates old face-framed `face+pose` images: refresh
with the same command to reframe app-owned portraits without rebuilding identities.
Cache reuse checks the person geometry and contract as well as the crop kind.

Set `portraits.selection.anchor_only: true` to require the actual portrait face to pass
the current clustering anchor pose and visibility rules, in addition to its
identity check. The default is false. `portraits.quality.max_pose_deviation` and portrait
quality weights remain ranking preferences; this opt-in gate uses
`matching.face.anchors.quality.max_pose_deviation`, `min_anchor_visible_face_fraction` and
the existing auxiliary admission floor. Anchor quality and strict reference
authority are separate: an accepted attachment can illustrate its identity when
its source face qualifies as an anchor.

VTT and retained-crop selection, decoded video faces, prefetch and cached portraits
all apply the gate. Video eligibility uses the actual decoded face, including
when its seek hint was auxiliary. Positive anchor visibility requires a current
FaceOcc measurement even with `portraits.weights.visibility: 0`; unavailable
measurements cannot authorize a new anchor-only portrait. Rejected sources record
`portrait_anchor_required` or `portrait_anchor_visibility_unavailable`. If no
eligible anchor exists, no new image is published; an already published image
is preserved. The option invalidates portrait eligibility/ranking caches only,
works with `recognition portraits`, and never requires an identity rebuild or
rescan. With no configured anchor restrictions, every otherwise eligible face
passes the same unrestricted anchor policy.

`matching.face.views.corroborated_view_enrichment` (default true) can extend the frozen
historical query bank with a `corroborated_view`. Its accepted attachment cohort
needs at least `max(view_min_support, min_merge_support)` independent cues from
two distinct scenes. Every query must independently match either two strict-rooted
views whose support sets are disjoint, or one strict view with complete support
in two other source scenes. The single-root route requires at least the same
minimum number of distinct timestamps in **each** source scene and forbids using
any source scene as a query scene. Both routes preserve mixed rivals/co-presence
checks and the immutable, non-transitive appearance radius. Overlapping prototype
aliases cannot manufacture independent confirmations.

Promotion is derived from latest complete, currently eligible durable memberships.
The proof records its parent route, original strict and source-scene witnesses, bounded query witnesses,
scene count and thresholds. Losing support, ownership, quality or a current scan
revokes the next capture and invalidates its matching cache. Promoted views remain
query-only and never parent another promotion or vote for strict identity merges.

Promotion keeps separate hypotheses for disjoint rooted parent pairs. A later
query confirmed through another pair cannot intersect away an existing valid
proof. Each hypothesis retains its bounded appearance exemplars as candidate
representatives, including supported edge looks beside the dominant medoid.
Every emitted representative needs its own narrow independent-query support and
two-scene proof. Identical vectors/pose representatives are deduplicated. Current
source validation still revokes obsolete proofs; no old automatic label supplies
authority. Query witness vectors, pair hypotheses and emitted exemplars share
the explicit reference byte budget and fail rather than silently losing coverage.

Full and mixed selections finalize one rooted generation inside the same private
replay. Before replay, previous attachment observation IDs freeze query eligibility,
without using old labels as positive evidence. They cannot seed new cores or supply
birth votes. A prior supported look from one query scene can defer a nearby new
birth only after fresh validation against two disjoint rooted parents. Such hints
contain no owner or positive matching authority. They prevent a new local core
from stealing selected source views while a foreign query scene awaits its second
scene. After final auxiliary source ownership, stage one first revalidates deferred
queries against roots, then validates accepted attachments, combining selected and
unselected scenes, then freezes the supported appearances. Stage two uses that bank
to resolve remaining queries before the single durable commit and publication.
Its output cannot supply another promotion or change a promoted bank's root owners.
Invalidated source references remain visible as rivals but cannot authorize new
claims. There is no second full scene replay for the new generation, no repeated
enrichment until convergence, and no rescanning or model inference. Stable completed
generations reuse the cache. Checkpoints retain strict-prefix recovery and the
query-admission dependencies; an interrupted final stage resumes from retained
evidence and performs fresh validation. Strict checkpoint coverage counts current
anchor-admitted observations; auxiliary rows remain available for the later stages.
Historical label binding excludes rejected observations from its reference set,
while retaining cross-scope, model-namespace and incomplete-scan checks. A rejected
old member cannot train a reference or block the selected repair of its identity.

A previous promotion plan can only defer a new near-appearance birth. It cannot
assign an owner, override previously established strict witnesses, or authorize a
match after its parents fail fresh validation. Deferred queries are resolved only
against the final current bank; obsolete hints are cleared with the completed
plan. A freshly valid one-scene negative hint can be cached while its query remains
withheld; it does not request another identical run or become a positive view.
A stale hint that deferred a birth still requests a clean retry when it loses
fresh validation; newly accepted promotion queries no longer require a later run.
Scans and embeddings are retained. Matching policy advances to v35.

The local anchor bank uses the same broad/narrow cohort builder as the global
bank. Independently supported reserve centres remain available beyond the primary
`max_view_prototypes_per_person` priority, including context-validated additions.
There is no silent 32-cohort discovery cutoff. Scratch exemplars and emitted
medoids are checked against `matching.face.samples.references.max_reference_mb`; insufficient memory
fails explicitly. Matching policy v30 rebuilds decisions from retained embeddings.

Frozen view-query support can cross prototype boundaries of the same identity
through overlapping hypotheses. An observed query supplies each immutable centre;
every witness must fit half the angular cohesion radius around that centre.
Overlaps preserve boundary votes without combining hypotheses into a transitive
support component. Distinct scene/time votes remain mandatory; reference-training
cues never vote as queries. Existing support for one exact prototype remains
valid and takes one streaming pass. Only unsupported query vectors are retained
within the explicit reference budget; additional witnesses are streamed from
fresh metadata/embedding row factories, without loading crops. All proofs freeze
before callers can update those rows. The same query-support
rules apply to local anchors, auxiliary views, historical and selected pools,
and partial-cohort repairs. Every original source witness remains in recovery
proofs. A rival matching only one member of a whole query cohort still vetoes
that member: rivals cannot disappear because they do not explain its peers.
These changes advance the matching policy to v34 and reuse retained embeddings.

The complete appearance bank preserves independently supported narrow looks
inside broader coherent cohorts. Narrow looks use half the angular view-match
radius and must satisfy the usual independent support minimum themselves.
Broad cohorts remain available when their individual variants are too sparse.
Every retained centre participates in both matching and rival competition, within
the explicit reference memory budget. Identity ranking uses the strongest valid
strict or view evidence per person; a winning view remains attachment authority.
An independently unambiguous strict proof keeps strict authority even when a
stronger own view improves ranking. Its original strict score, cohesion and
margin must still beat all mixed rivals and raw-anchor competition. An own view
below its match threshold cannot hide a valid strict match.

`matching.face.anchors.quality.min_visible_face_fraction` and
`matching.face.samples.minimum.min_visible_face_fraction` optionally qualify retained
FaceOcc measurements for their respective roles. Anchor admission requires both
the anchor pose and visibility limits; a frontal but occluded face may instead
query as auxiliary. Faces below the auxiliary limit remain explicitly withheld.
The auxiliary limit must not exceed the anchor limit. Both explicit limits must
be at least `face.quality.min_visible_face_fraction`, the hard scan floor: lower
that floor first to retain more occluded queries. Null role limits preserve the
previous admission rules. Positive role limits request FaceOcc even with a zero
scan floor. Changing positive role thresholds reuses scans and embeddings;
enabling previously unmeasured visibility requires new measurements. Missing or
obsolete measurements never seed a visibility-restricted core. Portrait visibility
preferences remain independent of these identity roles.

Cold replay freezes bounded appearance cohorts for each selected scene before
creating identity labels. Quality, independent anchor support, appearance
diversity and frozen compatible looks determine their priority; physical cue
times still determine support and co-presence. Source auxiliary confirmation
uses that same frozen appearance order, so moving an otherwise identical shot
earlier or later cannot change which source core was already established.
The order metadata is rebuilt independently of replay checkpoint restoration.

A wholly contained view query remains attachment-only in a joint replay too.
All strict merger nominations and their rivals are frozen before those
attachments, and affected components cannot nominate a new merger afterwards.
Final direct auxiliary ownership revalidates selected source-view attachments.
Near-existing coherent queries can be withheld as
`historical_potential_identity` rather than immediately publishing another
person; normal match/cohesion thresholds and rival margins are unchanged.

With `cross_scene_provisional`, durable withheld queries are also reconsidered
against the entire frozen external bank, including when only an unrelated scene
is selected. Query votes are independent of reference-training votes. External
and current queries can jointly reach view support, but recovered observations
remain attachments and never train this pass. Source witness ownership is
checked again in the final transaction; backfill and selected membership changes
roll back together. Stash publication still follows the explicit selection.

Historical core demotions also require the final owners of their selected
source witnesses. If a source changes owner, its private plan is discarded and
matching is retried once without the invalidated source views. The first RAM
replay closes before retrying; raw scans and embeddings are reused. A second
unstable attempt fails before rebuilt memberships are committed.

Global reference repairs and unresolved-pool backfills enqueue affected scenes
in the face database, atomically with their derived memberships. A normal run
publishes only its selected scenes and acknowledges only successful publication.
To inspect or publish the remaining debt without scanning, loading models,
refreshing portraits or writing names, use:

```bash
python stash_ai.py recognition sync-identities
python stash_ai.py recognition sync-identities --scene-id 123 --apply
python stash_ai.py recognition sync-identities --apply
```

The default action is an offline read-only JSON plan for the configured model
and Stash scope. It separates identities with current memberships, retained
profiles without latest memberships, and queued scene-link repairs. The ordinary
fragmentation report exposes the same publication state. Missing publication
journals and mismatched managed links also reveal debt from before this queue
existed; only `--apply` adopts those inspected repairs into durable requests.
`--apply` reuses the
existing ownership checks and journaled ADD/REMOVE recovery. It requires the
queued policy, latest complete scan, queue revision and current assignments to
remain valid before writes. Successful scenes are acknowledged; blocked or
uncertain scenes remain queued. New queue revisions cannot be erased by an older
publication. App-managed stale links can be replaced, manual same-model links
and profile metadata survive, and profiles are never merged or deleted by this
action. Retained empty profiles remain available for reversible identity splits.

## Automatic portraits and operator names

Candidate eligibility is rechecked when the compatible identity references
change, using retained embeddings without rescanning media. VTT portraits can
illustrate an accepted assignment under the current matching policy, including
one accepted earlier in this run before a later scene added competing references.
Historical membership alone cannot override a current rejected assignment.
Decoded video faces and cached video embeddings require an independent identity
comparison. Existing confirmed/manual associations and images are preserved.
Older cached video portraits without retained identity vectors keep their bytes
but cannot authorize a new upload; no replacement inference runs merely to
upgrade their cache. Unknown name writes are reconciled against their original
journal payload and ownership, independently of a later names YAML edit.

Portraits default to the original VTT thumbnail used for an accepted recognition
observation. The retained geometry and embedding select and verify that person's
face without running detection/recognition a second time. Semantic visibility
assessment runs its own FaceOcc segmentation once per observation and is cached
independently. The selected scene's VTT/sprite content is checked against the scan
fingerprint before extracting a
wider crop from the full thumbnail. No video stream, FFmpeg process or video
decoder is used in this mode. All retained observations belonging to the accepted
person in the selected scans are evaluated, without `max_candidates`, a sharpness
shortlist or temporal/scene sampling. Quality targets and visibility are soft
ranking preferences: a blurry, angled, partially covered or incomplete face can
still supply the best available portrait. Only the highest-ranked usable
observation is encoded as a JPEG; an unavailable source tries the next ranked
observation. If no original thumbnail is usable, the retained recognition crop
supplies a smaller fallback portrait. Other candidates retain metadata, not additional JPEG BLOBs.
Existing prepared JPEGs remain reusable and rejected caches keep their bytes for
provenance. Only one scene's source/sprite cache is active at a time.
The portrait can have lower source resolution than a video frame; no upscaling
invents additional detail.

Configure framing and JPEG encoding in `facial_recognition.yml`:

```yaml
version: 2
portraits:
  enabled: true
  source:
    kind: vtt  # Default; use video for FFmpeg frame extraction.
  crop:
    margin: 0.35  # Add 35% of face width/height on each side.
    aspect_ratio: null  # Keep natural crop; 0.8 enforces 4:5 for faces and bodies.
  output:
    jpeg_quality: 90  # Integer 1..100.
  quality:
    min_visible_face_fraction: 0.8
  weights:
    visibility: 3.0
```

The default requested VTT portrait is 1.7 times the detected face width/height,
compared with the recognition cache crop's 1.3 times. Padding is limited by the
full thumbnail boundaries. An optional width/height aspect ratio expands the
crop, shifts it inside the source, and reduces padding if necessary. The ratio
remains an output constraint even when the available source cannot contain the
entire face at that ratio: the crop stays centered on the face within the source.
Body portraits apply the same ratio to the retained person crop, using measured
nose/eye/ear points with clearance to keep the head when possible. Without usable
head points, a shorter crop keeps the top of the person. Cropping cannot restore
a head already outside the retained source. `null` keeps natural framing.
`jpeg_quality` is preferred: encoding lowers JPEG quality and, if needed, image
dimensions to fit `storage.max_crop_mb`, without upscaling. Common ratios use
exact integer multiples; arbitrary ratios or very small images are limited by
whole-pixel rounding. Cached JPEGs with incompatible dimensions are regenerated
from available source pixels rather than reused.
Changing that budget invalidates only the portrait encoding policy. Compatible,
digest-verified cached JPEG pixels can be re-encoded without another face-model
call or video decode; the historical JPEG and source coordinates are preserved.
Confirmation/removal reads use the separate supported historical-image ceiling
of 8 MiB, so lowering today's encoding budget cannot block acknowledgement or
explicit removal of an older owned photo. Exact ownership and digest checks
still protect manual replacements and unresolved writes.
Actual crop coordinates,
source rectangle, observation/cue, input fingerprint, quality, identity scores
and source type remain in face SQLite. The cue time describes the VTT cue, not
a claimed decoder-measured video timestamp. Fallbacks record both
`requested_source` and actual `source_type` (`vtt` or `recognition_crop`), with
the actual crop coordinate system and fallback reason.

### Best available quality targets

The existing YAML keys retain their names and default values, but portrait
minimums/maximums now describe preferred quality rather than eligibility:

| Portrait setting | Default | Effect on ranking |
| --- | ---: | --- |
| `min_face_pixels` | 96 | Extra resolution penalty below the preferred shorter face-box side; 0 removes this penalty |
| `min_sharpness` | 25 | Extra sharpness penalty below the preferred Laplacian variance; 0 removes this penalty |
| `max_pose_deviation` | 0.35 | Increasing penalty above the preferred five-landmark pose estimate |
| `min_exposure`, `max_exposure` | 30, 225 | Favor mean brightness near the center of the preferred range |
| `max_clipped_fraction` | 0.15 | Extra exposure penalty above the preferred fraction of near-black/white pixels |
| `min_visible_face_fraction` | 0.80 | Extra visibility penalty below the preferred visible-face estimate |

Resolution, sharpness, pose and visibility also retain their general quality
scores. Each preference affects its corresponding nonnegative `weights` entry;
zero weight removes that component from the main score. The score is a bounded
weighted average. Equal scores use general face quality, then stable time/ID
ordering. All weights may be zero: selection then uses general quality without
loading optional models. For example, a 70px face below a 96px target remains a
candidate; a sharp, frontal 120px face usually scores better, while the 70px face
can still be the only available portrait.

Missing optional visibility/expression measurements contribute zero to their
enabled components. They do not imply an uncovered face or neutral expression,
and do not prevent a portrait. Initialization failures are attempted once per
component per run and reported as a warning; a later run can retry. Missing,
Identity loaders initialize only the selected detector and recognizer; optional
quality paths are verified by their own lazy loaders. Invalid or unavailable
optional model assets also produce warnings and unavailable
cues, without accepting unchecked files. Detector/recognizer assets and the
required recognition runtime remain mandatory. Strict CUDA
model routing remains unchanged, with no implicit neural CPU fallback.
Detection `min_face_pixels`, `min_sharpness` and `score_threshold` remain hard
recognition filters, independently of these portrait preferences.

The chosen metadata records `quality_shortfalls`, optional `quality_errors`,
actual encoding settings and any adjusted padding/ratio/downscaling. Run summaries
count `portraits_quality_limited` and `portraits_fallback`, including preserved
portraits. These diagnostics help tune detection and group matching without
hiding imperfect accepted faces. If no identity-verified image pixels can be
recovered, the existing owned image is preserved; a missing image is skipped.

### Portrait expression, mouth and eye preferences

Four optional nonnegative weights (0..100) rank visible facial cues:

| Weight | Score favored by a positive weight |
| --- | --- |
| `neutral_expression` | Weaker visible expression activity, including smile/frown, brow, cheek, mouth and jaw cues |
| `closed_mouth` | Lower MediaPipe `jawOpen`: score `1 - jawOpen` |
| `open_mouth` | Higher MediaPipe `jawOpen`: score `jawOpen` |
| `open_eyes` | Neither eye blinking: score `1 - max(eyeBlinkLeft, eyeBlinkRight)` |

All four default to **0**, preserving the existing ranking and avoiding expression
model downloads/inference. For a preference for neutral expression, closed mouth
and open eyes, add these keys to your existing weights:

```yaml
version: 2
portraits:
  weights:
    neutral_expression: 3.0
    closed_mouth: 2.0
    open_mouth: 0.0
    open_eyes: 2.0
```

These scores join the quality weighted average after identity verification.
They are soft preferences, not absolute
precedence or rejection thresholds: a sufficiently better portrait can still win
with a smile/open mouth. Prefer one mouth weight; equal positive `open_mouth` and
`closed_mouth` weights sum to a constant and cancel the mouth preference. All
ranking weights may also be zero, using general quality as the tie-breaker.
Expression-only ranking is supported, with general quality breaking equal scores
and deterministic time/ID ordering for the video's initial hints.

Two pinned ONNX exports of [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
produce a mesh and blendshape coefficients. The original Google bundle is
`face_landmarker/float16/1`; the unchanged-weight exports come from
`thedemons/mirayu-models` at `dc44090caa7ce49c81139e4fd7c22b63f81943a3`:

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| `face_landmarks_detector.onnx` | 4921002 | `de2cda3bdaab8ba37d8f1236a7091c301c1cff4dd0460f5b43c03df042c6d9c0` |
| `face_blendshapes.onnx` | 1883471 | `24ce8a9c501db3d2c6d32958d1b9d867d3b15de3de3a896cdc8d28641b30b608` |

They load lazily once per recognition run through the shared application ORT.
**`execution.gpu.device: cuda` requires both graphs to use CUDA**, and disables CPU
execution-provider fallback at node level for these graphs. An unavailable CUDA
provider, failed GPU initialization or unsupported GPU graph is an error, not a
silent CPU expression run. Explicit `cpu` and `auto` remain supported; only `auto`
permits automatic CPU fallback. Measured device/provider provenance is stored with
the expression evidence. Preprocessing and the small scoring calculations use
CPU; both neural networks run on the configured provider.

Downloads use the existing pinned Hugging Face cache, exact size/checksum checks
and atomic installation. Existing custom/invalid files are not overwritten.
The models are Apache-2.0; upstream/export attribution is in
[expression-attribution.md](expression-attribution.md).

Analysis uses a fixed square RGB crop 1.5 times the larger face-box side, aligned
to the eye line and mapped to float32 NHWC 256px RGB in 0..1. Its framing is
independent of portrait `margin`, `aspect_ratio` and JPEG quality. The mesh must
confirm face presence, and five mesh anchors must agree with the
identity-verified YuNet face; a neighboring or ambiguous face cannot provide its
expression. The mesh is mapped back to original-image pixels; the upstream 146
point subset is then passed to the 52-score blendshape model. The
`neutral_expression` score is the explicit heuristic
`1 - max(activity coefficients)` over the brow/cheek/squint/wide-eye, jaw, mouth
and nose-sneer channels listed in `expression.ACTIVITY`. Eye blink and gaze are
excluded from that activity list. It is not MediaPipe's `_neutral` output or an
emotion/probability classification. Neither `jawOpen` nor eye blink is a literal
percentage of mouth/eye opening. Mouth and expression criteria overlap, so their
weights are not independent measurements of confidence.

VTT mode measures every otherwise eligible selected observation, with no new
shortlist or candidate cap. It caches raw coefficients, derived scores and a
model/preprocessing contract and runtime provenance in the observation's existing SQLite quality
JSON. Changing weights/framing reuses valid measurements without another identity
scan or expression inference; no FFmpeg is used. Video mode measures the actual
decoded face, retaining its evidence in portrait metadata. Changed quality targets
and weights rescore compatible cached frames/JPEGs with fresh identity checks
without video decoding or repeated successful inference. Missing optional cues
on a verified cached video JPEG can be retried from those pixels without a seek;
metadata records `quality_source: cached_portrait`. Other video policy changes
may need decoding again; the existing video candidate and offset limits still apply.

The weights and measurement contract participate in portrait policy identity.
The next selected-scene run updates an application-owned Stash image if a
different winner produces different JPEG bytes, and repeated runs do not repeat
that update. Manual replacements remain protected. Missing, invalid, unmatched or
failed expression measurements lower ranking confidence and record an unavailable
cue; they do not withhold the portrait. Disabling all four
weights reranks using the original quality criteria and keeps cached expression
evidence. No face-identity namespace or SQLite schema migration is introduced.

Offline tests use synthetic coefficient fixtures. The explicit Linux/Windows
model smoke verifies both ONNX models, face association and mouth/eye output
on the checksum-pinned public-domain Grace Hopper image. These checks do not
calibrate neutral-expression accuracy or ranking weights for your VTT library.
CUDA provider routing and refusal of CPU fallback are covered by offline adapter
tests; hosted CPU jobs do not claim measurements on actual CUDA hardware.

`portraits.source.kind: video` retains the video-frame mode. Install system FFmpeg
(or configure `portraits.source.ffmpeg`); absence or unsupported/budget-limited video
is recorded on the video candidate while links remain eligible. `time_offsets`,
`max_candidates`, `timeout_seconds`, `max_video_mb` and `ffmpeg` in the portraits section are
used only by the video source. When no usable identity-verified video portrait is
found, selection tries accepted VTT observations and then retained recognition
crops. Metadata records the fallback rather than claiming a video frame/time.

`portraits.quality.min_visible_face_fraction` is a number in `[0, 1]`, default `0.8`:
prefer at least 80% estimated visible facial surface. `portraits.weights.visibility`
is a nonnegative ranking weight, default `3.0`, which favors more visible faces.
A zero target removes the extra shortfall penalty while retaining the base
visibility score. A zero weight disables portrait visibility inference and removes
its main ranking contribution. FaceOcc is still required when the independent
`face.quality.min_visible_face_fraction` gate is positive. Set that threshold to 0
as well to disable all visibility downloads/inference. Unknown YAML keys still fail validation.

The pinned MIT-licensed FaceOcc v1.0.0 SafeTensors model predicts visible face
versus background/occluder pixels. Its U-Net/ResNet18 runs on the existing
application torch/torchvision CPU or CUDA profile, with no SMP/timm dependency or
ImageNet weight download. `execution.gpu.device: cuda` stays strict; only explicit `auto`
allows CPU fallback. One lazy segmentation model is shared across selected people
and closed at the end of the invocation. Visibility failures record an unavailable
cue and warning while ranking other available quality; they never imply that a
face is uncovered and do not block its portrait.

The application reports the fraction of mask-positive pixels (`logit > 0`) within
a fixed elliptical face region derived from the detector box, excluding artificial
padding. The model input is a fixed square crop around that box, rotated upright
using the semantic eye landmarks and resized to RGB 256x256 in one resampling
pass with ImageNet normalization. This removes in-plane roll, including 90/180/270
degrees; it does not turn a profile into a frontal face. The original source-box
ellipse and real-source coverage are mapped through the same transform, so
rotation cannot remove an occluder from the denominator or count padded pixels.
Portrait `margin`, aspect ratio and JPEG
encoding do not change this denominator. This is a model estimate of visible
surface, not an exact anatomical percentage or an occluder type classifier. Hair,
beards, glasses, pose, box/alignment error and low-resolution thumbnails can affect it;
the default target is not calibrated to an operator's VTT library. The application
does not infer hidden features from landmark placement.

VTT measurements, contract/model digest, pixel counts and device are retained in
observation quality and selected portrait metadata in the face SQLite. Changing
framing, ranking weights or the visibility threshold reuses these measurements
without repeating face recognition or segmentation. A changed source/model contract
requires a fresh measurement. Video measures the actual decoded face and retains
that result in portrait metadata. Attribution and pinned inference provenance are
in [faceocc-attribution.md](faceocc-attribution.md).

On the next explicit `recognition run`, changed portrait settings re-evaluate
selected accepted people and refresh their application-owned images. The face
scan and its embeddings remain reusable when only portrait settings change.
Refresh requires the live image bytes to still match the last acknowledged
application image digest. Existing manual images and manual replacements,
including replacements served at the same URL, are preserved. New selected
scenes can improve a portrait under the same policy; a worse candidate does not
replace a better acknowledged image. Changed preferences re-evaluate the best
available choice. Identical JPEG bytes cause no remote image write; changed
policy/winner provenance can still be stored. Quality shortfalls and unavailable
sources/models never remove an owned portrait. Setting `enabled: false` removes
the application-owned image for selected accepted people, preserving manual
images/replacements. Changes to `source`, framing, quality targets or ranking are included in
the portrait policy identity. A lost update response is reconciled against the
new image digest; seeing the old image does not authorize a blind retry.

Removal uses the same intent-before-dispatch image journal, records the policy
and old image digest, and freshly verifies ownership/image bytes before clearing
Stash with `image: ""`. The acknowledgement and a fresh performer read must both
confirm absence. Removed JPEG evidence remains in SQLite; no person, scene link
or manual metadata is deleted. An unchanged disabled policy does not
repeat the removal, and re-enabling the policy can refill the portrait
from retained evidence. The summary reports `portraits_removed` when a removal
is confirmed. Lost removal responses are reconciled by observing absence; if the
old image remains, `--retry-unresolved-images` explicitly permits another attempt.

Source/decoder failures try the fallback chain before preserving the current
image. Configured aspect ratios constrain the crop even when some subject pixels
must be omitted; oversized JPEGs lower quality or dimensions while respecting
the ratio and byte budget.
There is no quality-rejection removal path.

If output says `Earlier image write is unresolved`, the previous attempt may
have reached Stash without a confirmed response. A normal rerun acknowledges it
without another write when Stash serves the submitted JPEG. If the image is still
missing or matches the image from before that attempt, explicitly allow another
attempt for the selected scenes:

```bash
python stash_ai.py recognition run --scene-id 123 --retry-unresolved-images
# Retry only one existing performer's portrait, without scene recognition:
python stash_ai.py recognition portraits --performer-id 456 --retry-unresolved-images
```

Use the same `--face-config`, `--faces-db` and `--device` overrides as the original
run, if any. This invocation-only flag retries unresolved portrait writes for
currently accepted, application-owned performers in the selected scope. It
revalidates identity and image ownership, reuses eligible cached JPEGs, and
checks the live image again immediately before dispatch. A changed/manual image
is preserved. It does not retry uncertain performer creation, linking or name
writes. No YAML key, database reset or deletion of retained images is needed.
Run history records the flag and successful `portraits_retried` count; the new
image intent retains one bounded snapshot of the preceding uncertain attempt.
A lost response to the retry remains uncertain and needs fresh reconciliation
or another explicit invocation with the flag.

The baseline missing-image
contract is its own `/performer/ID/image?...&default=true` URL, not a guessed
placeholder filename. A fresh check immediately before the mutation also protects
images supplied during extraction. Stash has no atomic image/name compare-and-set
mutation: a server-side edit between this check and dispatch remains a race.

The video shortlist alternates selected scenes and spreads cue times. At most
`max_candidates` observations times `time_offsets` are evaluated; each selected
scene decoder shares one byte/time budget across its seeks. Only selected-scene
video is fetched, even for a historically known person. FFmpeg receives an owned
loopback range URL, not Stash credentials or signed media URLs. Authenticated
same-origin requests, video byte/deadline budgets and decoder log limits apply.
Requested and actual times use scene-relative seconds. The
decoder records raw frame PTS and the source timeline origin separately, using
the same FFmpeg invocation to normalize nonzero container starts before checking
seek consistency. Missing timestamps or a changing timeline origin exclude that
video frame and allow the fallback chain. Actual presentation times, identity comparisons, quality,
exclusions, crop/margin, policy and final bounded JPEG BLOBs stay in face SQLite.
The best evaluated crop under these budgets is not a global-best-frame claim.
Five-landmark geometry estimates pose/completeness; FaceOcc estimates semantic
visibility. Failed decoding may be retried on a later explicit run.

Configure `names.file: performer_names.yml` to opt into this separate optional file:

```yaml
version: 1
stash_scope: "http://localhost:9999"
performers:
  "123": "Example Performer"
```

Use the exact configured Stash base URL (or the scope digest printed in the bounded
starter). IDs must be quoted canonical positive strings. The complete file is
validated and snapshotted once before Stash/state/model initialization. Within
the run, only mapped accepted/bound selected IDs enter the name plan; all selected
targets are resolved before any name updates. An unknown selected target prevents
the whole name batch; unrelated mapping entries are not resolved or changed.
Updates change only the name, with fresh ownership/current-name checks, typed
responses and uncertain-outcome reconciliation. Equal names do not merge IDs.
Both YAML snapshots, resolved names/preconditions and history remain in face SQLite.

Normal output includes at most 50 selected published IDs/current names as a YAML
starter, also stored in SQLite. It never overwrites your mapping file or treats
these generated anonymous placeholders as rename instructions. No additional
command, manual portrait picker or persistent image/operation sidecar is required.
Image acknowledgement recovery compares the submitted JPEG digest to Stash's
served image. A server that transforms the image is treated as an unresolved or
conflicting result and its live image is preserved, rather than silently retried.

For queues, caches and FaceOcc batching, see [portrait performance](portrait-performance.md).

# Facial recognition

The recognition package is independent of the tag model registry and configuration.
The internal layers provide configuration, verified pinned HF assets,
CPU/GPU YuNet/SFace adapters, bounded VTT extraction and checkpointed face SQLite.
The only public operation is the explicit, finite `recognition run` command.
No public scan/cluster/review/apply commands are provided.

Use the same complete application profile as tagging on Python 3.14.7:

```powershell
uv pip install -r requirements-gpu.txt -c constraints/gpu-py314.txt --index-strategy unsafe-best-match
# CPU alternative in a separate environment:
uv pip install -r requirements-cpu.txt -c constraints/cpu-py314.txt --index-strategy unsafe-best-match
```

Optional portrait expression inference uses the same ONNX Runtime CPU/GPU
distribution as recognition. No MediaPipe/TensorFlow package or graphics runtime
is added to the application profiles.

With `STASH_URL` and optional `STASH_API_KEY` set for Stash 0.31.1 or later:

```powershell
python stash_ai.py recognition run --scene-id 6306
python stash_ai.py recognition run --scene-id 6306 --scene-id 6307 --face-config facial_recognition.yml --faces-db faces_state.sqlite3
python stash_ai.py recognition run --all-scenes
```

The last command explicitly snapshots all scene IDs for this one invocation.
There is no watcher, tagging hook or background recognition process. Ordinary
tagging, merge, cleanup and `--models all` neither start recognition nor read its
YAML. Missing pinned weights are installed automatically on recognition runs.
Eligible identities become anonymous performers and receive additive scene
links. Weak identities remain recorded with reasons and do not block clear ones.

Ownership uses an exact installation/person token in performer details, never a
name-only match. Each mutation intent is committed to the face SQLite before
dispatch. Lost creation/link/merge responses are reconciled by fresh Stash reads;
unresolved outcomes are withheld without blindly retrying. Existing manual links
are preserved with Stash's `ADD` mutation. Selected-scene runs do not resume links
for other scenes. Exit 0 means the finite selection completed; exit 1 records
partial/failed work, and interruption returns 130 from the root launcher.

`facial_recognition.yml` is a versioned face-only configuration. Numeric matching
thresholds are cosine similarities, not calibrated probabilities. The initial
0.65 match/0.60 cohesion defaults prioritize withholding uncertainty; they do not
claim measured accuracy on a user's VTT library. OpenCV's example LFW threshold
is not used as a Stash-thumbnail accuracy guarantee.

All relative model, database and names paths resolve against the selected YAML
directory. Missing default YAML uses bundled defaults; an explicitly selected
missing file fails. Values are immutable during an invocation. The default
`faces_state.sqlite3` is reserved for face information and is independent of
tagger state/history. A names/portrait/threshold setting is not a raw model identity.

Recognition defaults to `models.device: cuda`, including when default YAML is absent.
Both YuNet detection and SFace/AuraFace neural recognition use the application's
ONNX Runtime CUDA provider. CPU image decoding, alignment/NMS and SQLite/identity
matching remain local host work. CPU runs explicitly set `models.device: cpu` or
use `recognition run --scene-id ID --device cpu`. Only an explicit `auto` permits
CPU fallback; the CUDA default fails clearly if unavailable or if a session
initializes only CPU. CUDA libraries are preloaded through the same policy as
ONNX taggers, and actual detector/embedding devices are recorded in face SQLite. Weights are checked against immutable revision-bound SHA256 values;
existing custom/corrupt files are never silently overwritten.

Normal core tests do not download models. Recognition CPU CI jobs install the complete application CPU profile and
load pinned detector/recognizers and FaceOcc on Linux and Windows. A public-domain
portrait with a synthetic lower-face covering tests the real visibility mask;
this is a behavior smoke, not domain accuracy calibration. GPU provider routing,
strict fallback handling and device overrides are tested without GPU hardware;
these hosted jobs do not claim actual CUDA inference. This is a runtime,
alignment and embedding-shape smoke test, not an accuracy benchmark.

## Internal media and state

Each media acquisition has one elapsed budget covering redirects, headers and
body reads, including trickling responses. The video bridge shares one budget
across its seeks and cancels owned socket reads when closed. GraphQL mutations
also use one bounded attempt: expiry before dispatch is a known failed write;
after possible dispatch it is unknown and requires fresh-state reconciliation.
No elapsed timeout automatically retries a mutation. Owned HTTP workers have
shared bounded capacity; expired results cannot authorize subsequent writes.

Each selected scene verifies the VTT/sprite content before reusing observations.
The sprite cache keeps at most two active sheets. Completed cue checkpoints
survive an interrupted inference; incomplete
scans cannot publish identities. Partial sampling and rejected/no-face outcomes
remain explicit coverage information, not proof that a person is absent.

Recognition has no byte ceiling on VTT/sprite inputs, no fixed VTT cue-count
or sprite-sheet count ceiling, and no pixel ceiling on decoded images or detector
input. Optional `execution.max_cues` still selects an evenly distributed sample; `0` processes
all cues in the selected scene. The retired `max_vtt_bytes`, `max_media_bytes`,
`max_image_pixels` and `max_sheets` execution keys have been removed from the
configuration and bundled template. Remove them from existing local YAML before running;
unknown keys are rejected rather than ignored.

Tagging and recognition share a Pillow reader that disables decompression-bomb
pixel thresholds for Stash images, including subsequent crop checks. Large-image
warnings no longer clutter output; other warnings and image/rectangle validation
remain active. Video portraits use native frame dimensions without pixel-driven
downscaling.

Scene performer associations use Stash's `BulkUpdateIds.ids` with mode `ADD`,
preserving unrelated performers. On the next explicit run, a legacy unconfirmed
link with the old malformed input and its exact recorded HTTP 400 can recover
after fresh ownership and scene reads; new intents record their input contract.
Timeouts, connection failures, other HTTP errors and unrelated uncertain writes
remain protected from blind retries. HTTP 400 GraphQL validation messages are
reported with bounded credential-redacted detail instead of only the endpoint.

One face SQLite stores model/scan provenance, detections, normalized embeddings,
bounded crop BLOBs, accepted identities, Stash ownership/bindings, candidate
portraits, mutation records, progress and run history. Its independent schema
refuses tag/history/foreign databases and hard-linked aliases. Internal backup
uses SQLite's consistent backup API; no result/history/image sidecars are needed.
Recognition schema v1 upgrades atomically to v2 to retain identity merge aliases
and their evidence. Existing observations, IDs, performer bindings and history
are preserved. An incomplete v1 database is refused before migration.

Face completeness/pose currently use measurable geometry and five-landmark
ordering/symmetry estimates. They do not prove that every feature is unoccluded.
Sharpness is measured on a normalized grayscale crop; actual face dimensions,
exposure and clipping are retained separately.

## Automatic identity policy

Matching is incremental: selected-scene vectors can reuse compatible retained
people from other scenes without fetching their media. New identities need the
configured distinct cue-time support.
Insufficient, near-threshold or ambiguous matches are withheld automatically.

The best and next coherent identity compete even when the runner-up falls just
below the match threshold. Cohesion is checked against bounded representative
samples, avoiding transitive single-link unions. This sampled check is not a
proof that every historical pair matches. Reference bytes, samples and identity
counts have explicit limits; no N-by-N distance matrix is allocated.

### Consolidating identity fragments

Each uncached matching pass also checks whether the selected evidence supports
joining existing identities in the same Stash instance and model namespace.
At least `recognition.min_support` distinct cue times in that selected scan must
match both groups at `recognition.match_threshold`, with the usual sample
cohesion check. The average of cross-group sample similarities must reach
`match_threshold`, and their minimum must reach `clustering.cohesion_threshold`.
Different assigned faces in one retained cue block a merge; selected-scene
assignments are recorded before this check so new co-occurrence also counts.
Samples are bounded by `max_samples_per_person` for each original group. While
combining several groups in a pass, all constituents remain available for the
cross-check: A-B and B-C similarity cannot override incompatible A-C evidence.
This is a sampled rule, not an identity probability or proof about every frame.

The oldest accepted recognition identity wins by SQLite `people.created`, with
lexical identity ID as a deterministic tie-break. Its UUID, ownership token and
Stash performer ID stay stable. Members and derived candidate assignments from
all affected scenes move to it in one SQLite transaction. Retired identities
remain as durable aliases, are excluded from matching/reference budgets and
cannot create another performer. Unrelated groups are not merged merely because
an older database contains them: independently timed evidence from the selected
scan is required. Supported merge decisions survive later configuration changes;
they are not automatically split again when thresholds increase.

For example, if four retained fragments mutually pass the cross-check and
three new cue times match them, the three younger identities retire into the
oldest. A query that matches two otherwise incompatible people remains ambiguous
and does not merge them. Raising or lowering thresholds reruns these decisions
using valid retained vectors; it does not require another recognition inference.

Owned published profiles are consolidated with Stash's native `performerMerge`
(`source`, `destination`, `values` from the
[Stash v0.31.1 performer schema](https://github.com/stashapp/stash/blob/v0.31.1/graphql/schema/types/performer.graphql)).
Its transactional
[repository implementation](https://github.com/stashapp/stash/blob/v0.31.1/pkg/sqlite/performer.go)
transfers scene, image, gallery and tag associations and deletes the younger
profile. Associations in unselected historical scenes move as part of that
merge; their VTT/video media are not fetched. The oldest destination keeps its
name, image and other scalar metadata. Details/notes and ownership tokens from
both profiles are retained in the destination. Unpublished local fragments do
not cause throwaway profiles to be created. Name mappings should continue to
use the surviving performer ID; mappings for retired IDs no longer apply.

Merge intent is committed before dispatch. A fresh read must confirm the
destination ownership and carried merge marker and the source's disappearance.
After a timeout/interruption, those same facts reconcile the original intent;
absence of the source alone cannot confirm success. An unresolved outcome or
changed ownership withholds the merge without a blind retry or a duplicate
profile. Subsequent selected runs resume relevant pending merges. A valid pair
can still merge when a third rival makes the selected scene ambiguous, without
adding an unproven scene link. Multi-scene metadata targets are remapped to the
survivor before names/portraits run; its existing image follows the usual manual
image and configuration ownership rules. Historical portrait bytes/metadata and
mutation intents remain in SQLite as evidence.

Run summaries report `identities_merged` (committed local consolidations),
`identities_merged_in_stash` (acknowledged profile merges) and
`identity_merges_observed` (reconciled previous acknowledgements). Merge evidence
in `identity_merges` retains supporting times, policy, sampled scores and
constituent reference fingerprints. The matching cache fingerprints only active
references; repeating an unchanged run does not repeat merges or inference.

| Setting | Effect on merging |
| --- | --- |
| `recognition.match_threshold` | Minimum mean match to each endpoint and mean cross-group similarity. |
| `recognition.min_support` | Required distinct supporting cue times in a selected scan. |
| `recognition.ambiguity_margin` | Withholds competing assignments; cannot relax merge evidence requirements. |
| `clustering.cohesion_threshold` | Minimum representative similarity, including cross-group comparisons. |
| `clustering.max_samples_per_person` | Bounds each group's retained representatives and the final survivor's lookup sample. |
| `clustering.max_people_per_lookup` | Caps active compatible identities, including provisional groups; aliases are excluded. |
| `clustering.max_reference_bytes` | Caps the bounded reference vectors; exceeding it rolls back that scene's matching and local merges. |

### Reference budget

`clustering.max_reference_bytes` caps the sum of the float32 embedding array
bytes used as references in a lookup, including provisional new groups while
matching the selected scene. Compatible retained people from the same Stash
instance and model namespace participate even when their scenes are not selected.
Each person contributes at most `max_samples_per_person` vectors (default 32).
The default limit is 134217728 bytes (128 MiB); the supported range is 1 MiB
through 1 GiB. Exceeding it fails that scene's matching instead of silently
dropping people or samples. The error includes required bytes, configured limit
and the numbers of people/groups and samples reached at the failure.

SFace uses 128 float32 components (512 bytes per vector), while AuraFace uses
512 (2048 bytes per vector). For example, with 32 samples for every person:

| Compatible people | SFace reference bytes | AuraFace reference bytes |
| --- | --- | --- |
| 1000 | 15.625 MiB | 62.5 MiB |
| 5000 | 78.125 MiB | 312.5 MiB |
| 10000 | 156.25 MiB | 625 MiB |

People with fewer retained samples use less. Raising the byte limit only permits
a larger reference set; it does not reserve memory, increase samples per person,
improve the similarity scores or speed up matching. More actual references mean
more comparisons. This budget excludes Python/SQLite overhead, image buffers,
model memory and GPU VRAM, so leave room for them in system RAM.

During `recognition run`, the matching status shows reference MiB used/limit,
percentage, run peak, people/groups, sample count and stage. Snapshots are
throttled to once a second, with immediate stage-boundary and budget-failure
updates. A per-scene line and the final summary also expose usage, including
when `--progress-style off` disables the live status. For example:

```text
Face references scene 6306: 78.12/128.00 MiB (61.0%) peak=79.00 MiB (61.7%) people=5000 samples=160000 stage=retained
```

The peak is the highest logical reference demand across matching scenes in this
run, including provisional groups that may later be withheld and the first
over-budget request if one fails. It is not process RSS or simultaneous Python
heap allocation. `loading` is a partial reference load, `matching` includes
provisional groups, `merging` checks supported consolidations using the retained
references, `retained` is the bounded persisted set after successful
matching, and `cached` means the matching decisions were reused after reference
validation. `no_observations` uses zero references because matching was skipped;
it does not mean the database contains no people. Peaks are maxima, never sums
across scenes. The last snapshot and run peak are retained in the same SQLite's
`progress` (`references` phase) and `runs.summary.references`; these metrics need
no extra files. History writes occur after the matching transaction
so reporting cannot commit partially formed candidates.

Keep 128 MiB while the reported peak has comfortable headroom. If it approaches
the limit or a scene fails for this budget, and system RAM is available, increase
it to 268435456 (256 MiB) or 536870912 (512 MiB). The independent
`max_people_per_lookup` limit still applies. Changing this setting currently
changes the grouping policy fingerprint, so the next run recomputes matching
from retained vectors without repeating neural inference for unchanged scans.

Developer-only `evaluate_pairs` accepts explicit local same/different-person
ground truth and reports false matches/misses for that supplied dataset. Synthetic
vector fixtures prove grouping rules, not real-world face-recognition accuracy.
The default policy is labeled uncalibrated for a user's domain in stored provenance.

## Automatic portraits and operator names

Candidate eligibility is rechecked when the compatible identity references
change, using retained embeddings without rescanning media. Cached VTT and
video portrait images also require current identity eligibility before a new
publication. Existing confirmed/manual associations and images are preserved.
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
shortlist or temporal/scene sampling. Quality, identity and visibility gates run
before the final weighted ranking. Only the highest-ranked usable observation is
encoded as a JPEG; an unavailable source or impossible crop tries the next ranked
observation. Other eligible candidates retain metadata, not additional JPEG BLOBs.
Existing prepared JPEGs remain reusable and rejected caches keep their bytes for
provenance. Only one scene's source/sprite cache is active at a time.
The portrait can have lower source resolution than a video frame; no upscaling
invents additional detail.

Configure framing and JPEG encoding in `facial_recognition.yml`:

```yaml
portraits:
  enabled: true
  source: vtt             # Default; use video for FFmpeg frame extraction.
  margin: 0.35            # Add 35% of face width/height on each side.
  aspect_ratio: null     # Keep padded face shape; 0.8 requests a 4:5 portrait.
  jpeg_quality: 90       # Integer 1..100.
  min_visible_face_fraction: 0.80
  weights:
    visibility: 3.0
```

The default requested VTT portrait is 1.7 times the detected face width/height,
compared with the recognition cache crop's 1.3 times. Padding is limited by the
full thumbnail boundaries. An optional width/height aspect ratio expands the
crop, shifts it inside the source, and reduces padding if necessary; an
impossible ratio that would cut the face is skipped. Actual crop coordinates,
source rectangle, observation/cue, input fingerprint, quality, identity scores
and source type remain in face SQLite. The cue time describes the VTT cue, not
a claimed decoder-measured video timestamp. Existing portrait quality gates and
ranking weights apply to both modes and can also be changed in YAML.

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
portraits:
  weights:
    neutral_expression: 3.0
    closed_mouth: 2.0
    open_mouth: 0.0
    open_eyes: 2.0
```

These scores join the existing weighted average after identity, size, sharpness,
pose, exposure and visibility gates. They are soft preferences, not absolute
precedence or rejection thresholds: a sufficiently better portrait can still win
with a smile/open mouth. Prefer one mouth weight; equal positive `open_mouth` and
`closed_mouth` weights sum to a constant and cancel the mouth preference. All
ranking weights together must have a positive sum. Expression-only ranking is
supported, with deterministic time/ID ordering for the video's initial hints.

Two pinned ONNX exports of [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
produce a mesh and blendshape coefficients. The original Google bundle is
`face_landmarker/float16/1`; the unchanged-weight exports come from
`thedemons/mirayu-models` at `dc44090caa7ce49c81139e4fd7c22b63f81943a3`:

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| `face_landmarks_detector.onnx` | 4921002 | `de2cda3bdaab8ba37d8f1236a7091c301c1cff4dd0460f5b43c03df042c6d9c0` |
| `face_blendshapes.onnx` | 1883471 | `24ce8a9c501db3d2c6d32958d1b9d867d3b15de3de3a896cdc8d28641b30b608` |

They load lazily once per recognition run through the shared application ORT.
**`models.device: cuda` requires both graphs to use CUDA**, and disables CPU
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
decoded face, retaining its evidence in portrait metadata. When only weights
change, compatible cached frames/JPEGs are rescored with fresh identity checks
without video decoding or neural inference. Other video policy changes and frames
without valid measurements may need decoding again; the existing video candidate
and offset limits still apply.

The weights and measurement contract participate in portrait policy identity.
The next selected-scene run updates an application-owned Stash image if a
different winner produces different JPEG bytes, and repeated runs do not repeat
that update. Manual replacements remain protected. Missing, invalid, unmatched or
failed expression measurements withhold the portrait update and preserve the old
owned image rather than deleting it as a policy rejection. Disabling all four
weights reranks using the original quality criteria and keeps cached expression
evidence. No face-identity namespace or SQLite schema migration is introduced.

Offline tests use synthetic coefficient fixtures. The explicit Linux/Windows
model smoke verifies both ONNX models, face association and mouth/eye output
on the checksum-pinned public-domain Grace Hopper image. These checks do not
calibrate neutral-expression accuracy or ranking weights for your VTT library.
CUDA provider routing and refusal of CPU fallback are covered by offline adapter
tests; hosted CPU jobs do not claim measurements on actual CUDA hardware.

`portraits.source: video` retains the video-frame mode. Install system FFmpeg
(or configure `portraits.ffmpeg`); absence or unsupported/budget-limited video
is recorded as an image skip while links remain eligible. `time_offsets`,
`max_candidates`, `timeout_seconds`, `max_video_bytes` and `ffmpeg` in the portraits section are
used only by the video source. A source failure never silently falls back to the
other mode.

`portraits.min_visible_face_fraction` is a number in `[0, 1]`, default `0.8`:
require at least 80% estimated visible facial surface. `portraits.weights.visibility`
is a nonnegative ranking weight, default `3.0`, which favors more visible faces
among candidates that meet the gates. A zero threshold removes the visibility
gate; a zero weight removes its ranking contribution. Set both to zero to disable
visibility model downloads/inference entirely. Unknown YAML keys still fail validation.

The pinned MIT-licensed FaceOcc v1.0.0 SafeTensors model predicts visible face
versus background/occluder pixels. Its U-Net/ResNet18 runs on the existing
application torch/torchvision CPU or CUDA profile, with no SMP/timm dependency or
ImageNet weight download. `models.device: cuda` stays strict; only explicit `auto`
allows CPU fallback. One lazy segmentation model is shared across selected people
and closed at the end of the invocation. Enabled visibility failures withhold the
portrait and report a metadata failure; they never imply that a face is uncovered.

The application reports the fraction of mask-positive pixels (`logit > 0`) within
a fixed elliptical face region derived from the detector box, excluding artificial
padding. The model input is a fixed square crop around that box, resized to RGB
256x256 with ImageNet normalization. Portrait `margin`, aspect ratio and JPEG
encoding do not change this denominator. This is a model estimate of visible
surface, not an exact anatomical percentage or an occluder type classifier. Hair,
beards, glasses, pose, box/alignment error and low-resolution thumbnails can affect it;
the default gate is not calibrated to an operator's VTT library. The application
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
including replacements served at the same URL, are preserved. Unchanged policy
does not cause an image write, and a changed policy yielding identical JPEG bytes
only updates stored provenance. If a complete evaluation finds no replacement
meeting the new requirements, the old application-owned image is removed from
Stash. Setting `enabled: false` also removes the application-owned image for
selected accepted people. Manual images/replacements are preserved in both
cases. Changes to `source`, framing, quality filters or ranking are included in
the portrait policy identity. A lost update response is reconciled against the
new image digest; seeing the old image does not authorize a blind retry.

Removal uses the same intent-before-dispatch image journal, records the policy
and old image digest, and freshly verifies ownership/image bytes before clearing
Stash with `image: ""`. The acknowledgement and a fresh performer read must both
confirm absence. Removed JPEG evidence remains in SQLite; no person, scene link
or manual metadata is deleted. An unchanged rejecting/disabled policy does not
repeat the removal, and relaxing/re-enabling the policy can refill the portrait
from retained evidence. The summary reports `portraits_removed` when a removal
is confirmed. Lost removal responses are reconciled by observing absence; if the
old image remains, `--retry-unresolved-images` explicitly permits another attempt.

A media/source/decoder failure is an incomplete evaluation, not proof that no
portrait qualifies. Such failures retain the current image and report a metadata
failure rather than deleting it. Visibility model failures also retain it. An
impossible configured aspect ratio or crop byte limit is a framing rejection,
so a complete evaluation with no alternative removes the old owned image.

If output says `Earlier image write is unresolved`, the previous attempt may
have reached Stash without a confirmed response. A normal rerun acknowledges it
without another write when Stash serves the submitted JPEG. If the image is still
missing or matches the image from before that attempt, explicitly allow another
attempt for the selected scenes:

```bash
python stash_ai.py recognition run --scene-id 123 --retry-unresolved-images
```

Use the same `--face-config`, `--faces-db` and `--device` overrides as the original
run, if any. This invocation-only flag retries unresolved portrait writes for
currently accepted, application-owned performers in the selected scenes. It
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
seek consistency. Missing timestamps or a changing timeline origin withhold the
portrait. Actual presentation times, identity comparisons, quality,
exclusions, crop/margin, policy and final bounded JPEG BLOBs stay in face SQLite.
The best evaluated crop under these budgets is not a global-best-frame claim.
Visibility remains the documented five-landmark estimate, not semantic occlusion
detection. Failed decoding may be retried on a later explicit run.

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

## Optional AuraFace recognizer

SFace remains the default. Both CPU/GPU application profiles already include
AuraFace's runtime: set `models.recognizer: auraface` in `facial_recognition.yml`
and invoke the same `recognition run --scene-id ID`; no additional requirements
file or ORT package is installed. Exactly one ORT variant is required. CPU is
runtime-tested on Python 3.14.7 on Linux/Windows. The default CUDA device is strict;
CUDA hardware execution is not covered by hosted CPU CI. Actual detector and
embedding providers/runtime/preprocessing provenance are stored in SQLite.

Only the pinned publisher's `glintr100.onnx` recognition weight plus YuNet are
downloaded; no InsightFace dependency, default model pack, landmarks, age/gender
weights or external recognition service is used. The pinned model card and
`LICENSE.md` declare Apache-2.0 for AuraFace; library licenses are separate.

The upstream [ArcFace adapter](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/arcface_onnx.py)
and [five-point alignment](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py)
define a least-squares similarity transform to the 112px ArcFace reference,
BGR-to-RGB conversion, `(pixel-127.5)/127.5` float32 NCHW and normalized embeddings.
The pinned AuraFace graph starts with Conv/PRelu (no built-in initial Sub/Mul),
accepts `data` with dynamic batch and `3x112x112`, and returns `1333` with `1x512`.
Size/hash, runtime I/O, known affine alignment and exact RGB normalization are
checked. This alignment/preprocessing has its own namespace; the SFace namespace
and existing evidence are unchanged.

Switching models computes selected-scene observations in the new namespace;
vectors and ownership bindings are never compared/merged across recognizers.
If this Stash instance already has published identities in another namespace,
new unbound identities are withheld with `incompatible_published_namespace`,
while retaining their analysis. Existing compatible bindings can still be reused.
There is no automatic migration, duplicate-profile creation or hidden review
command. Use the developer pilot for comparison before choosing a model for a new
installation; future cross-model binding/migration is outside this release.

### Measured runtime and pilot scope

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
The helper runs the real grouping policy in disposable in-memory SQLite and
reports embedding/accepted coverage, withheld/ambiguous observations, pair false
merges and misses (including withheld), pair threshold errors, time and asset size.
It makes no Stash/media calls or changes to published membership. Labels,
configuration, derived results and scope stay in the existing face SQLite's run
progress/history; no evaluation state sidecars or manual normal-use workflow are
needed. There is no claimed real-library accuracy until such labeled data exists.


## Shared CPU/GPU runtime and cache compatibility

Pinned files, normalized vector spaces and the SFace/AuraFace namespace hashes
remain unchanged when selecting a device. CUDA uses the original OpenCV SFace
alignCrop and raw RGB float32 pixels; normalization remains inside the SFace graph.
YuNet follows OpenCV's BGR/divisor-32 padding, stride decoding, integer-box NMS and
bounded top-K contract. ONNX input/output declarations are adapted in memory for
YuNet spatial dimensions, and legacy initializer-inputs are frozen to their
original values. No neural nodes or initializer values are changed, no transformed
weights are stored, and original assets are checksum-verified first. Transformation
digests and active providers are recorded as runtime provenance. Native CPU ORT
smoke checks validate non-square YuNet inputs and SFace embedding agreement with
OpenCV; CPU/GPU floating-point differences are expected. Device selection alone
does not request a new namespace, model switch or duplicate-profile publication.

# Facial recognition

Set `portraits.anchor_only: true` to require the actual portrait face to pass
the current clustering anchor pose and visibility rules, in addition to its
identity check. The default is false. `portraits.max_pose_deviation` and portrait
quality weights remain ranking preferences; this opt-in gate uses
`clustering.max_anchor_pose_deviation`, `min_anchor_visible_face_fraction` and
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

`clustering.corroborated_view_enrichment` (default true) can extend the frozen
historical query bank with a `corroborated_view`. Its accepted attachment cohort
needs at least `max(view_min_support, min_merge_support)` independent cues from
two distinct scenes. Every query must independently match two strict-rooted
views whose support sets are disjoint, pass all mixed rivals/co-presence checks,
and remain inside an immutable, non-transitive appearance radius. One parent
with overlapping prototype aliases cannot supply two confirmations.

Promotion is derived from latest complete, currently eligible durable memberships.
The proof records both parents, original strict witnesses, bounded query witnesses,
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
medoids are checked against `clustering.max_reference_bytes`; insufficient memory
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

`clustering.min_anchor_visible_face_fraction` and
`clustering.min_auxiliary_visible_face_fraction` optionally qualify retained
FaceOcc measurements for their respective roles. Anchor admission requires both
the anchor pose and visibility limits; a frontal but occluded face may instead
query as auxiliary. Faces below the auxiliary limit remain explicitly withheld.
The auxiliary limit must not exceed the anchor limit. Both explicit limits must
be at least `detection.min_visible_face_fraction`, the hard scan floor: lower
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

## SCRFD five-landmark detector choices

The default detector for new configurations is **SCRFD-10G KPS**. This release
also supports SCRFD-2.5G KPS and the explicitly selected legacy YuNet.
Switching detectors changes the recognition namespace, prevents reuse of
old scan/embedding evidence for that detector, and triggers a new scan;
it does **not** delete historical embeddings or identities.

```yaml
models:
  detector: scrfd_10g       # default; alternatives: scrfd_2_5g, yunet
  recognizer: lvface_b     # or adaface_ir50, auraface, sface
  device: cuda
```

SCRFD runs in ONNX Runtime on CPU or CUDA with a 640 × 640 letterboxed,
BGR-to-RGB detector input normalized by `(pixel-127.5)/128`. The
nine-output ONNX models use strides 8/16/32, two anchors per location,
five landmarks and per-image NMS. Bounding boxes and landmarks are
converted back into original VTT coordinates; recognizers still receive
the unchanged 112 × 112 ArcFace aligned input. `10G` and `2.5G` are
compute tiers, **not** gigabytes of downloaded data.

The SCRFD artifacts are pinned to Hugging Face commit
`d168e9d89c1687119307f528cbc67eb257cf0021`; 10G weights are
16,923,827 bytes, and 2.5G weights are 3,291,737 bytes. Both have strict
size/SHA-256 integrity checks; only the selected weights are downloaded.

**Licensing:** the HF wrapper advertises Apache-2.0, whereas upstream
InsightFace pretrained model weights have noncommercial research-use
restrictions. Confirm licensing for your intended usage before a
commercial deployment; wrapper metadata does not supersede upstream terms.

The `recognition audit-alignment` command now uses the detector named in
`models.detector`. To compare two variants, run the audit with separate
YAML configurations and **different new output directories**. Compare
actual landmark positions and 112px aligned crops on same-person and
different-person cues, not only face detection benchmark scores.

## Read-only five-landmark alignment audit

Use the opt-in `recognition audit-alignment` action to inspect detector landmarks
and the shared ArcFace 112 × 112 crop for the detector selected in
`models.detector` (YuNet or SCRFD). It never loads face recognizer weights,
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
are eligible. `execution.max_cues` still limits source cue selection, while
`detection.max_faces_per_frame` remains a detector-level limit.

The audit applies the same hard detection gates as `recognition run`:
detector score, minimum source face pixels, `detection.min_sharpness`,
alignment validity, complete-face and pose restrictions. When
`detection.min_visible_face_fraction > 0`, it additionally loads **FaceOcc**
and checks the visibility threshold, just as recognition scanning does.
With visibility disabled (`0`), no FaceOcc asset is loaded. Rejected faces
are **counted with their rejection reason but not exported as PNGs**.
The manifest (format 2) includes `detection_rejections`, counters for
detected/rejected/passing faces, exact applied `detection_policy`, saved
face quality including optional visibility, and each image's geometry.
This audit validates detection acceptance, **not** embedding, clustering,
or the accuracy of identity matching.

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

## Orphan cleanup and switching recognition models

`recognition run` now permits other recognizer namespaces to have published
profiles. For **successfully rebuilt, selected scenes only**, publication attaches
the current recognizer's identities and detaches verified recognition-owned
performer links from *other* model namespaces. Old performer profiles, portraits,
local face evidence and links on unselected/failed scenes are retained. Existing
manual (non-recognition-owned) performers remain attached. Unlike ordinary
same-model stale-link cleanup, even a manually added association to an owned
**other-model** performer is detached for selected scenes. Ambiguous ownership or
uncertain unlink results fail the selected scene's publication rather than blindly
removing unrelated links. `recognition portraits` does not change scene links.

To inventory Stash profiles carrying recognition ownership tokens that have
**no matching identity in the selected face database**:

```powershell
python stash_ai.py recognition cleanup --faces-db faces_state.sqlite3
```

The default is read-only; an explicit apply deletes the verified orphan
**performer profiles** (which also removes their Stash associations):

```powershell
python stash_ai.py recognition cleanup --faces-db faces_state.sqlite3 --apply
```

This separate command never loads recognizer models, never scans scenes and
never runs implicitly on `recognition run`. The existing SQLite file is required:
a missing path does **not** create an empty database that might make every
profile appear orphaned. The inventory includes recognizer-owned profiles from
any installation represented by a valid recognition ownership token on this
Stash instance. Use the complete database for that instance: another independent
recognition installation will otherwise look orphaned. Profiles with manual or
unrecognized details, or with *any* identity token / performer binding still in
that database, are protected. Each deletion is checked against fresh Stash data
and recorded as a durable operation; uncertain dispatch is never blindly retried.
The dry run and apply record their counts in the face database's run history.

## Alternative ONNX face recognizers (research weights)

The `models.recognizer` setting supports `sface`, `auraface`,
`lvface_b` (ByteDance LVFace-B Glint360K, 512D, ~456 MB) and
`adaface_ir50` (AdaFace IR50 MS1MV2, 512D, ~174 MB). All share the
YuNet detector and use ONNX Runtime with the existing `models.device`
CPU/CUDA policy. No extra PyTorch/InsightFace inference dependencies.

```yaml
models:
  detector: yunet
  recognizer: lvface_b      # or adaface_ir50
  device: cuda
  download_missing: true
```

These models are fetched **only when explicitly selected**, from immutable
Hugging Face revisions and checked by SHA-256 plus exact byte count. Both
use an ArcFace 112×112 five-landmark similarity-aligned crop:
LVFace-B uses **RGB** and AdaFace IR50 uses **BGR**; each normalizes
`(pixel - 127.5) / 127.5` and returns L2-normalized 512D embeddings.
The separate embedding namespace prevents mixing any embeddings or identity
evidence between the models, even when both produce 512D vectors. Switching
models triggers a new scan; cached embeddings cannot be converted.

**Weights licensing matters:** LVFace code is MIT, but the upstream model
README explicitly restricts its **pretrained weights** to *non-commercial
research*. The independent AdaFace ONNX exporter notes that its MS1MV2
checkpoint carries dataset/research-use restrictions distinct from the
MIT-licensed AdaFace code. Neither source grants unambiguous permission
for commercial production. Check permissions before using either outside
research, and do not redistribute weights in this repository.

**Calibration:** the example recognition/clustering cosine thresholds
are *not* validated for LVFace or AdaFace. Compare labeled same-person
and different-person samples with the existing internal offline pilot
and `recognition report` diagnostics, then tune independently. Do not carry AuraFace thresholds over as a
measured recommendation. Larger weights and new recognizers do not
automatically recover detail lost in small VTT thumbnails.

Source: https://huggingface.co/bytedance-research/LVFace
and https://huggingface.co/globalnebula/adaface-ir50-ms1mv2-onnx


The recognition package is independent of the tag model registry and configuration.
The internal layers provide configuration, verified pinned HF assets,
CPU/GPU YuNet/SFace adapters, bounded VTT extraction and checkpointed face SQLite.
The public operations are the explicit, finite `recognition run` command and
`recognition portraits` for refreshing existing identities' portraits.
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
action. Normal `recognition run` still requires `--scene-id` or `--all-scenes`.
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

Recognition uses the same console renderer as tagging. Terminals show a scene
bar and a second bar for VTT frames in the active scene, with elapsed time,
throughput and estimated remaining time. Publication has its own scene bar;
names and portraits have a performer bar. ETAs describe the current stage,
not the whole invocation, and remain unavailable until work has completed.
The frame bar starts at the saved checkpoint cursor; cached scenes show all
frames reused without treating them as newly inferred work. Frame progress
updates independently of the configured SQLite checkpoint interval.

Identity rebuild has separate progress for publication reconciliation, copying
selected evidence, matching scenes and their face observations, binding historical
identities, saving memberships/candidates, and collecting results. A large scene
therefore advances its observation counter before the scene finishes. Cached
rebuilds show validation work instead of claiming that matching ran again.
The apply bar counts one transaction and completes only after its commit; inner
record counters describe preparation inside that transaction.

Scene selection, model asset verification, name preflight/starter reads and
portrait candidate checks also expose work counters. VTT portraits count ranked
observations; video portraits count candidate frame checks. Known totals show
stage throughput and ETA after completed work is observed. Sources without a
known total, such as paginated all-scene discovery, remain indeterminate and show
completed units and elapsed time. Nested detail bars are terminal-only; redirected
output keeps bounded status heartbeats and major-stage milestones.

Redirected output keeps phase milestones and infrequent status heartbeats,
including scene/frame counts and stage ETAs, without per-scene frame logs.
`--progress-style verbose` adds per-scene quality/reference details and the
performer names YAML starter. The starter remains recorded in face SQLite
history in every style. `--progress-style off` keeps errors and the final
summary, including aggregate rejection reasons and reference usage. Repeated
scan/publication errors are grouped; failed scenes still count as inspected
work and are reported separately from successful scans and reused evidence.

There is no watcher, tagging hook or background recognition process. Ordinary
tagging, merge, cleanup and `--models all` neither start recognition nor read its
YAML. Missing pinned weights are installed automatically on recognition runs.
Eligible identities become anonymous performers and receive additive scene
links. Weak identities remain recorded with reasons and do not block clear ones.

Ownership uses an exact installation/person token in performer details, never a
name-only match. Each mutation intent is committed to the face SQLite before
dispatch. Lost creation/link/unlink and legacy merge responses are reconciled by fresh Stash reads;
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
Stored source URLs omit username/password userinfo, queries and fragments.
Recognition diagnostic URLs use the same sanitized source identity; authentication
still uses the original request/header path. Selected cached metadata and copied
retry-error snapshots are sanitized without reloading models or rewriting images.

Each selected scene verifies the VTT/sprite content before reusing observations.
The sprite cache keeps at most two active sheets. Completed cue checkpoints
survive an interrupted inference; incomplete
scans cannot publish identities. Partial sampling and rejected/no-face outcomes
remain explicit coverage information, not proof that a person is absent.

The detector and recognizer are initialized lazily once per invocation and shared
by selected-scene scans and video portrait inference. A fully cached run does not
load these models. Scene cleanup releases its borrowed backend; the invocation
closes the actual models on completion or interruption. An inference failure
discards the failed backend so a later scene can initialize a fresh instance.

`execution.prefetch_frames` defaults to 4 ready thumbnail crops. A CPU producer
decodes/crops images ahead of inference, preserves cue order and owns independent
copies until consumed. The queue retains at most that many ready crops, plus the
producer and consumer's in-flight frames; `0` uses synchronous frame preparation.
`execution.prefetch_scenes` defaults to `true` and verifies/downloads one next
selected scene while the current scene scans. Each input owns isolated read-only
HTTP sessions; only the current and one next input are retained. Set it to `false`
to disable scene lookahead independently of frame preparation.

Both options leave model/scan identities unchanged, so existing observations are
reused. Producers never run models, write SQLite or publish Stash mutations.
Queued frames are speculative until inference completes: failures/interruption
save only completed cues, cancel media reads and close queued images/readers.

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

Scene performer associations use Stash's `BulkUpdateIds.ids` with modes `ADD`/`REMOVE`,
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
The source connection must have no active transaction before backup; callers
must explicitly commit or roll back first. Backups publish a complete staged
file exclusively and preserve any existing destination. A setup failure removes
the owned staging file; a process crash can leave a hidden staging file.

Opening state validates column defaults, required constraints and the installation
identity before migration or interrupted-run recovery. Unknown triggers and views
are refused. Close the SQLite owner on its creating thread: a rejected close or
a failed close that leaves SQLite live retains the writer lock.
Recognition schemas v1, v2 and v3 upgrade atomically to v4, retaining merge
aliases/evidence and cross-scene support and adding original sample ownership for
reversible identity binding. Existing observations, IDs, performer bindings and history
are preserved. An incomplete v1 database is refused before migration.
After acquiring this database's exclusive writer lock, a new owner marks prior
`running` recognition histories interrupted regardless of instance scope or
selected scene order. Their saved summaries/progress and pending or unknown
Stash intents remain intact. Runs started by the current owner are left active;
another writer cannot acquire the database while that owner holds it.

Face completeness/pose currently use measurable geometry and five-landmark
ordering/symmetry estimates. They do not prove that every feature is unoccluded.
Sharpness is measured on a normalized grayscale crop; actual face dimensions,
exposure and clipping are retained separately.

## Recognition observation quality

Detection quality gates run before embedding and grouping. They keep unreliable
profile/covered faces from seeding identity fragments without lowering matching,
ambiguity, cohesion or reflection-merge thresholds. Rejected observations retain
geometry, measured quality and a reason in SQLite, but no embedding/crop; they
cannot become reference samples, supply distinct-time support or support merges.
The run prints per-scene rejection counts, saves the reason breakdown in detection
progress, and includes `observations_rejected` in its summary, including reused scans.

| `detection` setting | Default | Meaning |
| --- | ---: | --- |
| `min_face_pixels` | 32 | Minimum shorter face-box side in source pixels |
| `min_sharpness` | 10 | Minimum Laplacian variance on the normalized grayscale crop; 0 disables |
| `max_pose_deviation` | 0.50 | Maximum five-landmark pose estimate in `[0, 1]`; `null` disables this gate |
| `min_visible_face_fraction` | 0.80 | Minimum FaceOcc visible-face estimate in `[0, 1]`; 0 disables recognition segmentation |
| `require_complete_face` | true | Reject face boxes extending beyond the source thumbnail |

The pose gate also requires valid, ordered landmarks inside the face box. Its
dimensionless score combines nose/eye/mouth geometry; it is not an angle in degrees
or a calibrated head-pose measurement. Completeness checks image boundaries, not
semantic occlusion. Visibility uses the same fixed FaceOcc crop, ellipse and model
contract as portraits. It is an estimate, not an anatomical percentage; hair,
beards, glasses, pose and small thumbnails can affect it. These starting defaults
are not calibrated on the operator's library. Inspect rejected examples before
tightening them further; filtering can miss people seen only in difficult poses.

Recognition visibility is independent of `portraits.enabled` and
`portraits.weights.visibility`. When its threshold is positive, FaceOcc becomes
a required recognition asset. Missing/invalid assets or failed/invalid measurements
fail the scene without publishing its partial scan; completed cue checkpoints
remain resumable on a later explicit run. An unavailable measurement never counts
as an uncovered face. Cheap size/sharpness/geometry gates run before segmentation.
One lazy visibility model is shared across selected scenes and portraits; accepted
VTT observations retain the raw measurement so portrait ranking does not repeat it.
If `require_complete_face: false` permits a partial face whose visibility region
extends outside the real source, that observation is rejected as
`face_visibility_unmeasurable`, without an embedding/crop. Other valid cues continue.
This local geometry rejection does not turn a broken or unavailable model into a
successful scan.

Fresh retained face JPEGs also obey `storage.max_crop_bytes` (minimum 1024 bytes).
Encoding starts at quality 90, then reduces JPEG quality and, if needed, dimensions
without upscaling. `crop_bounds` retains source coordinates; `crop_encoding`
records the actual JPEG quality, pixel size and downscaling. This applies even
when portrait publication is disabled; normal-budget JPEG bytes stay unchanged.
When a retained crop supplies a fallback portrait, its face box and landmarks
are mapped to the recorded encoded dimensions. Older full-size crops retain
their original coordinate contract.

Changing detection gates or the enabled visibility model contract starts a new
selected-scene scan, while preserving the recognizer namespace and existing
performer bindings. Unchanged completed scans reuse their decisions. Portrait-only
preferences still do not invalidate recognition scans, and imperfect portraits
remain eligible among accepted observations. Earlier raw scans and histories are
retained. The identity replay described below
replaces selected-scene memberships, so superseded/rejected selected samples no
longer remain in the active reference set. Published profiles are preserved for
stable labels and future splits; stale managed links are reconciled separately.

To explicitly restore the earlier size/sharpness-only observation policy:

```yaml
detection:
  max_pose_deviation: null
  min_visible_face_fraction: 0
  require_complete_face: false
```

## Offline identity face export

Export **every current member observation** of one retained identity, including
auxiliary/view attachments and faces beyond `clustering.max_samples_per_person`.
Use the internal identity ID from `recognition report`, or the linked numeric
Stash performer ID:

```sh
python stash_ai.py recognition report-identity --performer-id 123 --output-dir identity-123
python stash_ai.py recognition report-identity --identity-id <identity-uuid> --face-config facial_recognition.yml --faces-db faces_state.sqlite3 --output-dir identity-faces
```

The selected YAML determines the recognizer namespace. If a performer ID is
ambiguous across Stash instances, add `--scope <scope>` or select an internal
`--identity-id`. A retired identity/performer resolves through its recorded merge
chain to the surviving identity; the JSON preserves both requested and resolved
IDs and the merge evidence. Scope/model boundaries are validated before export.

The output directory must be new. It contains PNG files such as
`scene-6306_cue-000080_anchor-same-scene_face-000001.png` and a `manifest.json`.
Cue indices are **0-based**, matching `audit-alignment --cue-index`; the final
face ordinal makes names unique even for several faces in the same cue or
multiple retained scans. Categories include `anchor`, `anchor-same-scene`,
`anchor-view`, `anchor-cross-scene`, `auxiliary`, `auxiliary-view`, and `unknown`.
They describe the recorded assignment path. A separate JSON field describes
anchor eligibility and reference selection under the **current** configuration;
neither category nor current reference selection reconstructs the historical
reference set used to make the assignment.

Each manifest sample links its image to the scene, scan, cue, timestamp, geometry,
quality measurements and recorded candidate decisions. Decisions expose their
status, policy hash, reason, cosine scores, representative limit and any retained
same-scene, cross-scene, view-prototype or merge proof. Current-policy evidence is
distinguished from historical policies; missing accepted evidence remains
`membership_only` with an unknown role. Scores are not probabilities. Missing
cohesion values, runner-up identity IDs and overwritten decisions are not invented.
Original identity labels and recorded merge evidence are included for context.

This command is offline and reads a single consistent SQLite snapshot without
the recognition writer lock. It never fetches scenes, loads weights, reruns
detection/matching, or changes SQLite/Stash. Every linked observation is included,
even if it falls below today's detection gates: filtering would hide part of the
identity being audited. There is no export sample cap. PNGs preserve the decoded
pixels of retained recognition crops (normally JPEG), including any compression
or downscaling already applied during scanning; they are not newly extracted
original thumbnails or aligned model inputs. Missing/unreadable crops remain
explicit manifest entries. The summary counts them and the command returns exit
code **1** for an incomplete image export (**0** when all members were exported).
Filesystem write failures abort instead of being reported as missing crops.

Images and JSON are sensitive local biometric artifacts. On POSIX the new output
directory has mode `0700` and its files `0600`; raw embeddings, installation
tokens, source URLs and complete run configurations are not exported.

## Two-identity merge parameter report

Compare two identities by their numeric Stash performer IDs, including identities
in different scenes:

```powershell
python stash_ai.py recognition report-merge --performer-id 123 --performer-id 456 --face-config facial_recognition.yml --faces-db faces_state.sqlite3 --output merge-report.json
# Omit --output to print JSON to stdout.
# Internal identity UUIDs remain available, including identities without a Stash performer:
python stash_ai.py recognition report-merge --identity-id <left-uuid> --identity-id <right-uuid> --output merge-report.json
```

Provide exactly two `--performer-id` values or exactly two `--identity-id` values;
the selectors cannot be mixed. Performer IDs are resolved from the retained local
face database, without contacting Stash. If a performer ID occurs in multiple Stash
scopes, add `--scope <scope>` or use internal identity IDs. The selected YAML
determines the recognizer namespace. Missing identities in that namespace/scope
are errors; IDs never fall back to another model or instance.

The report includes requested and resolved identity/performer IDs, including when
both inputs already resolve to one identity. The two IDs must share the recognizer
namespace and Stash scope. Predecessor IDs
follow recorded merge aliases; two aliases of one person return `already_merged`.
The command opens one read-only SQLite snapshot and uses retained embeddings.
It does not load models, fetch media, contact Stash, create run history or change
memberships. The output path must be new.

`recommendations` contains the parameter name, current and proposed values,
direction of change, and a copyable `yaml` / structured `yaml_patch`. Each proposal
has passed the production frozen-component planner, including all compatible
strict rivals, co-presence, complete-component checks and ambiguity margins.
`merged_identity_ids` lists the whole component; `additional_identity_ids` makes
any effect on a third identity explicit. Proposals with fewer additional identities
and fewer changed parameters sort first. These are **global settings**, not rules
limited to this pair.

The strict-core route checks `recognition.match_threshold`,
`clustering.cohesion_threshold`, `recognition.ambiguity_margin` and the co-presence
threshold. The confirmed-view route also checks the view thresholds and margin,
independent merge support and the minimum cues needed to form views. Joint patches
respect YAML constraints and round threshold bounds downward before verification.
The report tries both relaxing failed gates and tightening gates to isolate a pair;
changed view gates rebuild the view catalogue before checking the proposal.

Both routes require `clustering.anchor_view_matching` and an anchor pose/visibility
restriction. Enabling the feature is proposed when its prerequisite already exists;
the report does not invent an anchor restriction or simulate ordinary chronological
nomination merges. `paths` records current blockers, measured scores, competing
identities, attempted patches and their verification. `identities` shows member,
strict-authority and anchor-quality counts plus the actual bounded references.
Missing references and legally unattainable thresholds are explicit; attachment-only,
superseded or incompatible evidence is never promoted into strict proof.

`eligible_under_current_parameters` means this frozen catalogue already permits
consolidation. `verified_parameter_changes` means at least one tested YAML patch
permits it; `no_verified_proposal` means the bounded search found none. It is not an
exhaustive proof that no possible configuration could work. At most four measured
view pairs and two threshold strategies per route are searched.

Verification applies to the **current frozen references**, not an entire cold
rebuild: replay ordering, identity births, changing references, attachment decisions
and final label reconciliation can change the result. Apply a selected patch and
rebuild the retained scenes of both identities together; `next_action.scene_ids`
lists them. Portrait refresh alone cannot merge identities. The report also
preserves configured person/reference memory budgets rather than silently sampling
away competing identities.

## Offline scene fragmentation report

To inspect suspected splits between recognized identities without Stash, model loading,
GPU inference, or an identity rebuild, run:

```powershell
python stash_ai.py recognition report --scene-id 6306 --output fragmentation-6306.json
# Or print JSON to standard output:
python stash_ai.py recognition report --scene-id 6306
# Widen the similarity range or include more ranked pairs:
python stash_ai.py recognition report --scene-id 6306 --min-mean 0.35 --limit 50
```

The report uses `--face-config` and `--faces-db` to select the same model and
threshold policy as the completed recognition run; if the latest completed scan
does not contain matching-policy candidates, it fails instead of silently mixing
in stale results. In rare multi-Stash databases, `--scope` disambiguates scene
IDs. `--output` must name a **new** JSON file (no overwrite). Without it, JSON is
written to stdout. The database is opened in SQLite read-only mode using a
consistent snapshot; the reporter never acquires the recognition writer lock,
creates run history, downloads models, or calls Stash.

The JSON contains selected-scene identities and performer IDs, counts of accepted,
withheld and rejected observations, candidate decision reasons, and ranked
potential-fragment pairs. The diagnostic also includes rejection reason counts
and up to 50 timestamped withheld observations with their recorded best-two
scores/reasons. Each pair includes cross-group mean/minimum cosine
scores over the *currently retained bounded representatives*, the observation
IDs and VTT timestamps for the least similar pair, and the direct numerical
blockers for `recognition.match_threshold`,
`clustering.cohesion_threshold` and observed co-presence. The report also
rechecks any *recorded* merge nominations against current representatives and
reports how many independent scene/time cues still qualify. Missing nominations
do **not** prove that evidence is missing: there is no complete persisted log
of every rejected or never-proposed merge.

Additional read-only diagnostics include:

- `observation_quality`: retained versus rejected counts by reason, with
  measured face pixels, pose deviation, sharpness and visible fraction (min,
  median, p90, max and missing counts). `detection_gates` shows the active
  size/pose/sharpness/visibility thresholds; `rejected_examples` provides up to
  eight cue IDs/times and measured quality values per rejection reason without
  images. These are recorded measurements, not predictions of how a relaxed
  filter would perform.
- `identities[].representatives`: available/selected sample counts and truncation,
  the **canonical scene/cue selection order** (not quality-ranked), counts per
  source scene, measured-quality summaries and per-sample IDs, cue timestamps,
  scenes and quality metrics. Crops and raw embeddings are not exported.
- `potential_fragments[].score_diagnostics`: pair-score distribution (p10,
  median and p90), three worst reference comparisons with provenance and a
  separate cross-check using only the *selected bounded references* originating
  from the requested scene. If the reference cap excludes those scene samples,
  the scene-only comparison may be empty even when the identity is present.
- `potential_fragments[].same_scene_merge`: actual effective local
  match/cohesion thresholds, strictly accepted cue counts for each side, definite
  numeric/count blockers and separate recorded local nominations under the
  scene-specific evidence policy. This does **not** revalidate the strict core
  against all identities, replay chronological nominations or prove that a pair
  will merge. `can_merge: null` remains intentional.

Both global and local support diagnostics report how many stored nominations were
found and how many pass the **current bounded pair** checks. Missing nominations
or a possible core by raw cue count do not prove a merge would succeed or fail.
Legacy quality metadata that lacks a measurement is reported as missing, never 0.

By default, pairs with cross mean below `match_threshold - 0.20` are excluded.
`--min-mean` adjusts this search floor (`-1..1`), while `--limit` (1..500,
default 20) bounds output size but **not** the number of evaluated pairs.
The report compares only identities actually assigned within the selected
scene, though representatives can come from their other retained scenes.
Large crowds have quadratic pair comparison cost, unlike the production
demand-driven merge path. It does not discover pairs solely among withheld
observations or infer that two performers are the same real person.

**Interpretation:** a pair with no numeric blocker is only a candidate for
review. It is not a predicted merge. Full cold replay still depends on
chronological nominations, revalidated merge support, ambiguous third-party
matches, constituent history and co-presence (including withheld observations).
The reported indicative per-pair thresholds are *not* calibrated recommendations
or guarantees. No edits to thresholds, memberships or published performers occur.

## Automatic identity policy

Every explicit `recognition run` prepares the complete selected evidence set before
publishing identities. It then replays grouping from an empty, temporary SQLite,
ordered by numeric scene ID, cue and detection ID. Previous people, memberships,
merge aliases and accumulated merge votes never seed this replay. Only the current
complete scan of each selected scene enters it; older detection policies and
superseded media cannot contribute duplicate observations.

The resulting partition is equivalent to a cold run of **the same selected scenes,
media, model and configuration**. Scene argument order and prior threshold changes
do not change that partition. `--all-scenes` supplies the library-wide input set;
`--scene-id` rebuilds only the indicated subset and does not promise a library-wide
repartition. Failed scenes remain untouched and the invocation reports partial
work; they are never treated as empty/no-person scenes.

Near-threshold or ambiguous matches are withheld. `recognition.min_identity_support`
controls acceptance of a new group (default 1 distinct cue time in a scan).
`recognition.min_merge_support` independently controls consolidation evidence
(default 3 distinct scene/time observations). Proofs accumulate across scenes
**inside the current replay**, are revalidated against its current representatives,
and are rebuilt when settings change. The same time in the same scene counts once,
even with duplicate detections. Historical `merge_support` rows remain as recovery
history but do not vote in a new replay.

Mean similarity must reach `match_threshold`; every bounded representative must
also pass `clustering.cohesion_threshold`. The best and next coherent identity
compete even when the runner-up falls just below the match threshold. A-B and B-C
similarities cannot override incompatible A-C evidence. Possible reflections retain
the stronger `cooccurrence_merge_threshold` cross-sample requirement. These are
sampled similarity checks, not identity probabilities or universal accuracy claims.
Co-presence includes compatible observations withheld for ambiguous identity,
not just accepted memberships; increasing ambiguity cannot remove this safeguard.
Exact duplicate geometry/embeddings in one cue count as one physical observation.
Co-presence is contextual evidence, not an unconditional different-person rule:
strongly supported reflections may still merge. The matching-policy revision
replays grouping from cached embeddings without invalidating completed scans.

### Same-scene matching preference

After all selected scenes have completed ordinary matching and identity merges,
one final pass can attach borderline, withheld observations to a person already
strongly established in the **same Stash scene / VTT**. The defaults are:

```yaml
clustering:
  same_scene_match_bonus: 0.03
  same_scene_min_support: 3
```

The bonus lowers **both** local thresholds: effective
`match_threshold = max(0, match_threshold - bonus)` and
`cohesion_threshold = max(0, cohesion_threshold - bonus)`. For example,
with 0.65 match, 0.60 cohesion and a 0.03 bonus, local checks require at least
0.62 mean and 0.57 minimum representative cosine similarity. The bonus accepts
0..0.10; zero disables both local passes. This adjusts thresholds, not
similarity scores or probabilities.

The first final pass attaches withheld observations only to a strictly established
person in the **same scene**. `same_scene_min_support` is the number of distinct
strictly accepted cue times establishing the local core (1..1000; default 3).
Duplicate cue times do not help. Core observations must remain unambiguous under
the original strict thresholds. The local check retains the original
`ambiguity_margin` and `cooccurrence_merge_threshold`; it checks competing
groups and cue peers using the same lowered local cohesion and match thresholds.

The second final pass can consolidate *existing distinct identities*, but only
when both have strictly accepted observations in the same scene, at least one
has `same_scene_min_support` independent cue times, and at least
`recognition.min_merge_support` independent nominations from **that same scan**
pass both effective thresholds against all constituents. All original bounded
representatives remain frozen across local consolidations and selected scenes:
A-B and B-C cannot conceal an incompatible A-C endpoint. Distinct co-present
faces retain the original stronger reflection requirement. Attaching withheld
observations never creates local anchors or merge-support votes. Local mergers run
**after** strict replay, so they never loosen the matching of subsequent scenes.

Each weak attachment retains raw scores, strict-reference fingerprints and
supporting cue IDs/times in `same_scene_context`. Local consolidation records
`same_scene_merge` in the source candidates with threshold, support and
constituent evidence. The contextual stages expose scene/observation progress;
retuning replays cached embeddings without new inference. These are heuristic
thresholds, not accuracy guarantees, and a too-low local threshold can still
create false-positive identities.

### Retuning and stable performer labels

After the cold partition is complete, a separate one-to-one binding step assigns
historical person IDs to its groups. Original physical sample ownership is stored
in `identity_origins`, independently of the current partition. The oldest matching
identity wins a consolidation. If it splits, its largest overlapping piece retains
its ID, and historical retired profiles can reactivate for other pieces. IDs never
cause two cold groups to combine. Compatible unselected references can help reuse
an unambiguous historical label for a new scene; they do not seed grouping or alter
the selected partition. New groups with no suitable unused label receive new IDs.

Both stricter and looser `recognition`/`clustering` settings can therefore split,
join or withhold groups. Stored compatible embeddings are reused. Changing detection
settings creates current selected scans (and may require new inference); returning
to a compatible earlier scan reuses it. Selected old memberships are replaced in
one transaction, including when all current observations are rejected or withheld.
Unselected memberships, raw evidence and histories are preserved. Reference limits
are checked before the replacement commits; a failed replay leaves the previous
partition intact. No all-person distance matrix is allocated. The replay streams
embeddings into disk-backed temporary SQLite without copying JPEGs; this scratch
is automatically removed on close, and the face SQLite remains the durable state.
An unchanged selected input/policy/reference context reuses the committed partition.

### Reversible Stash synchronization

For each successfully rebuilt selected scene, current owned performers are resolved
and linked before stale application links are removed with `BulkSceneUpdateInput.performer_ids`
`ADD`/`REMOVE` operations. Unrelated performers and manual associations are preserved.
Only journaled application links to freshly verified owned profiles qualify for
removal. Future observed links preserve whether the application added them; a
pre-existing manual link is not adopted, and a manual re-addition after an
acknowledged removal is preserved. Legacy journaled links retain their recorded
application ownership. Names, notes, images, galleries and other manual metadata
are not transferred or deleted by consolidation.

Normal recognition no longer calls destructive `performerMerge`: inactive profiles
remain available so a later split can reuse their IDs and metadata. Equivalence to
a cold run concerns the grouping and managed selected-scene links, not deletion of
historical profiles or equality of numeric Stash IDs. `scene_links_removed` reports
confirmed stale-link removals. The `rebuild` history phase records the exact scan
selection and matching policy used for the partition.

Every ADD/REMOVE has an intent committed before dispatch and a fresh association
check. Lost removal acknowledgements are reconciled by observing absence. If the
performer remains linked, the unresolved operation blocks replacement writes and
is not blindly retried, including after another threshold change. Ownership edits
also withhold synchronization rather than unlinking a manually edited profile.
Metadata/portrait selection runs only after binding; a historical portrait that
lost its supporting membership cannot outrank a current eligible face simply
because its saved quality score was higher.

Legacy destructive merges remain in the recovery journal. An uncertain legacy
merge touching selected people must first be confirmed by its carried marker,
both ownership tokens and disappearance of the source; otherwise the selection
is withheld. Already deleted legacy profiles cannot regain their old Stash IDs:
a split allocates a separate owned profile rather than treating an inherited source
token on the old survivor as a new binding. Unrelated pending legacy merges are not
resumed by a selected-scene run.

Older YAML `recognition.min_support` remains a deprecated compatibility alias for
both support minimums; remove it before specifying either split key.

| Setting | Effect on merging |
| --- | --- |
| `recognition.match_threshold` | Minimum mean match to each endpoint and mean cross-group similarity. |
| `recognition.min_identity_support` | Minimum distinct cue times to accept a new person (default 1). |
| `recognition.min_merge_support` | Minimum supporting (scene, cue-time) observations across selected scans (default 3). |
| `recognition.ambiguity_margin` | Withholds competing assignments; cannot relax merge evidence requirements. |
| `clustering.cohesion_threshold` | Minimum representative similarity, including cross-group comparisons. |
| `clustering.same_scene_match_bonus` | Local match + cohesion threshold reduction for withheld assignment and supported same-scene identity merges (default 0.03; 0 disables; range 0..0.10). |
| `clustering.same_scene_min_support` | Strong independent cue times in that scene before the bonus applies (default 3; range 1..1000). |
| `clustering.max_samples_per_person` | Bounds each group's retained representatives and the final survivor's lookup sample. |
| `clustering.max_people_per_lookup` | Caps active compatible identities, including provisional groups; aliases are excluded. |
| `clustering.max_reference_bytes` | Caps the bounded reference vectors; exceeding it rolls back that scene's matching and local merges. |

### Pose-aware anchor and auxiliary matching

By default, older local YAML configurations omit `clustering.max_anchor_pose_deviation`
(`null`), preserving their established matching behavior. The distributed example
YAML opts in at `0.20`; this is **not a yaw angle** or a calibrated reliability
threshold. It is the existing normalized five-landmark `pose_deviation` score.
Measure rejections and fragmentation on real VTT observations before tuning.

When both limits are enabled, `clustering.max_anchor_pose_deviation` must be at
most `detection.max_pose_deviation`: the detection limit is the outer hard gate.
For example, 0.20/0.50 admits quality-qualified samples through 0.50 while only
those through 0.20 become anchors; values between are auxiliary. Equal limits
are valid but leave no pose-based auxiliary interval.

With the gate enabled, retained non-rejected observations at or below the configured
anchor limit are eligible to create identities, serve as bounded reference samples,
and support strict or same-scene identity merges. Higher-pose observations remain
`auxiliary`: they cannot create new people, increase anchor references or nominate
merges. After the strict cold replay and same-scene pass finish, auxiliary faces
may attach to already established groups using unchanged cosine match, cohesion,
ambiguity and co-presence checks. `clustering.auxiliary_matching: false` withholds
all auxiliary observations. `clustering.auxiliary_min_support` counts independent
(scene, timestamp) observations assigned to the same anchor identity. Observations
with missing or invalid pose measurements are auxiliary when the gate is enabled.

Rebuild copies retained pose quality alongside embeddings to its RAM database, so
retuning this gate invalidates identity matching but does **not** re-run YuNet or
AuraFace on scans whose detection policy and media have not changed. Profile-only
people may remain deliberately unpublished. Portrait ranking is independent of
this identity-evidence policy.

### Cross-scene provisional identity confirmation

`recognition.cross_scene_provisional: true` opts into a post-strict-replay
pass. It can attach a withheld `candidate_group` to an existing, strictly
confirmed identity from **another** selected scene if all samples meet the
normal (not discounted) match, cohesion and ambiguity thresholds, with
co-presence protection. It can also combine unconfirmed anchors from different
scenes only with mutually unambiguous, complete-link pairwise agreement and
at least `min_identity_support` distinct (scan, time) cues. Same-scene-only
singletons remain withheld. A new provisional match cannot train further
matches or merge any existing published identities in that replay. The
bundled YAML opts in; the default for custom configurations is off.

For targeted offline diagnostics:

```bash
python stash_ai.py recognition report --scene-id 1 --compare-scene-id 2 --face-config facial_recognition.yml --output comparison.json
```

The `cross_scene_comparisons` section includes numeric scores against confirmed
performers as well as the highest-scoring **withheld-to-withheld anchor pairs**
across the two scenes. All data comes from retained embeddings, no rescan or
inference, and the report never modifies SQLite. High scores are hypotheses,
not automatic merge authorization. `multi_view_diagnostics` groups stored
view-stage exclusion reasons by scene; `not_evaluated` is reported separately.

### Verified contextual view enrichment

`clustering.context_view_enrichment: true` is optional (off by default,
enabled in the bundled YAML). A `same_scene_identity` observation may
enrich a confirmed anchor's view collection **only** if its similarity to
an already frozen strict view medoid is at least `view_match_threshold`,
its pose hint agrees, and at least two independent contextual cues pass
complete-link `view_cohesion_threshold`. It cannot create a core, change
strict references or be used as a merge nomination. The augmented prototype
is frozen before withheld observations are retried, so matches cannot
bootstrap further matches in the same rebuild.

### Supported anchor subclusters for same-scene context

`clustering.anchor_view_matching: true` opts in to **strict-core anchor
view matching**. This is separate from the auxiliary multi-view pass. After
ordinary strict matching and same-scene merge handling, accepted anchor
observations (`known_identity` or `candidate_group` only) are grouped
into frozen, complete-link appearance subclusters, scoped to the selected
scene and identity. Each subcluster requires at least the larger of
`view_min_support` and `same_scene_min_support` distinct (scan, seconds)
cues. Duplicate cue detections never manufacture support. Supported anchor
medoids do not create identities or participate in merge nominations.

Previously withheld *anchor* observations with reason
`insufficient_match_or_cohesion` may attach to one confirmed same-scene view
when matching exceeds both `recognition.match_threshold` and
`view_match_threshold`, identity-level ambiguity meets the stronger margin,
the pose hint agrees. Confirmed views can match observations anywhere
within the same selected scene, regardless of elapsed time. Independent
faces of other identities in the same cue veto the attachment; same-cue
embedding conflicts are also checked. At least
`view_attach_min_support` separate cues must agree on the **same
identity and exact prototype**. A successful attachment never becomes
another prototype in this rebuild, and cannot become historical merge support.
No extra face inference, schema change, or rescan is required to retune it.

This can recover anchor observations that were withheld because a diverse
identity's full reference collection failed the global minimum similarity.
It **does not automatically fuse already accepted people** on the basis of
temporal proximity or one favourable prototype. Fragment pairs with mean
similarity near 0.45 and no independent cross-identity evidence (such as
the supplied scene 2 diagnosis) remain subject to the existing strict
merger rules. Lowering global cosine thresholds to force a merge is unsafe.
Temporal proximity is neither required nor evidence of identity. Independent
(scan, seconds) cues still prevent repeated detections at one timestamp from
artificially satisfying support requirements.

Omitting `anchor_view_matching` preserves the previous behaviour; the
bundled YAML opts in.

### Confirmed multi-view identity prototypes

Opt in with `clustering.view_prototypes_enabled: true` and a configured
`clustering.max_anchor_pose_deviation`. This adds **several independently
confirmed appearance subclusters per identity**, not several identities.

After strict anchor replay and the existing auxiliary pass, the matcher derives
bounded per-view medoids from auxiliary faces **already matched to an anchor
identity**. Each view requires `view_min_support` distinct scan/timestamp
observations and pairwise exemplar agreement at
`view_cohesion_threshold`. A rough signed yaw signal from retained YuNet
landmarks divides left/right/other orientations before similarity grouping;
it is not a head-angle estimate, and embedding similarity still determines
the appearance subcluster. Up to `max_view_prototypes_per_person` strongest
views are kept, within `max_reference_bytes`.

In a second frozen pass, previously unmatched auxiliary faces may match these
confirmed views. Match strength must reach at least the higher of `view_match_threshold` and
`recognition.match_threshold`; increasing the global match threshold can
never inadvertently weaken view matching. Best and
runner-up scores are aggregated **per identity**, not per view, and must have
at least `view_ambiguity_margin` separation. Same-cue co-presence remains
strict; view-only attachments need `view_attach_min_support` independent
cues. New view-only attachments are **never used to bootstrap additional
prototypes within that replay**. The view prototypes never create people,
enter strict anchor reference snapshots, or contribute to identity mergers.

The view prototypes are derived deterministically during selected replay from
retained embeddings and candidate proof; no extra persistent schema,
model execution, or rescanning is required when adjusting matching settings.
The bundled YAML opts in, but configurations omitting the setting retain the
previous behavior. This deliberately cannot recover a completely side-profile
person without some initial anchor-confirmed observations. Tune the thresholds
on labelled examples before broad publishing: multi-view max similarity
increases false-positive exposure and is not a calibrated probability.

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
crop, shifts it inside the source, and reduces padding if necessary. If the
requested ratio would cut the face, the crop uses the available padded face
shape instead. `jpeg_quality` is preferred: encoding lowers JPEG quality and,
if needed, image dimensions to fit `storage.max_crop_bytes`, without upscaling.
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

`portraits.source: video` retains the video-frame mode. Install system FFmpeg
(or configure `portraits.ffmpeg`); absence or unsupported/budget-limited video
is recorded on the video candidate while links remain eligible. `time_offsets`,
`max_candidates`, `timeout_seconds`, `max_video_bytes` and `ffmpeg` in the portraits section are
used only by the video source. When no usable identity-verified video portrait is
found, selection tries accepted VTT observations and then retained recognition
crops. Metadata records the fallback rather than claiming a video frame/time.

`portraits.min_visible_face_fraction` is a number in `[0, 1]`, default `0.8`:
prefer at least 80% estimated visible facial surface. `portraits.weights.visibility`
is a nonnegative ranking weight, default `3.0`, which favors more visible faces.
A zero target removes the extra shortfall penalty while retaining the base
visibility score. A zero weight disables portrait visibility inference and removes
its main ranking contribution. FaceOcc is still required when the independent
`detection.min_visible_face_fraction` gate is positive. Set that threshold to 0
as well to disable all visibility downloads/inference. Unknown YAML keys still fail validation.

The pinned MIT-licensed FaceOcc v1.0.0 SafeTensors model predicts visible face
versus background/occluder pixels. Its U-Net/ResNet18 runs on the existing
application torch/torchvision CPU or CUDA profile, with no SMP/timm dependency or
ImageNet weight download. `models.device: cuda` stays strict; only explicit `auto`
allows CPU fallback. One lazy segmentation model is shared across selected people
and closed at the end of the invocation. Visibility failures record an unavailable
cue and warning while ranking other available quality; they never imply that a
face is uncovered and do not block its portrait.

The application reports the fraction of mask-positive pixels (`logit > 0`) within
a fixed elliptical face region derived from the detector box, excluding artificial
padding. The model input is a fixed square crop around that box, resized to RGB
256x256 with ImageNet normalization. Portrait `margin`, aspect ratio and JPEG
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
image. Impossible configured aspect ratios adapt to the available face shape;
oversized JPEGs lower quality or dimensions while respecting the byte budget.
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
Each selected scene must use observations from one retained scan revision;
mixing versions is rejected before model preparation to avoid false co-presence.
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

### Rebuild performance diagnostics

Set `execution.profile_rebuild: true` to save bounded `rebuild_profile` aggregates
in recognition run history, including failed/interrupted attempts. The default
is false. Measurements contain no SQL text, scene/person IDs, images or vectors.
They cover copy, reference loading, scene matching, contextual checks, merge
support/compatibility, rechecks, binding and final apply. Work counters include
validated vectors and group/sample comparisons. SQL buckets count statement
execution and transaction exits; statement seconds exclude later lazy cursor
consumption. Stage seconds are inclusive and nested, so do not add them together.
Record runtime, observation/person counts and warm versus cold state when
comparing runs. Profiling has overhead and does not alter matching policy or
require fresh inference. Ordinary progress remains throttled independently.

Cold replay now owns a RAM-resident canonical reference index for its immutable
selected observation catalogue. Membership inserts, moves and retractions update
that private index; unchanged representatives reuse validated vectors. Known
references remain frozen inside each scene. The exact canonical sample order,
reference fingerprint and logical reference budget are preserved. General
reference loads use bounded person batches, including full membership-scope
validation outside the representative cap.

`execution.rebuild_memory_bytes` is a lazy working-index budget (8 GiB by default),
not an upfront allocation or the existing representative-vector limit. It can be
increased on hosts with more RAM, such as a 96 GB system. If the canonical index
cannot fit its conservative per-observation allowance, replay uses the existing
ordered queries without a new disk-spill cache. A failed replay discards its
private index; it is never reused after rollback or by another invocation.

Similarity scoring uses bounded native NumPy blocks for sufficiently large
catalogues. A conservative floating-point error interval may reject only groups
that cannot meet the relevant threshold/cohesion condition. Every remaining
comparison uses the original scalar dot products and reductions, preserving
stored scores, ambiguity, ties and decisions. Small catalogues keep the scalar
path. Native block caches are invocation-owned and byte-bounded, including their
retained input owners; no approximate-nearest-neighbor or truncated rival list is
introduced. Same-scene contextual matching retains its full cohort of rivals.

Cold replay and each standalone scene match own a RAM-only evidence cache. Cue
rows are narrow and vectors/signatures are validated lazily, preserving existing
short-circuit failure behavior. Exact peer scores are reused only for the same
observation and immutable representative version. Contextual merge checks share
verified latest-scan heads and complete multi-face evidence, including withheld
observations and later selected scenes. Newer partial or differently configured
scans still supersede old evidence. The cache uses a bounded share of the working
budget and recomputes evicted data instead of spilling to disk.

### CPU workers during identity rebuild

See [the retained-evidence benchmark and CPU diagnostics guide](recognition-parallel-benchmark.md)
for read-only local baseline collection and computed-versus-applied progress.

`execution.rebuild_workers` controls the rebuild compute pool independently of
`portrait_cpu_threads` and the optional one-reader `rebuild_prefetch_scenes`.
The default `1` keeps serial execution. `0` chooses up to sixteen workers,
leaving two available CPUs free where possible. Explicit values from 1 to 256
are capped by available CPUs divided by `rebuild_native_threads` (1..16).
These are logical CPU budgets, not core pinning or guaranteed utilization.

For an initial comparison on a 24-core host, try 1, 8, 16 and 20 workers with
one native BLAS thread per worker. For example:

```yaml
execution:
  rebuild_workers: 16
  rebuild_native_threads: 1
  rebuild_queue_bytes: 67108864
  profile_rebuild: true
```

This is a tuning starting point, not a measured fastest setting. Worker startup,
Python loops, matrix packing, memory bandwidth and serial decisions can dominate
small workloads. Sufficiently large immutable reference catalogues use bounded
read-only matrix shards and query batches. Small catalogues, mutable inputs and
work that cannot fit admission keep the existing serial score path.

The coordinator still visits scenes and observations in canonical order. It
alone grows new candidate groups, makes merge/support/co-presence decisions,
claims historical person/performer labels and writes SQLite. Workers calculate
frozen-reference scores used by assignment, post-merge rechecks, merge
nominations, final same-scene rankings and historical fallback. Native batch
scores may only prove rejection; finalists retain the scalar decision values.
Demanded component-pair compatibility proofs also use immutable CPU jobs when
multiple constituent pairs are needed. Each proof keeps the directional scalar
dot/mean/min order; the coordinator validates reference ownership and imports
both positive and negative proofs into the existing bounded cache. A small
batch is retired before an incompatible endpoint stops later work. No global
all-pairs identity graph is scheduled.

For longer demanded support/context streams, bounded CPU blocks compute exact
constituent predicates. The coordinator still selects the authoritative latest
scans, reduces votes in scan/cue/observation order, counts each scene/time once
and applies distinct-signature co-occurrence rules. Mutable direct membership
is checked at decision time before contextual work. A complete small batch is
retired before exposing its ordered results, allowing early exit without
abandoning an active CPU stream. Invalid vectors keep their original point of
failure, and small/oversized streams use the scalar path. These ephemeral masks
are not persisted or reused across component changes; shared versioned mask
caching remains separate work. Cue-peer/rival checks remain on the coordinator.

Rebuild uses retained embeddings and never loads or invokes GPU models. CPU
workers receive neither models nor SQLite connections. Scan and portrait model
ownership remains separate from this CPU pool; invocation-wide unification of
all model owners is tracked separately in #849/#850.

`rebuild_queue_bytes` bounds queued CPU jobs including estimated scratch/results;
its effective value is capped at `rebuild_memory_bytes / 16`. Packed matrix
caches, the bounded current query batch and sparse-result accumulator retain
their separate bounded ownership. This field is not a total-process RSS cap or
unified accounting of every rebuild cache. No disk-backed working cache is
introduced. Native BLAS limits are restored only after their workers stop;
incomplete shutdown retains borrowed owners under the existing quarantine rule.

With profiling enabled, `rebuild_profile.cpu` records requested/effective workers,
native limits, queue capacity/peak, submitted/applied blocks and ordered wait.
Successful worker wall seconds are summed parallel task durations, not elapsed
run time. Worker-thread CPU seconds exclude native helper-thread CPU time; the
benchmark additionally records process CPU time. Worker metrics do not retain
scene IDs, embeddings or SQL. Progress remains based on coordinator-applied
scenes, so prefetched computations cannot advance the bar early.

Execution budgets do not change matching policy or semantic checkpoint identity.
A restored checkpoint resumes only the applied strict prefix and can use a
different worker count. Changing workers on an already valid completed rebuild
can reuse the result, so it does not measure cold-rebuild scaling.

Run the synthetic cold/cached comparison independently of production state:

```powershell
python -m benchmarks.benchmark_recognition_parallel --scenes 480 --people 96 --dimensions 512 --workers 1 8 16 20 --repeats 3
```

The fixture contains repeated orthogonal identities and omits model inference,
media, API latency and realistic face/merge distributions. Its timings are not a
forecast for a 9,000-scene library. The JSON includes semantic hashes, effective
budgets, stage profiles, process CPU and RSS after each trial. Use representative
retained evidence and inspect the limiting stage before promoting a default.

### Strict reference authority

Membership and reference authority are separate. Strict matching and independently
confirmed cross-scene anchors may supply references; same-scene bonus and view-only
attachments may not. The latest retained candidate decision controls authority,
including after retuning. Membership-only legacy anchors retain their former role.
Identity reports record `reference_role` for rebuilt assignments. Additional people
in a cue do not themselves veto recovery: destination occupancy and competing
embeddings still protect against confusing two people or reflections. Auxiliary
view matching chooses a pose-compatible prototype before ranking each identity.

### Representative strict anchors

The bounded reference bank selects quality and appearance/scene coverage from
all strict members, using at most twice `max_samples_per_person` in selection
scratch. Later scenes can replace redundant early cues. Duplicate scene/time
measurements do not fill the bank. Canonical output ordering and the same selector
are used for SQL loading, RAM snapshots and historical label comparisons.
`reference_selected` in identity reports describes the bank after that rebuild;
it is distinct from eligibility (`reference_role`).

Pose quality now measures yaw/pitch in the eye coordinate frame. Roll remains a
reported measurement but does not by itself disqualify an anchor after alignment.
The scanning quality contract is versioned: older scans remain retained history,
and a new invocation measures the updated contract rather than reusing stale
pose admissions. This may require a new scan on the first run after upgrading.

### Global confirmed anchor views

With `clustering.anchor_view_matching`, a final frozen catalogue shares strict
appearance cohorts across selected scenes. `view_min_support`,
`view_cohesion_threshold`, `view_match_threshold`, `view_ambiguity_margin`,
`view_attach_min_support` and `max_view_prototypes_per_person` control these
cohorts too. Contextual and recovered members cannot train this catalogue.
Independent attachment support can come from different scenes; singletons and
competing identities remain withheld. Reports include the medoid, support IDs,
score and thresholds in `global_view` and mark these members as attachments.

Established fragments can consolidate through independently supported strict
cohorts even when other appearances fail whole-bank cohesion. The frozen graph
forms components from independently supported strong cohort pairs. Every pair
in a component must have strong evidence before any member moves. Weak external
neighbours remain competition evidence: they block a proposed component only
when their score falls within its internal evidence ambiguity margin. Co-present faces require
the stronger reflection threshold. No graph is recomputed from merged output.
`global_view_merge` records complete-component size, context and pair witnesses.
This consolidation is reversible replay, not a destructive Stash performer merge.

View exemplars use a bounded quality/diversity pool rather than the first eight
confirmations. Every independent confirmed cue contributes to medoid selection;
later evidence can replace redundant early samples. Pose hints admit neighbouring
neutral/side categories only within 0.02 of the 0.12 signed-eye-distance boundary.
Embedding, support, ambiguity and co-presence requirements remain in force.

### Selected runs and frozen global identities

A selected-scene rebuild still computes its strict partition from selected
evidence. Before label binding it reads a frozen reference bank from unselected
scenes, using only current members with strict authority, compatible model/scope
and quality admission, and latest complete scan evidence. No unselected images are
loaded and no unselected observations, candidates or scene links are rewritten.
Historical labels from selected scenes remain eligible for label-only reuse
across a rescan, preserving owned performer names and portraits.

Several independently accepted local fragments may adopt the same global label
when every retained representative matches frozen references unambiguously.
Co-present fragments cannot both claim that label. Withheld selected anchors can
also reuse an established strict identity; view-only matches require independent
attachment support and never become strict training references. Reports include
`historical_reference`, the matching path, source medoids and dependency hash.

The selected result cache includes the unselected strict reference context, so
reference repairs or superseding scans invalidate stale label proposals. Strict
replay checkpoints remain selected-only; the external binding stage is recomputed
after restore. Stash publication continues to update only selected scene links
and reuses verified owned performers, preserving manual names and associations.

### Simultaneous claims and deferred global identities

All matching passes resolve simultaneous pending claims before counting support.
Distinct detections claiming one person in the same cue must satisfy the stronger
reflection similarity threshold; exact duplicate detections remain one sample.
Conflict details are retained as `decision_conflict` in the identity report.

A coherent new local group is not sufficient evidence for a new global person
when compatible historical references provide ambiguous competing identities.
Such groups remain withheld with `historical_ambiguous_identity` and a
`global_identity_decision` describing the rivals and dependency context. Existing
selected ownership remains eligible for stable label reuse. These decisions are
reversible and do not rewrite unselected scene memberships or Stash profiles.

### Shared frozen references

A final strict retry rechecks earlier withheld anchors against the completed
selected strict catalogue. Its assignments are attachments, not new training
samples. Auxiliary queries can also use compatible frozen external cores and
views; reference owners are resolved to final global labels before attachment,
including when the selection has no anchors of its own.

With view prototypes enabled, auxiliary observations independently matched to a
core established before their source cue can corroborate a frozen appearance
view. A new, unowned local anchor hypothesis may attach to that earlier view
before receiving a global identity. Existing ownership, distinct-cue support,
competing anchors and co-presence remain checked. Such attachments cannot train
strict references, promote further views or form transitive merge chains. This
separates the geometry admission role from permission to train references.

### Production evidence and whole-flow evaluation

The fragmentation report uses the production strict reference admission and
quality/diversity selector, including compatible historical scans and current
reference authority. `reference_context` fingerprints that bank. Memberships
without strict references remain visible as `no_strict_references`; attachment
samples cannot distort strict similarity or local-core counts. Deferred global
rivals and simultaneous claim conflicts are included in withheld examples.
`confirmed_view_evidence` uses the same cohort checks as global consolidation;
pair evidence alone still does not establish complete-component merge safety.
Cross-scene comparisons include `production_match` from the frozen matcher.

For labelled local evaluation, `stash_ai.recognition.evaluation.evaluate_identities`
accepts records with `observation`, `scene`, explicit `truth` and `predicted`
(identity ID or `None`). It reports global and within-scene fragmentation, extra
fragments, mixed-identity merges, false split/merge observation pairs, coverage,
and pair precision/recall. Withheld samples lower coverage/recall and never form
one artificial identity. These metrics describe only the supplied ground truth.

`compare_identity_runs(records, predictions)` compares named observation-to-ID
maps from joint, separate, reordered or retuned runs. Comparisons use partitions
on common observations, so UUID renaming alone is equivalent. Missing keys mean
unprocessed observations; explicit `None` means withheld. Coverage changes and
partition changes are reported separately. Synthetic regression vectors verify
policy logic and do not establish accuracy on real video media.

### Global view resolution before label binding

Scene-local strict confirmations are frozen before auxiliary views propose
repairs of selected anchor hypotheses. Only an established core in the source
scene can train such a view; numeric scene IDs do not define a cross-scene
chronology. Old automatic ownership does not veto the repair. Historical label
votes are applied afterwards, preserving owned performer records. Co-presence,
independent support, competing strict anchors and frozen no-chain checks still
apply. Recovered observations remain attachments and cannot train another view.


### Durable verified auxiliary views and quality retuning

Selected runs also read auxiliary confirmations from unselected scenes. Each
source must still be an accepted direct `auxiliary_identity` membership and
must match its owner unambiguously against current frozen strict references.
Only independently supported coherent cohorts become `verified_view` references.
Historical/view/context attachments cannot train them. Source membership,
latest-scan supersession, current quality and rival identity checks are repeated
on every capture; the external cache dependency includes the durable auxiliary
evidence. Historical assignment proofs include `view_evidence` with authority,
medoid and support pointers. Cross-scene report matching reads this bank too.

A scanning policy hash controls detection-cache reuse, not embedding compatibility.
An unselected observation from a different detection policy may remain a reference
when its model namespace is unchanged and its retained current pose/alignment
contracts, detector score and measured quality satisfy the current gates. Missing
visibility measurements or old geometry/pose contracts exclude that fallback.
Rejected observations are never resurrected. No extra inference or schema
migration is required; raw observations and decisions remain the source of truth.


### Supported cohorts and unresolved outliers

A failed whole-group historical match can now recover independently matched
cohorts without publishing another identity for their unmatched observations.
All queries use the original frozen bank, preserve identity-level competition
and co-presence checks, and repeat exact-medoid support checks after resolving
simultaneous claims. Recovered members are attachments. Unmatched observations
remain `cohort_unresolved`; reports include the frozen source, proposed independent
support and total support in `cohort_resolution`.

Selected-fragment repairs use the existing global-view neighbour shortlist;
small groups without a confirmed view are considered separately. A selected
repair requires a strict majority of independent cue times, an unresolved
remainder, and no target that is itself being repaired. Higher coverage resolves
unequal proposals; equal cycles remain unchanged. This avoids replay-order chains
and avoids another exhaustive all-identity scan for unrelated selected people.

View training now accepts adjacent pose categories near the same uncertainty
boundary used by matching. Every observed direction must remain compatible;
neutral evidence cannot bridge opposite left/right directions. Independent
support is pooled before promotion instead of being split across a hard bucket
boundary. Embedding cohesion and promotion thresholds remain unchanged.


### Consolidation competition diagnostics

Global consolidation now uses one frozen plan for runtime and read-only reports.
A strong neighbour chain with incompatible endpoints remains blocked; adding an
uncompetitive weak outsider no longer makes a complete strong component fail.
A near-threshold outsider or a high-scoring view with insufficient merge support
can still veto the component through the ambiguity margin. Competition is checked
against every member's weakest supported internal pair, including its other
appearance views. No merged output is used to recompute the graph.

`global_view_consolidation` records `incomplete_strong_component`,
`competing_view` or `no_supported_pair`, component size/context, supported degree
and the nearest external rival's score and gap. Isolated people with no candidate
neighbours do not receive redundant decision payloads. `global_view_merge` format
3 carries the same plan evidence. Scene fragmentation pairs expose this read-only
assessment as `global_view_component`; `can_merge` remains unset because a report
is not a selected replay or authorization to merge Stash performers.

### Independent scene cores

With `anchor_view_matching` enabled, a cold replay first establishes each scene's
strict cores independently. Previously replayed scenes cannot capture part of a
later scene's local core. One frozen complete-component pass reconciles compatible
strict cores using the ordinary strict thresholds, original cross-sample evidence,
outsider competition and the stronger co-presence threshold. Appearance-only
consolidation keeps its existing stronger view thresholds and support minimum.

Auxiliary observations before the first supported source core can use the complete
source scene. An already established source core retains precedence over later
provisional fragments, preventing a later anchor hypothesis from stealing the
auxiliary evidence that resolves it. A small accepted core without its own promoted
view may also attach as a whole to the independently frozen view bank. These
`selected_view_identity` attachments never train another view. Stable labels are
bound only after this partition is resolved; scan inputs and model inference are
unchanged. Matching policy v20 invalidates earlier derived decisions/checkpoints.

### Durable strict confirmations and reserve appearances

Direct auxiliary decisions record `strict_confirmation`: the frozen strict
witness observation IDs, owner and effective thresholds. Historical promotion
revalidates those exact witnesses under current quality, membership, latest-scan
and rival checks. Adding a different correct appearance of the same owner cannot
invalidate the original proof through a diluted global mean. Removed or reassigned
witnesses revoke the proof. Older decisions without this field are independently
revalidated against current strict appearance cohorts; recovered attachments never
become witnesses.

Production matching and fragmentation reports retain reserve confirmed cohorts.
`max_view_prototypes_per_person` controls the priority set: frequent and diverse
appearances are examined first, with the remaining confirmed appearances available
to matching and competition. It is not a semantic cutoff that can create a new
identity merely because its matching appearance ranked fifth. The former first-32
discovery cutoff is also removed in this complete mode. Candidate and retained
evidence are checked against `max_reference_bytes`; exceeding the budget fails
explicitly instead of silently dropping a rare view. Matching policy v21 records
these changes; the scan and model caches remain reusable.

### Global reference registry and incomplete rescans

Historical strict cores and current independent scene cores now share a frozen
complete-component reconciliation plan. Ordinary strict compatibility and supported
strict appearance witnesses may nominate a pair, but every pair in a resulting
component must have its own valid proof. A mixed chain cannot consolidate by
reusing an intermediate union. Historical and selected co-presence are checked
together. Existing view-only query matches remain attachments instead of acquiring
strict authority through a provisional local identity.

The oldest historical label wins a supported component. Original physical sample
ownership, raw observations/crops and performer records are retained. Canonical
memberships and derived decision provenance may be repaired outside the selected
scenes in the final SQLite transaction; publication and stale-link reconciliation
in Stash still touch only the selected scenes. When an earlier historical core is
recognized by a newly available frozen source view, it becomes
`global_reference_attachment`, so its recovered appearance cannot train the next
recovery. Reports expose read-only `reference_registry` proposals and identity
exports retain the committed evidence.

A newer partial scan, or completed replacement without resolved membership,
temporarily hides old references. Those old observations can defer a possible
duplicate as `reference_bank_unavailable`; they cannot positively assign a person.
Unrelated new identities remain eligible. Latest scan state and gap evidence are
cache dependencies, so completing/restoring the source lets the next selected
replay resolve the deferred observations. Matching policy v22 records this flow.

### Unresolved support across runs

With `recognition.cross_scene_provisional: true`, the latest unresolved anchor
observations are a durable support pool. A later selected cohort revalidates their
current quality/model compatibility and complete-scan status. Superseded, rejected,
other-namespace and auxiliary observations cannot establish a person. Independent
support counts `(scan, seconds)`, so repeated detections at one cue do not supply
additional votes. Candidate exemplars use the bounded quality/diversity selector;
new identities require a coherent frozen union rather than a weak bridge chain.

Views frozen from independent current strict cores or source-confirmed auxiliary
evidence can also recover older pending queries. Reference training cues never
count as query votes. Exact-medoid query support, global rivals, known membership
and simultaneous cross-view claims are checked before acceptance. New pool results
never train another view in the same pass. A final auxiliary reassignment that
revokes the frozen source owner invalidates its pool claims before commit.

Confirmed derived assignments, including still-unresolved older witnesses, commit
atomically with the selected replay and preserve original physical ownership.
Strict births use `cross_scene_provisional`; recovery through a view uses
`pooled_view_identity` and remains attachment-only. Late direct matches to an already
confirmed strict core also remain attachments. Raw scans/images are not changed or
reinferred. Stash links and portraits are still published only for selected scenes;
select an older backfilled scene to publish its existing identity association.

Pool input changes participate in external cache dependencies, and the committed
fingerprint includes successful backfills. Reports/exported decisions include
`provisional_pool` support, thresholds, witnesses, authority and deferral evidence.
The feature remains disabled when `cross_scene_provisional` is false. Matching
policy v23 invalidates older derived decisions without changing scan/model keys.

Recovered membership-only or damaged legacy decisions receive explicit attachment
authority, so the legacy strict fallback cannot silently promote their evidence.
Strict-core summary vectors remain immutable for reuse by the packed score cache.


### Offline identity evaluation on retained VTT observations

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


### Supported pairs inside incomplete identity components

A connected but incomplete component may contain an unambiguous supported pair.
The frozen planner now proposes reciprocal unique best pairs and checks their
internal score against **every** outside identity, including the rest of the
original component. Equal alternatives and insufficient margins remain blocked.
Pair proposals are disjoint; merged output never extends another proposal.
Reports expose `partition_strategy`, `parent_component` and the closest rival.

Production applies these repairs only after all assignment/enrichment phases.
It freezes strict and confirmed-view proposals together and rejects incompatible
overlapping proposals. This prevents an early pair merge from training a later
recovery in the same replay. The first implementation finalizes selected strict
cores only; identities tied to unselected historical owners or carrying non-strict
members require their existing reconciliation paths. Current matching policy is
versioned so old rebuild decisions are recomputed from retained embeddings.

### Recovering uncertain pose from VTT geometry

A final query-only pass may match a neutral/side landmark category mismatch to
a frozen strict or independently verified view. Opposite side categories and
invalid geometry do not receive this exception. It preserves embedding, raw
rival, ownership, visibility and simultaneous-face checks and requires at least
two distinct query timestamps (or the configured higher attachment support).
Recovered observations record `pose_recovery` evidence and
`pose_uncertain_identity`; they remain attachments and cannot seed another look.
The pass uses only retained VTT observations and invalidates matching caches,
without invalidating scans or running inference.

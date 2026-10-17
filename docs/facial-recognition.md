# Facial recognition

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

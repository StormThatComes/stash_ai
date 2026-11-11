# Identity matching and references

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Recognition observation quality](#recognition-observation-quality)
- [Automatic identity policy](#automatic-identity-policy)

For day-to-day commands start with the [recognition guide](../facial-recognition.md).
This reference explains how retained observations become identities.

## Recognition observation quality

Detection quality gates run before embedding and grouping. They keep unreliable
profile/covered faces from seeding identity fragments without lowering matching,
ambiguity, cohesion or reflection-merge thresholds. Rejected observations retain
geometry, measured quality and a reason in SQLite, but no embedding/crop; they
cannot become reference samples, supply distinct-time support or support merges.
The run prints per-scene rejection counts, saves the reason breakdown in detection
progress, and includes `observations_rejected` in its summary, including reused scans.

| `face.quality` setting | Built-in default | Meaning |
| --- | ---: | --- |
| `min_face_pixels` | 32 | Minimum shorter face-box side in source pixels |
| `min_sharpness` | 10 | Minimum Laplacian variance on the normalized grayscale crop; 0 disables |
| `max_pose_deviation` | 0.50 | Maximum five-landmark pose estimate in `[0, 1]`; `null` disables this gate |
| `min_visible_face_fraction` | 0.80 | Minimum FaceOcc visible-face estimate in `[0, 1]`; 0 disables recognition segmentation |
| `require_complete_face` | true | Reject face boxes extending beyond the source thumbnail |

The shipped [YAML template](../../facial_recognition.yml) can override these built-in
defaults; for example, its face visibility floor is 0.50. Existing local files
retain their configured values.

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
A lazy visibility model is reused within each active phase (or across phases in
`retain` mode); accepted VTT observations retain the raw measurement so portrait
ranking does not repeat it.
If `require_complete_face: false` permits a partial face whose visibility region
extends outside the real source, that observation is rejected as
`face_visibility_unmeasurable`, without an embedding/crop. Other valid cues continue.
This local geometry rejection does not turn a broken or unavailable model into a
successful scan.

Fresh retained face JPEGs also obey `storage.max_crop_mb` (minimum 0.0009765625 MB).
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
version: 2
face:
  quality:
    max_pose_deviation: null
    min_visible_face_fraction: 0
    require_complete_face: false
```

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

Near-threshold or ambiguous matches are withheld. `matching.face.anchors.min_identity_support`
controls acceptance of a new group (default 1 distinct cue time in a scan).
`matching.face.cross_scene.min_merge_support` independently controls consolidation evidence
(default 3 distinct scene/time observations). Proofs accumulate across scenes
**inside the current replay**, are revalidated against its current representatives,
and are rebuilt when settings change. The same time in the same scene counts once,
even with duplicate detections. Historical `merge_support` rows remain as recovery
history but do not vote in a new replay.

Mean similarity must reach `match_threshold`; every bounded representative must
also pass `matching.face.cohesion_threshold`. The best and next coherent identity
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
version: 2
matching:
  face:
    same_scene:
      match_bonus: 0.03
      min_support: 3
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
`matching.face.cross_scene.min_merge_support` independent nominations from **that same scan**
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
embeddings into private in-memory SQLite without copying JPEGs; this scratch
is released on close, and the face SQLite remains the durable state. Replay pages
share the [rebuild memory budget](performance.md#replay-memory-and-checkpoints).
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
| `matching.face.match_threshold` | Minimum mean match to each endpoint and mean cross-group similarity. |
| `matching.face.anchors.min_identity_support` | Minimum distinct cue times to accept a new person (default 1). |
| `matching.face.cross_scene.min_merge_support` | Minimum supporting (scene, cue-time) observations across selected scans (default 3). |
| `matching.face.ambiguity_margin` | Withholds competing assignments; cannot relax merge evidence requirements. |
| `matching.face.cohesion_threshold` | Minimum representative similarity, including cross-group comparisons. |
| `matching.face.same_scene.match_bonus` | Local match + cohesion threshold reduction for withheld assignment and supported same-scene identity merges (default 0.03; 0 disables; range 0..0.10). |
| `matching.face.same_scene.min_support` | Strong independent cue times in that scene before the bonus applies (default 3; range 1..1000). |
| `matching.face.samples.references.max_samples_per_person` | Bounds each group's retained representatives and the final survivor's lookup sample. |
| `matching.face.samples.references.max_people_per_lookup` | Caps active compatible identities, including provisional groups; aliases are excluded. |
| `matching.face.samples.references.max_reference_mb` | Caps the bounded reference vectors; exceeding it rolls back that scene's matching and local merges. |

### Pose-aware anchor and auxiliary matching

By default, older local YAML configurations omit `matching.face.anchors.quality.max_pose_deviation`
(`null`), preserving their established matching behavior. The distributed example
YAML opts in at `0.20`; this is **not a yaw angle** or a calibrated reliability
threshold. It is the existing normalized five-landmark `pose_deviation` score.
Measure rejections and fragmentation on real VTT observations before tuning.

When both limits are enabled, `matching.face.anchors.quality.max_pose_deviation` must be at
most `face.quality.max_pose_deviation`: the detection limit is the outer hard gate.
For example, 0.20/0.50 admits quality-qualified samples through 0.50 while only
those through 0.20 become anchors; values between are auxiliary. Equal limits
are valid but leave no pose-based auxiliary interval.

With the gate enabled, retained non-rejected observations at or below the configured
anchor limit are eligible to create identities, serve as bounded reference samples,
and support strict or same-scene identity merges. Higher-pose observations remain
`auxiliary`: they cannot create new people, increase anchor references or nominate
merges. After the strict cold replay and same-scene pass finish, auxiliary faces
may attach to already established groups using unchanged cosine match, cohesion,
ambiguity and co-presence checks. `matching.face.auxiliary.enabled: false` withholds
all auxiliary observations. `matching.face.auxiliary.min_support` counts independent
(scene, timestamp) observations assigned to the same anchor identity. Observations
with missing or invalid pose measurements are auxiliary when the gate is enabled.

Rebuild copies retained pose quality alongside embeddings to its RAM database, so
retuning this gate invalidates identity matching but does **not** re-run YuNet or
AuraFace on scans whose detection policy and media have not changed. Profile-only
people may remain deliberately unpublished. Portrait ranking is independent of
this identity-evidence policy.

### Cross-scene provisional identity confirmation

`matching.face.cross_scene.provisional: true` opts into a post-strict-replay
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

`matching.face.views.context_view_enrichment: true` is optional (off by default,
enabled in the bundled YAML). A `same_scene_identity` observation may
enrich a confirmed anchor's view collection **only** if its similarity to
an already frozen strict view medoid is at least `view_match_threshold`,
its pose hint agrees, and at least two independent contextual cues pass
complete-link `view_cohesion_threshold`. It cannot create a core, change
strict references or be used as a merge nomination. The augmented prototype
is frozen before withheld observations are retried, so matches cannot
bootstrap further matches in the same rebuild.

### Supported anchor subclusters for same-scene context

`matching.face.views.anchor_view_matching: true` opts in to **strict-core anchor
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
when matching exceeds both `matching.face.match_threshold` and
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

Opt in with `matching.face.views.prototypes_enabled: true` and a configured
`matching.face.anchors.quality.max_pose_deviation`. This adds **several independently
confirmed appearance subclusters per identity**, not several identities.

After strict anchor replay and the existing auxiliary pass, the matcher derives
bounded per-view medoids from auxiliary faces **already matched to an anchor
identity**. Each view requires `view_min_support` distinct scan/timestamp
observations and pairwise exemplar agreement at
`view_cohesion_threshold`. A rough signed yaw signal from retained YuNet
landmarks divides left/right/other orientations before similarity grouping;
it is not a head-angle estimate, and embedding similarity still determines
the appearance subcluster. Up to `max_view_prototypes_per_person` strongest
views are kept, within `max_reference_mb`.

In a second frozen pass, previously unmatched auxiliary faces may match these
confirmed views. Match strength must reach at least the higher of `view_match_threshold` and
`matching.face.match_threshold`; increasing the global match threshold can
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

`matching.face.samples.references.max_reference_mb` caps the sum of the float32 embedding array
bytes used as references in a lookup, including provisional new groups while
matching the selected scene. Compatible retained people from the same Stash
instance and model namespace participate even when their scenes are not selected.
Each person contributes at most `max_samples_per_person` vectors (default 32).
The default limit is 128 MB; the supported range is 1..65536 MB.
Fractional limits are accepted; 1 MB = 1048576 bytes (MiB). Exceeding it fails that scene's matching
instead of silently dropping people or samples. The error includes required bytes, configured limit
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

For a larger library on a high-RAM host, an 8 GiB reference budget can be set with:

```yaml
version: 2
matching:
  face:
    samples:
      references:
        max_reference_mb: 8192  # 8 GiB; allocated only as references are loaded.
```

Existing YAML values remain unchanged after upgrading. Increasing this limit
does not require rescanning and does not change recognition thresholds.

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

With `matching.face.views.anchor_view_matching`, a final frozen catalogue shares strict
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
evidence are checked against `max_reference_mb`; exceeding the budget fails
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

With `matching.face.cross_scene.provisional: true`, the latest unresolved anchor
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

### Matching independently supported strict appearances

`matching.face.views.strict_appearance_matching` is enabled in the shipped YAML and
requires `anchor_view_matching`. Omitted settings retain the previous behaviour
so an existing configuration can opt in and compare both variants with
`recognition evaluate`.

A coherent strict appearance with at least `max(view_min_support,
min_merge_support)` distinct timestamps can explain queries using the ordinary
`match_threshold` and `cohesion_threshold`. Every retained appearance witness
participates in mean/minimum scoring; a single best image cannot authorize a
match. Raw competing anchors, ambiguity, pose, visibility and co-presence still
apply. Complete selected fragments can attach to such an appearance only after
every member passes; reciprocal proposals and chains stay unresolved. Remaining
withheld queries need at least two independent timestamps. Both routes remain
attachment-only and expose `strict_appearance` evidence (inside
`shared_reference` for a whole-fragment repair).

With this option, provisional local groups also refresh bounded representatives
using quality and appearance diversity. Their original seed stays pinned, so
replacement cannot walk away from its origin. Later accepted cues can replace
duplicate early samples within `max_samples_per_person`; duplicate timestamps
cannot fill that bank. Once the bank is full, refreshes batch at most eight accepted timestamps and
flush at the end of the scene. The sampler reads at most the cap plus eight
observations, avoiding a full selection after every subsequent face.
No VTT rescan or model inference is needed.


### Promotion from one independently supported strict look

The single-root alternative to two disjoint parent looks records
`one_strict_view_in_two_source_scenes_and_two_disjoint_query_scenes` in the
promotion proof. Its one parent must have strict authority; a verified auxiliary
or previously promoted view cannot use this alternative. Each of two source
scenes supplies `max(view_min_support, min_merge_support)` independent timestamps.
The new appearance needs its own support across two additional scenes, disjoint
from every scene contributing to the strict source cohort.

Source-scene observation IDs remain part of `source_ids` even when bounded
representative selection chooses identical embeddings from an earlier scene.
Deleting/reassigning those observations, collapsing their timestamps or
superseding a source scan revokes the next promotion. A full replay rebuilds this
evidence from current VTT observations. Output remains generation one,
attachment-only, and cannot supply another promotion root. The existing two-parent
route remains available; a single-root negative birth hint also requires two
query scenes. No schema change, additional input sampling or inference is needed.

### Frozen match completion

Recognition now evaluates the best admissible appearance cohort instead of the
first matching cohort. Local scene priorities receive one frozen reassessment;
this does not promote any observation into a reference. Joint strict-appearance
and neutral/side pose recovery requires independent queries and preserves all
raw-anchor, ownership, opposite-side and co-presence vetoes.

Incomplete identity components can recover complete subgroups larger than two.
Every pair needs original frozen evidence, every outsider remains a competitor,
and overlapping incompatible proposals veto each other. Subgroup search is
bounded to 128 nodes and a quarter of `max_reference_mb`; on exhaustion the
previous conservative pair/component route remains. Provisional cross-scene
confirmation uses the same bounded proposal logic.

After a final subgroup repair, one retry uses the pre-merge witnesses with
canonical labels. It does not construct a new merged centroid, relax thresholds,
or use recovered queries as training evidence. Recovered memberships remain
attachment-only and require at least two independent timestamps. Matching policy
version changes invalidate previous derived decisions; VTT scans remain reusable.

### Face-rooted body perspectives

Face prototypes distinguish a measured neutral pose from unknown landmarks.
The bounded primary set gives a supported new direction precedence when it
covers additional witnesses. Reserve cohorts remain available in complete replay.

Body collection saves a small RGB scene signature and a conservative anatomical
torso direction per observation. Scene replay can recover a body appearance
using two disjoint short VTT tracks, each bracketed by strict face/body anchors
of the same owner. A cut, long gap, ambiguous competing body or contradictory
face ends the track. One shot contributes at most one witness. Recovered members
are auxiliary, and the proof retains the exact face roots and body witnesses.
A complete body fragment moves only when all its members agree on the same
target. This route works within a scene; global reconciliation uses the usual
original-witness safeguards across scenes.

### Retained VTT scene continuity

Local SapiensID paths compare adjacent cue evidence using shot appearance,
torso geometry and alternative owners. A change in VTT sampling cadence alone
does not cut a path or exclude a labelled holdout episode. An adjacent cue
can still contain a cut. Face roots on either side of a path can identify it
when the strongest admissible path is separated from its rivals.

The fragmentation report includes recorded face/body and body/body split
examples and pending global merges with their required scene selections.
Replaying that selection revalidates current evidence. The optional solver
comparison `--compare-scene-strategies` tests families on training goals
within `--max-trials`; provide a budget above 14 for independent finalist
tuning. The holdout reports back, contact and entrant coverage separately.


The shipped run and solver profiles enable local SapiensID tracks, independent face endpoints,
evidence-weighted scene membership, face deduplication and adaptive face views. Their RF-DETR/PMPose
pipeline also enables duplicate-pose suppression and SAM-pose2seg refinement with masked identity
crops. The updated body collection contract recollects selected cues on the first run. A one-ended
body path can query an independently face-rooted look as auxiliary evidence. Global joins require
current independent face endpoints, bounded face nominations and revalidation of co-presence and
original body witnesses. The merge proof records its source scans and a deterministic example.

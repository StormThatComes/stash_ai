# Recognition reports and exports

The merge inspector's **Build a View and test its downstream matches** scenario
uses the same bounded formation diagnosis and proposal generator as the solver.
It lists independent witnesses, missing admission/authority/support/cohesion
conditions, and combined formation/attachment experiments. Its local/global
modality scenarios also share the solver's generator and keep the loaded global
similarity floors. These are replay hypotheses; a sampled cohort does not certify
an emitted View or a successful merge. See [mechanism experiments](solver.md#mechanism-construction-experiments).

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Offline identity face export](#offline-identity-face-export)
- [Two-identity merge parameter report](#two-identity-merge-parameter-report)
- [Offline scene fragmentation report](#offline-scene-fragmentation-report)

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

Each face/person crop also gets a `*-thresholds.png` companion with the retained
overlay above a readable admission table, and a `*-thresholds.json` sidecar.
`image.thresholds` links both files; `admission` in the manifest contains the same
checks. The clean and landmark-only PNGs retain their original pixels. Missing
crops still get the JSON explanation, without inventing an image.

The table shows the **measured value, threshold, comparison and pass/fail/unknown/
disabled status** for each gate. JSON additionally records the signed margin and
`parameter_requirement`: for a 72 px face, the size gate needs
`detection.min_face_pixels <= 72`; a detector score of 0.91 needs
`detection.score_threshold < 0.91` (strict comparison). These are boundaries for
individual gates, not a sufficient recipe for detecting or assigning that sample.

Detection settings are verified against the retained scan/person contract hash.
If the supplied configuration differs, the report searches saved run snapshots
for matching settings. A match verifies the settings, not which run created the
observation. If no matching snapshot exists, the table explicitly says
`current_configuration_only_historical_unavailable`. Anchor/auxiliary quality
checks always use the supplied configuration and do not rewrite the recorded role.
Faces show confidence, source pixels, sharpness, completeness, head pose,
visibility and geometry admission. Person evidence shows detector confidence and
the body anchor size gate; independent body observations additionally show
collection pixels/sharpness/keypoint count/overlap and body anchor quality.
Face-associated person evidence inherits the face gates and does not require the
independent body collection floors. Group results cover only the displayed quality
gates; support, matching, reference selection and association ambiguity can impose
additional requirements. NMS/result limits are frame context, not single-crop
pass/fail checks. Raw per-keypoint confidences were not retained, so their threshold
is displayed with `unknown`, never reconstructed from accepted coordinates.

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

The report also writes `*-annotated.png` copies using the same colors and drawing
code as `audit-alignment`: yellow boxes, cyan eyes, red nose and green mouth
corners. Face copies show the five retained face-detector landmarks. When a
retained person crop and compatible SapiensID evidence are available, its
`person-*-annotated.png` copy shows 17 magenta COCO pose points (indices 0–16)
and the five face points (17–21). Coordinates account for the crop's source
offset and any face JPEG downscaling. Points outside the crop and missing pose
measurements are omitted, never clamped to the border.

Each image's `annotations` entry links the copy, lists the drawn indices and
records its status. Older evidence without crop bounds or pose geometry keeps
its clean PNG with an explicit unavailable annotation status. The clean PNGs
and summary's face-export counts retain their existing meaning. Overlays show
recorded source measurements; they do not recreate SapiensID's canonicalized,
square-padded model input or trigger another scan.

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

For multi-set solving and JSON interpretation, see [merge goals](solver.md) and
[solver reports](solver-report.md).

Each identity manifest sample includes an `explanation` for the recorded face
or body assignment. It distinguishes the active configuration, evidence from
another matching policy, and membership with no retained decision. References
are typed face/body observation IDs. Track endpoints and the sighting between
them are labelled separately. A recorded predecessor links to the applicable
chain in the report's `identity_merges`; unrelated family merges are omitted.
Missing measurements and references remain unavailable.

### Desktop identity inspector

Open an English-language window on Windows or Linux with a graphical desktop:

```sh
python stash_ai.py recognition inspect-identity --performer-id 123
python stash_ai.py recognition inspect-identity --identity-id ID --observation-id OBSERVATION --faces-db faces_state.sqlite3
```

The scene and zero-based cue filters can be combined with an exact observation
ID. Select a result to see its stored crop, explanation and related merge.
Both desktop inspectors list only observations from each scene's active scan
matching the loaded YAML's detection settings, and body observations matching
its current collection settings. Retained historical evidence remains available
to reports and recorded reference images, but is not a current result.
The reference selector steps through recorded witnesses and track endpoints.
Each action uses a short read-only snapshot. Paging retrieves metadata only;
selecting a row reads its images on demand. Nothing is exported. Missing crops
or references remain visible as such. Tkinter and Pillow ImageTk are needed
only for desktop viewing; check the installation with `python -m tkinter`.

New matches record the historical references used before a group was updated,
the measured minimum similarity, applicable thresholds, and the scored rival
when available. A seed has no matching reference. Replay cluster IDs and
resolved identity IDs are labelled separately. Body banks may retain a bounded
list of example witnesses alongside the total reference count. A certified
lower bound is displayed as a bound; exact scores are shown only if measured.
Shared face-merge evidence is stored by content address and resolved on read.
Older decisions are still readable. The next selected rebuild reuses compatible
scan embeddings and populates the additional evidence.

### Desktop scene inspector

```sh
python stash_ai.py recognition inspect-scene --scene-id 123
python stash_ai.py recognition inspect-scene --scene-id 123 --scope SCOPE --face-config facial_recognition.yml --faces-db faces_state.sqlite3
```

The English-language Windows/Linux window opens a collapsible tree. Each
identity with accepted face or body membership in the entry scene has a branch
labelled with its Stash performer ID. Each identity contains separate collapsible
`Face` and `Body` groups. Expanding either group loads its recorded membership
states: `anchor`, `auxiliary`, `attachment` or `unknown`, where present. State
counts cover **current member observations across all scenes**, and each state
loads independent pages of 80 on expansion. Filtering happens before pagination.
Face states use the same decision categories as the detail panel: accepted
current-policy evidence takes precedence, merged predecessors are included, and
missing or conflicting evidence remains `unknown`. Body states use the recorded
member role. These are recorded roles, not eligibility under today's YAML. IDs
without a linked Stash performer are explicitly labelled `no Stash performer`.
The number in each branch title counts its observations in the entry scene only. Blue rows mark
that scene. The remaining rows provide the performer's wider recognition context.

Choose **Compare performers** in a scene with an assigned identity. Right-click a
performer or one of their observations and choose **Compare with…** to preselect A.
Use **Find B by ID…** to compare with a current Stash performer or identity UUID
from another scene in the same scope. Select a route to see its measured gates
and first blocker. **Inspect evidence** opens the recorded pair of cues with
geometry and available masks. Closing it returns to the same option.
Select A and B, including unpublished identities, then choose **Analyse paths**.
The comparison uses the loaded YAML and a consistent read of the recognition database.

- Face anchors, confirmed views, hybrid and independent body consolidation use catalogue evidence across the scope. Each route shows its blockers, measured scores where available and any additional identities in a verified component.
- Local routes consider face nominations, body-core formation and recorded face/body or track links in the entry scene. A directional route can demote A's anchors to auxiliary eligibility while retaining an eligible witness for B, or test the opposite direction. When both modalities can create separate cores, the view also offers combined face and body options. Support alternatives use independent evidence.
- Expand a route and select a YAML option to see the current and proposed values. **Copy YAML patch** copies that option in the loaded schema version. These settings affect the entire run.
- **Verify option by replay** runs the baseline and selected change against retained scans in the scope. It can take as long as a rebuild. The result shows whether both sets of observations reached one destination, how many were withheld, and which other identities shared their destinations. A frozen-planner result or a demoted anchor alone cannot establish that outcome.
- **Choose C…** adds an optional protected performer. Success requires every current selected observation of A and B to reach one owner and every observation of C to reach one separate owner.
- **Observation changes…** compares the baseline replay to the selected variant. Filter owner changes, anchor-to-auxiliary transitions or withheld observations, then open the recorded cue. The summary identifies a consolidation or an avoided core when the decision history ties it to the pair. Some runs retain only the final assignments.
- **Compare YAML variants…** accepts multiple options (Ctrl/Shift selection). Identical patches share one trial. **Entry and identity scenes** selects the entry scene plus all current scenes containing A, B, C or an automatic control for the selected options. Other scenes remain frozen history. **Full scope** replays every current scene in the selected scope. Preflight counts scenes and face/body observations before a replay starts.
- Compare the complete join, protection of C, unassigned observations, additional identities, patch size and elapsed time against one shared baseline per experiment. Open a row for its observation changes. **Validate on full scope** reruns that exact option with a full-scope baseline.
- **Minimize selected patch** starts with a successful option, removes settings and probes stricter thresholds using production replays. Every accepted probe preserves the join, the optional C constraint, and the original option's limits on collateral identities and coverage loss. The probe budget bounds additional trials. **Copy tested YAML** copies the selected result in the loaded configuration format. Quick results apply to their selected scenes.

The pair comparison also evaluates these strategies.

| Strategy | Evidence and replay |
| --- | --- |
| Follow a body through a turn | Follow a physically separated body path between compatible face endpoints in one shot. Inspect the first similarity blocker and the alternative body at that cue. |
| Resolve a crossing | Compare both endpoint assignments around a contact, the co-present partner and the interior assignments supported by the two-person tracker. Ambiguous face endpoints remain visible. |
| Prevent a birth and recover its samples | Combine a directional face/body core demotion with measured attachment gates in one replay. Inspect the earliest recorded seed/core member when available. |
| Isolate an influential anchor | Compare single face references and repeated-crop groups on frozen cores. Replay the best candidate as an auxiliary-only query. For local body cores, probe each endpoint of the weakest cross comparison and its repeated appearances. Diagnostic trials have no YAML to copy. |
| Bridge appearances through a View | Inspect A/intermediate, intermediate/B and direct A/B witnesses, their support, original core minima and the closest outside View. A bridge involving a third owner protects that owner during replay. |
| Compare local/global modality order | Try local body paths followed by global face/View matching and the converse local-face/global-body option. Inspect the recorded merge route at the configured global similarity floors. |
| Recover an unassigned bridge | Follow a shot-local path through currently unassigned body observations. Replay reports each watched observation's destination and the complete A/B outcome independently. |
| Challenge a strategy with its closest rival | Face centroids, confirmed View medoids, measured hybrid/body competitors and the co-present partner provide route-specific controls. Every observation of each selected control must remain at a single separate destination for that option to succeed. |

Expand a strategy to inspect its evidence and then select **Verify option by replay**.
**Compare YAML variants…** includes diagnostic replays and shows the first changed
A/B cue, its recorded decision, pair coverage and extra owners for the selected
result. Replays use retained VTT inputs. A frozen diagnostic reports what its
reference comparison established, while a replay reports final assignments on
the selected scenes.

`Unassigned in this scene` includes every current face without membership and
every current body without accepted membership, including withheld and
provisional tracks. An accepted auxiliary or attachment is already assigned
and does not enter this group. Accepted internal identities without a Stash ID
remain in their labelled identity branches. The unassigned group is local to
the entry scene, with only the configured body collection contract included.
Counts, search, review queues, summaries and direct observation navigation use
the same current-result filter. Changing YAML without rerunning recognition does
not make older observations current: incompatible face scans require a rerun,
and bodies collected under older settings are omitted.

`Load / refresh` rereads the selected YAML and validates it, preserving explicit
`--faces-db` and device overrides. Restored selections use the new configuration;
an obsolete selection is not reopened. Decision-chain and quality-gate labels
use the loaded YAML version's paths: for example, v2 shows
`body.collection.min_keypoints` rather than the internal
`sapiensid.collection.min_keypoints`. Numerical and non-configurable geometry
checks remain unchanged.

The unassigned tree is grouped as `Face` / `Body`, then by the displayed `State`.
State groups use the retained rejection reason (such as `blurred` or
`person_blurred`) when present, otherwise the membership state (such as
`unassigned`, `provisional` or `withheld`). Faces with a withheld decision for the
current matching policy also appear under `withheld`; historical attempts alone
remain `unassigned`.

`withheld` expands into recorded decision reasons, for example
`Body → withheld → body_anchor_keypoint_rules` or
`Body → withheld → unconfirmed_contact_face`. State and reason groups show counts;
reason groups are ordered by count, then name. A missing or invalid reason appears
under `unknown_reason`. Each reason loads its own pages of 80, and refresh/back/forward
preserve independently opened reason groups. Other states load observations directly.
Only states and reasons present in the entry scene are listed; empty kinds show an
explicit message. Counts and pages share the same filters. The reasons describe
retained decisions, and Details still identifies historical body decision policies.

Select an unassigned observation to inspect its crop, retained rejection,
matching attempts and available references. The detail panel also compares
retained measurements with detection/collection and anchor/auxiliary quality
gates, using the same admission projection as the identity report. Detection
settings are hash-verified when possible. Otherwise the UI labels them as
current YAML only. Current quality eligibility does not reconstruct a historical
matching decision. `Recorded evidence` opens the retained matching proofs.

For `unconfirmed_contact_face`, Details explains the recorded contact check:
crowding measurements, original head points and landmark error, head/torso
geometry, peer mask support/conflicts, and adjacent-cue witness rejections.
Missing clean neighbours, missing torso/shot evidence, low body similarity,
insufficient rival margin and temporal identity conflicts are reported separately.
Retained supporting and rival body IDs are available as references. Accepted bodies
also show whether same-cue anatomy, adjacent witnesses or independent clean
endpoints confirmed their contact link. This check precedes later track recovery;
the final membership remains the decision shown above it. Old decisions explicitly
report unavailable contact diagnostics instead of reconstructing an explanation.

If the full head/torso path fails, Details also shows the masked upper-body path:
aligned original eyes/nose, anatomically eligible ears, shoulders, excluded points
and their provenance, head/shoulder geometry, own-mask point/corridor coverage and
the largest gap. Per-peer checks distinguish a remote box intersection from a
missing nearby mask, competing head claim or foreign foreground in the head
region. These measurements are retained even when the head-point count fails,
so an untested mask is not confused with a rejected mask. `same_frame_upper_body`
identifies links confirmed with both shoulders. `same_frame_single_shoulder`
identifies the alternative with one original shoulder, independent original head
evidence, all five face landmarks inside the own mask, and connected mask support.
Details shows the face-axis geometry and landmark coverage for that alternative;
the missing shoulder is never synthesized. Disabled contact masks name the exact
recorded switch separately for the selected body and its peers. Suggested next
steps can enable `pose.mask.apply_to.contact_confirmation` and mask generation
without changing `body_admission` or `overlap`. If current YAML already enables
them, the suggestion requests fresh evidence instead of another YAML change.
Historical decisions remain historical; no missing mask or geometry is inferred.
Recollect affected scenes with recognition
after updating; old assignments and solver replay cannot invent the new proof.

For an unassigned observation, `Suggested next steps` precedes the recorded
evidence. It shows current YAML values, measured boundaries, copyable YAML
patches and the next run needed to test them. The patch uses the loaded YAML
layout (v1 or v2), and must pass configuration validation. Detection/collection
blockers come first and require fresh scan evidence. Matching or sample-role
changes require rebuilding assignments with the relevant scenes selected.

Alternative paths can preserve anchor requirements and try an auxiliary
attachment, qualify the sample as an anchor, or collect more independent
evidence without relaxing settings. Current-policy face candidate scores,
body-core support and frozen-body-bank scores can yield measured
support/similarity proposals.
Ambiguity, co-presence conflicts, missing measurements and historical matching
policies do not produce guessed threshold values. Regional keypoint rules are
explained without rewriting their `all`/`any` lists.

These are first-step hypotheses from retained evidence, **not replay-verified
assignments**. Clearing a quality gate does not prove identity ownership, and
YAML settings affect other observations too. The viewer does not apply patches,
run models, change memberships or bypass face/identity guards.

Assigned observations in both `inspect-scene` and `inspect-identity` show
`If this assignment is incorrect`. This offers alternative YAML changes to test
when you identify an incorrectly attached sample. Current-policy direct face
matches, anchor-view matches and measured body-bank/core attachments can suggest
stricter similarity, weakest-witness cohesion or rival-margin boundaries. Hybrid
query suggestions distinguish face and body scores; local body tracks use their
recorded weakest link to a root. An
inclusive similarity gate must be raised **above** the recorded score; equality
still accepts it. Same-scene face proposals account for the match bonus. Accepted
body cores can suggest more independent support, with proposals capped at 8.

Each alternative shows current values, a validated v1/v2 YAML patch, measured
evidence and the next run needed. Quality alternatives explicitly distinguish
excluding a sample globally from merely removing anchor eligibility: an auxiliary
can remain attached. Detection/collection changes require fresh scan evidence.
Regional keypoint rules are not rewritten. Test alternatives individually and
check other members for unwanted fragmentation after rebuilding.

The advice does not classify assignments as incorrect or guarantee detachment.
Another route may reattach the observation. Inherited merge membership directs
you to the merge chain, and face-associated bodies direct you to the linked face.
Historical/missing matching proofs, candidate seeds and certified similarity
lower bounds do not yield invented numeric matching thresholds.

The scene catalogue uses the current scan head, including a partial scan whose
status is shown. Older superseded scans do not populate the unassigned group.
Changing the Scene ID or using `Load / refresh` rebuilds the tree. Ambiguous
Stash scopes require `--scope`. Lists load metadata only and crops are decoded
on selection, using read-only SQLite snapshots and background workers.

Only observations retained in SQLite can be shown. Detections discarded before
storage, pruned observations and absent measurements cannot be recovered by
this viewer. Routine inspection reads retained evidence. The optional FaceOcc
replay below runs only for the face you select.

When a selected observation or reference has no readable stored crop,
`inspect-scene` now reconstructs its preview from Stash VTT. This includes
unassigned faces and bodies rejected before crop encoding. The existing
`STASH_URL` and `STASH_API_KEY` select the source instance. Opening the tree,
expanding groups and paging never contact Stash. Reconstruction starts only
when an observation or reference image is requested, on the detail worker.

The preview uses recorded geometry and is labelled `Reconstructed from VTT`.
Stash scope, the VTT/sprite fingerprint and cue must match the retained scan.
An existing pixel signature is also checked. Rejected detections commonly have
no crop signature; these previews explicitly say that only the source was
verified. Changed or unavailable sources, invalid bounds and signature
mismatches leave an explanatory placeholder while preserving decision details.
The current source configuration must select the same VTT cues as the original
scan. One source is reused in memory, with a sprite cache capped at 64 MiB.
Reconstructed previews are not written to the recognition database or a media
cache and do not change assignment, rejection or quality evidence.

Both inspectors enable `Show geometry` by default for selected observations and
reference previews. Yellow marks the face box and five retained facial landmarks;
green marks the body box and available COCO17 pose points and skeleton. A thicker
box and the caption identify the selected observation. Only explicitly linked
face/body geometry from the same scan and cue is added. The current collection
contract is preferred; ambiguous linked poses are omitted instead of combined.

Overlays are drawn on copies, mapping source coordinates through recorded crop
bounds and image scaling. Missing/outside points are omitted. A retained crop
without reliable source bounds stays clean with an explanation. VTT previews
use the verified reconstructed crop transform, after any pixel signature check.
The checkbox switches between annotated and clean pixels without database,
network or model work. Crops and recognition evidence are never modified.

In `inspect-scene`, select a face (including an unassigned `face_occluded`
observation) and click **Face visibility…**. The two synchronized panes show
the original VTT cue and the selected face on the same cue. The gold outline
marks the selected measurement region. Green marks counted FaceOcc pixels.
The region selector defaults to **Measured mesh** (or **Measured ellipse** when
explicitly configured). **Landmark ellipse (comparison)** and **Legacy box ellipse
(comparison)** are optional comparisons on the same logits. Only one region can
be selected; each newly opened observation resets to the measured region.
A comparison is unavailable if its geometry is invalid or extends beyond the
model crop. It cannot veto or replace an accepted mesh.

**Mesh landmarks** shows points from an accepted mesh. The caption identifies
both the requested measurement and the selected preview, plus the fraction of
real-source FaceOcc positives excluded by that preview. A failed mesh shows its
reason and the original image without drawing an ellipse or inventing a score.
Preview choices never change saved results.

**Show predicted face outside region (purple)** shows positive FaceOcc pixels
excluded by the chosen measurement region. These pixels do not enter visibility;
nearby skin can still participate in the separate sharpness mask. Purple may
also include false positives or another face. Switching regions, points and
pixel thresholds reuses the same predictions without further model inference.
Use the mouse wheel to zoom and drag either pane to pan.

The **Pixel logit threshold (preview)** slider starts at
`face.quality.faceocc_logit_threshold` from the loaded YAML. Drag it left to
include more pixels, or right to include fewer. Use the left and right arrow
keys on the slider for 0.01 steps. The green overlay and replay
fraction update from the same score map, with no additional VTT or model calls.
**Reset to YAML** restores the configured value. To use a preview value in
recognition, set it in your YAML and rerun the affected scenes.

The window displays the recorded `visible_face_fraction` alongside a fresh
FaceOcc result using the current configuration. The replay shows the visible
pixel count divided by the full selected-region pixel count in the 256 × 256 model
input. Source coverage shows how much of that region fell inside the cue.
A clipped region can lower the score even when all its in-frame pixels are
green. If alignment, minimum source coverage or model loading fails, the
window gives the reason. Mesh mode never substitutes an ellipse; the explicit
ellipse mode can retain its geometric outline as a diagnostic.

Replay requires access to the verified VTT source and uses the configured
device and FaceOcc asset download policy. It also works for observations
without a stored crop or score. Browsing other observations does not trigger
FaceOcc. Replaying a face leaves the saved score and recognition state intact.

For fully offline browsing, retain the earlier behavior with:

```sh
python stash_ai.py recognition inspect-scene --scene-id 123 --offline
```

`inspect-identity` remains offline. `report-identity --restore-crops` continues
to require retained pixel signatures for its exported identity evidence.

Inspector loading uses separate bounded metadata, observation-image and reference
workers. Metadata appears before network previews. New selections cancel queued
work and interrupt SQLite reads; in-flight HTTP requests have a bounded timeout
and obsolete results are discarded. Closing the window does not wait for HTTP.
`Retry images` retries only previews. Read caches are bounded and keyed by the
configuration and database/WAL revision. Scene lists use continuation keys and
retain at most 1,000 observation rows; `Earlier observations…` restarts an evicted
branch without accumulating an unlimited Tk tree.

`Search / filters` searches the complete scene-inspector scope (all retained
scenes of its performers plus the entry scene's unassigned observations), before
pagination. Filters include observation/performer/scene IDs, kind, state/rejection,
reason, current/historical/missing policy, missing retained crop and maximum
recorded ranking lead. Sort by scene/cue, state, lead or sharpness. An absent
measurement stays unknown. Clear filters restores the grouped tree.

Image panels have independent Fit/100% controls, cursor-centered wheel zoom and
drag panning; `Sync viewports` links them explicitly. Refresh restores open groups
and the selected observation where still present. Back/Forward retains up to 50
session locations including filters and viewports. Selections beyond a loaded
page are restored explicitly without walking every intervening page.

`Full cue` restores the verified VTT frame and displays retained detections from
that exact scan/cue (current body contract plus the selected historical body, if
applicable). Double-click a detection to inspect it. Cue arrows and `Nearby cues`
follow actual VTT order and timestamps; they do not infer continuity or tracking.
Source and available selected-crop signatures are checked before displaying the
full frame. Offline mode keeps retained crops available.

`Gallery` pages the selected state or search result in groups of 12 thumbnails.
`Pin A → B` holds any inspected observation in the second panel while navigating;
`Unpin` restores recorded references. Pinned user comparisons are labeled separately
from decision evidence. `Open observation…` resolves an ID only within the active
scope/model. Geometry layers independently control boxes, facial landmarks, pose
points and skeletons. Enable **Body mask** in **Geometry layers…** to overlay the
recorded SAM quality mask on the crop or full cue. This works for accepted and
rejected body observations, including a body linked to a selected face. Each
body caption identifies its mask or gives the recorded reason it was unavailable.
**Mask not retained** means the pixels are absent. If that observation has
recorded SAM prompts and a mask measurement, enable `storage.retain_body_masks`,
run recognition for that scene, then choose **Load / refresh**. A recovered bitmap
is labeled **Replayed quality mask**. **No quality mask (no_reliable_prompts)**
means SAM had no usable prompts for that pose. Hover over points for source
coordinates and retained confidence. Missing masks or point confidence are
explicitly unavailable; browsing never runs models or changes recognition data.

`Decision details` presents quality gates, retained matching stages, evidence IDs,
merge edges and final membership as a tree. Each stage identifies its configuration
source. Double-click evidence to inspect it. Recorded reference-bank selection
and role are displayed only when retained; membership or an anchor label does not
prove actual use in later comparisons. Recorded rivals keep unknown IDs/scores
unknown instead of substituting a newly calculated candidate.

`Calculate candidates` explicitly compares retained embeddings in the same scope
and model. Face references use strict-role/quality admission and the production
diversity sampler; body references use a bounded deterministic cue sample of
accepted anchors from the current collection contract. The selected observation
and same-cue references are excluded from scoring. Mean/minimum/maximum cosine
similarities, reference count, current owner and clickable best witness are shown.
These are exploratory comparisons, not probabilities or a replay of matching:
tracking, contextual gates, support and complete body-bank construction are not
rerun. Missing embeddings require re-collection; browsing does not run models.
Configured reference/person budgets apply; limited results and database changes
are disclosed. Closing or cancelling interrupts the calculation.

`Review queue` scans metadata across the complete scene-inspector scope, retaining
the top 500 candidates and showing 80 per page. Every priority has explicit reasons:
retained rejection (+5), recorded lead at most 0.03 (+4), face sharpness below the
current role floor (+3), historical/missing decision (+2), unknown role (+1).
This is a review heuristic, not an error probability or an automatic exclusion.

`Unassigned summary` streams the entry scene's retained unassigned metadata and
shows state counts plus individual gate failures/unknowns, measured ranges and
distance to the nearest failed boundary. Gate counts overlap; alternative anchor
and auxiliary paths remain separate. Historical versus current thresholds are
labeled. Double-click a state to open its filtered observations. Both aggregate
views run only when requested and read neither images nor embedding columns.

**View / Hybrid evidence…** opens the measured appearance matrix. Select a cell
to inspect its lowest retained comparison. The rival belongs to the frozen
component. The formation tab accepts several faces from one performer and
counts independent witnesses in their coherent groups. This evaluates the
selected subset under the loaded strict-view policy. Hybrid displays the two
face crops and the two face-associated body projections separately. Missing
crops or limiting witnesses are labelled explicitly.

**Attachment scenarios…** tests A → B or B → A on a private snapshot. View
queries counterfactually withhold the source faces and keep other reference
owners as competitors. Hybrid query treats the source as a new birth and uses
production claim resolution, including its complete-birth requirement.
The body-track timeline shows entry-scene observations by cue, face roots,
recorded breaks, orientation and linked blockers. Select a dot or open its row.
After a full replay, **A residuals** and **B residuals** locate missing samples,
separate destinations and assignments shared with an outside identity.

After a pair replay, **Observation changes…** opens at the first differing scene
and cue (numeric scene ID, then cue index). All changed observations at that cue
appear together. Select a row to compare owners, roles, routes and retained
references side by side. **Later linked changes** follows recorded references
within each trial through the selected identities. **Changed** includes route
changes even when ownership stayed the same.

Choose a mechanism and **Test stage necessity** to replay the same YAML and scene
selection with that stage enabled and omitted. Hybrid identity consolidation,
Hybrid query attachment, confirmed View consolidation, global View queries, strict
core consolidation, scene anchor View queries and auxiliary View attachment can
be isolated separately. You can also omit both Hybrid stages together, or the
confirmed View passes together (including the global whole-View attachment).
Other historical and pooled recovery routes remain eligible during this check.
The result reports stage entries, the A/B join, protected
C, additional identities and unassigned observations in each trial. A necessity
finding applies to the selected goal, observations and YAML. Open any changed cue
from the comparison to examine the retained evidence.

### Desktop performer inspector

`python stash_ai.py recognition inspect-performer --performer-id 123` opens the
shared observation inspector with performer-wide counts, scene/role grouping,
face/body and role filters, and paged current memberships. Internal identity IDs
and `--scope` are also supported. Groups cover the displayed page; counts cover
all current accepted observations. `--offline` disables VTT preview reconstruction.
Right-click an identity in the scene inspector and choose **Inspect performer**.
**Inspect scene** opens the selected observation's scene. Crops, geometry, masks,
galleries, reference comparisons and decision details reuse the existing viewer.

The performer graph projects a single read-only SQLite snapshot into typed face,
body, decision and recorded-View-evidence nodes. Aggregate match scores belong to
decision nodes. Displayed assignment roles are separate from reference authority;
`strict_route` still requires quality/reference selection before actual use.
Recorded View nodes identify a policy and witness set, not an inferred catalogue.
Historical/missing evidence and external references remain explicit. Each page has
at most 100 current observations and 500 total nodes; omitted context is counted.
No pixels, embeddings, model loading or recognition writes are needed to build it.

**Graph…** opens the bounded animated graph for the current scene/kind/role
filters. Previous/Next traverses all current observations; each page includes
its retained reference context. Double-click a group to expand it, or an
observation to inspect it in the shared crop/detail panel. Historical references
open in reference panel B. Drag nodes to pin them, right-click to unpin; drag the
background to pan and use the wheel to zoom. Fit, Pause/resume and Reset layout
control a deterministic layout that settles after a bounded number of updates.
Refresh retains positions of surviving node IDs. Expand/Collapse groups changes
presentation only, never reference authority or recognition membership.

Graph diagnostics are explicit actions. **Why here?** highlights recorded upstream
evidence; **Uses this evidence** follows downstream pointers. Rival and newly
computed similarity edges never establish dependency. **Weak connections** marks
structural bridges in the loaded partial graph, not incorrect assignments. Missing
references and pagination can create apparent bottlenecks. Removal effects need a
recognition replay because alternate routes may still attach the samples.

**Compare similarity…** calculates bounded top-three cosine neighbours among the
current observations on this page, separately for face/body. Same-cue and known
identical-pixel comparisons are excluded. Missing/invalid embeddings remain
unavailable. Dashed edges are new comparisons; their score is not a probability or
historical evidence. **Clear similarity** removes only that exploratory layer.
**Compare performer…** opens the existing merge-path analysis for another current
identity in the same scope/model, using a retained scene as the replay context.
View details show independently recorded support counts separately from retained
examples; multiple Views in a single decision are not fused into a fictitious View.

**Graph thumbnails and loading:** opening or refreshing the graph first shows a
cancellable loading panel. Evidence preparation uses an indeterminate indicator;
thumbnail decoding then reports completed observations / total, including missing
crops. Tk image preparation runs in short UI batches. The graph appears once the
current page and its retained reference thumbnails are ready; physics starts then.
The rest of the inspector remains responsive. Cancel restores the previous page,
if any; Refresh retries. Closing cancels work without waiting for completion.

Observation nodes display aspect-preserving retained crops up to 64 × 64 pixels.
Collapsed groups show one explicitly labelled **sample**, not a new portrait or
View medoid. Expand a group to see its observations. Role-coloured borders and
face/body labels remain visible; **Thumbnails** switches to the symbol view without
reading pixels again. Dragging, physics and zoom reuse the same images. Thumbnails
keep a readable screen size while zoom changes node spacing.

Only crops already retained locally are loaded, for the bounded graph page and its
references (at most 500 nodes). Missing/unreadable crops keep their symbol; hover
shows the reason and the loading summary counts them. This preload never fetches
Stash media or runs inference. A selected observation can still use the existing
verified VTT detail preview. Full-resolution crops are discarded immediately after
downsizing; the next page replaces the thumbnail set. A database revision change
invalidates the entire pending load rather than mixing images and stale evidence.

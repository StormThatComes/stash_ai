# Recognition reports and exports

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
The reference selector steps through recorded witnesses and track endpoints.
Each action uses a short read-only snapshot. Paging retrieves metadata only;
selecting a row reads its images on demand. Nothing is exported. Missing crops
or references remain visible as such. Tkinter and Pillow ImageTk are needed
only for desktop viewing; check the installation with `python -m tkinter`.

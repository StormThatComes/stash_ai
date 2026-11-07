# Reading merge solver reports

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

## Reading the solver JSON

`report-merge --solve` writes a compact diagnostic report to `--output` or stdout.
Start with `merge_diagnostics.sets`. Each entry names the performers, the first
trial that merged the set, and the stages and reasons most often observed when
it remained split. Counts distinguish trials from occurrences across replays.
Use `baseline_state` and `selected_state` as indices into `merge_diagnostics.set_states`
to compare coverage, fragments, withheld samples and blockers. Group indices are
zero-based (console “Set 1” has index 0).

Every completed trial under `simulation.trials` keeps its search parent, strategy,
changes since the preceding trial, input-relative `yaml_patch`, unmet conditions
by reason, and references to the set states it produced. Baseline, winner, best
trial per set and the first trial exhibiting a new per-set blocker also carry
sample assignment counts and bounded diagnostic examples when available.
`measured_query` examples show observed face/body scores and the thresholds
tested. `proposed_route` gives a suggested change, which may never have been
tested. Certified input problems are in `feasibility.groups[].input_certificates`.

The compact schema is identified by `solver_report_format: compact-v1`.
`best_result` contains the selected configuration changes. To export the previous
detailed schema, including individual observations and full decision traces, use
`--solver-report full`. The report mode applies only to the final JSON. Solver
checkpoints keep the evidence needed for resume, and `--yaml-output` exports
the winning configuration in either mode.

Repeated `--performer-id` or `--identity-id` selects one set. The original
exactly-two-ID diagnostic remains available without `--solve`. Set syntax is
mutually exclusive with those selectors. Alias overlap across sets is rejected.
Without `--solve`, set input resolves retained goals; `--simulate` evaluates the
baseline. No Stash writes, model loading, media access or source YAML writes occur.

Use repeatable **`--exclude-performer-scene PERFORMER_ID:SCENE_ID`** to tell the
solver that a declared person is absent from a specific scene:

```powershell
python stash_ai.py recognition report-merge --performer-set 47036,47034,47037 --exclude-performer-scene 47036:1234 --exclude-performer-scene 47036:5678 --solve --face-config input.yml --yaml-output corrected.yml --output corrected.json
```

The performer must belong to one of the selected goal sets (merged predecessor
IDs are resolved automatically). Because a set declares one person, the exclusion
applies to the **whole set**, including its merged or split destinations. A
single-performer set is supported. The excluded scene must have a complete latest
scan compatible with the input configuration; missing or incompatible evidence
is an input error, not proof that the person is absent.

An explicit negative overrides historical positive memberships in that scene for
this set only. Those observations are removed from its frozen preservation/core
goals, including under `--goal-policy strict`, but remain in the production replay
as ordinary evidence. Other scenes remain positive obligations. A fragment left
without positive observations is listed in `excluded_members`; a whole set left
without positive evidence in another scene is rejected. Thus an erroneous existing
scene link can be corrected without asking the solver to preserve it simultaneously.

Every trial checks all final face and accepted independent-body assignments in the
excluded scene. Frozen positive observation addresses and linked accepted body
alternatives track the person through rebuilds; UUID spelling and reuse after a
split do not identify a person. A different UUID, a body-only attachment, or a small leftover fragment
cannot bypass the exclusion. Linked bodies do not add positive consolidation votes.
Losing the person everywhere cannot satisfy the negative goal. Violations make a
trial inadmissible under both policies; among inadmissible trials, fewer unsatisfied
exclusions and forbidden assignments rank ahead of consolidation. Optional outside
impact bounds retain their existing priority.

Bounded face, hybrid-person and independent-body similarity measurements propose
tighter rejection thresholds; full production replay decides whether they preserve
the correct appearances. Reports and console progress show each exclusion's
`satisfied`, `matches`, missing observations and target representation. Exclusions
participate in checkpoint compatibility and report provenance. `--simulate` checks
the baseline, including with exactly two performer selectors.

These labels guide **offline configuration search**, not a runtime scene blacklist.
The exported YAML contains ordinary recognition parameters, with the result verified
on the retained snapshot. `best_effort` can still violate an exclusion; inspect the
reported unmet conditions. Counts cover every observation; JSON observation examples
are capped at 32 per condition to keep trial histories bounded. Recognition must be run with the selected YAML to apply
it, and changed media/evidence requires revalidation.

Use repeatable **`--exclude-performer-observation PERFORMER_ID:OBSERVATION_ID`**
when only a particular face or body sample is wrong and the person can still
appear in that scene. Copy the exact `observation_id` from the corresponding
`samples` entry in a `report-identity` `manifest.json`; the PNG's `face-000001`
or `person-000001` suffix is an export counter, not the observation ID.

```powershell
python stash_ai.py recognition report-merge --performer-set 47036,47034 --exclude-performer-observation 47036:724cd36c-2b66-45bd-b287-e4505417b6cb --solve --face-config input.yml --yaml-output corrected.yml --output corrected.json
```

The exclusion applies to the whole goal set after performer alias resolution.
It removes only that sample's retained face and compatible linked body addresses
from positive goals; other observations in the **same scene** remain positive.
A face+body association is checked in both directions, so switching from its face
to its body cannot bypass the constraint. Independent body-only samples are also
supported. Face/body forms of the same exclusion are deduplicated. These addresses
remain in the real replay and can join another person or remain unassigned.
Changing a cluster UUID or losing the target person cannot satisfy the exclusion.

Unknown IDs, another scope/recognizer, historical scans, incompatible scan/body
contracts and disabled body identities for a requested body sample are input errors.
After a rescan, regenerate the identity report and select its current observation
ID. A set left without positive evidence is rejected. During evaluation a missing
negative observation is a failure, not a successful exclusion. All goal policies,
including `strict` and `identity-physical`, enforce this rule.

Observation and scene exclusions can be combined. Both contribute to solver
ranking, threshold proposals, checkpoint compatibility and JSON/YAML provenance.
The `observation_exclusions` trial results list `requested_observation_ids`,
`satisfied`, matching/missing counts and final `assignments` (assigned, unassigned
or missing, with destination identity IDs). Assignment examples are capped at 32;
`assignments_truncated` marks a longer list. Console progress includes each
observation exclusion. As with scene exclusions, these are offline goals rather
than runtime overrides: only a full replay validates the exported ordinary YAML.

Each trial starts from the same read-only retained history in an independent RAM
database and calls the ordinary production rebuild. All compatible complete latest
scans in the scope participate, including competitors. `--solve` excludes scenes
whose latest scan is incomplete or incompatible with the current scan policy,
instead of aborting all trials because of one unavailable scene. Replay never
substitutes an older compatible scan for an excluded current scan. Excluded scenes remain in frozen
history and go through the ordinary production reference admission rules.

Latest means the scan selected by the most recent `run` that checked that scene's
current media and scan settings, including cache reuse. For example, running
configuration A, then B, then A again makes the cached A scan current without
recomputing its embeddings. Historical insertion order and identity origins remain
unchanged. A newly created scan supersedes that selection even when incomplete.
Existing databases need one run with the intended configuration to record a
previously reused scan as current; no evidence deletion or schema migration is needed.

`simulation.input_selection` lists every excluded scene and scan, its reason,
completion counters, and stored/expected policy hashes. The console announces a
partial input scope before baseline and lists blockers at the end. Missing goal
observations always appear in raw coverage diagnostics; whether they violate a
goal depends on the policy below. Empty performers cannot satisfy a goal
vacuously. Goals may be fully solved on the available selection while unrelated
scenes are excluded; this does not certify a run including those excluded scenes.

The scan policy covers detection settings, `execution.max_cues`, and whether
visibility measurements are required. Changes to `sapiensid.matching.*` alone do
not invalidate a scan. When inputs are incompatible, use their original scan
settings or run recognition on the reported scenes with the current configuration.
Incomplete scans require finishing collection. If no compatible complete latest
scan is available, the solver writes a `no_compatible_scans` diagnostic report,
runs no trials, exports no YAML, and exits 2. Plain `--simulate`, evaluation and
threshold tuning retain their requirement that the whole requested selection be
compatible; their error now identifies the affected scenes and reasons.

The default **`--goal-policy identity`** (version 3) treats each set as one declared
person and previous memberships as noisy fragments. It preserves that person's
appearances in the set's original scenes and rewards consolidation, without
joining different sets' substantial core evidence. A weak fragment may disappear
or split if other fragments of the same set still cover its scenes. Consequently,
`represented=19/20` can beat a fragmented `20/20` baseline and can even accompany a
full solution. Destination IDs and input performer order do not matter.

Goals freeze evidence roles, independent units and thresholds **before any trial**:

- Core evidence has production strict-reference authority, passes the input YAML's
  anchor eligibility rules, and comes from complete, non-rejected observations.
  Reference eligibility is a confidence hint, not a verified identity label.
  Other memberships, including auxiliary and contextual attachments, are soft.
- Evidence uses the existing independent timestamp/crop matching. Repeated pixels
  do not gain votes; duplicate copies assigned to different destinations vote for
  neither. Retained raw sample diagnostics remain separate.
- Core units are pooled across the set before independent counting. Only a set
  with no eligible core falls back to all its membership evidence. An auxiliary-only
  fragment cannot impose an additional core quorum on a set that has references.
- Every original scene needs at least one independent membership unit retained in
  a destination owned by this set. Another input fragment can supply that unit;
  another person merely appearing in the same scene cannot. Different destinations
  may cover different scenes in a partial solution. This includes soft-only scenes
  and historical scenes missing from the replay selection: losing the only link to
  one is an unmet condition. An entirely empty set cannot pass; an empty historical
  fragment within a nonempty, covered set does not independently veto progress.
- A destination belongs to its sole substantial core owner, or, when there is no
  such owner, to the sole set contributing original memberships there. An ambiguous
  soft overlap cannot establish scene coverage for either set.
- Full consolidation requires **one destination** covering every original scene
  and retaining `max(min(2,n), n-max(1,floor(0.2*n)))` of the **set's** `n` frozen
  core units. Thus 1/1, 2/2, 2/3, 4/5 or 8/10 suffice. Scene coverage alone cannot
  certify two substantial core fragments as merged. Raising trial admission
  thresholds cannot remove evidence from the frozen goal.
- Soft units retained in any destination of the same set are not lost. Missing,
  withheld, ambiguous or foreign-set units contribute to set-level soft loss.
  The allowance `min(n, max(3, floor(0.2*n)))` now applies to pooled soft units and
  is diagnostic: both soft loss and excess are late tie-breakers after consolidation,
  never preservation gates. Dropping soft evidence cannot improve the frozen
  common-destination score; ties prefer retaining it.
- A destination conflicts across sets when, for a scene in each set, its core
  support there **exceeds that scene's allowed outlier count**. Small noisy
  core minorities and soft corrections do not alone prove an identity collision.
  Auxiliary-only fragments supply conflict fallback only when the whole set lacks
  core. Pooled independent scene cohorts prevent a large scene from diluting a
  smaller scene's substantial collision; repartitioning historical performers
  cannot change ownership. `strict_disjoint` and
  `strict_conflicts` still expose every raw cross-set overlap.

These are explicit heuristics, versioned in `goal_policy`, not verified truth
labels or a guarantee that minority assignments are errors. Fully solved under
this policy means preserved, consolidated identity evidence within these limits;
it does not mean perfect assignment of every sample. Feedback query labels use
frozen per-scene scoring evidence, including membership fallback for a soft-only
scene, so the search can still propose recovery of such a scene. Rebuilt set
ownership follows substantial core support or exclusive memberships without a
core owner. Production matching continues to process all samples.

Admissible candidates preserve every set's scene coverage and keep the sets
disjoint under the chosen policy. They always outrank
inadmissible candidates; full success outranks partial success. If none qualify,
the solver still selects the best completed trial, including baseline, as
`best_effort`. Identity-mode ranking then prioritizes separation, worst/mean
scene coverage and fewer cross-set core pairs. Next come worst/mean set shared
coverage and scene-balanced pair consolidation. Shared coverage weights original
scenes equally, using their frozen reference units (membership fallback in a
scene without references); removing samples never shrinks the denominator.
Pair consolidation weights each ordered pair of different scenes equally; a
single scene uses its independent-unit pair fraction.
Soft losses, budget excess and pooled core retention break remaining ties.
Member-level `represented`, `consolidated`, `identity_components` and
`core_shared_coverage` remain diagnostics and do not veto a better set result.

Optional solver-only bounds `--max-outside-coverage-loss`,
`--max-outside-split-pairs`, `--max-outside-joined-pairs` and
`--max-new-outsider-observations` accept nonnegative counts relative to the
baseline replay. A candidate exceeding a bound cannot outrank one satisfying
the bounds, even when it improves consolidation. Unspecified bounds are disabled.
New outsiders are counted as new (goal set, observation) links, so replacing one outsider with another still consumes the limit.
These measure collateral partition changes, not verified recognition errors.

Use **`--goal-policy strict`** for the previous objective: every original observation
must remain present and assigned; any cross-set sample overlap is a conflict.
Strict ranking uses assigned coverage before complete-performer consolidation,
pair/shared coverage and fragment count. This option applies to the solver and
set evaluation; the legacy two-ID catalogue diagnostic is unchanged. Policy and
frozen evidence are included in checkpoint compatibility. Start a fresh solve
after this code update or a policy change; existing checkpoints cannot be resumed.

Before a fully solved configuration exists, ties prefer fewer goal-to-outsider
attachments, external coverage losses, splits and joins, then confirmed local
stability and fewer/smaller normalized parameter changes. External partition
changes are not accuracy estimates; labels of other people are unknown.

Compact console output identifies the search phase, the first changed parameter against the previous winner, and the most frequent sampled blocker per unresolved set (up to three sets). Where retained evidence supplies an address, the message includes its scene and cue. A partial diagnostic budget is reported once per affected signal.

Held-out evaluation reports baseline and winner replays separately, followed by each omitted cue window. Its summary compares fragments and false joins for all labels and for back/entangled slices, then lists lost and joined pairs for the windows.

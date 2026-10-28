# Replay-verified merge goals

`report-merge --solve` searches one shared configuration for one or more explicit
sets of Stash performer IDs. Members of each set are intended to be one person;
different sets must remain separate. These are evaluation constraints, never
forced assignments inside recognition.

```powershell
python stash_ai.py recognition report-merge --performer-set 123,456,789 --performer-set 234,567 --solve --max-trials 500 --yaml-output merged.yml --output merge-goals.json
```

Use the same input YAML for scanning and solving. `--face-config input.yml`
selects the input configuration; omitting it loads `facial_recognition.yml` from
the current directory (or built-in defaults if absent). `--yaml-output merged.yml`
only names the result file and never selects an input configuration. Incompatible
input diagnostics include the resolved configuration and database paths.

Repeated `--performer-id` or `--identity-id` selects one set. The original
exactly-two-ID diagnostic remains available without `--solve`. Set syntax is
mutually exclusive with those selectors. Alias overlap across sets is rejected.
Without `--solve`, set input resolves retained goals; `--simulate` evaluates the
baseline. No Stash writes, model loading, media access or source YAML writes occur.

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
These measure collateral partition changes, not verified recognition errors.

Use **`--goal-policy strict`** for the previous objective: every original observation
must remain present and assigned; any cross-set sample overlap is a conflict.
Strict ranking uses assigned coverage before complete-performer consolidation,
pair/shared coverage and fragment count. This option applies to the solver and
set evaluation; the legacy two-ID catalogue diagnostic is unchanged. Policy and
frozen evidence are included in checkpoint compatibility. Start a fresh solve
after this code update or a policy change; existing checkpoints cannot be resumed.

Ties prefer fewer goal-to-outsider attachments, external coverage losses, splits
and joins, then confirmed local stability and fewer/smaller normalized parameter changes. External partition changes are
not accuracy estimates; labels of other people are unknown.

Search measures strict representative similarities, explores joint thresholds,
margin/support gates and retained role-quality gates, then refreshes candidate
thresholds from rebuilt representatives. Each trial is verified by full replay.
Budgets bound attempted replays, including baseline. Search is heuristic and
non-exhaustive. Neither budget exhaustion nor an unsuccessful search proves
that no possible configuration works. The selected result is the best among
completed trials, not a guarantee of the global optimum. If no changed candidate
beats baseline, the current configuration is explicitly selected as unchanged.

The default budget is **500 completed replays, including baseline**. It is a
ceiling, not a target or an exhaustive Cartesian grid. `--max-trials 1` evaluates
only baseline. The separate `recognition tune` Cartesian-grid command retains its
own default and limit.

With `sapiensid.enabled: true`, the search tunes all four matching gates:
`sapiensid.matching.face_floor`, `person_threshold`, `ambiguity_margin`, and
`min_support`. At the production frozen query stage, a read-only observer measures
eligible goal evidence against all reference cores **before face-floor nomination**.
This includes rejected queries of auxiliary-only fallback performers that have no
strict catalogue identity. In strict goal mode all original goal observations are eligible.
It records face/body boundaries, reference and independent query support, missing
features/core guards, and competitor scores. No observer is installed in ordinary
runs, and enabling diagnostics does not change production decisions.

Measured proposals clear multiple blockers together: for example a query can need
a lower face floor, lower body threshold, and lower independent support in the
same replay. Body modes and original query cores are measured independently;
alternative cores are not treated as simultaneous obligations. Quality proposals
use retained pose, visibility, size and sharpness measurements. Missing measurements
and invalid alignment remain explicit; no embeddings or quality values are invented.

Within-set acceptance bounds are compared with between-set/outsider rejection
bounds. The report lists `separation_intervals`: a threshold must be at most the
weakest desired score and strictly above the strongest unwanted score to separate
those frozen witnesses through that gate. Support uses integer boundaries, scores
use six decimal places. Feasible intervals produce joint proposals that preserve
other blocking gates. Overlapping intervals explain why that single scalar gate
cannot separate these witnesses at the search resolution. They **do not prove**
that no shared YAML configuration works: competitors, reference reconstruction,
other routes and combined gates still require full replay.

Independent probes remain available when diagnostics are incomplete. Face/body
thresholds are tested below and above their current value by `.02`, `.05`, `.1`,
`.2`, and `.01`, interleaved with measured boundaries. Margin and support have
separate probes. Coordinates and joint proposals are interleaved, including face
parameters, so a large pair list cannot consume all early probes. Dense measured
boundaries are spread through the search order.

Only a beam of **at most five diverse completed configurations** gets adaptive
descendants. The beam keeps the highest-ranked result, prefers different final
partitions, and reserves room for different parameter routes on a plateau. A
repeated inferior configuration does not create another unbounded search branch.
Up to 13 baseline explorations warm up the active families while baseline is
still the best result. An improvement ends the initial baseline-only priority:
the new winner is expanded in the next available normal slot, subject to the
reserved baseline slot. Baseline exploration continues once per four completed
normal trials, and at least every other adaptive parent preference goes to the
current winner when its streams have candidates. Other parents have a separate
rotation. Bidirectional and robustness trials do not advance these scheduling
counters; alternating lanes cannot repeatedly skip the winning parent. Family allocation
favours observed improvements, with exploration and a starvation guard, instead
of alternating hybrid and face trials regardless of usefulness. Duplicate or
invalid proposals do not consume a replay budget slot.

After 64 completed trials without an improvement in goal/outside-impact metrics
(with a smaller patience for smaller budgets, at least 24), the solver refines
around the retained beam. Refinement tries measured brackets, smaller steps and
reverting unnecessary changes. Up to 24 completed refinement trials run; material
improvement resumes exploration. Otherwise it tries at most eight distant restart trials once, refines again,
and performs the reserved local stability checks before stopping. Patch-size tie-breaks alone do not reset patience.
The configured replay limit and proposal-attempt guard always remain in force.

`search.stop_reason` explains budget exhaustion, stagnation, exhausted candidates
or interruption. `beam_width`, `peak_beam`, `beam_trials`, `pruned_trials`,
`stagnation_patience` and `refinement_budget` expose the search limits.
`search.executed_by_family` and each trial's `search_family` show actual allocation.
`search_phase` is `baseline_exploration`, `adaptive`, `refinement`, `restart`, or
`robustness`. `search_parent` and `search_strategy` explain the origin, including
`measured_blockers`, `coordinate_probe`, `boundary_probe`, `crossover`,
`local_refinement`, `restart_probe`, `robustness_probe`, and optional `tpe`. `search.tested_hybrid_values` includes
only completed trials, including baseline, never interrupted proposals.

Each trial's `solver_feedback` contains query-stage blockers, separation intervals
and retained quality measurements. The selected trial's feedback is also available
in `explanation.solver_feedback`. Query comparisons include all competing cores,
but the report retains desired routes and compact unwanted witnesses rather than
a full corpus matrix. Configured SapiensID candidate-pair and plan-memory limits
also bound the query observer. If that diagnostic budget runs out, it discards the
partly measured query, reports `truncated`, `truncation_reasons` and
`skipped_query_visits`, and continues with complete hints and independent probes.
An incomplete competitor scan never supplies a supposedly safe gap. Production
replay and reference-pair evidence retain their existing explicit budget errors.

Hybrid proposals use qualified retained face/body banks, independent cue support,
confirmed body appearance modes, original merged-core guards, co-presence, and
competitors both in other requested sets and outside the goals. Joint proposals
clear multiple measured gates together, including a measured competitor margin;
they do not average face and body scores. Supported body modes contribute a
threshold that covers every mode on both sides, avoiding an unnecessarily low
minimum across unrelated clothing looks. All these hints remain subject to full
replay, including mode reconstruction under the new thresholds. Tightening as
well as loosening gates is explored around measured score/support boundaries.

After each trial, observation IDs map the original sets to their rebuilt
identities, including split members and cross-set conflicts. The sets remain
separate when collecting feedback. New hybrid measurements and proposals feed
the next trials. No goals are passed to the production matching rules. Missing
body features or original-core evidence are reported; reducing `min_support`
never invents independent observations. Model, pose/association, collection,
consolidation scope, execution and storage settings are not tuned. Existing
compatible person embeddings are required; the solver never loads SapiensID or
downloads weights to fill gaps.

`measured_hybrid_evidence` describes the initial retained state. Each trial's
`hybrid_solver_evidence` contains the rebuilt measurements, independent support,
body-mode boundaries, missing references, pair blockers and nearest competitors.
These diagnostics are hints, not impossibility proofs. Hybrid diagnostics inspect
only pairs touching the goals, retaining target-pair proofs and the nearest
competitor per target instead of an all-library pair matrix. The configured
SapiensID `reference_bytes`, `plan_bytes` and `max_candidate_pairs` budgets also
bound this work; exceeding them fails explicitly.

Reports include final destinations, missing/withheld counts, conflicting trials,
recorded decision reasons and available numeric decision measurements. Measured
reference pairs include their weakest observation IDs; these are diagnostic
witnesses, not standalone proofs that a whole replay must merge.

`best_result` contains the selected trial, its YAML patch, `changed`, `admissible`,
`fully_solved`, and `unmet_conditions`. Its `quality` is `fully_solved`,
`partial_solution` (policy requirements preserved, incomplete internal merges),
or `best_effort` (preservation or separation also fails). Identity-mode unmet
conditions identify `empty_identity_set`, `set_scene_not_preserved`,
`incomplete_identity_set` and `cross_set_merge`, with pooled core support and
per-scene destinations. Member representation and soft-budget excess are diagnostic
fields, not unmet conditions. Strict mode retains missing/withheld/split/empty sample conditions.
`explanation.unresolved` repeats this list. `goal_diagnostics` identifies the
policy, strict success/separation and soft losses; YAML comments include this
distinction even when identity-mode `fully_solved` is true. For compatibility,
`recommendation` is populated only when a changed configuration wins;
`best_result` also represents an unchanged baseline. The top-level status remains
`baseline_retained` for an admissible partial baseline, while its quality is
`partial_solution`.

`--yaml-output` requires `--solve` and by default writes the selected complete configuration,
including best-effort and unchanged-baseline results. YAML comments identify the
trial, quality and unmet condition types; the JSON report contains the details.
For a partial input scope, YAML comments also identify the reduced validation
scope. Use the report's `run_argv` to replay exactly the validated scenes;
including excluded or recollected scenes requires revalidation.
Relative model/database/names paths become absolute so the exported file can be
relocated. Outputs must be distinct new files.
For a full export, `run_argv` gives the exact scene selection as an argument array, avoiding shell
quoting differences. Code/configuration/evidence hashes record the tested state;
changed inputs require revalidation. Stash publication itself is not simulated.

Use `--yaml-mode diff` to save only changed keys relative to the solver's **input
baseline**, including its defaults and command-line overrides. This is independent
of the mutation relative to the previous trial shown in console logs:

```powershell
python stash_ai.py recognition report-merge --face-config facial_recognition.yml --performer-set 123,456,789 --performer-set 234,567 --solve --yaml-output merge-diff.yml --yaml-mode diff --output merge-goals.json
```

For example, a winner changing only one SapiensID gate produces this YAML body:

```yaml
sapiensid:
  matching:
    face_floor: 0.76
```

The diff omits unchanged keys, path rewrites and schema metadata. An unchanged
baseline produces `{}`. Comments retain the winning trial, unmet conditions,
partial input scope and baseline configuration hash. Merge the changed leaf values
into the original configuration, keeping other keys and input overrides; the diff
is **not a standalone `--face-config` file**. The JSON report records
`yaml_export_mode: diff` and `yaml_diff_baseline` with the baseline path and hash.
It omits `run_argv` for a diff; use `simulation.selection` for the validated scenes
when running with the merged configuration. `--yaml-mode full` retains the default
complete, relocatable output.

Both formats support atomic winner updates, Ctrl-C, partial results and checkpoints.
With `--checkpoint` and no `--yaml-output`, the automatically named `.best.yml`
also uses the selected format. Repeat the same `--yaml-mode` on `--resume`; a
format change is rejected before updating the winner file. No YAML is written
until a baseline trial has completed.

Progress and the final best-result summary go to stderr, leaving JSON stdout
parseable. At the start of each trial, the solver prints every configuration
change relative to the **previously executed trial**, with the full YAML key and
old/new values. This includes restoring a parameter to the input configuration
when switching search branches. Skipped duplicate or invalid proposals do not
become the comparison point. `changes_vs` names that previous trial; `parent`
separately names the search ancestor, which may be an older trial or baseline.
For example:

```text
Merge solver baseline: start (input configuration)
Merge solver trial-1: start changes_vs=baseline parent=baseline
  sapiensid.matching.face_floor: 0.8 -> 0.785
Merge solver trial-2: start changes_vs=trial-1 parent=baseline
  clustering.max_anchor_pose_deviation: 0.2 -> 0.4
  sapiensid.matching.face_floor: 0.785 -> 0.8
```

Stage messages are throttled, so the last printed stage may precede a failure.
On a rebuild exception, the CLI also prints `Recognition replay trial-N; stage:`
with the last entered stage and up to five file/line/function locations. This
diagnostic is available with progress disabled and contains no frame locals or
embedding contents. The failed trial is not completed or exported; the last
completed winner remains in the YAML file.

Completed JSON trial records contain the same `previous_trial` and
`changes_from_previous` data (`parameter`, `previous`, `current`). The existing
`changes` and `yaml_patch` remain relative to the input configuration for export.
Every completed trial logs `best=trial-N best_number=N` (baseline is
number 0), using the same ranking as YAML export. JSON trial records carry
`best_trial` and `best_trial_number`. Each completion header is followed by a
summary and sample-detail lines per set describing **that trial's result**, even if a previous trial remains
the winner. The final summary uses the same fields for the selected winner and
also states the stop reason, replay count and unmet conditions. For example:

```text
Merge solver trial-12: complete disjoint=True best=trial-7 best_number=7 goal_policy=identity strict_disjoint=True
  Set 1: scenes_preserved=4/4 set_shared_coverage=50.0% set_fully_merged=False consolidated=2/4 identity_components=[2,2] represented=4/4 core_shared_coverage=50.0% soft_lost_or_reassigned=0 soft_budget_excess=0 largest_complete=2/4 complete_components=[2,2] assigned=8/8 (100.0%) shared_coverage=50.0% fragments=2 missing_obs=0 withheld_obs=0 missing_performers=0 withheld_performers=0 split_performers=0 empty_performers=0
```

`complete` in the header means that the trial finished. The old unnamed `(k,n)`
pairs are replaced by these explicit metrics:

- In identity mode, `scenes_preserved=k/n` exposes set preservation,
  `set_shared_coverage` is the strongest scene-normalized common destination, and
  `set_fully_merged` checks one destination's scene links and pooled core quorum.
  The JSON adds `missing_scenes`, `scene_support`, `set_core_destinations` and
  `set_core_required` for diagnosis.
- `consolidated=k/n` counts represented performers in the largest
  shared destination; `identity_components` shows these counts for every destination.
  `represented=k/n` counts performers meeting core and scene requirements, even if
  still separate. `core_shared_coverage` is equally weighted core support in the
  strongest common destination. `soft_lost_or_reassigned` and `soft_budget_excess`
  expose losses outside the set and excess over its pooled allowance. The member
  diagnostics and soft allowance do not determine admissibility or full success.
- `largest_complete=k/n`: the largest number of whole input performers sharing
  one output identity. A performer counts only if every original observation is
  present, assigned and in that identity. Four separate intact performers give
  `1/4`; two merged pairs give `2/4`; all four together give `4/4`. Destination ID
  and input order do not matter. The denominator counts unique identities after
  alias resolution.
- `complete_components`: counts of whole performers per output identity, sorted
  descending and including singletons. `[2,2]` exposes progress in both pairs;
  partially assigned or split performers are absent from this list.
- `assigned`: assigned/total original observations and their percentage, including
  missing observations in the denominator. `shared_coverage` instead averages each
  performer's share in the strongest common destination, weighting performers
  equally. These measures show partial progress when `largest_complete` is zero.
- `fragments`: the number of output identities containing any of the set's
  observations, including fragments of incomplete performers.
- `missing_obs` and `withheld_obs`: original observations absent from replay and
  present observations left unassigned, respectively. `missing_performers` and
  `withheld_performers` count affected performers; `split_performers` counts those
  assigned to multiple identities; `empty_performers` counts those with no original
  observations. These performer categories can overlap.

If four performers each have 99 of 100 observations in a shared identity and one
withheld observation, the log shows `largest_complete=0/4 complete_components=[]`
alongside `assigned=396/400 (99.0%) shared_coverage=99.0% withheld_obs=4
withheld_performers=4`. It can also show `consolidated=4/4 identity_components=[4]
represented=4/4`. Full identity-mode success depends on preserved set scenes and
the common destination's pooled core quorum, even when `represented` is lower.
`largest_complete` remains strict in both modes. `disjoint` is checked across sets
under the selected policy; internal consolidation alone does not certify separation.
`strict_disjoint` in identity-mode headers shows the original all-sample check.

Each set also lists the requested performer/identity IDs and resolved Stash
performer IDs, followed by face and SapiensID sample counters. For example:

```text
    Members: performer_ids=["900","901"] resolved_performer_ids=["900","901"] roles=input_config target=<output-identity-UUID> target_ties=1
    Face anchor: assigned=3/3 (100.0%) target=3/3 other=0 withheld=0 missing_obs=0 cross_set=0 fragments=1
    Face auxiliary: assigned=2/3 (66.7%) target=1/3 other=1 withheld=1 missing_obs=0 cross_set=1 fragments=2
    SapiensID anchor: ready=2/3 assigned=2/2 (100.0%) target=2/2 other=0 withheld=0 missing_obs=0 cross_set=0 fragments=1 unavailable=1 no_evidence=0 incompatible=0
    SapiensID auxiliary: ready=2/3 assigned=1/2 (50.0%) target=0/2 other=1 withheld=1 missing_obs=0 cross_set=1 fragments=1 unavailable=0 no_evidence=1 incompatible=0
```

`roles=input_config` means cohorts are fixed before search: anchors have strict
reference authority and satisfy the input configuration's anchor quality gates;
auxiliary samples satisfy query gates but not anchor gates. Both require complete,
non-rejected source observations. Other samples, including contextual attachments
with anchor-quality faces, rejected samples and samples below query gates, appear
in an `other` line when present. These are eligibility cohorts, not a claim that
every anchor was selected into a capped reference bank or kept that role in a
trial. Changing trial quality thresholds does not change their denominators.

- `assigned` counts final memberships, including assignments to other fragments.
  `withheld` means present in replay but unassigned; `missing_obs` means absent
  from the replay selection. Assigned + withheld + missing equals the denominator.
- All face/body role lines use the same `target`: in identity mode, a destination
  satisfying the full set goal if one exists, then highest scene-normalized evidence
  coverage and pooled core support. Strict mode prefers complete performers, then
  normalized evidence coverage and sample count. Exact ties use a stable UUID
  order; `target_ties` reports their count. This is a display choice and never
  forces recognition to use that identity. `target` counts samples there;
  `other` counts assigned samples in all other destinations. Their sum is assigned.
- `cross_set` counts samples whose destination also contains original samples
  from another requested set. It is a raw overlap diagnostic, can overlap
  `target`/`other`, and can be nonzero for tolerated noise in identity mode.
- SapiensID rows describe the **body evidence attached to the same observations**,
  using the same frozen roles and final identity assignments. `ready=n/total`
  counts retained features with the expected input contract. SapiensID assignment
  fractions use only these ready observations as their denominator. `unavailable`
  means collection recorded no usable body feature; `no_evidence` means no record;
  `incompatible` means only other contracts exist. A ready feature can still be
  absent from replay, reported as `missing_obs`. When disabled, the log explicitly
  prints `SapiensID: disabled`.

SapiensID does not have an independent identity assignment in the current hybrid
pipeline. A ready body's assigned status does **not** prove SapiensID caused that
match; the production decision evidence describes the route. Counts are raw
observations, not independent support votes or correctness estimates. These
diagnostics do not affect the solver objective or selection of the winner.
The same data is stored in each group's `sample_assignment` in JSON and restored
from checkpoints. The final summary, including Ctrl-C, uses the selected winner's
counts, even when a later trial performed worse. Metadata is read once during
goal resolution without loading body vectors, crops, models or additional media.

`--progress-style off` suppresses solver messages. Ctrl-C during replay search
preserves completed trials and their best result, writes that winner to
`--yaml-output` when supplied, marks the result interrupted, and exits 130. This
also applies during local refinement or after a losing trial; the last trial is
not automatically the winner. Incomplete trials are never eligible for selection. If no trial
completed, status is `no_completed_trials`, `best_result` is null and no YAML is
exported. Interrupting initial goal preparation may stop before a report exists.
Runtime uses configured rebuild workers and RAM budgets; GPU
inference is not repeated. Performance on a large library must be measured locally.

Solver diagnostics retain a configuration/evidence fingerprint, producing trial,
active parameters, and a trace combining terminal assignments, actual hybrid
query choice/co-presence/support decisions, and measured reference blockers.
Descendants use their parent's proposal and option snapshots. Diagnostic query
slots rotate across sets, performers and scenes before replay-order traversal;
partial comparison budgets never certify an unmeasured competitor gap.
Run `python -m benchmarks.benchmark_merge_solver --max-trials 100` for deterministic
face, auxiliary, disjoint-set, missing-body, quality and body-mode cases. Its JSON
records best-objective trajectories, first solution, elapsed time and 10/50/100
trial milestones. Synthetic timings do not predict a user's library runtime.

Joint separation retains each unwanted witness's face/body/support vector and
searches bounded combinations where every desired route passes all gates and
each unwanted route fails at least one gate. Scalar overlap no longer suppresses
multi-gate proposals. `joint_constraints` records unresolved witnesses and pruning;
these are frozen diagnostics, not global infeasibility proofs. Strong fragment
bridges, bounded crossovers between beam states and coupled refinement complement
the full-set proposal. Production complete-core guards always remain authoritative.

Search now distinguishes parameter values and blocker signatures within the beam.
Pair consolidation fractions break ties after the selected policy's
common-destination metrics; preservation and set separation retain priority. Mutation magnitudes use
parameter scales (fractional thresholds, support, pixels and sharpness). Operator
and family allocation account for improvements, replay time and rejected proposals.
One restart of at most eight completed experiments explores distant measured
values before final stagnation; it never discards the best completed result or
extends `--max-trials`. Search metadata includes operator statistics and restart
counts. Every trial still logs its changes against the previous executed trial.

With multiple sets and at least one set containing multiple performers, search
also approaches the problem from a permissive configuration. Starting with
`trial-2`, every other exploration slot is offered to this alternate lane while
the goal is unsolved. The first normal trial and the existing baseline probes
remain available. The lane alternates broad joint relaxations with attempts to
repair their results; it does not wait for stagnation or the late restart.

The initial profiles relax SapiensID matching together (`face_floor` at most
0.2, `person_threshold` at most 0.35, zero ambiguity margin, support 1), face
matching together (match thresholds at most 0.35, cohesion thresholds at most
0.25, zero ambiguity margins and minimum support), and both routes together.
They also include available measured admission-gate relaxations. Existing
lower thresholds are preserved. Detection, feature extraction and model
contracts are unchanged; missing features remain missing. These profiles are
experiments, not recommended settings until production replay evaluates them.

An independent frontier retains up to three completed states ranked by
continuous within-set consolidation and coverage (pooled core evidence and
set scene coverage in identity mode). A state mixing two sets can
remain a parent even when the normal beam rejects it. Repairs use positive and
negative witnesses, joint/per-family interpolations towards a completed disjoint
state, and individual gate reversions. A disjoint but incomplete state instead
gets `frontier_expansion` proposals: matching gates are relaxed jointly and
individually beyond the fixed seed values, using measured thresholds and
proportional steps. These proposals retain the parent's useful changes;
baseline hints cannot silently tighten them again. The strongest alternate
frontier state gets every other parent preference, with separate rotation for
the other states. Interpolation only proposes experiments:
matching is not assumed monotonic, and every candidate receives a full replay.
This can cross a plateau where changing one gate leaves `largest_complete=1/N`.

The console identifies these trials with `Search: phase=bidirectional` and
`strategy=relaxed_seed`, `frontier_expansion`, `separation_repair` or `boundary_bridge`, in addition
to the exact mutations. JSON includes `search.bidirectional` and the
`bidirectional` phase count. The lane has its own bounded allowance (at most
64 completed replays, approximately one third of smaller budgets, with a minimum
allowance of four when the total budget permits). All trials still share the
global limit, default 500, and the final robustness allowance. It is disabled
for a single set and stops offering candidates once a full solution is found.

The separate parent-selection rank does not change the exported winner rank.
A cross-set merge under the selected policy cannot displace a completed disjoint result, and Ctrl-C keeps
the best completed winner rather than the last permissive experiment. Checkpoint
resume reconstructs both frontiers and skips completed replays. As with other
solver code changes, a checkpoint from an older version requires a fresh solve.

Replay preparation now copies scoped retained history once into a temporary,
immutable disk SQLite seed. Each trial receives an independent RAM backup, checks
the same memory limit and runs production rebuild. No assignments or decisions
from another trial are reused. The seed is deleted when replay exits.

Measured boundaries supply interval representatives in a separate search lane.
A frozen-witness preview orders small batches of directed proposals by measured
separation and acceptance. It never drops a candidate or certifies a merge;
independent coordinate probes and full replay remain authoritative, including
when diagnostic sampling is incomplete.

Search guidance prioritizes missing target scenes and pooled-core deficits.
Complementary component bridges precede redundant dense-scene connections;
repair proposals use the current best disjoint parent's measurements. Query
sampling rotates least-visited evidence across sets, members and scenes, even
when a pass cannot afford one query per set. The rotation state is restored with
the existing journal. Terminal decisions are read by the indexed observation
address and explicit latest-row order, across matching policies.

Alternative reference routes for a query are OR choices, not simultaneous
acceptance obligations. A bounded constraint frontier combines useful routes
while retaining real competing witnesses. Missing body evidence does not imply
that the goal is impossible: measured face-only query/reference boundaries can
still propose face and retained-quality changes. Production replay remains the
only authority for accepting any proposed configuration.

The solver selects compatible latest scans before measuring initial proposals.
Face-bank statistics use bounded float64 tiles and cache immutable vector
content, including initial measurements before baseline replay. Goal addresses
and scene cohorts are compiled once; vote memoization is local to a single
evaluation and cannot reuse assignments from a previous trial.

Historical threshold pools retain at most 128 values per parameter, except when
protecting live parent settings and their immediate boundary neighbours requires
more. Fresh measurements take precedence over stale options; extrema and spread
across the range remain represented. Direct witness proposals remain available
independently of this bounded historical pool.

### Checkpoints and local stability

`--yaml-output` is atomically refreshed after every completed improvement,
including an unchanged baseline and partial solutions. The file must initially
be new; external edits are detected before subsequent replacement.

Add `--checkpoint solve.json` to persist a small manifest and a sibling
`solve.json.trials/` directory of completed results and parent diagnostics. If
no YAML path is supplied, the winner is saved as `solve.json.best.yml`.
Resume with the same command plus `--resume`. Keep both the manifest and trial
directory. The solver reconstructs its frontier, seen proposals and operator
statistics from the journal; completed production replays are not repeated.
It verifies all retained database tables, goals, input configuration, code and
search settings first. Changed code, evidence, parameters or budgets require a new
checkpoint. JSON report `--output` still requires a new path; omit it during
resumable work or choose a new report filename on each invocation.

The manifest is committed before the winner file, so resume can finish an
interrupted winner update. Checkpoints retain diagnostics and observation IDs,
not crop images. They use incremental per-trial files instead of rewriting the
entire history after every trial. `search.restored_trials` reports reused trials.

For budgets of at least 20, up to eight trials are reserved for full production
replays near two leading configurations (at most ten percent of the budget).
Two or more successful nearby replays give a stability preference only after
goal and outside-impact scores tie. The `robustness` report lists actual checks;
missing checks mean unknown stability, never an assurance. These probes share
the global trial limit and do not prove robustness on future media.


### Optional learned proposals

Install `pip install -r requirements-solver.txt` and add
`--solver-sampler tpe --solver-seed 42`. Default `heuristic` mode has no Optuna
dependency. TPE proposes at most one candidate per five completed trials after
13 warmup trials; other candidates continue to come from the guided search.
Rejected or duplicate TPE proposals do not consume a replay slot or replace
heuristic proposals indefinitely. The usual default total limit remains 500.

The sampler uses up to eight active tunable dimensions, mixes integer support,
boolean and numeric thresholds, and learns from completed production trials.
Its training target is the ordinal order of the solver's full lexicographic
ranking, not a weighted sum that exchanges a hard constraint for consolidation.
Every proposal still passes configuration/contract validation and full production
replay. Seeds, dependency version and mode are part of checkpoint compatibility.
The journal reproduces the same proposals when reconstructing a completed prefix.

`python -m benchmarks.benchmark_merge_solver --max-trials 100 --sampler compare`
compares both modes on six deterministic retained-evidence fixtures, reports
objective trajectories, first-solution time and outcomes at 10/50/100 trials.
These synthetic cases test mechanisms; they do not establish accuracy or a
speedup on a real library. In the recorded comparison both modes fully solved
five cases; the missing-body case remained partial. TPE did not improve the first
solution count on these fixtures, so it remains opt-in. See
`benchmarks/results/merge_solver_comparison.json` for measured counts and timing.

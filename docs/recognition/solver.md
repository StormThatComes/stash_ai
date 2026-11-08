# Merge goals and solver search

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Find a feasible merge, then tighten it](#find-a-feasible-merge-then-tighten-it)
- [Targeted merge trials](#targeted-merge-trials)

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

The solver caps minimum support counts at **8** for recognition, clustering and
enabled SapiensID routes. Global body matching also respects `max_references`
when it is smaller than 8. If an input YAML requests more support, the solver
uses the capped count starting with the baseline replay and shows the adjustment
in the trial report and exported YAML diff. This limit applies only to the solver.

The solver keeps `matching.body.anchors.keypoint_rules` exactly as configured
during numeric tuning, including custom point lists and empty rules. To also
compare the two predefined whole-rule alternatives on the winning numeric
configuration, add `--compare-anchor-rules`. These replays share the existing
`--max-trials` budget. [Regional anchor rule profiles](../sapiensid-hybrid.md#regional-body-anchor-keypoints)

Configuration examples use YAML v2. Diagnostic field names may use the internal
v1 paths (for example, `sapiensid.identities.anchor_keypoint_rules`); exported
winner YAML and diffs follow the input file's layout.

See [solver report interpretation](solver-report.md) for JSON fields and blockers,
[parallel execution](solver-parallel.md) for CPU/RAM controls and
[feasibility and checkpoints](solver-feasibility.md) for recovery.

## Find a feasible merge, then tighten it

The first fully solved replay (including an already solved baseline) starts a
dedicated `tightening` phase. Remaining ordinary search slots strengthen the
solution instead of treating unchanged fragmentation as stagnation. The best
completed configuration is still saved immediately; interruption or a failed
probe cannot replace it with an unverified boundary.

Among fully solved configurations the ranking preserves the existing collateral
impact priorities, then prefers restoring relaxed admission requirements,
preserving face support of frozen face references, stronger face-core matching, face views/attachments, the hybrid face floor, and
body matching, in that order. Within each tier, bounded normalized strengths are
summed (integer support uses `value / (value + baseline_scale)`). Switching more
frozen face references to body-assisted matching than in the first feasible
solution incurs a penalty before threshold strength; simply dropping auxiliary
observations earns no route-preference reward. Original goal
quality, local stability and smaller YAML changes break remaining ties. Thus a
loose input YAML does not win merely because it needs fewer edits. All positive
sets, negative scene/observation constraints and explicit outside limits still
have to pass the full production rebuild.

The first feasible replay's accepted production routes determine which matching
gates receive a strength reward. Unused changed gates are restored towards the
input policy rather than driven to an arbitrary domain extreme. Admission gates
(anchor/auxiliary quality, pose and body keypoint rules) are also restored only
up to their input requirements: filtering out additional evidence is not rewarded
as stronger matching. Booleans and null opt-outs are not scalar tightening axes.
The active scope is frozen for comparable trial rankings and included in
`search.tightening.matching_parameters`.

Tightening tries measured values and brackets a passing setting with an observed
failing setting, down to integer or six-decimal resolution. A bracket is reused
only with identical other parameters; clustering is not assumed globally
monotone. Joint probes cover cohesion/match dependencies, effective face/view
thresholds, hybrid gates and learned blocker groups. Every proposed point must
pass replay; no interpolated configuration is exported. Parallel trials drain
pending results when those results are needed for the next boundary proposal.
`tightening_frontier_exhausted` means the current local proposals were exhausted,
not that a global optimum has been proved. The total trial ceiling and the
reserved ablation/robustness budget still apply.

Targeted planning gives measured face routes priority while preserving slots for
hybrid and body fallbacks. After two consecutive preferred face proposals for a
set, an available fallback gets a turn. A face proposal that reduces the current
ambiguity margin, or goes below the configured hybrid face floor when SapiensID
is enabled, is marked `uncertain_face` and follows available hybrid proposals.
This does not compare face and body cosine scores directly or modify production
matching: face evidence still wins first where its guards pass; body evidence
can resolve ambiguity or missing faces. Anchor references, direct auxiliary
attachments and supported views retain their production authority rules.

`search.tightening` and the final console summary report the first feasible
trial, completed tightening replays and parameter scope. These are local
experiments on retained evidence, not a false-merge guarantee for future samples.

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

A beam of **three to eight diverse completed configurations** gets adaptive
descendants. Its width follows partition and blocker diversity over the last
16 completed trials; repeated plateaus shrink it, diverse outcomes expand it. The beam keeps the highest-ranked result, prefers different final
partitions, and reserves room for different parameter routes on a plateau. A
repeated inferior configuration does not create another unbounded search branch.
Up to 13 baseline explorations warm up the active families while baseline is
still the best result. An improvement ends the initial baseline-only priority:
the new winner is expanded in the next available normal slot, subject to the
reserved baseline slot. Baseline exploration continues once per four completed
normal trials, and at least every other adaptive parent preference goes to the
current winner when its streams have candidates. Other parents have a separate
rotation. Bidirectional, minimization and robustness trials do not advance these scheduling
counters; alternating lanes cannot repeatedly skip the winning parent. Family allocation
favours normalized improvement at the first changed objective and new partition/
blocker information per replay second, with exploration and a starvation guard.
This scheduling reward never changes the lexicographic winner ranking. Duplicate or
invalid proposals do not consume a replay budget slot.

While no fully solved configuration exists, after 64 completed trials without an improvement in goal/outside-impact metrics
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
`search_phase` is `baseline_exploration`, `adaptive`, `bidirectional`, `refinement`,
`restart`, `tightening`, `minimization`, or `robustness`. `search_parent` and `search_strategy` explain the origin, including
`measured_blockers`, `coordinate_probe`, `boundary_probe`, `crossover`,
`local_refinement`, `restart_probe`, `tighten_boundary`, `tighten_joint`, `parameter_ablation`, `robustness_probe`, and optional `tpe`. `search.tested_hybrid_values` includes
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
version: 2
matching:
  hybrid:
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

## Targeted merge trials

After baseline replay, `--targeted-trials 8` (default) first tries measured routes
for each unsolved set. The allowance is capped at one third of non-baseline
slots, rounded down, with one targeted trial allowed in a two-trial run. These
trials consume `--max-trials`; they do not add rebuilds outside that budget.
`--targeted-trials 0` disables this lane. To disable both early lanes, also set
`--feasibility-trials 0`.

`solver_feedback.targeted.sets` reports current fragments, their scene links,
missing target scenes, pooled core deficit, and observed blockers per historical
performer. Performers are diagnostic addresses, not preservation obligations:
identity mode still permits weak samples or an entire historical identity to
disappear when the pooled set retains the required core and scene support.
Strict mode still requires every original observation.

Candidates use measured face, hybrid and independent-body boundaries. Query
routes can lower face/body/support gates together; admission routes can recover
retained samples filtered by quality, including body samples that currently have
no identity. Linked face/body alternatives count once. No missing model features
are fabricated and measurement contracts remain unchanged.

The lane rotates between sets and reads the best completed parent's current
diagnosis. After a useful admission or merge, subsequent routes therefore use
the rebuilt references and fragments. Equivalent YAML patches share one replay;
complementary bridge proposals can connect multiple fragments at once. A connected
pair graph is only a candidate, never proof that production complete-core checks
will accept the whole set. Every candidate runs the full production replay with
all competitors, set-separation requirements, scene exclusions and outside-impact
limits. The winner uses the original goal policy.

Console trial headers identify the set, route, involved performers and blockers;
JSON includes `search_target`, `search.targeted` and the bounded candidate list.
At most 16 candidates per set and 64 overall are retained. If no usable route
remains, or the allowance is consumed, search falls through to construction
probes and ordinary optimization. A new best state may offer additional targeted
routes within the remaining allowance. Failed targeted trials or truncated
diagnostics never establish global impossibility. Resume requires the same
`--targeted-trials` setting as the original checkpoint.

The retained-evidence benchmark accepts `--targeted-trials 0` or `8` for comparison.
In three alternating runs per mode with `--max-trials 12`, both modes solved all
six synthetic cases. Counts below include baseline:

| Case | First solution, targeted off | First solution, targeted on |
| --- | ---: | ---: |
| Face | 2 replays | 2 replays |
| Hybrid query | 2 | 2 |
| Disjoint sets | 6 | 2 |
| Missing body | 2 | 2 |
| Admission quality | 3 | 2 |
| Body modes | 2 | 2 |

For disjoint sets, median cumulative replay time to first solution fell from
0.529 s to 0.171 s; for admission quality, from 0.118 s to 0.089 s. The diagnosis
has overhead: the same-count face case took 0.124 s versus 0.101 s, and full
12-trial runtime including polishing did not consistently improve. These small
synthetic fixtures do not establish a speedup on a real library. Raw repetitions
and timing scope are in `benchmarks/results/merge_solver_targeted_comparison.json`.

### Validating a finalist on back views and entangled people

Pass `--solver-validation holdout.json` to `report-merge --solve`. The solver
excludes all versions of the listed scenes from its private training snapshot,
selects its winner and then replays the held-out scenes. The default is the
configured recognition database. An optional `database` path can point at a
separate holdout database with the same scope and encoder namespace. Example:

```json
{"scenes":["42"],"observations":{
  "face-A":{"truth":"Alice","slices":[]},
  "observation-A":{"truth":"Alice","slices":["back","entangled"]},
  "observation-B":{"truth":"Bob","slices":["entangled"]}},
 "partner_sets":[["face-A","observation-A","observation-B"]],"series":[["42",8,12]]}
```

The report compares baseline and winner on the supplied observations, including
coverage, extra fragments, false merge pairs and pair recall. Include both people
in an entangled cue and keep difficult negatives. Up to three inclusive, zero-based
cue windows are removed one at a time in disposable full replays. If omitted from
the manifest, windows come from the winner's recorded body-view tracks when
available. The omission report excludes removed labels from both comparison
metrics and counts lost and newly joined relations among the remaining labels.

Partner triples report recovered target relations and false joins to the named
partner, including counts per performer. Multiple projections of a person in the
same cue have one vote per relation. Recovery requires agreement across those
projections, and any false join counts. Primary metrics and omission deltas use
accepted production memberships. Provisional proposals have separate metrics.

A multi-scene holdout also replays the frozen finalist one scene at a time in two
opposite orders, ranking scenes by usable face cue coverage and labelled rear cues.
Each order reports the lost and newly joined pairs and coverage changes against the joint replay.
Cue omissions run twice: once as a rejected observation and once as an empty physical cue.
Both retain VTT cue markers, and both exclude omitted labels from the comparison.

To measure turns, label one performer's front, back and returning front in the same scene.
For crossings, label both people before, during (`entangled`) and after the crossing.
Rates use observed labelled sequences; zero available sequences produce an undefined macro rate.
A missing ID during contact can recover at the end. A swap to the partner fails that crossing.
Body-track diagnostics include the scene, cue, peer, measured value and its actual threshold when available.

### Physical cue and contact validation

The holdout report includes one vote per scene, cue and labelled performer in
`metrics.physical`. A face and body assigned to different IDs produce a
projection conflict and an unresolved physical vote. Sequence reports keep
the continuity count and also report `own_id_turns` out of `anchored_turns`.
The latter requires an independent accepted strict face reference and checks
whether a partner shares that ID during the annotated turn.

An optional `visible_people` object in the validation manifest can label a
missing body using source VTT pixel coordinates. Its keys are selected scene
IDs and zero-based cue numbers. For example:

```json
{"visible_people":{"123":{"121":[{"truth":"A","point":[70,90]},{"truth":"B","point":[160,100]}]}}}
```

The `detection` report counts expected people, missing and duplicate poses,
and ambiguous poses independently of the membership replay. Existing
`observations` still label the samples used to evaluate identity decisions.
Explicit omission windows take precedence over automatic scenario selection.

The holdout groups each labelled visible performer and cue into one outcome. The
report counts correct identities, foreign identities, new fragments, withheld
people and missed detections. An assigned person without a unique strict face
reference has a separate unknown-reference outcome. Two-sided turns require
strict faces at both boundaries of the continuous episode. Single-sided and
remote face references have separate counts.

When a visible labelled pair has two distinct retained bodies and a usable
face association, the holdout also swaps their face associations in one private
replay. It rebuilds the hybrid projections and reports changed memberships in
later cues and other selected scenes. The source database is opened read-only.

Contact interior outcomes distinguish correct, withheld, partner, other foreign and
unscorable references. Endpoint recovery remains a separate measure. A labelled strict face
may set `"reference": true` to score a split owner relative to a designated reference.
The independent verdict records coverage and baseline/finalist counts in JSON and exported YAML.
Up to three distinct contact episodes are perturbed separately in both private replays.
Threshold proposals use a bounded two-person path when both goal owners are labelled
in consecutive cues of the same measured shot. The proposal is validated by the usual full replay.

### Comparing body recognition strategies

Add `--compare-scene-strategies` to a run with `--solve --solver-validation holdout.json`.
It replays the finalist on the same retained embeddings with tracklets off and on,
each with the body scene bonus disabled, fixed or evidence-weighted. The positive
bonus uses the finalist's effective value, then the face scene bonus, then 0.03.
The `holdout_validation.scene_strategies` rows give the effective settings,
physical coverage, accepted fragments per labelled performer, own-ID turns,
contact exits and partner swaps. These comparisons do not retune the finalist.

Use `--goal-policy identity-physical --physical-pair FACE_ID:BODY_ID` to label
one training vote whose face and body have different baseline assignments.
Repeat the option for more pairs. Both observations must come from the same
current scan and cue. The face must belong to a selected goal. The label
overrides its automatic physical alternative during goal scoring.

Each omitted cue window now has a `damage` result for baseline and winner.
It counts correct same-person pairs lost outside that window and incorrect
cross-person pairs introduced by its omission, grouping projections by physical
person and cue. The independent verdict includes both measures.

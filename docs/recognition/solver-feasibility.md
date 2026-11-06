# Solver feasibility and checkpoints

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Initial construction feasibility](#initial-construction-feasibility)
- [Checkpoints and local stability](#checkpoints-and-local-stability)
- [Optional learned proposals](#optional-learned-proposals)

## Initial construction feasibility

After targeted trials, the solver checks whether the requested sets can actually
be constructed with the retained evidence and enabled identity routes. A joint
construction witness from baseline or a targeted trial skips permissive probes.
`--feasibility-trials 6` (default) permits up to six initial production replays;
these consume the existing `--max-trials` budget, including failed resource-limited
attempts. The probe allowance is capped at half the non-baseline slots (rounded
down, with one probe permitted in a two-trial run), reserving time for ordinary
tuning. `--feasibility-trials 0` disables these early probes but keeps construction
diagnostics on ordinary trials. Baseline always consumes one trial. A small total
budget may prevent all six profiles from being tried.

The deterministic profiles try matching/support floors, admission floors, both
together, and separate face, hybrid and independent-body routes. Duplicates and
unchanged configurations are skipped. Profiles preserve detection/model/input
contracts, disabled evidence routes, absent optional gates and already more
permissive baseline settings. They cannot recover features that were never
retained. Rejection hints from forbidden scenes cannot overwrite these liberal
matching floors.

The initial phase stops as soon as a completed replay constructs **all** sets,
even if that replay mixes different sets or violates a forbidden scene. If
baseline already provides this witness, no initial probe is needed. Completed
probes feed the normal beam and alternate frontier without another rebuild;
an overmerge is immediately available for repair. Winner selection and YAML
export continue to use the full original goal policy and outside-impact limits.

The JSON `feasibility` report and `Merge feasibility:` console summary distinguish:

| Status | Meaning |
| --- | --- |
| `possible` | One completed production replay constructs all sets and satisfies the full goals. |
| `possible_with_conflicts` | One replay constructs all sets but violates set separation, a scene exclusion or an outside-impact limit. |
| `possible_goal_mismatch` | Construction succeeded, but the selected goal policy does not credit it; for example, accepted linked body evidence replaced frozen face samples. |
| `blocked_inputs` | A required frozen scene witness or core quorum is unavailable in selected retained inputs and enabled routes. This is a narrow input certificate, not a claim about recollected data or other mechanisms. |
| `not_demonstrated` | No joint witness has been observed. Finite unsuccessful probes, interruption and resource limits do not prove impossibility. |

Each set reports its own witness, best observed scene/core support, available
input counts and trial-specific blocker reasons. Separate successful witnesses
for different sets do **not** establish one jointly feasible configuration.
`joint_witness_trial` identifies such a joint witness when one exists. Observed
matching blockers are diagnostic evidence, never global infeasibility certificates.

Construction counts an accepted, compatible, same-scene body linked to the
original face identity as an alternative address of the **same** frozen unit.
It never doubles support; conflicting destinations cannot vote. This diagnostic
can therefore detect face-to-body handoff even when strict all-sample goals or
frozen positive face goals reject it. It does not change those goals. Strict
reports retain separate `construction_evidence` using the identity-level pooled
core and scene criteria; strict acceptance still requires every original sample.

Known resource-limit errors during initial probes are reported in `failed_probes`
with the stage and exception details, and search continues from completed states.
They are checkpointed and not retried on resume. Baseline and ordinary-search
errors, unexpected exceptions and invalid configuration still fail normally.
`search.attempted` includes these failed probes; `search.executed` counts completed
replays. Incomplete probes cannot become winners. Resume requires the same
`--feasibility-trials` setting as the original checkpoint.

With at least one fragmented or multi-scene set, including a single set, search
also maintains an alternate permissive frontier after this initial phase.
Initially every other exploration slot is offered to this alternate
lane while the goal is unsolved. Recent improvements increase its frequency;
four unsuccessful experiments reduce it. At most two consecutive alternate
trials run while normal exploration remains available. The existing baseline probes
remain available. The lane alternates broad joint relaxations with attempts to
repair their results; it does not wait for stagnation or the late restart.

The alternate seeds use legal domain floors plus broad, median and quartile boundaries from the current
measurements and proposals, separately for SapiensID, face matching and both
routes together. Floors are tried even when measurements exist; rejection hints
cannot silently replace them. Seeds include measured admission-gate relaxations. Existing
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
individually beyond the starting values, using measured thresholds and
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
allowance of four when the total budget permits), additionally bounded by goal
complexity so a small single-set problem does not consume 64 alternate trials.
All trials still share the
global limit, default 500, and the final robustness allowance. It is disabled
only when there is no fragmented or multi-scene goal, and stops offering
candidates once a full solution is found.

The separate parent-selection rank does not change the exported winner rank.
A cross-set merge under the selected policy cannot displace a completed disjoint result, and Ctrl-C keeps
the best completed winner rather than the last permissive experiment. Checkpoint
resume reconstructs both frontiers and skips completed replays. As with other
solver code changes, a checkpoint from an older version requires a fresh solve.

Replay preparation now copies scoped retained history once into a temporary,
immutable disk SQLite seed. Each trial copies mutable history and ID/scan metadata
to private RAM; indexed, read-only views share the face/body embeddings and
thumbnail evidence from the seed. Production rebuild retains its bounded private
selected-face working database. No assignments or decisions from another trial
are reused. The seed is deleted when replay exits. Worker initialization sends the
fixed corpus and goals once; subsequent tasks carry configuration and sampling state.

Measured boundaries supply interval representatives in a separate search lane.
A frozen-witness preview orders small batches of directed proposals by measured
separation and acceptance. It never drops a candidate or certifies a merge;
independent coordinate probes and full replay remain authoritative, including
when diagnostic sampling is incomplete.

All generated candidates pass a shared retained-parameter domain projection
before deduplication and production validation. This preserves detection floors,
integer support limits and anchor/auxiliary and cohesion/match dependencies.
Six-decimal boundary proposals use the inclusive final matching comparisons;
nomination tolerances do not establish acceptance. Co-occurring blockers on an
actual query or pair supply additional cross-family interaction proposals and
refinement groups, alongside the existing fixed groups.

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
Face-bank statistics use bounded float64 tiles and share exact numerical results
by immutable vector content, including between worker trials. Diagnostic face
pairs are bounded by the existing candidate, witness and plan-memory budgets;
`face_measurement_coverage` reports measured and total pairs, work limits and
whether coverage is complete. Large goal sets visit a deterministic ring of
owners before denser pairs. These sampled hints never truncate production vetoes
or certify acceptance. Catalogue and body-report caches require the same live
database generation and policy, and callers receive private report proofs. Goal addresses
and scene cohorts are compiled once; vote memoization is local to a single
evaluation and cannot reuse assignments from a previous trial.

Historical threshold pools retain at most 128 values per parameter, except when
protecting live parent settings and their immediate boundary neighbours requires
more. Fresh measurements take precedence over stale options; extrema and spread
across the range remain represented. Direct witness proposals remain available
independently of this bounded historical pool.

## Checkpoints and local stability

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

For budgets of at least 20, four to 24 trials (at most one quarter of the budget)
are reserved for final polishing, scaled to changed and active parameters. First,
up to half this allowance tests reverting individual parameters and related
groups to baseline. Every ablation is a full production replay; a simpler patch
is accepted only through the ordinary winner ranking. The next ablations use
the current winner. Remaining slots validate two leading configurations, covering
different coordinates before their second side, boolean flips, nearby measured
boundaries and simultaneous perturbations of related gates. Domain projection
and dependency validation also apply to these probes.
Two or more successful nearby replays give a stability preference only after
goal and outside-impact scores tie. The `robustness` report lists actual checks;
missing checks mean unknown stability, never an assurance. These probes share
the global trial limit and do not prove robustness on future media.


## Optional learned proposals

Install `pip install -r requirements-solver.txt` and add
`--solver-sampler tpe --solver-seed 42`. Default `heuristic` mode has no Optuna
dependency. After 13 warmup trials, TPE initially proposes one candidate per
five completed trials. Recent improvements shorten the interval to two; four
unsuccessful proposals extend it to eight. Other candidates continue to come
from the guided search; a due TPE slot takes precedence over the alternate lane.
Rejected or duplicate TPE proposals do not consume a replay slot or replace
heuristic proposals indefinitely. The usual default total limit remains 500.

The sampler uses up to eight tunable dimensions: four prioritized dimensions and
four deterministically rotating dimensions when more than eight are available.
It respects the shared domains, mixes integer support,
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


Checkpoint input signatures cover all retained scan versions in the selected
recognizer scope, face/body measurements, membership and role history, original
ownership, aliases, referenced pixel thumbnails and current-scan heads. Portrait
crops, run/publication journals and evidence from other scopes are excluded.
Configuration, goals, solver settings and implementation code remain part of the
compatibility signature; checkpoints from earlier code require a fresh solve.

Optional shared numerical cache failures fall back to the computed exact result
and increment `shared_numeric_errors`. Typed resource-limit failures remain
failed trials, including across process boundaries; invalid evidence is not
classified by matching words in an exception message. Enabled prefetch reserves
both current and pending scene vectors under the shared working budget before
allocating them.

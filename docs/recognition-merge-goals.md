# Replay-verified merge goals

`report-merge --solve` searches one shared configuration for one or more explicit
sets of Stash performer IDs. Members of each set are intended to be one person;
different sets must remain separate. These are evaluation constraints, never
forced assignments inside recognition.

```powershell
python stash_ai.py recognition report-merge --performer-set 123,456,789 --performer-set 234,567 --solve --max-trials 64 --yaml-output merged.yml --output merge-goals.json
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
observations prevent complete coverage certification and are included in the
best-effort result's unmet conditions. Empty performers cannot satisfy a goal
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

A candidate is admissible only when every goal member has retained observations,
all are present and assigned, and no final identity contains observations from
multiple sets. Admissible candidates always outrank inadmissible ones, and full
success outranks partial success. If none are admissible, the solver still selects
the best completed trial, including baseline, as `best_effort`.

The score next prioritizes disjoint sets, the worst set's assigned fraction, the
average assigned fraction, and fewer observation pairs joined across sets. Then
it prefers the worst set's fraction of fully consolidated performers, the total
consolidated performers, common-destination coverage (worst set, then average),
and fewer fragments. Assignment and common-destination coverage average fractions
per performer, so a performer with many samples cannot dominate. These continuous
metrics distinguish progress even when all complete-performer counts are zero.
Ties prefer fewer goal-to-outsider attachments, external coverage losses, splits
and joins, then fewer/smaller parameter changes. External partition changes are
not accuracy estimates; labels of other people are unknown.

Search measures strict representative similarities, explores joint thresholds,
margin/support gates and retained role-quality gates, then refreshes candidate
thresholds from rebuilt representatives. Each trial is verified by full replay.
Budgets bound attempted replays, including baseline. Search is heuristic and
non-exhaustive. Neither budget exhaustion nor an unsuccessful search proves
that no possible configuration works. The selected result is the best among
completed trials, not a guarantee of the global optimum. If no changed candidate
beats baseline, the current configuration is explicitly selected as unchanged.

With `sapiensid.enabled: true`, the search tunes all four matching gates:
`sapiensid.matching.face_floor`, `person_threshold`, `ambiguity_margin`, and
`min_support`. It starts with a joint measured hybrid proposal, when available,
then prioritizes independent single-gate probes before continuing the joint-pair
list. Face/body thresholds are tested below their current value by `.02`, `.05`,
`.1`, `.2`, and `.01`, interleaved with measured boundaries; corresponding increases
are explored as well. Values stay within `[0, 1]`. Margin and support have their
own decrease/increase probes. Every measured value remains eligible: dense nearby
boundaries are spread out in the search order instead of retaining only the three
nearest values.

These probes also run when no within-set pair proposal exists. In particular,
auxiliary/query samples may have usable body features while being absent from the
strict reference catalogue. Lowering `face_floor` can recover those observations;
absence of a reference-pair proposal must not disable hybrid search. Missing body
features themselves are never manufactured by probing thresholds.

Hybrid and face trials alternate. Each family starts with up to 13 unique baseline
explorations (subject to the total budget and available candidates), then alternates
baseline exploration with adaptive descendants. Improving descendants cannot
starve the independent baseline sweep, and a long joint-pair list cannot delay the
first three rounds of single-gate probes. All candidates still require full replay
and use the same coverage, separation and outside-impact ranking.

The baseline still counts toward `--max-trials`, so a budget of one only evaluates
the current configuration. `search.executed_by_family` and each trial's
`search_family` show which configurations were actually tested. `search_phase`
(`baseline_exploration` or `adaptive`) and `search_parent` explain how each trial
was generated. `search.executed_by_phase` summarizes this allocation, and
`search.tested_hybrid_values` lists the actual values used by completed trials,
including baseline. It excludes proposed but untested or interrupted candidates.

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
`partial_solution` (coverage and separation preserved, incomplete internal merges),
or `best_effort` (coverage or separation also fails). Unmet conditions identify
empty performers, observations missing from latest scans, withheld observations,
split performers, incomplete sets, and cross-set merges, with identities and
counts. `explanation.unresolved` repeats this list. For compatibility,
`recommendation` is populated only when a changed configuration wins;
`best_result` also represents an unchanged baseline. The top-level status remains
`baseline_retained` for an admissible partial baseline, while its quality is
`partial_solution`.

`--yaml-output` requires `--solve` and writes the selected complete configuration,
including best-effort and unchanged-baseline results. YAML comments identify the
trial, quality and unmet condition types; the JSON report contains the details.
For a partial input scope, YAML comments also identify the reduced validation
scope. Use the report's `run_argv` to replay exactly the validated scenes;
including excluded or recollected scenes requires revalidation.
Relative model/database/names paths become absolute so the exported file can be
relocated. Outputs must be distinct new files.
`run_argv` gives the exact scene selection as an argument array, avoiding shell
quoting differences. Code/configuration/evidence hashes record the tested state;
changed inputs require revalidation. Stash publication itself is not simulated.

Progress and the final best-result summary go to stderr, leaving JSON stdout
parseable. The summary includes per-set complete counts, assigned observations,
common-destination coverage, missing/withheld counts and unmet conditions.
`--progress-style off` suppresses solver messages. Ctrl-C during replay search
preserves completed trials and their best result, marks the result interrupted,
and exits 130. Incomplete trials are never eligible for selection. If no trial
completed, status is `no_completed_trials`, `best_result` is null and no YAML is
exported. Interrupting initial goal preparation may stop before a report exists.
Runtime uses configured rebuild workers and RAM budgets; GPU
inference is not repeated. Performance on a large library must be measured locally.

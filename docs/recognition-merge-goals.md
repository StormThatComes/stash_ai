# Replay-verified merge goals

`report-merge --solve` searches one shared configuration for one or more explicit
sets of Stash performer IDs. Members of each set are intended to be one person;
different sets must remain separate. These are evaluation constraints, never
forced assignments inside recognition.

```powershell
python stash_ai.py recognition report-merge --performer-set 123,456,789 --performer-set 234,567 --solve --max-trials 64 --yaml-output merged.yml --output merge-goals.json
```

Repeated `--performer-id` or `--identity-id` selects one set. The original
exactly-two-ID diagnostic remains available without `--solve`. Set syntax is
mutually exclusive with those selectors. Alias overlap across sets is rejected.
Without `--solve`, set input resolves retained goals; `--simulate` evaluates the
baseline. No Stash writes, model loading, media access or source YAML writes occur.

Each trial starts from the same read-only retained history in an independent RAM
database and calls the ordinary production rebuild. All latest complete scans in
the scope participate, including competitors. Incompatible latest scan contracts
require rescanning. Old members absent from the latest selection prevent complete
coverage certification. Empty performers cannot satisfy a goal vacuously.

A candidate is admissible only when every goal member has retained observations,
all are present and assigned, and no final identity contains observations from
multiple sets. The score prioritizes full success, the worst set's fraction of
fully consolidated performers, the total consolidated performers, and fewer
fragments. Ties prefer fewer external coverage losses, splits and joins, then
fewer/smaller parameter changes. External partition changes are not accuracy
estimates; labels of other people are unknown.

Search measures strict representative similarities, explores joint thresholds,
margin/support gates and retained role-quality gates, then refreshes candidate
thresholds from rebuilt representatives. Each trial is verified by full replay.
Budgets bound attempted replays, including baseline. Search is heuristic and
non-exhaustive. Neither budget exhaustion nor an unsuccessful search proves
that no possible configuration works. A best partial solution is explicitly
reported; the source configuration is retained if no admissible improvement wins.

With `sapiensid.enabled: true`, the search tunes all four matching gates:
`sapiensid.matching.face_floor`, `person_threshold`, `ambiguity_margin`, and
`min_support`. It starts with a joint measured hybrid proposal, when available,
then alternates hybrid and face trials. Each family has its own adaptive queue;
a large face threshold grid cannot consume the budget before hybrid trials run.
The baseline still counts toward `--max-trials`, so a budget of one only evaluates
the current configuration. `search.executed_by_family` and each trial's
`search_family` show which configurations were actually tested.

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

`--yaml-output` requires `--solve` and writes a complete configuration only for a
selected changed solution. Relative model/database/names paths become absolute
so the exported file can be relocated. Outputs must be distinct new files.
`run_argv` gives the exact scene selection as an argument array, avoiding shell
quoting differences. Code/configuration/evidence hashes record the tested state;
changed inputs require revalidation. Stash publication itself is not simulated.

Progress goes to stderr, leaving JSON stdout parseable. `--progress-style off`
suppresses solver messages. Ctrl-C during replay search preserves completed
trials, marks the result interrupted, and exits 130. Incomplete trials are never
eligible for selection. Interrupting initial goal preparation may stop before a
report exists. Runtime uses configured rebuild workers and RAM budgets; GPU
inference is not repeated. Performance on a large library must be measured locally.

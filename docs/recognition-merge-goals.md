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

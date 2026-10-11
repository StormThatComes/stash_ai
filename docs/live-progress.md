# Live console progress

`--progress-style auto` is the default. A terminal uses coordinated tqdm bars
and one bounded status area. Redirected output prints phase/bar milestones and
infrequent liveness messages. `compact` keeps the short layout; `verbose` permits
additional details; `off` disables decorative progress while retaining warnings,
errors and normal command summaries.

The console has one dedicated rendering owner. HTTP/CPU workers only publish
bounded snapshots; execution is never moved to the rendering thread. The owner
continues rendering while the command is blocked on Stash or worker results.
All bars and the renderer close on normal completion, error and Ctrl+C.

Phase elapsed time and time since a completed work unit are separate. Redraws,
heartbeats and task submission do not claim completion. Unknown totals remain
indeterminate. HTTP read status shows attempts, backoff and remaining caller
budget; mutations retain their single-attempt policy. SQL/GraphQL strategy
transitions report only sanitized reason classes.

The presenter keeps at most 32 status fields, 64 diagnostic counters and a bounded
64-entry queue of synchronous output actions. It does not retain scene payloads,
perform Stash requests, traverse cache objects or modify inference/evidence.

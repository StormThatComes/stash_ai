# Live console progress

`--progress-style auto` is the default. A terminal uses coordinated tqdm bars
and one bounded status area. Redirected output prints phase/bar milestones and
infrequent liveness messages. `compact` keeps the short layout; `verbose` permits
additional details; `off` disables decorative progress while retaining warnings,
errors and normal command summaries.

`recognition run` shares this owner with two work rows (selected scenes and VTT
frames in the active scene) followed by a status area. Publication
and names/portraits replace the work rows with their own stage totals. Remaining
time estimates cover each measured stage; redirected frame updates appear in
throttled heartbeats rather than one log entry per frame or scene. Recognition
quality/reference details and the names YAML starter require `verbose`.
The final recognition error summary retains at most six groups and two examples
per group (350 characters each), with exact counts and overflow totals. Distinct
reasons sharing an exception type remain available in auto/compact/off output.
The same sanitized, format-1 structure is stored in recognition run history under
`summary.errors`, including collected failures before interruption. Optional
console/snapshot failures cannot skip independent run-history finalization.

Terminal status separates phase, activity (`Stage`), current work, timing,
progress, rates and references into labeled rows. Activity appears directly
after the phase, before counters or ETA fields. Rows wrap to the current output
terminal width and resize without leaving stale activity or resource rows.
The status area uses at most 12 rows, with an overflow marker for exceptionally
large snapshots. Redirected heartbeats retain their single-line log format.

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

Threshold preflight separates inspected, checked, planned, unchanged, absent
evidence and blocked pairs, including per-model subtotals. No apply starts after
a failed preflight. Application advances after each successful transaction, so a
rolled-back batch contributes zero committed pairs. Dry runs label overlay changes
as planned. `--rethreshold-spool-limit-mb MIB` optionally aborts preflight before
application; zero keeps the existing unlimited behavior. Spool bytes are SQLite
pager allocation measured after a spool commit, rather than a filesystem free-space
estimate or a promise of durable threshold application.

Parallel merge measures workers executing evidence chunks, rather than treating
submitted futures as busy processes. A fixed shared ring observes chunks offered
to ordered map, executing chunks, computed chunks not yet received, and actual
successful/failed scene computation. Chunk capacity includes map's transient extra
submission; ready scene envelopes are separate from ready chunks. Ordered-head age
starts at worker entry, never submission. The parent still receives results in
order and publishes serially. Sparse transitions use bounded lock waits; if a
terminated process holds the lock, telemetry becomes unavailable.

Evidence/publication throughput and ready-backlog change use a bounded 30-second
window and show warming-up until enough time has been observed. Publication rate
advances at the durable local commit boundary. Cache snapshots read scalar retained,
configured and effective budgets plus pressure/eviction counters without traversing
entries or invoking eviction. Parent RSS is sampled at most once per second;
child RSS and accelerator utilization are unavailable. Cache budgets describe
headroom policy, not an operating-system memory limit. Inference reuses its buffer
and cache rows; other phases use one resource row below the status area.
Work rows and the bounded status area occupy at most 17 rows; redirected
liveness remains infrequent and diagnostics use stderr.

After interruption, stopping milestones describe only entered shutdown operations:
checkpoint saving, cancellation, worker termination, HTTP closure and spool release.
Cleanup failures remain visible without replacing the original error. One terminal
outcome separates confirmed mutation batches from failed/unknown acknowledgements
and from locally committed scene/pair counts. It never retries a mutation.

`--run-summary-json NEW.json` optionally writes schema version 1 for inference,
merge or cleanup. The path is exclusively reserved before execution; existing
files are never overwritten. Completed, failed, interrupted and planned outcomes
include observed counts, known not-started work, unresolved work, recovery state,
phase/timing data and configuration/selection fingerprints. Unknown affected-row
counts are null. Raw paths, URLs, payloads and error examples are omitted. Export
failure is diagnostic and cannot replace an existing execution error. Reporting
uses observed state without extra Stash reads, journal writes or inference.

The summary reservation uses an advisory lock released automatically when the
process exits; the final JSON name appears only after a complete validated
document is atomically published. A failed export leaves that name available
for another explicit run. Existing or concurrently created files stay intact.
Setup failures report setup rather than an unentered finalize phase, and
deliberately skipped phases include bounded reasons in both progress styles.

An illustrative local Python 3.14.7 scalar-update measurement (100,000 updates)
took 17.9 ms with progress off and 75.5 ms with auto: approximately 0.58 µs extra
per update. 100,000 throttled resource polls took 10.5 ms. These are coordination
microbenchmarks, rather than an end-to-end inference or real-terminal performance
guarantee; model/API costs and terminal speed vary.

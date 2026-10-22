# Live console progress

`--progress-style auto` is the default. A terminal uses coordinated tqdm bars
and one bounded status area. Redirected output prints phase/bar milestones and
infrequent liveness messages. `compact` keeps the short layout; `verbose` permits
additional details; `off` disables decorative progress while retaining warnings,
errors and normal command summaries.

`recognition run` uses a numbered phase plan: selection, lazy model configuration,
VTT scanning, identity rebuild, Stash publication, performer selection, names
preflight, metadata and names starter (nine phases with metadata enabled).
`recognition portraits` has its own shorter plan. Phase numbers describe the
execution order, rather than a percentage of total runtime. Branches with no
eligible work explicitly report why they were skipped; a failed rebuild skips
publication with that failure reason.

Two work rows show the current scene/person total and the observations, references
or candidates inside that work. Publication and names/portraits replace these rows
with their own totals. Each bar owns its count, throughput and remaining-time
estimate; these metrics are not repeated in the terminal status area. The status
instead shows the numbered phase, named rebuild step, current activity, elapsed
phase time and time since completed work **in that phase**. Reference memory and
detailed CPU queue counters require `verbose`. Estimates cover the active work
row only; there is no whole-run ETA. Unknown totals remain indeterminate.

Redirected recognition output prints phase start/outcome messages and one named
work snapshot at most every 15 seconds, including during continuously busy work.
Each snapshot gives the active row labels, counts, rates and row-local ETAs.
Per-bar start/end milestones and rebuild-step transitions require `verbose`;
normal runs do not log two extra milestones for every small replay operation.
Recognition quality/reference details and the names YAML starter also require
`verbose`. `off` retains summaries and errors without decorative output.

Recognition also measures identity replay (copy, matching, historical binding,
transactional apply and cached validation), name preflight, model assets, and
portrait candidates. Scene-level work uses the first row; observations,
references or candidate checks use the second. UI refreshes are throttled and
record totals use scalar counts rather than materializing embeddings/JPEGs.
`off` and standalone calls skip optional total queries. Unknown totals remain
indeterminate; a failed transaction never completes its apply bar.
Local appearance grouping reports both selected scenes and successfully grouped
observations, using an optional scalar observation count. Provisional identity
comparisons report blocks and groups; frozen consolidation reports competition
checks and component application separately. Rebuild dependency validation,
checkpoint restoration, strict cores, appearance banks, historical reconciliation,
auxiliary assignment and final reference validation retain a named step even when
SQL/native work has no measurable total. Nested work restores its parent's labels
and counters, and completed detail rows clear their own counters.
`Validate rebuild input catalogue` keeps the scene total on the first row and
shows face-observation hashing and assignment-evidence validation on the second,
with per-batch observation totals, throughput and ETA. A scene batch completes
only after both required passes succeed.
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

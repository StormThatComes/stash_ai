# Parallel merge solver

The solver keeps recognition thresholds and retained-input contracts unchanged.
Execution options have a joint CPU ceiling (1–16), a trial count and an optional
memory admission ceiling. Each trial reserves two rebuild database ceilings,
reference/plan ceilings and its numerical queue. This conservative admission
estimate is not a hard operating-system RSS limit. Lower memory admission reduces
concurrency; it never silently removes references or competitors.

Run `python -m benchmarks.benchmark_solver_parallel --max-trials 16` for repeated
1/2/4/8/16-CPU comparisons. These synthetic fixtures measure search behavior, not
real-library accuracy or guaranteed speedup. Solver profiling is opt-in and
includes rebuild phases, SQL, numerical worker utilization and process CPU time.

Implementation stack: resources/profiling; isolated evaluation; parallel search;
shared inputs/cache; face/hybrid kernels; complete body proofs and rivals.

## Usage

```powershell
python stash_ai.py recognition report-merge --performer-set 900,901 --solve --solver-cpus 16 --solver-trials 4 --solver-memory-gib 96 --profile-solver --checkpoint solver.json
```

Default execution remains one trial using the existing rebuild settings. Explicit
solver execution limits BLAS to one native thread and allocates inner workers
from the joint CPU ceiling. Baseline may use the whole CPU allocation. Trial
workers use threads for local cohorts to avoid nested process pools.

`--solver-scheduling batch` generates bounded batches and applies results in
submission order. `async` refills on completed results; its trajectory depends on
completion timing. The journal records dispatch/retirement order so either mode
can resume its original trajectory. Pending configurations are deduplicated.
Completed worker results are checksummed before application; interruption retains
finished results, cancels queued work and signals running workers cooperatively.
Unexpected worker exceptions propagate. Resource-limited trials are reported as
failed and never become winners. Only the coordinator writes the winning YAML.

## Console progress

The console first prints the effective CPU/trial limits, scheduling mode and
trial budget. Baseline runs before the parallel trials. `trial-N: start` describes
the submitted configuration; a spawned worker may still be starting up.

Every five seconds while waiting, the coordinator prints pool counts and one
status line per pending trial. For example (illustrative values):

```text
Merge solver workers: pool running=1 queued/starting=0 ready=1 retired=0 limit=2 elapsed=65s scheduling=batch waiting_for=trial-1
Merge solver trial-1: status pid=1234 elapsed=60s stage=Match selected scene stage_elapsed=12s last_update=1s ago scene=7 | Rebuild selected identities=6/33 scene | Match face observations=128/400 item | last_completed=1s ago
Merge solver trial-2: status result ready; awaiting application
```

- `running` counts trials whose worker has reported its start. `queued/starting`
  means a worker start has not yet been observed, including process startup.
- `ready` counts finished/restored results waiting for application. `retired`
  counts parallel results handed to the coordinator, excluding baseline.
- Worker lines show PID, trial and stage durations, scene when available, and
  up to two current work counters. Unknown totals are shown as `?`; there is no
  fabricated overall trial percentage or ETA.
- `last_update` is the age of the last observed stage/work update;
  `last_completed` is the age of the last observed increase in completed work.
  Periodic worker telemetry does **not** reset those ages. A long age can mean
  blocking SQL/native work or a phase without counters; it does not diagnose a
  deadlock. `last_report` separately shows how recently the worker reporting
  thread sent its snapshot, without claiming computational progress.
  Counters are sampled and may lag very short bursts of work.
- `ready evaluation=finished` (or `failed`) is printed once as soon as the
  coordinator collects a result, even if batch ordering delays its application.
  `waiting_for=trial-N` identifies that ordering dependency. The existing
  `complete` line and set details appear after application, with trial duration,
  completed-trial count (including baseline), and the best applied result.

Preparation, prediction collection and solver feedback are named stages too.
Worker snapshots use a bounded, nonblocking channel and are retried once a second;
slow output cannot create an unbounded telemetry backlog. Only the coordinator
writes console messages to stderr, so JSON stdout stays clean. Scheduling,
checkpoint order and winner selection do not consume telemetry; batch order
stays unchanged and async search still depends on completion timing.
`--progress-style off` disables the telemetry channel and worker reporting
threads. `--profile-solver` is not required for live progress.

Each trial preserves full production replay, chronological identity updates,
competitors, exclusions and original-core guards. Parallel ablations are separate
experiments, never automatically combined. More trials per second do not imply a
better result per second: compare both quality-at-time and time-to-quality.

Spawned trials share content-addressed packed numerical banks and matrix products
through read-only memory mappings in a disposable directory beside the seed.
A SQLite reservation ledger bounds published arrays; contention falls back to
exact local computation. Private replay databases still contain their own retained
rows. Shared files contain no executable pickle and never cache merge decisions.
The directory outlives workers and is removed after their shutdown, including on
interruption. With sufficient inner CPU budget the next scene is prefetched on an
isolated read-only connection; its CPU slot is deducted from numerical workers.

Face diagnostics use the same CPU allocation as the rebuild. Complete reference
banks are measured concurrently, preserving float64 tile order, weakest-witness
tie breaking and coordinator-owned caches. Hybrid nomination multiplies bounded
centroid tiles concurrently; every incident rival remains eligible. A batch
retires completely before exposing its results, so consumers can safely request
exact proofs from the same pool. Oversized work uses the original serial path.

Global body consolidation now batches exact proofs across candidate pairs,
original cores and inherited face vetoes. Witness loading and comparison-budget
charges remain on the SQLite owner. Workers retain the original float32 dot and
accumulation order. Both all-competitor centroid passes use bounded parallel
matrix tiles; every rival within the rounding envelope still receives an exact
proof. Near thresholds, weak rivals and missing original witnesses keep their
existing acceptance rules. No top-k truncation or approximate merge acceptance
is introduced.

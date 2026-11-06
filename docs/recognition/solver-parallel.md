# Parallel merge solver

[Documentation index](../README.md) · [Recognition workflow](../facial-recognition.md)

- [Usage](#usage)
- [Resource profiles](#resource-profiles)
- [Console progress](#console-progress)

The solver keeps recognition thresholds and retained-input contracts unchanged.
Execution options have a joint CPU ceiling (1–16), a trial count and an optional
memory admission ceiling. Each trial has one shared working budget for its
history database, inner replay database, reservations and evictable cache.
Additional reference/plan/queue headroom covers allocations outside that accounting.
The configured rebuild limit is a maximum: the solver divides its total memory
budget across requested trials and enforces a smaller working limit when needed.
This conservative admission estimate is not a hard operating-system RSS limit. If CPU or memory admission
cannot accommodate the requested trial count, the command fails before baseline
or candidate trials instead of silently reducing concurrency. The error reports
the requested/admissible counts and every insufficient resource, including CPU
budget/availability or RAM budget, per-trial estimate and required total.

Run `python -m benchmarks.benchmark_solver_parallel --max-trials 16` for repeated
1/2/4/8/16-CPU comparisons. These synthetic fixtures measure search behavior, not
real-library accuracy or guaranteed speedup. Solver profiling is opt-in and
includes rebuild phases, SQL, numerical worker utilization and process CPU time.

Implementation stack: resources/profiling; isolated evaluation; parallel search;
shared inputs/cache; face/hybrid kernels; complete body proofs and rivals.

## Usage

```powershell
python stash_ai.py recognition report-merge --performer-set 900,901 --solve --face-config facial_recognition_solver.yml --solver-cpus 16 --solver-trials 4 --solver-memory-gib 64 --profile-solver --checkpoint solver.json
```

Default execution remains one trial using the existing rebuild settings. Explicit
solver execution limits BLAS to one native thread and allocates inner workers
from the joint CPU ceiling. Baseline may use the whole CPU allocation. Trial
workers use threads for local cohorts to avoid nested process pools.

`--solver-cpus 16` alone still runs one trial. Add `--solver-trials 4` to
request up to four simultaneous candidate trials. Preparation and baseline
must finish before these trials can start; neither phase is a parallel search.

`--solver-scheduling batch` generates bounded batches and applies results in
submission order. `async` refills on completed results; its trajectory depends on
completion timing. The journal records dispatch/retirement order so either mode
can resume its original trajectory. Pending configurations are deduplicated.
Completed worker results are checksummed before application; interruption retains
finished results, cancels queued work and signals running workers cooperatively.
Unexpected worker exceptions propagate. Resource-limited trials are reported as
failed and never become winners. Only the coordinator writes the winning YAML.

## Resource profiles

The repository provides two complete v2 configurations for a host allocating
16 CPUs and 64 GiB RAM to recognition:

| Setting | `facial_recognition.yml` (run) | `facial_recognition_solver.yml` (solver) |
| --- | --- | --- |
| Rebuild working maximum | 40 GiB | 10 GiB per trial |
| Evidence / matrix cache | 8 / 4 GiB | 2 / 1 GiB per trial |
| CPU work queue | 1 GiB | 256 MiB per trial |
| Face / body reference limits | 4 / 4 GiB | 2 / 2 GiB per trial |
| Body matching plan limit | 2 GiB | 512 MiB per trial |
| Rebuild workers / cohort backend | 16 / processes | CLI allocation / threads |
| Hybrid nomination device | measured CPU or CUDA (`auto`) | CPU |
| Inner replay storage | SSD | SSD; mutable trial history still uses RAM |

The solver YAML does **not** set the number of trials or the total CPU/RAM
budget. Pass all three flags from the example above. At 16 CPUs and four trials,
the planner assigns four CPU slots to each trial: a coordinator and three inner
slots, one of which is used for scene prefetch. Baseline runs alone with the
larger CPU allocation but the same per-trial memory limit.

The conservative maximum admission is **14.75 GiB per trial**: 10 GiB working,
2 GiB face references, 2 GiB body references, 0.5 GiB plans and 0.25 GiB queue
headroom. Four trials therefore admit at most **59 GiB**, leaving 5 GiB of the
explicit 64 GiB budget unassigned. The 2 GiB evidence cache is already inside
the working budget, and the 1 GiB matrix cache is inside the evidence cache.
Admission may reserve less for references when the retained input is small.
These are not OS RSS caps or a guarantee that every 10,000-scene replay fits.
If a trial exceeds its working limit, reduce the trial count and raise its
configured working maximum only within the total machine budget.

Both files are standalone templates with the same model and matching settings;
neither inherits a locally edited configuration. For an existing tuned setup,
copy your current YAML to `facial_recognition_solver.yml`, then copy only
`execution.rebuild` and `matching.face.samples.references.max_reference_bytes`
from the shipped solver profile. Keep both files in the same directory so
relative database and model paths keep their meaning. Preserve your model,
detection, matching and anchor-rule settings when transferring solver results
back to the run configuration; keep the run resource limits.

The 96 GiB `replay_snapshot_bytes` setting is a per-image disk ceiling. Source
snapshots, parallel private replay images, caches and the recognition database
can coexist, so it does not enforce a combined 300 GB SSD budget. Measure disk
usage on retained subsets before running all 10,000 scenes concurrently.
Rebuild checkpoints are disabled for solver trials; use `--checkpoint` for
solver search recovery. `--profile-solver` controls trial profiling.

## Console progress

Before copying the report snapshot, the console prints `execution: request`
with the requested CPU/trial limits and a reminder that baseline runs alone.
After resource planning, `execution: plan` prints the admitted limits, before
checkpoint validation, diagnostic measurements and replay preparation. Previously
this plan was hidden until all those stages completed.

An admitted plan preserves `requested_trials`. Insufficient resources produce
an error with a nonzero exit code, including with `--progress-style off`.
No baseline, candidate trial, checkpoint or result export is started by an
admission failure. Automatic memory admission uses 80% of available RAM.
The per-trial amount is conservative admission accounting, not measured RSS.
`rebuild_memory_gib` shows the effective per-trial working limit and
`configured_rebuild_memory_gib` shows the original YAML maximum. For example,
a 30 GiB configured maximum and a 64 GiB total solver budget with four trials
give each trial at most 16 GiB including additional headroom; the solver no
longer demands two full 30 GiB allocations per trial. Both private RAM databases
and other registered work share that trial's working limit, with SQLite page
ceilings updated as reservations change. Source snapshots remain on disk under
`execution.rebuild.replay_snapshot_bytes`.

The baseline uses the same effective memory limit as candidate trials, including
spawned workers and reused worker caches. Runtime limits are included in checkpoint
signatures and never replace the user's YAML values or the winning configuration.
The 1 MiB schema floor plus headroom must fit every requested trial; otherwise
admission fails before evaluation. Admitting a budget does not guarantee the input
fits it: a larger replay fails with a typed resource error naming the owner, actual
shared limit and reservation totals. Reducing trial count increases available
memory per trial. Automatic serial execution retains its existing behavior when
the configured ceiling exceeds detected available RAM. An explicit
`--solver-memory-gib` replaces automatic admission and must be chosen to fit
the machine; it does not allocate or enforce an operating-system RAM limit.

`trial-N: start` describes the submitted configuration; a spawned worker may
still be starting up. `--progress-style off` hides both startup and worker lines.

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

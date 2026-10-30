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

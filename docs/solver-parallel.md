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

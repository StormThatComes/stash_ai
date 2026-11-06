# Stash AI documentation

Start with the [quick start](../README.md#quick-start). Commands in every guide
are run from the repository root. This index separates current workflows from
historical implementation notes and measurements.

## Tagging

| Task | Guide |
| --- | --- |
| Run inference, reuse evidence and publish results | [Tagging workflow](tagging/README.md) |
| Configure models, temporal rules and categories | [Configuration](tagging/configuration.md) |
| Understand retention, rethresholding and input verification | [State and cache](tagging/state-and-cache.md) |
| Re-merge stored evidence | [Merge](tagging/merge.md) |
| Preview exact publication changes | [Dry-run publication plan](tagging/publication-preview.md) |
| Exclude tags and remove published results | [Exclusions](tagging/exclusions.md) · [SQL cleanup](tagging/cleanup.md) |
| Manage marker organization | [Marker groups](tagging/marker-groups.md) · [Occurrence points](tagging/occurrences.md) |
| Diagnose reuse, progress and bottlenecks | [Run diagnostics](tagging/diagnostics.md) · [Merge profiling](tagging/merge-profiling.md) |
| Review model utility, tags and missing cache | [Reports](tagging/reports.md) |
| Tune workers, queues and memory | [Performance](tagging/performance.md) |

## Recognition

| Task | Guide |
| --- | --- |
| Set up and run recognition; migrate YAML | [Recognition workflow](facial-recognition.md) |
| Choose face and pose models | [Face models](recognition/models.md) · [Person detectors and pose backends](recognition/pose-backends.md) |
| Configure body evidence and hybrid matching | [SapiensID](sapiensid-hybrid.md) |
| Understand anchors, auxiliary observations and views | [Identity matching](recognition/identity-matching.md) · [Reliability](recognition/reliability.md) |
| Inspect detections and compare parameter values | [Detection audits](recognition/detection.md) |
| Refresh portraits, crop policy and names | [Portraits](recognition/portraits.md) · [Portrait performance](recognition/portrait-performance.md) |
| Export evidence and inspect fragmentation | [Reports](recognition/reports.md) |
| Tune merge goals | [Solver search](recognition/solver.md) · [Solver JSON](recognition/solver-report.md) |
| Control parallel trials and resume | [Parallel solver](recognition/solver-parallel.md) · [Feasibility and checkpoints](recognition/solver-feasibility.md) |
| Maintain evidence and remove orphans | [Recognition database](recognition/storage.md) |
| Investigate run time, memory and recovery | [Execution](recognition/performance.md) · [Benchmarking](recognition/benchmarking.md) |
| Evaluate accuracy and retained evidence | [Evaluation](recognition/evaluation.md) |

## Maintenance and development

- [Application updates](maintenance/updating.md) and [legacy migrations](maintenance/legacy-migrations.md)
- [Tagging database health, upgrades and compaction](maintenance/database.md)
- [Fresh database initialization](maintenance/database-initialization.md)
- [Interrupted operations and recovery](maintenance/interrupted-operations.md)
- [Console progress](maintenance/progress.md) and [network deadlines](maintenance/network.md)
- [Tests, CI and documentation conventions](development.md)
- [Architecture and module responsibilities](ARCHITECTURE.md)
- [Benchmarks](../benchmarks/README.md) and [dependency profiles](../constraints/README.md)
- [Security](../SECURITY.md), [third-party notices](../THIRD_PARTY_NOTICES.md),
  [FaceOcc attribution](recognition/faceocc-attribution.md) and [expression-model attribution](recognition/expression-attribution.md)

## Historical records

[Archived PR series and measurements](archive/README.md), [validation reports](validation/README.md)
and [saved benchmark results](../benchmarks/results/README.md) describe specific
earlier revisions. Their test counts, timings and merge-order instructions are
historical records, not the current application specification.

The updater installs the root README and Markdown/text files under `docs/`.
It does not install tests or benchmark programs. For those, use a repository checkout.

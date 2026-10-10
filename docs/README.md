# Stash AI documentation

Documentation is organized into current user guides and historical
measurement artifacts.

## Current documentation

- [Quick start and configuration](../README.md)
- [Current user guide](README_multimodel.md)
- [Module boundaries and class responsibilities](ARCHITECTURE.md)
- [SQL cleanup](UPGRADE_cleanup_sql.md)
- [Performance and memory](UPGRADE_performance.md)
- [Merge and fast path](UPGRADE_merge.md)
- [Exclusions and cleanup](UPGRADE_exclusions.md)
- [Occurrence points](UPGRADE_occurrences.md)
- [Marker groups](UPGRADE_marker_groups.md)
- [Abandoning interrupted operations](UPGRADE_abandon_pending.md)
- [Initializing a fresh database](UPGRADE_db_init.md)
- [Merge diagnostics](UPGRADE_diagnostics.md)
- [Benchmarks](../benchmarks/README.md)

The `UPGRADE_*.md` filenames are historical. Their contents document
the current behavior of the corresponding features; they are not
installation instructions for a particular old release.

## Historical artifacts

Files under [validation/](validation/README.md) and saved results in
`benchmarks/results/` document specific earlier measurements. They can
help with regression analysis, but historical timings and test counts
are not the current project specification.

Application code lives in the `stash_ai/` package. Root-level files
include the `stash_ai.py` and `update_stash_ai.py` launchers,
YAML configuration, and default `tagger_sources.sqlite3` state database.
The updater does not download documentation, tests, or benchmarks.

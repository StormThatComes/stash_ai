# Tagging database maintenance

[Documentation index](../README.md)

Tagging uses `tagger_sources.sqlite3`. Recognition has a separate [database guide](../recognition/storage.md).
Back up the state database and configuration before an upgrade.

## Commands

```powershell
python .\stash_ai.py --db-report
python .\stash_ai.py --db-check
python .\stash_ai.py --db-update --dry-run
python .\stash_ai.py --db-update
python .\stash_ai.py --db-repair --dry-run
python .\stash_ai.py --db-compact --dry-run
python .\stash_ai.py --db-compact
```

Use `--state-db PATH` to select a different tagging database. To adopt existing
Stash results into a new database, use [database initialization](database-initialization.md).

### Database health, upgrades, and compaction

For the separate facial recognition database, use `python stash_ai.py recognition
db-status` to inspect table/index sizes and estimated savings, and `python
stash_ai.py recognition db-vacuum` to compact it. `--dry-run` previews the operation;
`--faces-db` selects an existing file. See [recognition database maintenance](../recognition/storage.md#recognition-database-status-and-vacuum).

`--db-update` upgrades the database selected by `--state-db` from
schema **v18 or newer** to the current **v27**, across all scopes.
Normal execution and `--dry-run` require the current schema; new databases
start on v27. Migration adds database-rebuild state but does not adopt
existing Stash results. Adoption is authorized only by an explicit
`--db-init` on a new/empty database.

The updater uses the regular writer lock and runs offline, without Stash or
model configuration. It retains numeric thresholds on current results,
publication ownership, pending publication/cleanup and recovery state.
Older results are marked for reinference and their checkpoints removed.
Snapshot versions v4/v5/v6/v7 are converted in one pass, preserving shared
evidence. Retired algorithm settings are removed while geometry and evidence
references stay intact. Updating cannot reconstruct lost detections.
Pre-format-5 inference requires a rerun.

Result metadata is populated after conversion. Frames in active results
must be chronologically ordered. For databases predating v23, unordered
records are marked for reinference and dependent publications receive
`requires_rebase`. Stored payloads are not sorted or re-signed; ordered
format-5 results remain valid. Reads and writes reject missing provenance,
threshold and retention-floor metadata rather than silently dropping votes.

Normal cleanup accepts only the current fingerprint. An interrupted cleanup
with a legacy fingerprint requires an explicit offline conversion using
**the same YAML files and output scope** (`--cleanup-only`,
`--cleanup-markers-only`, or `--cleanup-scene-tags-only`). The tool validates
every per-model policy, updates only the pending key, and retains the
generation, masks and recovery state:

```bash
python -m stash_ai.tools.cleanup_pending_migration --state-db /path/to/tagger_sources.sqlite3 --cleanup-only --marker-core-exclusions marker_core_exclusions.yml --scene-tag-exclusions scene_tag_exclusions.yml --dry-run
```

Remove `--dry-run` after checking the plan, then resume the original cleanup.
The migration tool neither contacts Stash nor loads models.

For databases **older than v18**, first use compatible commit
[`c5bcd887bafe76b7ee2a9de944666799c9a0fc30`](https://github.com/StormThatComes/stash_ai/commit/c5bcd887bafe76b7ee2a9de944666799c9a0fc30)
in a separate directory with its dependencies. Back up the database and
point the bridge at the same state file:

```bash
git clone https://github.com/StormThatComes/stash_ai stash-ai-upgrade-bridge
git -C stash-ai-upgrade-bridge checkout c5bcd887bafe76b7ee2a9de944666799c9a0fc30
python stash-ai-upgrade-bridge/stash_ai.py --db-update --state-db /path/to/tagger_sources.sqlite3
```

Then run `--db-update` from the current installation. The compatibility
commit converts historical series/imports to v20, which the current
release can upgrade directly.

`--db-update --dry-run` reports source and target schema versions without
migration, file writes or locking. Databases older than v18 receive an
upgrade-path message; foreign or newer schemas are rejected. Payload
conversion, retiring active results, and updating the version execute
within one transaction. Missing-table creation is idempotent, and an
interrupted upgrade can be retried.

`--db-check` performs a deep, read-only consistency check across every
database scope. On older schemas, it only checks the SQLite integrity
and schema preflight, and requires `--db-update` before deep validation.
In addition to a full SQLite `integrity_check`, it validates:
- `results` against `result_index`, including recorded last-inference signatures;
- decoding of persistent BLOBs and valid model/output kinds;
- checkpoint/frame/analysis relationships and checkpoint dimensions;
- cleanup/recovery generations;
- required scene, frame and marker-interval fields, tags, times,
  probabilities and saved temporal statistics;
- checkpoint frames, including `scene_tags`, and float32 values
  (rejecting `NaN`, infinities and values outside `[0, 1]`).

Active results must satisfy the current inference contract. Legacy
intervals and incomplete statistics are only accepted in archived evidence.
Imports and `score: null` assessments are rejected. Retired
`legacy_imports` and `refinement_progress` tables are flagged without
reading their payloads. Some missing snapshot/recovery associations are
warnings. Pending publication/cleanup and refresh requests are treated as
in-progress work, not automatically as errors.

The check does not connect to Stash, load models, repair data, or modify
anything. A progress bar displays the current phase (e.g. SQLite integrity,
results/index, state payloads, checkpoints). Exit code `0` indicates no
logical errors, `1` indicates inconsistencies; warnings alone do not fail it.

`--db-report` needs neither Stash nor models. It reports schema version,
SQLite `quick_check`, DB/WAL/SHM size, free pages, result counts grouped by
model/output kind, and checkpoint, refresh, pending/recovery and merge-check
counts for the current scope. It does not print tag names, scene IDs or scores.

`--db-compact` runs SQLite `VACUUM` under the normal writer lock. It first
checks database integrity, free disk space, and the absence of unfinished
merge/cleanup operations. `--dry-run` runs only the preflight and estimates
space reclaimable from unused pages. Recovery evidence and checkpoints
are preserved.

### Offline schema updates and repairs

`--db-update` checks schema shape and completed data revisions first.
An already current, complete database returns without decoding
inference BLOBs or running SQLite `quick_check`. The small metadata/key
scan can grow with result count. Required historical conversions
(minimum schema v18) run automatically before recording completion.
A previously unrecorded data revision may require a one-time full pass.
A DDL-only upgrade with completed data revisions does not require
a global repair.

Use `--db-repair` for an explicit full scan on the current schema:
it repairs derived metadata and identical scene aggregate duplicates
while preserving archived/publication proofs. Conflicting duplicates
stop the operation and roll it back. Investigate semantic corruption
with `--db-check`; a fast preflight cannot prove arbitrary BLOB contents.
Update and repair both work offline and support `--dry-run`; the repair
preview uses a temporary SQLite copy. Neither command runs models,
contacts Stash, or globally requests reinference.

Repair spools canonical BLOB rewrites and signature replacements into
a disk-backed temporary SQLite database with a bounded page cache.
It scans each original archive record once and applies evidence,
metadata and publication-reference changes in one transaction.
Interruption before commit rolls back the entire repair.
Available memory must still fit one decoded evidence/publication record,
and temporary disk space must fit rewritten BLOBs.

CLI maintenance shows the active phase and scanned/written row counts.
A current-schema update that restores metadata or completes a data
revision reports that work rather than claiming no changes were
needed. A complete no-op creates no progress bar and performs
no full scan.

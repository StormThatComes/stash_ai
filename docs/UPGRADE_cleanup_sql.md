# SQL cleanup for large libraries

This guide describes the current cleanup path for large Stash libraries.

## Strategies

`--cleanup-strategy auto` is the default:
- tries read-only `querySQL` on Stash 0.31.1+ after validating the
  required SELECT/schema;
- otherwise falls back to GraphQL scene reads.

`--cleanup-strategy sql` requires SQL candidate discovery and
fails if it is unavailable. `--cleanup-strategy scenes` forces
GraphQL discovery.

```powershell
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql --dry-run
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql
```

SQL is **read-only**. The application never runs direct
`DELETE`/`UPDATE` statements on Stash's database; every
modification uses the Stash API.

## Cleanup scope

Supported modes:
- `--cleanup-only` — scene tags and markers;
- `--cleanup-markers-only`;
- `--cleanup-scene-tags-only`.

Global entries in `scene_tag_exclusions.yml` may remove
excluded tag definitions and their relationships.
`marker_core_exclusions.yml` applies only to managed marker
core labels and does not remove the tag definition.

Partial cleanup modes preserve tag definitions and only
modify the selected relationship type.

## Why SQL is faster

The SQL path:
- uses fixed parameterized SELECT statements;
- paginates by key/ID instead of COUNT+OFFSET;
- fetches candidate IDs instead of full marker objects;
- avoids retaining library-wide lists in memory;
- re-verifies state before API writes.

For model-dependent rules, only scenes requiring an exact
recalculation are fetched. `SceneReader` loads their tags
and markers in batches of `SCENE_PAGE_SIZE` scenes
(default 100, maximum 1000), with no individual GraphQL
request per scene and no sprite/VTT path reads.
The next page is fetched only after processing the previous
one. Scene state is refreshed before writes; unchanged
scenes need no extra read.

If snapshot SELECT is unavailable or fails mid-run, remaining
detail reads fall back to GraphQL without reprocessing completed
scenes. This also applies under `--cleanup-strategy sql`:
SQL is mandatory for candidate discovery, but detail reads
have their own fallback. `scenes` remains GraphQL-only.
Dry-run uses the same batching and preserves previews of
global removals even during scene re-reads.

## Batch sizes

| Option | Default | Range |
|---|---:|---:|
| `--cleanup-page-size` | 2000 | 1–10000 |
| `--cleanup-write-batch-size` | 200 | 1–1000 |

If server mutations are slow, reduce the write batch:

```powershell
python .\stash_ai.py --models all --cleanup-only --cleanup-strategy sql --cleanup-write-batch-size 50
```

## Retries and resume

Read-only GraphQL/SQL can be retried after transient errors.
GraphQL mutations are **not automatically retried** because
Stash might have completed the write before a timeout.

After interruption, rerun the same operation with the same
state and policy. The application rereads current objects
and continues with those still requiring changes.

To abandon and apply new rules instead:

```powershell
python .\stash_ai.py --models all --cleanup-only --abandon-pending
```

See [abandoning interrupted operations](UPGRADE_abandon_pending.md).

## Dry-run

`--dry-run` writes neither to Stash nor to the source database.
With a supported current database, the source is opened
read-only and a temporary SQLite overlay stores the preview:
the entire state file is not copied into RAM.

## Pruning unused tag definitions

After successful cleanup, prune unused leaf tags:

```powershell
python .\stash_ai.py --models all --cleanup-only --prune-empty-tags
```

Or run pruning alone:

```powershell
python .\stash_ai.py --prune-empty-tags-only --dry-run
```

Obviously used tags are rejected first. Full Stash count
verification is performed only for remaining candidates.

## Diagnostics

Cleanup summaries include SQL-read counts, fetched rows,
API writes and object/tag mutations. For synthetic comparisons:

```powershell
python -m benchmarks.benchmark_cleanup_sql --markers 100000
```

The benchmark never connects to a production Stash instance.

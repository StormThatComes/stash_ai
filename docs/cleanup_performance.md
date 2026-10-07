# Popular-tag deletion: measured decision

The global cleanup path keeps one confirmed native `tagsDestroy` mutation per
tag after deleting its primary markers. It does **not** first detach every
remaining marker/scene association. The proposed extra step did not reduce total
cost in the indexed SQLite benchmark, even with API, network, object loading and
hooks excluded from the detachment cost.

## Source contract

The benchmark follows the relevant implementation in Stash v0.31.1:

- [tagsDestroy resolver](https://github.com/stashapp/stash/blob/v0.31.1/internal/api/resolver_mutation_tag.go)
  runs native tag destruction in a transaction and retains the tag-destruction hooks.
- [Tag.Destroy](https://github.com/stashapp/stash/blob/v0.31.1/pkg/sqlite/tag.go)
  clears the tag image, rejects remaining primary marker references and destroys
  the existing tag; [repository deletion](https://github.com/stashapp/stash/blob/v0.31.1/pkg/sqlite/repository.go)
  ultimately deletes the tag row.
- [Association migration 35](https://github.com/stashapp/stash/blob/v0.31.1/pkg/sqlite/migrations/35_assoc_tables.up.sql)
  gives marker, scene and image tag associations foreign-key cascades, composite
  primary keys and tag-ID indexes.
- [Database connection](https://github.com/stashapp/stash/blob/v0.31.1/pkg/sqlite/database.go)
  uses WAL, NORMAL synchronization, foreign keys and immediate write transactions.

The disposable fixture models those three association tables and preserves
manual scene/image tags, marker owner tags, objects and primary tags. The blocked
tag is only an extra tag: its primary markers have already been removed by the
preceding cleanup stage. Both strategies must produce the same complete fixture
hash and pass foreign-key checks.

## Measurements

Python 3.14.7, SQLite 3.53.1, three repeats per case, batches of 200. Fixture
construction/copy, validation and hashing are excluded from timings. Each case
uses a fresh database; strategy order alternates. Recorded trials are in
[tag_cascade_2026-10-07.json](../benchmarks/results/tag_cascade_2026-10-07.json).

| Marker associations / scene associations | Cascade total, median | Prior detach total, median | Longest cascade transaction, median | Longest detach transaction, median | Native mutations: cascade / detach |
|---|---:|---:|---:|---:|---:|
| 100,000 / 1,000 | 125.6 ms | 187.1 ms | 125.5 ms | 1.58 ms | 1 / 506 |
| 1,000,000 / 10,000 | 1,260.1 ms | 1,723.0 ms | 1,260.1 ms | 2.34 ms | 1 / 5,051 |

The detached variant performs keyset discovery through the tag/rowid index,
then deletes each object's target join in bounded transactions and finally
destroys the tag. It intentionally omits Stash's object reads, updates, mutation
responses and hooks: this is an optimistic storage-cost model. Native mutation
counts describe the calls a corresponding API strategy would need; those calls
are not executed. For illustration, **assuming** 5 ms overhead per call adds
25.25 seconds for the 5,050 additional calls in the million-marker case. This is
an arithmetic scenario, not a measured API latency.

Detachment reduces the longest write transaction but increases total work by
37–49% in these cases before API overhead. These measurements do not establish
end-to-end Stash latency or plugin-hook cost. Keep the native cascade as the
default. Reconsider optional bounded detachment only if a real Stash workload
shows that write-lock duration is the limiting problem and its additional API
cost is acceptable.

## Behavior retained

All Stash writes still use native mutations with strict confirmation. Cleanup
intent remains durable until completion, ambiguous writes are not retried by the
transport, and partial-output cleanup still uses bounded `REMOVE` mutations.
The request heartbeat from PR7 makes a long native deletion visible.

The unused `StateStore(cleanup_preview=...)` parameter is removed. Read-only
preview behavior already follows `dry_run`; its source database, pending intent
checks and sparse overlay behavior are unchanged. No state schema, raw inference
format, re-inference or migration changes are needed in this PR.

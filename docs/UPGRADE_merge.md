# Merge: fast path and Stash reads

Merge reconstructs published state from retained evidence.
`--merge-only` can run without loading models.

## Fast path

After successful publication, a scene receives a
`merge_check`. The next run can mark it `merge_reused`
and skip expensive computations if:
- publication policy matches;
- the managed tag catalog matches;
- the scene's tag/marker fingerprint matches;
- no publication is pending.

New results, changes to YAML, publication code, hierarchy,
or Stash state trigger reconciliation.

## Batched SQL reads

When Stash provides compatible `querySQL`, fixed read-only
SELECT statements fetch tags/markers without the overhead
of many GraphQL resolvers.

With `--merge-only`, scenes may be paginated by increasing
ID without COUNT/OFFSET. During ordinary inference,
GraphQL still supplies VTT/sprite paths while tag/marker
state can be attached in batches.

Stash 0.31.1+ must provide required hierarchy fields and
`end_seconds`; point markers use `end_seconds: null`.
If read-only SQL fails, merge can fall back to GraphQL.
All writes use the Stash API.

## Final pruning of unused tags

After successful merge, read-only SQL excludes tags
associated with scenes or markers before full usage
counts are fetched over GraphQL. This applies to
empty marker groups, excluded tag definitions and
`--prune-empty-tags`. One fresh ID/name catalog supports
both pruning phases.

SQL only narrows the candidate set. Full usage counts,
children, group ownership and pre-deletion checks remain
in Stash API calls. In dry-run, candidates also include
groups that would lose markers through the planned merge;
other-scene usage still protects them.

If SQL capability detection, validation or catalog reads
fail, pruning uses the GraphQL path. Invalid results from
later candidate filtering abort pruning. The summary
reports `merge_prune_sql_enabled` or
`merge_prune_graphql_fallback`.

## Temporal indexes

Repeated lookups of occurrences within the same interval
use timeline indexes and caches of shared results.
Unusual overlapping legacy timelines retain the safe
reference implementation.

## Publication

The plan is saved locally before mutations. After
successful synchronization, the pending snapshot
commits together with the updated merge check.

Read-only GraphQL can retry transient failures.
Mutations are not automatically retried.

## Useful commands

```powershell
# Normal merge without models
python .\stash_ai.py --merge-only

# One scene
python .\stash_ai.py --merge-only --scene-id 123 --verbose

# Force recomputation despite a matching publication cache
python .\stash_ai.py --merge-only --scene-id 123 --force-merge

# JSONL profile
python .\stash_ai.py --merge-only --scene-id 123 --force-merge --profile-merge merge-profile.jsonl
```

In the summary, inspect `merge_reused`,
`merge_graphql_fallback`, policy/Stash/cache change
reasons and timing of Stash reads and merge stages.

For profile details see
[merge diagnostics](UPGRADE_diagnostics.md).

"""Bounded late query batches; admission and fallback semantics are unchanged."""
from itertools import islice
from .historical_references import rank_many
from .pose_policy import is_anchor, query_allows


def ranked_queries(rows, history, config, *, appearance, relaxed):
    grouping = config.data['clustering']
    eligible = (row for row in rows if (grouping['auxiliary_matching'] or is_anchor(row['quality'], grouping))
                and query_allows(row['quality'], grouping))
    iterator = iter(eligible)
    while batch := list(islice(iterator, 16)):
        ordinary = list(rank_many(batch, history, config))
        unresolved = [row for row, result in ordinary if result is None]
        fallback = {row['id']: result for row, result in rank_many(unresolved, history, config,
            relaxed_pose=relaxed or not appearance, strict_appearance=appearance,
            quarantine_only=appearance and not grouping['strict_appearance_matching'])}
        for row, result in ordinary:
            yield row, result, fallback.get(row['id'])

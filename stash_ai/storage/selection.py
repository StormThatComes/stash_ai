"""Bounded SQL selection shared by lightweight scope queries."""
from itertools import batched


def selected_rows(store, sql, params, scenes=None, *, column='scene', suffix=''):
    if scenes is None:
        yield from store.connection.execute(sql + suffix, params)
        return
    selected = tuple(dict.fromkeys(str(scene) for scene in scenes))
    for group in batched(selected, 800):
        placeholders = ','.join('?' for _ in group)
        yield from store.connection.execute(sql + f' AND {column} IN ({placeholders})' + suffix,
                                            (*params, *group))


def evidence_scenes(store, requested=None):
    """Keep explicit order while checking presence in bounded indexed queries."""
    wanted = tuple(dict.fromkeys(map(str, requested))) if requested else None
    present = {row[0] for row in selected_rows(store,
        'SELECT DISTINCT scene FROM ' + store.table('results') + ' WHERE scope=?',
        (store.scope,), wanted)}
    return [scene for scene in wanted if scene in present] if wanted is not None else sorted(present)

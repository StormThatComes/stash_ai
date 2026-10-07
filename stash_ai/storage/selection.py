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

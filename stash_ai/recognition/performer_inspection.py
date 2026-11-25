"""Performer-wide metadata using exactly the inspector membership filters."""
from contextlib import closing
from .identity_inspection import _select, member_query, bodies_available
from .inspection_runtime import open_readonly


def summary(reader):
    with closing(open_readonly(reader.config.database)) as db:
        requested, person = _select(db, *reader.selector)
        kinds = ['face', 'body'] if bodies_available(db) else ['face']
        sql, args = member_query(person, reader.face_policy, kinds,
                                filters=(None, None, None), accepted_only=True, config=reader.config)
        groups = [dict(r) for r in db.execute(
            'SELECT scene,kind,state,COUNT(*) AS count FROM (' + sql +
            ') GROUP BY scene,kind,state ORDER BY CAST(scene AS INTEGER),scene,kind,state', args)]
        return dict(requested=requested, person=person, groups=groups,
                    observations=sum(g['count'] for g in groups), scenes=len({g['scene'] for g in groups}))

"""Global read-only views from durable direct auxiliary confirmations.

Revalidate against current frozen strict references; view-only attachments never
participate. The persisted observations and decisions are the source of truth,
so removing support or superseding a scan invalidates the next capture.
"""
from .clustering import vector,co_presence_allows
from .database import encoded
from .global_views import IdentityView
from .view_prototypes import confirmed_views


def historical_views(db,config,scope,namespace,selected,groups,usage,cohorts=(),aliases=None):
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    if not groups:
        return ()
    by_person={g.identifier:g for g in groups}
    if not cohorts:
        from .global_views import catalogue
        from .scanning import scan_identity
        cohorts=catalogue(db,config,scope,namespace,usage=usage,minimum=1,complete=True,
            excluded_scenes=[item['scene'] for item in selected],scan_policy=scan_identity(config,namespace))
    rows=db.execute('SELECT m.person,o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality,'
        '(SELECT c.payload FROM candidates c WHERE CAST(CASE WHEN json_valid(c.payload) THEN json_extract(c.payload,\'$.observation\') END AS TEXT)=o.id '
        'ORDER BY c.rowid DESC LIMIT 1) AS decision '+
        _source(config,fallback_sql(db,config,namespace),verified=True)+
        ' ORDER BY m.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id',
        parameters(selected,config,scope,namespace))
    def validated():
        for row in rows:
            row=dict(row)
            row['person']=(aliases or {}).get(row['person'],row['person'])
            if row['person'] not in by_person:
                continue
            from .strict_confirmations import revalidate
            reference=revalidate(db,row,config,scope,namespace,selected,groups,cohorts,aliases)
            if reference is not None and co_presence_allows(
                    db,row,reference,groups,vector(row),config):
                yield row
    grouping=config.data['clustering']
    promotion=dict(grouping,view_min_support=max(grouping['view_min_support'],grouping['auxiliary_min_support']))
    prototypes=confirmed_views(db,None,promotion,remaining_bytes=grouping['max_reference_bytes']-usage.bytes,
                               rows=validated(),complete=True)
    output=[]
    for view in prototypes:
        samples={row['id']:vector(row) for row in db.execute('SELECT id,dimensions,embedding FROM observations '
            'WHERE id IN (SELECT value FROM json_each(?))',(encoded(view.support),))}
        samples[view.medoid]=view.group.samples[0]
        examples=tuple(samples[identifier] for identifier in view.support)
        usage.add(examples,kind='Historical verified view evidence')
        output.append(IdentityView(view.group,view.pose,view.medoid,view.support,view.support_count,
                                   examples,'verified_view'))
    return tuple(output)

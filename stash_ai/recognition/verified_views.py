"""Global read-only views from durable direct auxiliary confirmations.

Revalidate against current frozen strict references; view-only attachments never
participate. The persisted observations and decisions are the source of truth,
so removing support or superseding a scan invalidates the next capture.
"""
from .clustering import vector,co_presence_allows
from .database import encoded
from .global_views import IdentityView
from .view_prototypes import confirmed_views


def historical_views(db,config,scope,namespace,selected,groups,usage):
    from .historical_references import History,rank,_source,parameters
    from .reference_compatibility import fallback_sql
    if not groups:
        return ()
    history=History(groups,(),{},'revalidated-historical-strict-cores')
    by_person={g.identifier:g for g in groups}
    rows=db.execute('SELECT m.person,o.id,o.scan,o.cue,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality '+
        _source(config,fallback_sql(db,config,namespace),verified=True)+
        ' ORDER BY m.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id',
        parameters(selected,config,scope,namespace))
    def validated():
        for row in rows:
            if row['person'] not in by_person:
                continue
            result=rank([row],history,config)
            if result is not None and result[0]==row['person'] and co_presence_allows(
                    db,row,by_person[row['person']],groups,vector(row),config):
                yield row
    grouping=config.data['clustering']
    promotion=dict(grouping,view_min_support=max(grouping['view_min_support'],grouping['auxiliary_min_support']))
    prototypes=confirmed_views(db,None,promotion,remaining_bytes=grouping['max_reference_bytes']-usage.bytes,
                               rows=validated())
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

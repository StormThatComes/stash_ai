"""Superseded evidence can defer birth, but cannot positively identify a face."""
from .clustering import Group,vector
from .database import encoded


def latest_clause():
    return ("EXISTS (SELECT 1 FROM scans latest WHERE latest.scope=s.scope AND latest.namespace=s.namespace "
        "AND latest.scene=s.scene AND latest.rowid>s.rowid AND NOT EXISTS (SELECT 1 FROM scans n "
        "WHERE n.scope=latest.scope AND n.namespace=latest.namespace AND n.scene=latest.scene AND n.rowid>latest.rowid) "
        "AND (latest.status<>'complete' OR NOT EXISTS (SELECT 1 FROM observations x JOIN members y ON y.observation=x.id "
        "WHERE x.scan=latest.id))) AND NOT EXISTS (SELECT 1 FROM scans newer JOIN observations x ON x.scan=newer.id "
        "JOIN members y ON y.observation=x.id WHERE newer.scope=s.scope AND newer.namespace=s.namespace "
        "AND newer.scene=s.scene AND newer.rowid>s.rowid AND newer.status='complete')")


def capture(db,selected,config,scope,namespace,usage):
    from .historical_references import _source,parameters
    from .reference_compatibility import fallback_sql
    from .reference_selection import select
    source=_source(config,fallback_sql(db,config,namespace),gaps=True)
    args=parameters(selected,config,scope,namespace)
    output=[]
    for person, in db.execute('SELECT DISTINCT p.id '+source+' ORDER BY p.id',args):
        rows=db.execute('SELECT o.id,o.scan,o.seconds,o.geometry,o.dimensions,o.embedding,o.quality,s.scene '+
            source+' AND p.id=? ORDER BY s.scene,o.cue,o.id',args+(person,))
        samples=select(rows,config.data['clustering']['max_samples_per_person'])
        if samples:
            ids,vectors=zip(*samples)
            usage.add(vectors,kind='Unavailable reference witnesses')
            output.append(Group(person,tuple(vectors),True,tuple(ids)))
        if len(output)>config.data['clustering']['max_people_per_lookup']:
            raise ValueError('Unavailable identity lookup exceeds configured person budget.')
    return tuple(output)


def defer(replay,group,history,config):
    if not history.unavailable:
        return False
    owned=replay.execute('SELECT 1 FROM members m JOIN sample_keys k ON k.observation=m.observation '
        'JOIN origins o ON o.sample_key=k.sample_key WHERE m.person=? LIMIT 1',(group.identifier,)).fetchone()
    if owned:
        return False
    import numpy as np
    threshold=config.data['recognition']['match_threshold']-config.data['recognition']['ambiguity_margin']
    rivals=[g.identifier for g in history.unavailable if any(float(np.dot(a,b))>=threshold
        for a in group.samples for b in g.samples)]
    if not rivals:
        return False
    proof=dict(format=1,frozen=True,context=history.context,unavailable_identities=rivals,
               outcome='deferred_birth',threshold=threshold)
    replay.execute("UPDATE candidates SET person=NULL,status='withheld',payload=json_set(payload,"
        "'$.prior_reason',json_extract(payload,'$.reason'),'$.reason','reference_bank_unavailable',"
        "'$.global_identity_decision',json(?)) WHERE person=?",(encoded(proof),group.identifier))
    replay.execute('DELETE FROM members WHERE person=?',(group.identifier,))
    return True

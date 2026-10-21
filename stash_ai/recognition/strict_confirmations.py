"""A direct auxiliary confirmation depends on its original strict witnesses.

Adding another appearance of the same owner must not revoke that proof. Removed,
reassigned, rejected, incompatible or superseded witnesses do revoke it. Current
global rival anchors still veto promotion. Legacy decisions are independently
revalidated against current strict appearance cohorts, never recovered views.
"""
import json

from .clustering import Group, vector
from .database import encoded


def revalidate(db,row,config,scope,namespace,selected,groups,cohorts):
    from .historical_references import History,rank,_source,parameters
    from .reference_compatibility import fallback_sql
    try:
        proof=json.loads(row['decision']).get('strict_confirmation')
    except (ValueError,TypeError,AttributeError):
        return None
    owner=row['person']
    if proof is not None:
        if (not isinstance(proof,dict) or proof.get('format')!=1 or proof.get('identity')!=owner
            or not isinstance(proof.get('support_ids'),list) or not proof['support_ids']
            or any(not isinstance(i,str) for i in proof['support_ids'])
            or len(set(proof['support_ids']))!=len(proof['support_ids'])):
            return None
        witnesses=db.execute('SELECT o.id,o.dimensions,o.embedding '+
            _source(config,fallback_sql(db,config,namespace))+' AND p.id=? '
            'AND o.id IN (SELECT value FROM json_each(?)) ORDER BY o.id',
            parameters(selected,config,scope,namespace)+(owner,encoded(proof['support_ids']))).fetchall()
        if len(witnesses)!=len(proof['support_ids']):
            return None
        candidates=(Group(owner,tuple(vector(w) for w in witnesses),True,tuple(w['id'] for w in witnesses)),)
    else:
        candidates=tuple(Group(owner,v.examples,True,v.support) for v in cohorts if v.group.identifier==owner)
        candidates+=tuple(g for g in groups if g.identifier==owner)
    rivals=tuple(g for g in groups if g.identifier!=owner)
    for candidate in candidates:
        result=rank([row],History((candidate,)+rivals,(),{},'original-strict-witnesses'),config)
        if result is not None and result[0]==owner:
            return candidate
    return None

"""Portable late-phase payloads reference authoritative retained observations."""
from .clustering import Group,vector
from .view_prototypes import ViewPrototype


def encode(early,deferred):
    return dict(early=[dict(person=v.group.identifier,pose=v.pose,medoid=v.medoid,
        support=list(v.support),support_count=v.support_count,authority=v.authority,
        promotion=v.promotion) for v in early], deferred=sorted(deferred))


def decode(db,payload):
    if set(payload)!={'early','deferred'} or not isinstance(payload['early'],list) or not isinstance(payload['deferred'],list):
        raise ValueError('Invalid late checkpoint payload.')
    views=[]
    for row in payload['early']:
        if not isinstance(row,dict) or row.get('pose') not in ('left','right','other'):
            raise ValueError('Invalid late checkpoint view.')
        sample=db.execute('SELECT dimensions,embedding FROM observations WHERE id=? AND rejected IS NULL',(row['medoid'],)).fetchone()
        if sample is None or not db.execute('SELECT 1 FROM people WHERE id=?',(row['person'],)).fetchone():
            raise ValueError('Late checkpoint view lost its source.')
        support=row['support']
        if (not isinstance(support,list) or not support or any(not isinstance(i,str) for i in support)
                or type(row['support_count']) is not int or row['support_count']<len(set(support))):
            raise ValueError('Invalid late checkpoint support.')
        for identifier in support:
            if not db.execute('SELECT 1 FROM observations WHERE id=? AND rejected IS NULL',(identifier,)).fetchone():
                raise ValueError('Late checkpoint witness is unavailable.')
        views.append(ViewPrototype(Group(row['person'],(vector(sample),),True),row['pose'],row['medoid'],
            tuple(support),row['support_count'],row['authority'],row['promotion']))
    if any(not isinstance(i,str) for i in payload['deferred']):
        raise ValueError('Invalid deferred checkpoint hints.')
    return tuple(views),set(payload['deferred'])

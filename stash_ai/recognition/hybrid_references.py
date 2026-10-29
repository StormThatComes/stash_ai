"""Independent body selection; face quality/ownership remain admission gates."""
from .person_signal import contract
from .clustering import reference_admission
from .reference_roles import strict_sql
from .reference_compatibility import fallback_sql
from .scanning import scan_identity
from .reference_selection import select
from .database import encoded


def body_vector(row):
    import numpy as np
    value=np.frombuffer(row['embedding'],dtype='<f4')
    if value.shape!=(4096,) or not np.isfinite(value).all() or abs(float(np.linalg.norm(value))-1)>.001:
        raise ValueError('Invalid retained SapiensID feature.')
    return value


def _rows(db, config, scope, namespace, owners):
    admission=reference_admission(config.data['clustering'],scan_policy=scan_identity(config,namespace),
                                  compatibility=fallback_sql(db,config,namespace))
    sql=('SELECT m.person AS owner,o.id,o.scan,o.seconds,o.geometry,o.quality,s.scene,4096 AS dimensions,e.embedding,e.metadata AS person_metadata '
        'FROM members m JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
        'JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
        "WHERE m.person IN (SELECT value FROM json_each(?)) AND s.scope=? AND s.namespace=? AND s.status='complete' "
        "AND o.rejected IS NULL AND e.status='ready' AND "+strict_sql()+' AND '+admission+
        ' ORDER BY m.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id')
    args=(contract(config),encoded(list(owners)),scope,namespace)
    from .person_anchor import pixel_gate
    rows=db.execute(sql,args)
    if config.data['sapiensid']['identities']['anchor_min_person_pixels'] is None:return rows
    return (row for row in rows if pixel_gate(row['person_metadata'],config.data['sapiensid']['identities'])['eligible'])


def _select(records,config):
    chosen=select(records,config.data['clustering']['max_samples_per_person'],vector_reader=body_vector,quality_reader=body_quality)
    wanted={i for i,_ in chosen}
    from .hybrid_modes import body_modes
    result=BodyRows(r for r in records if r['id'] in wanted)
    result.modes=body_modes(records,config)
    from .independent_evidence import Votes
    votes=Votes()
    for row in (records if result.modes else result):votes.add(row)
    result.support=len(votes)
    return result


def catalogue_rows(db,config,scope,namespace,owners):
    """One ordered SQL cursor for all owners; budget complete mode witnesses."""
    from itertools import groupby
    result={};used=0
    for owner,rows in groupby(_rows(db,config,scope,namespace,owners),key=lambda r:r['owner']):
        records=[]
        for row in rows:
            used+=2*len(row['embedding'])+len(row['quality'])+len(row['person_metadata'])+1024
            if used>config.data['sapiensid']['execution']['reference_bytes']:
                raise ValueError('SapiensID reference_bytes budget exceeded by mode witnesses.')
            records.append(row)
        result[owner]=_select(records,config)
    return result


def body_rows(db,config,scope,namespace,owner):
    return catalogue_rows(db,config,scope,namespace,[owner]).get(owner,BodyRows())


class BodyRows(list):
    modes=()


def body_quality(row):
    """Rank the measured body input, retaining neutral legacy metadata."""
    import json, math
    metadata=json.loads(row['person_metadata'])
    person=metadata.get('person')
    if not person:return 0.
    bounds=person.get('bounds',())
    if len(bounds)!=4:return 0.
    width,height=bounds[2]-bounds[0],bounds[3]-bounds[1]
    if min(width,height)<=0:return 0.
    points=person.get('keypoints',())
    coverage=sum(len(p)==2 and min(p)>=0 for p in points)/17
    score=float(person.get('score',0.))
    if not math.isfinite(score):return 0.
    pixels=min(1.,math.log1p(math.sqrt(width*height))/math.log1p(512))
    return pixels*min(1.,max(0.,score))*min(1.,coverage)

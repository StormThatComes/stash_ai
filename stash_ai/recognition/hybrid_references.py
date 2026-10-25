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


def body_rows(db, config, scope, namespace, owner):
    admission=reference_admission(config.data['clustering'],scan_policy=scan_identity(config,namespace),
                                  compatibility=fallback_sql(db,config,namespace))
    sql=('SELECT o.id,o.scan,o.seconds,o.geometry,o.quality,s.scene,4096 AS dimensions,e.embedding,e.metadata AS person_metadata '
        'FROM members m JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
        'JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
        "WHERE m.person=? AND s.scope=? AND s.namespace=? AND s.status='complete' "
        "AND o.rejected IS NULL AND e.status='ready' AND "+strict_sql()+' AND '+admission+
        ' ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id')
    args=(contract(config),owner,scope,namespace)
    chosen=select(db.execute(sql,args),config.data['clustering']['max_samples_per_person'],vector_reader=body_vector,quality_reader=body_quality)
    wanted={i for i,_ in chosen}
    return [r for r in db.execute(sql,args) if r['id'] in wanted]


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

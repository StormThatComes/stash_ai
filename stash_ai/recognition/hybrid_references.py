from .shared_person_evidence import expression
body_embedding = expression('embedding')
"""Independent body selection; face quality/ownership remain admission gates."""
from .resource_errors import ResourceLimit
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


def _query(db, config, scope, namespace, owners,ids=None):
    admission=reference_admission(config.data['clustering'],scan_policy=scan_identity(config,namespace),
                                  compatibility=fallback_sql(db,config,namespace))
    sql=(f"SELECT 'person' AS evidence_kind,m.person AS owner,o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,s.scene,4096 AS dimensions,{body_embedding} AS embedding,e.metadata AS person_metadata "
        'FROM members m JOIN observations o ON o.id=m.observation JOIN scans s ON s.id=o.scan '
        'JOIN person_evidence e ON e.observation=o.id AND e.contract=? '
        "WHERE m.person IN (SELECT value FROM json_each(?)) AND s.scope=? AND s.namespace=? AND s.status='complete' "
        "AND o.rejected IS NULL AND e.status='ready' AND "+strict_sql()+' AND '+admission+
        ' ORDER BY m.person,CAST(s.scene AS INTEGER),s.scene,o.cue,o.seconds,o.id')
    args=(contract(config),encoded(list(owners)),scope,namespace)
    if ids is not None:
        sql=sql.replace(' ORDER BY m.person', ' AND o.id IN (SELECT value FROM json_each(?)) ORDER BY m.person')
        args+= (encoded(ids),)
    return sql,args


def _rows(db,config,scope,namespace,owners,ids=None):
    sql,args=_query(db,config,scope,namespace,owners,ids)
    from .person_anchor import anchor_gate
    rows=db.execute(sql,args)
    return (row for row in rows if anchor_gate(row['person_metadata'],config.data['sapiensid']['identities'],
        collection=config.data['sapiensid']['collection'])['eligible'])


def _select(records,config):
    chosen=select(records,config.data['clustering']['max_samples_per_person'],vector_reader=body_vector,quality_reader=body_quality)
    wanted={i for i,_ in chosen}
    from .hybrid_modes import body_modes
    result=BodyRows(r for r in records if r['id'] in wanted)
    result.modes=body_modes(records,config,partial=True)
    result.unsupported_modes=sum(map(len,result.modes)) < len(records)
    from .independent_evidence import Votes
    votes=Votes()
    for row in (records if result.modes else result):votes.add(row)
    result.support=len(votes)
    return result


def catalogue_rows(db,config,scope,namespace,owners):
    """Stream each owner through a repeatable disk spool before selection."""
    from itertools import groupby
    from .disk_body_modes import Records,ComparisonBudget
    from .progress import work
    from .independent_evidence import Votes
    result={};used=0
    with work('Build complete body appearances',unit='comparison') as progress:
        budget=ComparisonBudget(config,progress)
        from .dependencies import cached
        from .derived_cache import current,fingerprint
        from .config import digest
        store=current()
        for owner in sorted(owners):
            spool=Records();keep=False
            try:
                from .body_mode_cache import load,save,witnesses
                query=_query(db,config,scope,namespace,[owner]) if store else None
                policy=digest(['body-source-v4-partial',config.data['sapiensid']['matching'],config.data['sapiensid']['identities'],config.data['clustering']])
                from .local_dependencies import available as local_clocks
                from .dependencies import version
                policy=digest([policy,version(db,'heads')]) if store else policy
                signed=cached(db,'body-source',owner,policy,lambda:fingerprint(db,[query],policy),
                    dependency='owner' if local_clocks(db) else 'global',dependency_key=owner if local_clocks(db) else '*') if store else None
                found=load(spool,signed,config) if store else None
                saved_witnesses=witnesses(signed) if found is not None else None
                if found is not None and saved_witnesses is None:found=None
                if found is None:
                    for row in _rows(db,config,scope,namespace,[owner]):spool.add(row)
                    chosen=select(spool.rows(),config.data['clustering']['max_samples_per_person'],
                                  vector_reader=body_vector,quality_reader=body_quality)
                    ids=[i for i,_ in chosen]
                    modes,support=spool.modes(config,budget)
                    save(spool,signed,modes,support,ids,config)
                else:modes,support,ids=found
                selected=BodyRows(spool.selected(ids) if found is None else _rows(db,config,scope,namespace,[owner],ids))
                selected.modes=modes
                selected.witnesses=tuple(row[0] for row in spool.db.execute('SELECT id FROM records ORDER BY ordinal')) if found is None else tuple(saved_witnesses)
                selected.unsupported_modes=sum(map(len,selected.modes)) < len(selected.witnesses)
                votes=Votes()
                for row in selected:votes.add(row)
                selected.support=support if selected.modes else len(votes)
                used+=sum(2*len(r['embedding'])+len(r['quality'])+len(r['person_metadata'])+1024 for r in selected)
                used+=sum(len(i)+64 for i in selected.witnesses)
                if used+2*1024**2>config.data['sapiensid']['execution']['reference_bytes']:
                    raise ResourceLimit('SapiensID reference_bytes budget exceeded by selected witnesses.')
                result[owner]=selected;keep=bool(selected.modes)
            finally:spool.close(keep=keep)
    return result


def body_rows(db,config,scope,namespace,owner):
    return catalogue_rows(db,config,scope,namespace,[owner]).get(owner,BodyRows())


class BodyRows(list):
    support=0
    modes=()
    witnesses=()
    unsupported_modes=False


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

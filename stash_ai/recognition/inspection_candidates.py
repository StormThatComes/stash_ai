"""Explicit exploratory comparisons of retained vectors; never matching replay."""
from contextlib import closing
import hashlib
import heapq
import json

from .inspection_runtime import open_readonly, check_cancelled, stamp
from .identity_inspection import TABLES
from .models import namespace
from .clustering import vector as face_vector
from .body_identity import vector as body_vector
from .body_observations import contract


def _body_sample(rows, maximum):
    """Bounded deterministic cue coverage; not a production body-bank replay."""
    pool={};heap=[]
    for row in rows:
        check_cancelled()
        key=(row['scan'],row['cue'])
        rank=int.from_bytes(hashlib.blake2b(json.dumps(key).encode(),digest_size=8).digest(),'big')
        if key in pool or len(pool)==maximum and rank>=-heap[0][0]:continue
        if len(pool)==maximum:
            _,old=heapq.heappop(heap);del pool[old]
        pool[key]=row;heapq.heappush(heap,(-rank,key))
    return [(r['id'],body_vector(r)) for _,r in sorted(pool.items())]


def calculate(config,kind,address,scope, *, top=5):
    import numpy as np
    from .reference_selection import select
    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    table,members=TABLES[kind];body=kind=='body';revision=stamp(config.database)
    result=dict(kind=kind,rows=[],compared=0,invalid=0,truncated=False,changed=False,
        note='Exploratory cosine similarities, not probabilities or a replay of the assignment. '
             'No models, thresholds or memberships are changed. Same-cue/self references are excluded from scoring.')
    with closing(open_readonly(config.database)) as db:
        fields='o.id,o.scan,o.cue,o.embedding'+(',o.contract' if body else ',o.dimensions')
        query=db.execute(f'SELECT {fields} FROM {table} o JOIN scans s ON s.id=o.scan WHERE o.id=? AND s.scope=? AND s.namespace=?',
                         (address,scope,namespace(config))).fetchone()
        if query is None:raise ValueError('Observation unavailable in this scope/model.')
        if query['embedding'] is None:
            result['note']+=' No retained embedding: re-collection would be required.';return result
        if body and query['contract']!=contract(config):
            result['note']+=' Historical body contract differs from current YAML; comparison unavailable.';return result
        vector=body_vector if body else face_vector
        try:sample=vector(query)
        except ValueError:
            result['note']+=' Selected embedding is invalid.';return result
        owner=db.execute(f'SELECT person FROM {members} WHERE observation=?'+(" AND status='accepted'" if body else ''),(address,)).fetchone()
        owner=owner[0] if owner else None
        maximum=(config.data['sapiensid']['global_matching']['max_references'] if body else config.data['clustering']['max_samples_per_person'])
        budget=(config.data['sapiensid']['execution']['reference_bytes'] if body else config.data['clustering']['max_reference_bytes'])
        # The face diversity sampler keeps several bounded reserve pools. Account
        # conservatively for vectors plus metadata, independently of GUI caches.
        permitted=int(budget//((sample.nbytes+4096)*16))
        if permitted<1:raise ValueError('Configured reference memory budget is too small for comparison.')
        if maximum>permitted:maximum=permitted;result['truncated']=True
        selection=("m.status='accepted' AND m.role='anchor' AND o.contract=?" if body else
                   strict_sql()+' AND '+anchor_sql('o',config.data['clustering']))
        base=(f' FROM {table} o JOIN {members} m ON m.observation=o.id JOIN scans s ON s.id=o.scan '
              "WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND o.rejected IS NULL AND o.embedding IS NOT NULL AND "+selection)
        args=(scope,namespace(config),contract(config)) if body else (scope,namespace(config))
        owners=db.execute('SELECT p.id,p.performer FROM people p WHERE p.scope=? AND p.namespace=? '
            'AND NOT EXISTS (SELECT 1 FROM identity_merges WHERE source=p.id) AND EXISTS (SELECT 1'+base+' AND m.person=p.id) '
            'ORDER BY (p.id=?) DESC,p.id',(scope,namespace(config),*args,owner))
        ranked=[];owner_result=None
        for person in owners:
            check_cancelled()
            if result['compared']>=config.data['clustering']['max_people_per_lookup']:
                result['truncated']=True;break
            result['compared']+=1
            fields='o.id,o.scan,o.cue,o.seconds,o.quality,o.geometry,o.embedding,s.scene'+('' if body else ',o.dimensions')
            def records():
                for raw in db.execute('SELECT '+fields+base+' AND m.person=? ORDER BY CAST(s.scene AS INTEGER),s.scene,o.cue,o.id',(*args,person['id'])):
                    check_cancelled()
                    try:
                        value=vector(raw)
                        if value.shape!=sample.shape:raise ValueError('Incompatible dimensions')
                    except ValueError:result['invalid']+=1;continue
                    # Only retain the selected cue exclusion; no unbounded ID map.
                    yield dict(raw)
            chosen=(_body_sample(records(),maximum) if body else select(records(),maximum))
            selected=any(i==address for i,_ in chosen)
            scores=[]
            for identifier,value in chosen:
                cue=db.execute(f'SELECT scan,cue FROM {table} WHERE id=?',(identifier,)).fetchone()
                if identifier==address or (cue['scan'],cue['cue'])==(query['scan'],query['cue']):continue
                scores.append((float(np.clip(np.dot(sample,value),-1.,1.)),identifier))
            entry=dict(identity=person['id'],performer=person['performer'],owner=person['id']==owner,
                score=float(np.mean([v for v,_ in scores])) if scores else None,
                minimum=min((v for v,_ in scores),default=None),maximum=max((v for v,_ in scores),default=None),
                count=len(scores),witness=max(scores)[1] if scores else None,selected=selected)
            if entry['owner']:owner_result=entry
            if scores:
                ranked.append(entry);ranked.sort(key=lambda r:(-r['score'],r['identity']));del ranked[top:]
        if owner and owner_result is None:
            person=db.execute('SELECT performer FROM people WHERE id=? AND scope=? AND namespace=?',(owner,scope,namespace(config))).fetchone()
            if person:owner_result=dict(identity=owner,performer=person[0],owner=True,score=None,minimum=None,maximum=None,count=0,witness=None,selected=None)
        if owner_result and all(r['identity']!=owner for r in ranked):ranked.append(owner_result)
        result['rows']=ranked
        result['note']+=(' Face references use the current strict-role/quality admission and production diversity sampler.' if not body else
                         ' Body references are a bounded deterministic cue sample of accepted anchors from the current collection contract; this is not the production body-bank selection.')
        result['note']+=' Rejected, auxiliary-only or embedding-less owners may have no comparable references.'
    result['changed']=revision!=stamp(config.database)
    return result

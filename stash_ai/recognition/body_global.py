"""Frozen, complete-component person consolidation; missing faces are not contradictions."""
import json
import time
import numpy as np

from .body_identity import Scores,anchor,independent,vector,policy,available
from .body_observations import contract
from .database import encoded
from .identity_merges import canonical
from .scan_heads import predicate as current_scan
from .progress import tracking

ROUTES=('body_global','body_scene')


def defaults():
    return dict(enabled=False,match_threshold=.90,cohesion_threshold=.85,
                ambiguity_margin=.10,min_support=3,max_references=32)


def _select(rows,maximum,settings):
    ranked=sorted(rows,key=lambda r:(-json.loads(r['quality']).get('quality',{}).get('sharpness',0),r['id']))
    # Independent views in a deterministic order within the reference budget.
    chosen=independent(ranked,settings)[:maximum]
    if not chosen and ranked:chosen=[ranked[0]]
    return chosen


def banks(db,config,scope,namespace):
    expected=contract(config);grouped={};cost=0;settings=config.data['sapiensid']['identities']
    sql=('SELECT b.id,b.scan,b.cue,b.seconds,b.face,b.quality,b.embedding,m.person,s.scene '
         'FROM body_observations b JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
         "WHERE s.scope=? AND s.namespace=? AND s.status='complete' AND b.contract=? AND b.rejected IS NULL "
         "AND m.status='accepted' AND m.role='anchor' AND m.policy=? AND "+current_scan()+' ORDER BY b.id')
    for raw in db.execute(sql,(scope,namespace,expected,policy(config))):
        row=dict(raw);person=canonical(db,row['person'])
        cost+=len(row['embedding'])*3+len(row['quality'])+2048
        if cost>config.data['sapiensid']['execution']['reference_bytes']:
            raise ValueError('Global body references exceed sapiensid.execution.reference_bytes.')
        row['_vector']=vector(row);grouped.setdefault(person,[]).append(row)
    output={}
    for person,rows in grouped.items():
        refs=_select(rows,config.data['sapiensid']['global_matching']['max_references'],settings)
        output[person]=dict(rows=rows,refs=refs,cores=[refs],missing_original=False,
                            support=len(independent(rows,settings)),faces=[],face_ids=[])
    # Aliased source witnesses retain veto power, but do not nominate candidates
    # or add new support votes. Never silently drop an unavailable endpoint.
    for alias in db.execute('SELECT source,target,evidence FROM identity_merges'):
        proof=json.loads(alias['evidence']);target=canonical(db,alias['target'])
        if proof.get('route') not in ROUTES or target not in output:continue
        for ids in proof.get('cores',{}).values():
            core=[]
            for identifier in ids:
                raw=db.execute('SELECT b.id,b.scan,b.cue,b.seconds,b.face,b.quality,b.embedding,s.scene '
                    'FROM body_observations b JOIN scans s ON s.id=b.scan WHERE b.id=? AND b.contract=? '
                    'AND b.rejected IS NULL AND '+current_scan(),(identifier,expected)).fetchone()
                if raw is None or raw['embedding'] is None:output[target]['missing_original']=True;continue
                cost+=len(raw['embedding'])*3+2048
                if cost>config.data['sapiensid']['execution']['reference_bytes']:
                    raise ValueError('Original body witnesses exceed sapiensid.execution.reference_bytes.')
                row=dict(raw);row['_vector']=vector(row);core.append(row)
            if core:output[target]['cores'].append(core)
    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    for person,bank in output.items():
        # Face vectors are only compared to face vectors; never to SapiensID.
        for raw in db.execute('SELECT o.id,o.dimensions,o.embedding FROM observations o JOIN members m ON m.observation=o.id '
            'JOIN scans s ON s.id=o.scan WHERE m.person=? AND o.rejected IS NULL AND '+strict_sql()+' AND '+
            anchor_sql('o',config.data['clustering'])+' AND '+current_scan(),(person,)):
            from .clustering import vector as face_vector
            feature=face_vector(raw);cost+=feature.nbytes*3
            if cost>config.data['sapiensid']['execution']['reference_bytes']:
                raise ValueError('Face contradiction witnesses exceed sapiensid.execution.reference_bytes.')
            bank['faces'].append(feature);bank['face_ids'].append(raw['id'])
        bank['cues']={(r['scan'],r['cue']) for core in bank['cores'] for r in core}
        bank['cues'].update((r['scan'],r['cue']) for r in db.execute('SELECT b.scan,b.cue FROM body_observations b '
            'JOIN body_members m ON m.observation=b.id JOIN scans s ON s.id=b.scan '
            "WHERE m.person=? AND m.status='accepted' AND "+current_scan(),(person,)))
    for alias in db.execute('SELECT target,evidence FROM identity_merges'):
        proof=json.loads(alias['evidence']);target=canonical(db,alias['target'])
        if proof.get('route') not in ROUTES or target not in output:continue
        for ids in proof.get('face_witnesses',{}).values():
            for identifier in ids:
                raw=db.execute('SELECT o.dimensions,o.embedding FROM observations o JOIN scans s ON s.id=o.scan '
                    'WHERE o.id=? AND o.rejected IS NULL AND '+current_scan(),(identifier,)).fetchone()
                if raw is None or raw['embedding'] is None:output[target]['missing_original']=True;continue
                from .clustering import vector as face_vector
                feature=face_vector(raw);cost+=feature.nbytes*3
                if cost>config.data['sapiensid']['execution']['reference_bytes']:raise ValueError('Original face witnesses exceed reference_bytes.')
                output[target]['faces'].append(feature);output[target]['face_ids'].append(identifier)
    return output


def report(db,config,scope,namespace):
    result=dict(enabled=config.data['sapiensid']['global_matching']['enabled'],contract=contract(config),pairs=[],decisions=[])
    if not result['enabled'] or not available(db):return result
    bank=banks(db,config,scope,namespace);settings=config.data['sapiensid']['global_matching']
    scores=Scores(config);identities=sorted(bank);pairs={};edges={p:set() for p in identities};plan_bytes=0
    with tracking(identities,'Compare global person identities',unit='identity') as people:
        for i,a in enumerate(people):
            for b in identities[i+1:]:
                first,second=bank[a],bank[b]
                mean,minimum=scores.compare(first['refs'],second['refs']);blockers=[]
                if mean<settings['match_threshold']:blockers.append('body_similarity')
                for left in first['cores']:
                    for right in second['cores']:
                        m,n=scores.compare(left,right);minimum=min(minimum,n)
                        if m<settings['match_threshold'] or n<settings['cohesion_threshold']:
                            if 'body_cohesion' not in blockers:blockers.append('body_cohesion')
                if min(first['support'],second['support'])<settings['min_support']:blockers.append('independent_support')
                if first['cues']&second['cues']:blockers.append('person_copresence')
                if first['missing_original'] or second['missing_original']:blockers.append('missing_original_body_witness')
                face_min=None
                for left in first['faces']:
                    for right in second['faces']:
                        if len(left)!=len(right):raise ValueError('Incompatible face namespaces in body consolidation.')
                        scores.count+=1
                        if scores.count>scores.limit:raise ValueError('Face contradiction checks exceed max_candidate_pairs.')
                        score=float(np.dot(left,right));face_min=score if face_min is None else min(face_min,score)
                if face_min is not None and face_min<config.data['sapiensid']['matching']['face_floor']:blockers.append('face_contradiction')
                pair=dict(people=[a,b],score=mean,minimum=minimum,face_minimum=face_min,
                    support=[first['support'],second['support']],
                    face_evidence='compared' if face_min is not None else 'unavailable',blockers=blockers)
                pairs[a,b]=pair;result['pairs'].append(pair)
                if not blockers:edges[a].add(b);edges[b].add(a)
                plan_bytes+=len(encoded(pair))+256
                if plan_bytes>config.data['sapiensid']['execution']['plan_bytes']:
                    raise ValueError('Global body plan exceeds sapiensid.execution.plan_bytes.')
    seen=set()
    for start in identities:
        if start in seen or not edges[start]:continue
        pending=[start];component=set()
        while pending:
            current=pending.pop()
            if current in component:continue
            component.add(current);pending.extend(edges[current]-component)
        seen.update(component);blockers=[];nearest=-1.
        if any(b not in edges[a] for a in component for b in component if a!=b):blockers.append('incomplete_body_component')
        for a in component:
            outsiders=[pairs[tuple(sorted((a,b)))]['score'] for b in identities if b not in component]
            runner=max(outsiders,default=-1.);nearest=max(nearest,runner)
            inside=min(pairs[tuple(sorted((a,b)))]['score'] for b in component if b!=a)
            if inside-runner<settings['ambiguity_margin'] and 'body_ambiguity' not in blockers:blockers.append('body_ambiguity')
        result['decisions'].append(dict(people=sorted(component),outcome='blocked' if blockers else 'merge',
            blockers=blockers,nearest_rival=nearest,
            cores={p:[r['id'] for core in bank[p]['cores'] for r in core] for p in sorted(component)},
            face_witnesses={p:bank[p]['face_ids'] for p in sorted(component)}))
    if len(encoded(result))>config.data['sapiensid']['execution']['plan_bytes']:
        raise ValueError('Global body proof exceeds sapiensid.execution.plan_bytes.')
    return result


def _scenes(db,people):
    encoded_people=encoded(list(people))
    bodies=db.execute('SELECT DISTINCT s.scene FROM body_members m JOIN body_observations b ON b.id=m.observation '
        'JOIN scans s ON s.id=b.scan WHERE m.person IN (SELECT value FROM json_each(?)) AND '+current_scan(),(encoded_people,))
    faces=db.execute('SELECT DISTINCT s.scene FROM members m JOIN observations o ON o.id=m.observation '
        'JOIN scans s ON s.id=o.scan WHERE m.person IN (SELECT value FROM json_each(?)) AND '+current_scan(),(encoded_people,))
    scenes={r[0] for r in bodies}|{r[0] for r in faces}
    for row in db.execute('SELECT target,evidence FROM identity_merges'):
        proof=json.loads(row['evidence'])
        if proof.get('route') in ROUTES and canonical(db,row['target']) in people:scenes.update(proof.get('scenes',()))
    return scenes


def reconcile(database,selected,config):
    """Caller owns the same transaction as local assignments; no partial plan."""
    db=database.connection;selection={s['scene'] for s in selected}
    first=db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
    scope,namespace=first
    for row in list(db.execute('SELECT m.* FROM identity_merges m JOIN people p ON p.id=m.source WHERE p.scope=? AND p.namespace=?',(scope,namespace))):
        proof=json.loads(row['evidence'])
        if proof.get('route') not in ROUTES:continue
        scenes=set(proof.get('scenes',()))|_scenes(db,[row['source'],row['target']])
        if scenes and scenes<=selection:
            db.execute('DELETE FROM identity_merges WHERE source=?',(row['source'],))
        else:
            # Recreated source cores need all original scenes to revalidate an
            # old alias. Do not publish a retired source or blindly trust it.
            db.execute("UPDATE body_members SET person=NULL,status='withheld',role='auxiliary',proof=? "
                'WHERE person=? AND observation IN (SELECT id FROM body_observations WHERE scan IN (SELECT value FROM json_each(?)))',
                (encoded(dict(reason='body_alias_requires_complete_selection',source=row['source'])),row['source'],encoded([s['scan'] for s in selected])))
    result=report(db,config,scope,namespace)
    from .publication_queue import enqueue
    proof_bytes=0
    for decision in result['decisions']:
        if decision['outcome']!='merge':continue
        people=decision['people'];scenes=_scenes(db,people)
        if not scenes<=selection:
            decision.update(outcome='blocked',blockers=['identity_outside_selected_scenes'],required_scenes=sorted(scenes,key=int));continue
        rows=list(db.execute('SELECT p.id,p.created,EXISTS(SELECT 1 FROM members WHERE person=p.id) AS has_face FROM people p '
            'WHERE p.id IN (SELECT value FROM json_each(?)) ORDER BY has_face DESC,p.created,p.id',(encoded(people),)))
        target=rows[0]['id']
        proof=dict(decision,route='body_global',scenes=sorted(scenes,key=int),contract=contract(config),policy=policy(config))
        proof_bytes+=len(encoded(proof))*(len(rows)-1)
        if proof_bytes>config.data['sapiensid']['execution']['plan_bytes']:
            raise ValueError('Body alias proofs exceed sapiensid.execution.plan_bytes.')
        for source in (r['id'] for r in rows[1:]):
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
            db.execute("UPDATE body_members SET person=?,role='auxiliary',proof=json_set(proof,'$.global_merge',json(?)) WHERE person=?",
                       (target,encoded(dict(source=source,target=target,route='body_global')),source))
            # Original face anchors retain vetoes in the ordinary hybrid system;
            # a body-based alias cannot promote them into new face references.
            db.execute('UPDATE members SET person=? WHERE person=?',(target,source))
            db.execute("UPDATE candidates SET person=?,payload=json_set(payload,'$.prior_reason',json_extract(payload,'$.reason'),"
                "'$.reason','body_global_identity','$.reference_role','attachment','$.reference_selected',json('false')) WHERE person=?",(target,source))
        for scene in scenes:enqueue(db,scope,namespace,scene,config,'body_global_merge')
    # A body-born identity may become entirely associated with an existing face
    # identity within the scene. Record that promotion as an explicit alias.
    origins={r[0] for r in db.execute('SELECT DISTINCT i.person FROM body_origins i JOIN body_observations b ON b.id=i.observation '
        'JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND '+current_scan(),(scope,namespace))}
    for source in sorted(origins):
        if canonical(db,source)!=source or db.execute('SELECT 1 FROM members WHERE person=? LIMIT 1',(source,)).fetchone():continue
        rows=list(db.execute('SELECT b.id,b.scan,b.cue,b.seconds,b.quality,m.person,m.status,s.scene FROM body_origins i '
            'JOIN body_observations b ON b.id=i.observation JOIN scans s ON s.id=b.scan LEFT JOIN body_members m ON m.observation=b.id '
            'WHERE i.person=? AND b.contract=? AND '+current_scan(),(source,contract(config))))
        targets={r['person'] for r in rows}
        if not rows or len(targets)!=1 or None in targets or source in targets or any(r['status']!='accepted' for r in rows):continue
        target=next(iter(targets));scenes={r['scene'] for r in rows}
        if not scenes<=selection or not db.execute('SELECT 1 FROM members WHERE person=? LIMIT 1',(target,)).fetchone():continue
        if len(independent(rows,config.data['sapiensid']['identities']))<config.data['sapiensid']['identities']['min_support']:continue
        proof=dict(route='body_scene',scenes=sorted(scenes,key=int),cores={source:[r['id'] for r in rows]},
                   contract=contract(config),reason='body_core_confirmed_by_face_bank')
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',(source,target,time.time(),encoded(proof),'local'))
    return result

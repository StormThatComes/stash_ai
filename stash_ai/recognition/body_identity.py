"""Scene-local person identities, with frozen face seeds and independent body anchors."""
import json
import time
import uuid
import numpy as np

from .body_observations import contract,enabled as collection_enabled
from .config import digest,plain
from .database import encoded
from .identity_merges import canonical
from .progress import tracking
from .person_anchor import pixel_gate,anchor_gate




def defaults():
    return dict(enabled=False,match_threshold=.85,cohesion_threshold=.80,ambiguity_margin=.1,
                min_support=3,attach_min_support=2,min_appearance_distance=4,
                anchor_min_keypoints=6,anchor_min_sharpness=25.,anchor_min_person_pixels=None)


def enabled(config):
    return collection_enabled(config) and config.data['sapiensid']['identities']['enabled']


def available(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='body_members' AND type='table'").fetchone() is not None


def policy(config):
    from .clustering import matching_policy
    return digest(['body-local-v7-global-appearances',contract(config),matching_policy(config),config.data['sapiensid']['identities']])


def vector(row):
    from .models import normalized_embedding
    data=row['embedding']
    if data is None or len(data)!=4096*4:raise ValueError('Invalid retained person embedding.')
    return normalized_embedding(np.frombuffer(data,dtype='<f4'),4096)


def quality(row):
    return json.loads(row['quality']).get('quality',{})


def anchor(row,settings):
    return anchor_gate(row['quality'],settings)['eligible']


def independent(rows,settings):
    """At most one witness per scan/cue; repeated pixels cannot add support."""
    from .independent_evidence import body_vote,representatives,unit
    candidates=[];votes=[];hashes=[];pixels={};seen=set()
    for row in sorted(rows,key=lambda r:(r['scan'],r['cue'],r['id'])):
        if row['id'] in seen:continue
        seen.add(row['id'])
        q=quality(row)
        vote=body_vote(row);physical=unit(vote)
        hashed=q.get('appearance_hash')
        hashed=int(hashed,16) if hashed else None
        fingerprint=pixels.get(physical)
        if fingerprint is None and hashed is not None:
            fingerprint=next((key for other,key in hashes
                if (hashed^other).bit_count()<settings['min_appearance_distance']),None)
        if fingerprint is None:
            fingerprint=encoded(physical)
            if hashed is not None:hashes.append((hashed,fingerprint))
        pixels[physical]=fingerprint
        # Retain every cue edge of an appearance. A greedy representative can
        # consume the sole cue of a different, otherwise independent crop.
        votes.append(dict(vote,quality=encoded({'quality':{'pixel_digest':fingerprint}})))
        candidates.append(row)
    chosen={r['id'] for r in representatives(votes)}
    return [r for r in candidates if r['id'] in chosen]


def copresent(first,second):
    return any(a['scan']==b['scan'] and a['cue']==b['cue'] and a['id']!=b['id'] for a in first for b in second)


def distinct_pixels(rows):
    """Repeated physical pixels contribute one score, independently of support."""
    seen=set();result=[]
    for row in rows:
        key=(quality(row).get('pixel_digest') if 'quality' in row else None) or ('observation',row['id'])
        if key not in seen:result.append(row);seen.add(key)
    return result


class Scores:
    """Exact weighted banks; identical vectors share computation, not votes."""
    def __init__(self,config):
        from collections import OrderedDict
        self.count=0;self.limit=config.data['sapiensid']['execution']['max_witness_comparisons']
        self.banks={};self.cache=OrderedDict()
        self.cache_limit=max(1,min(4096,config.data['sapiensid']['execution']['reference_bytes']//65536))

    def bank(self,rows):
        key=id(rows);state=self.banks.get(key)
        if state is None:
            state=dict(rows=rows,length=0,pixels=set(),vectors={});self.banks[key]=state
        # Scene banks grow append-only and remain frozen for attachments.
        for row in rows[state['length']:]:
            pixel=(quality(row).get('pixel_digest') if 'quality' in row else None) or ('observation',row['id'])
            if pixel in state['pixels']:continue
            state['pixels'].add(pixel)
            value=row['_vector']
            feature=row.get('_score_key')
            if feature is None:feature=row['_score_key']=value.tobytes()
            if feature in state['vectors']:state['vectors'][feature][1]+=1
            else:state['vectors'][feature]=[value,1]
        state['length']=len(rows)
        return state['vectors']

    def _charge(self):
        if self.count>=self.limit:
            raise ValueError('Person matching exceeds sapiensid.execution.max_witness_comparisons.')
        self.count+=1

    def certifies(self,row,rows,threshold):
        """Prove every cosine clears a threshold; never approximate rival scores.

        Each sample is within radius of the fixed pivot. Cauchy-Schwarz gives
        q.sample >= q.pivot - ||q|| * radius. The explicit float32 dot error
        allowance also covers the exact scorer's rounding. An inconclusive
        bound always falls back to every retained witness.
        """
        if not rows:return False
        bank=self.bank(rows);state=self.banks[id(rows)]
        # Preserve the exact score cache for repeated identical features.
        if len(bank)==1 and row['_vector'].tobytes() in bank:
            return self.compare([row],rows)[1]>=threshold
        if 'pivot' not in state:
            state.update(pivot=rows[0]['_vector'],radius=0.,max_norm=0.,cap_length=0)
        pivot=state['pivot'].astype('float64')
        for sample in rows[state['cap_length']:]:
            value=sample['_vector'].astype('float64')
            self._charge()
            state['radius']=max(state['radius'],float(np.linalg.norm(value-pivot))+1e-12)
            state['max_norm']=max(state['max_norm'],float(np.linalg.norm(value))+1e-12)
        state['cap_length']=len(rows)
        value=row['_vector'].astype('float64');norm=float(np.linalg.norm(value))+1e-12
        self._charge()
        unit=np.finfo(np.float32).eps/2
        operations=2*len(value);gamma=operations*unit/(1-operations*unit)
        lower=float(np.dot(value,pivot))-norm*state['radius']-gamma*norm*state['max_norm']-1e-10
        return lower>=threshold

    def compare(self,first,second):
        bank=self.bank(second);total=0.;minimum=float('inf');count=0
        for row in distinct_pixels(first):
            value=row['_vector']
            feature=row.get('_score_key')
            if feature is None:feature=row['_score_key']=value.tobytes()
            for other,(sample,weight) in bank.items():
                key=(feature,other)
                score=self.cache.get(key)
                if score is None:
                    self._charge()
                    score=float(np.dot(value,sample));self.cache[key]=score
                    if len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
                else:self.cache.move_to_end(key)
                total+=score*weight;count+=weight;minimum=min(minimum,score)
        return (total/count,minimum) if count else (-1.,-1.)


def scene_plan(db,scan,config):
    settings=config.data['sapiensid']['identities'];expected=contract(config)
    record=db.execute('SELECT * FROM scans WHERE id=?',(scan,)).fetchone()
    completed=db.execute('SELECT COUNT(*) FROM body_cues WHERE scan=? AND contract=?',(scan,expected)).fetchone()[0]
    if record is None or record['status']!='complete' or completed!=record['total']:
        raise ValueError('Independent person input is incomplete; run recognition to collect all selected cues.')
    from .reference_roles import strict_sql
    from .pose_policy import anchor_sql
    rows=[];cost=0
    for raw in db.execute('SELECT b.id,b.scan,b.cue,b.seconds,b.contract,b.face,b.geometry,b.quality,b.rejected,b.embedding,'
            'o.dimensions AS face_dimensions,o.embedding AS face_embedding,o.rejected AS face_rejected,'
            'm.person AS face_person,CASE WHEN o.id IS NOT NULL THEN '+
            strict_sql()+' AND '+anchor_sql('o',config.data['clustering'])+' ELSE 0 END AS face_anchor FROM body_observations b '
            'LEFT JOIN observations o ON o.id=b.face LEFT JOIN members m ON m.observation=o.id '
            'WHERE b.scan=? AND b.contract=? AND b.rejected IS NULL AND b.embedding IS NOT NULL ORDER BY b.seconds,b.id',
            (scan,expected)):
        row=dict(raw);cost+=len(row['embedding'])*3+len(row['quality'])+len(row['geometry'])+4096
        if cost>config.data['sapiensid']['execution']['reference_bytes']:
            raise ValueError('Person scene references exceed sapiensid.execution.reference_bytes.')
        row['_vector']=vector(row)
        from .clustering import vector as face_vector
        row['_face_vector']=(face_vector(dict(dimensions=row['face_dimensions'],embedding=row['face_embedding']))
            if row['face_embedding'] is not None and row['face_rejected'] is None else None)
        rows.append(row)
    from .body_guards import FaceGuard,resolve_claims
    guard=FaceGuard(db,config)
    scores=Scores(config);seeds={};decisions={}
    for row in rows:
        if row['face_person'] is None:continue
        person=canonical(db,row['face_person'])
        strong=bool(row['face_anchor']) and anchor(row,settings)
        decisions[row['id']]=dict(person=person,status='accepted',role='anchor' if strong else 'auxiliary',
                                  reason='measured_face_association')
        if strong:seeds.setdefault(person,[]).append(row)
    # Face-linked bodies are frozen before processing faceless queries.
    queries=[r for r in rows if r['id'] not in decisions]
    from .body_seed_modes import FrozenSeeds
    frozen=FrozenSeeds(seeds,scores,settings) if queries else None
    remaining=[]
    for row in queries:
        ranked=frozen.rank(row,guard)
        if ranked and max(r[5] for r in ranked)>=settings['match_threshold']-settings['ambiguity_margin']:
            choices=[r for r in ranked if r[4]]
            mean,minimum,person,bank,admissible,_=(choices or ranked)[0]
            runner=max((r[5] for r in ranked if r[2]!=person),default=-1.)
            good=admissible and mean-runner>=settings['ambiguity_margin']
            decisions[row['id']]=dict(person=person if good else None,status='accepted' if good else 'withheld',
                role='auxiliary',reason='frozen_face_body_bank' if good else 'ambiguous_or_insufficient_face_body_bank',
                score=mean,minimum=minimum,runner=runner)
        else:remaining.append(row)
    groups=[];weak=[]
    for row in remaining:
        if not anchor(row,settings):weak.append(row);continue
        # With one possible owner there is no rival ordering to approximate.
        # A lower bound above both gates proves the same complete-link decision.
        if (len(groups)==1 and not copresent([row],groups[0]) and guard.allows(row,groups[0])
                and scores.certifies(row,groups[0],max(settings['match_threshold'],settings['cohesion_threshold']))):
            groups[0].append(row);continue
        ranked=[]
        for index,group in enumerate(groups):
            mean,minimum=scores.compare([row],group)
            if not copresent([row],group) and guard.allows(row,group):ranked.append((mean,minimum,index))
        ranked.sort(key=lambda x:(-x[0],x[2]))
        if ranked and ranked[0][0]>=settings['match_threshold']:
            mean,minimum,index=ranked[0];runner=ranked[1][0] if len(ranked)>1 else -1.
            if minimum>=settings['cohesion_threshold'] and mean-runner>=settings['ambiguity_margin']:
                groups[index].append(row)
            else:decisions[row['id']]=dict(person=None,status='withheld',role='auxiliary',reason='body_group_ambiguous')
        else:groups.append([row])
    from .body_origins import physical_key, origins_for_scene,labels_for_groups
    origins_by_key=origins_for_scene(db,record)
    labels=labels_for_groups(db,record,groups,origins_by_key,settings)
    used=set(labels.values())
    for index,group in enumerate(groups):
        # Exact source observation origins, not previous assignments, preserve
        # IDs across reruns while allowing changed evidence to split a group.
        person=labels.get(index)
        if person is None:
            person=str(uuid.uuid5(uuid.NAMESPACE_URL,digest(['body-person',physical_key(record,group[0])])))
            attempt=0;seed=person
            while person in used:
                person=str(uuid.uuid5(uuid.UUID(seed),encoded(['split',physical_key(record,group[0]),attempt])))
                attempt+=1
        used.add(person)
        support=len(independent(group,settings));status='accepted' if support>=settings['min_support'] else 'provisional'
        for row in group:
            decisions[row['id']]=dict(person=person,status=status,role='anchor',reason='body_scene_core',independent_support=support)
    # Weak observations may attach to frozen accepted cores but never train them.
    banks=[(decisions[g[0]['id']]['person'],g) for g in groups if decisions[g[0]['id']]['status']=='accepted']
    for row in weak:
        if (len(banks)==1 and not copresent([row],banks[0][1]) and guard.allows(row,banks[0][1])
                and scores.certifies(row,banks[0][1],max(settings['match_threshold'],settings['cohesion_threshold']))):
            decisions[row['id']]=dict(person=banks[0][0],status='accepted',role='auxiliary',reason='body_core_attachment')
            continue
        ranked=sorted([(scores.compare([row],g),p,g) for p,g in banks],key=lambda x:(-x[0][0],x[1]))
        best=ranked[0] if ranked else None;runner=ranked[1][0][0] if len(ranked)>1 else -1.
        good=best and best[0][0]>=settings['match_threshold'] and best[0][1]>=settings['cohesion_threshold'] and best[0][0]-runner>=settings['ambiguity_margin'] and not copresent([row],best[2]) and guard.allows(row,best[2])
        decisions[row['id']]=dict(person=best[1] if good else None,status='accepted' if good else 'withheld',role='auxiliary',
            reason='body_core_attachment' if good else pixel_gate(row['quality'],settings)['reason'] or 'body_anchor_quality')
    resolve_claims(rows,decisions,config)
    if settings['anchor_min_person_pixels'] is not None:
        for row in rows:decisions[row['id']]['anchor_pixel_gate']=pixel_gate(row['quality'],settings)
    return record,rows,decisions


def rebuild(database,selected,config,result):
    if not selected or not available(database.connection):return result
    db=database.connection
    from .body_cache import cached,stamp as cache_stamp
    if cached(db,selected,config):return results(db,selected,result)
    # Plan every selected scene before replacing any derived state.
    plans=[]
    if enabled(config):
        with tracking(selected,'Build scene person identities',unit='scene') as items:
            plan_bytes=0
            for item in items:
                scan,rows,decisions=scene_plan(db,item['scan'],config)
                plan_bytes+=len(encoded(decisions))+len(rows)*128
                if plan_bytes>config.data['sapiensid']['execution']['plan_bytes']:
                    raise ValueError('Person scene plan exceeds sapiensid.execution.plan_bytes.')
                plans.append((scan,[dict(id=r['id']) for r in rows],decisions))
    from .publication_queue import enqueue
    with db:
        from .body_aliases import retire_disabled
        retire_disabled(database,selected,config)
        for item in selected:
            scan=db.execute('SELECT * FROM scans WHERE id=?',(item['scan'],)).fetchone()
            db.execute('DELETE FROM body_members WHERE observation IN (SELECT b.id FROM body_observations b '
                'JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND s.scene=?)',
                (scan['scope'],scan['namespace'],scan['scene']))
        stamp=policy(config)
        for scan,rows,decisions in plans:
            for row in rows:
                decision=decisions[row['id']];person=decision['person']
                if person is not None:
                    db.execute('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)',
                        (person,scan['scope'],scan['namespace'],time.time(),None,
                         f'stash-ai-recognition:{database.installation}:{person}'))
                    if decision['reason']=='body_scene_core':
                        db.execute('INSERT OR IGNORE INTO body_origins VALUES (?,?)',(row['id'],person))
                db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
                    (row['id'],person,decision['role'],decision['status'],stamp,encoded(decision)))
        for item in selected:
            scan=db.execute('SELECT scope,namespace,scene FROM scans WHERE id=?',(item['scan'],)).fetchone()
            enqueue(db,scan['scope'],scan['namespace'],scan['scene'],config,'body_scene_rebuild')
        if plans:
            from .body_global import reconcile
            reconcile(database,selected,config)
        cache_stamp(db,selected,config)
    return results(db,selected,result)


def results(db,selected,result):
    for groups in result.values():
        for group in groups:
            if group['person'] is not None:group['person']=canonical(db,group['person'])
    for item in selected:
        rows=db.execute('SELECT m.person,m.status,COUNT(*) AS observations FROM body_members m '
            'JOIN body_observations b ON b.id=m.observation WHERE b.scan=? GROUP BY m.person,m.status',(item['scan'],))
        result.setdefault(item['scan'],[]).extend(dict(person=r['person'],status='accepted' if r['status']=='accepted' else 'withheld',
            observations=r['observations'],evidence='person') for r in rows)
    return result


def publication_rows(db,scan):
    if not available(db):return []
    return db.execute('SELECT b.id,b.scan,b.cue,b.contract,b.rejected,b.embedding,m.person,m.status,m.role,m.policy,m.proof,'
        'p.scope,p.namespace,p.token FROM body_observations b JOIN body_members m ON m.observation=b.id '
        'LEFT JOIN people p ON p.id=m.person WHERE b.scan=? ORDER BY b.id',(scan,))

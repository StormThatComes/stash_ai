"""Scene-local person identities, with frozen face seeds and independent body anchors."""
from .resource_errors import ResourceLimit
import json
import time
import uuid
import numpy as np

from .body_observations import contract,enabled as collection_enabled
from .config import digest,plain
from .database import encoded
from .identity_merges import canonical
from .progress import tracking
from .person_anchor import pixel_gate,body_gate
from .face_landmark_fallback import body_face_sql




def defaults():
    from .face_landmark_fallback import defaults as fallback_defaults
    return dict(enabled=False, correct_contact_face_links=False, defer_contact_births=False, contact_birth_support=2, mixed_mask_score_penalty=0., contact_max_cues=12, contact_max_hidden=2, contact_max_people=4,tracklet_matching=False,body_first_matching=False,endpoint_matching=None,scene_bonus_mode='fixed',match_threshold=.85,cohesion_threshold=.80,ambiguity_margin=.1,
                track_similarity_threshold=None, turn_similarity_discount=.06, extended_track_graph=None, turn_recovery=None, scene_match_bonus=None, min_support=3,attach_min_support=2,min_appearance_distance=4,
                anchor_min_keypoints=6,anchor_min_sharpness=25.,anchor_min_person_pixels=None,
                anchor_keypoint_rules=dict(all=[],any=[]), anchor_face_landmark_fallback=fallback_defaults())


def turn_feature(settings, key):
    value = settings.get(key)
    return settings.get('tracklet_matching',False) and (settings.get('body_first_matching',False) if value is None else value)


def scene_bonus(data):
    value = data['sapiensid']['identities'].get('scene_match_bonus')
    return data['clustering']['same_scene_match_bonus'] if value is None else value


def enabled(config):
    return collection_enabled(config) and config.data['sapiensid']['identities']['enabled']


def available(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE name='body_members' AND type='table'").fetchone() is not None


def policy(config):
    from .rooted_witnesses import VIEW_POLICY
    from .clustering import matching_policy
    return digest(['body-local-v52-contact-confirmation',contract(config),matching_policy(config),
                   config.data['sapiensid']['identities'],VIEW_POLICY,'anonymous-track-lineage-v2'])


def vector(row):
    from .models import normalized_embedding
    data=row['embedding']
    if data is None or len(data)!=4096*4:raise ValueError('Invalid retained person embedding.')
    return np.frombuffer(normalized_embedding(np.frombuffer(data,dtype='<f4'),4096).tobytes(),dtype='<f4')


def quality(row):
    return json.loads(row['quality']).get('quality',{})


def anchor(row,settings):
    if settings.get('_inspection_omit_body_anchors') and row['id'] in settings['_inspection_omit_body_anchors']:
        return False
    return body_gate(row,settings)['eligible']


def independent(rows,settings,*,presorted=False,paged=False,reserve=lambda _:None):
    """At most one witness per scan/cue; repeated pixels cannot add support."""
    from .independent_evidence import body_vote,unit,Votes,stamp
    from .paged_references import rows as page_rows
    votes=Votes();seen=set()
    save=page_rows if paged else list
    distance=settings['min_appearance_distance']
    class_bytes=0
    if distance:
        # Freeze fingerprint classes before cue matching. A bridge hash must not
        # consume two otherwise independent observations just because it came first.
        rows=save(rows)
        by_pixel={}
        for row in rows:
            value=quality(row).get('appearance_hash')
            if value:
                pixel=unit(body_vote(row))
                number=int(value,16)
                by_pixel[pixel]=min(number,by_pixel.get(pixel,number))
        from .appearance_index import canonical_buckets
        classes=canonical_buckets(set(by_pixel.values()),distance)
        from .memory_size import size
        class_bytes=size([by_pixel,classes])
        reserve(class_bytes)
    def candidates():
        charge=0;count=0
        for row in (rows if presorted else sorted(rows,key=lambda r:(r['scan'],r['cue'],r['id']))):
            if row['id'] in seen:continue
            seen.add(row['id'])
            vote=body_vote(row);physical=unit(vote)
            fingerprint=(f'appearance:{classes[by_pixel[physical]]:016x}' if distance and physical in by_pixel
                         else encoded(physical))
            # Retain every cue edge of an appearance. A greedy representative can
            # consume the sole cue of a different, otherwise independent crop.
            vote=dict(vote,quality=encoded({'quality':{'pixel_digest':fingerprint}}))
            votes.add(vote);count+=1
            from .memory_size import size
            charge=class_bytes+size([vars(votes),seen]) if count & (count-1)==0 else charge+2048
            reserve(charge)
            yield dict(row=row,pixel=unit(vote),stamp=stamp(vote))
    candidates=save(candidates())
    len(votes)
    def freeze(value):return tuple(freeze(v) for v in value) if isinstance(value,(tuple,list)) else value
    def selected():
        chosen=set()
        for record in candidates:
            pixel=freeze(record['pixel']);timestamp=freeze(record['stamp'])
            if votes.owners.get(pixel)==timestamp and pixel not in chosen:
                chosen.add(pixel);yield record['row']
    return save(selected())


class SceneGroup(list):
    def __init__(self, row):
        super().__init__([row])
        self.cues = {(row['scan'], row['cue']): row['id']}

    def append(self, row):
        super().append(row)
        self.cues[row['scan'], row['cue']] = row['id']


def copresent(first,second):
    if isinstance(second, SceneGroup):
        return any((other := second.cues.get((row['scan'],row['cue']))) is not None and other != row['id']
                   for row in first)
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
        self.mask_penalty=config.data['sapiensid'].get('identities',{}).get('mixed_mask_score_penalty',0.)
        self.count=0;self.limit=config.data['sapiensid']['execution']['max_witness_comparisons']
        self.banks={};self.cache=OrderedDict()
        self.fingerprints=OrderedDict()
        self.cache_limit=max(1,min(4096,config.data['sapiensid']['execution']['reference_bytes']//65536))

    def bank(self,rows):
        key=id(rows);state=self.banks.get(key)
        if state is None:
            state=dict(rows=rows,length=0,pixels=set(),vectors={});self.banks[key]=state
        # Scene banks grow append-only and remain frozen for attachments.
        for row in rows[state['length']:]:
            pixel=(quality(row).get('pixel_digest') if 'quality' in row else None) or ('observation',row['id'])
            weight=int(pixel not in state['pixels'])
            state['pixels'].add(pixel)
            value=row['_vector']
            feature=row.get('_score_key')
            if feature is None:
                from .body_numeric import vector_digest
                feature=row['_score_key']=vector_digest(value)
            if feature in state['vectors']:state['vectors'][feature][1]+=weight
            else:state['vectors'][feature]=[value,weight]
        state['length']=len(rows)
        return state['vectors']

    def _charge(self):
        from stash_ai.runtime.cancellation import current
        token=current()
        if token is not None:token.check()
        if self.count>=self.limit:
            raise ResourceLimit('Person matching exceeds sapiensid.execution.max_witness_comparisons.')
        self.count+=1

    def face_similarity(self,first,second):
        from stash_ai.runtime.cancellation import current
        token=current()
        if token is not None:token.check()
        key=('face', *sorted((self._digest(first),self._digest(second))))
        score=self.cache.get(key)
        if score is None:
            from .body_numeric import dot
            score=dot(first,second,self._charge,'face-dot');self.cache[key]=score
            if len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
        else:self.cache.move_to_end(key)
        return score

    def _digest(self,value):
        from .body_numeric import vector_digest
        import weakref
        owner=value
        while isinstance(owner.base,np.ndarray):owner=owner.base
        if value.flags.writeable or not isinstance(owner.base,bytes):return vector_digest(value)
        entry=self.fingerprints.get(id(value))
        if entry is not None and entry[0]() is value:
            self.fingerprints.move_to_end(id(value));return entry[1]
        result=vector_digest(value)
        self.fingerprints[id(value)]=(weakref.ref(value),result)
        if len(self.fingerprints)>self.cache_limit:self.fingerprints.popitem(last=False)
        return result

    def certifies(self,row,rows,threshold):
        """Prove every cosine clears a threshold; never approximate rival scores.

        Each sample is within radius of the fixed pivot. Cauchy-Schwarz gives
        q.sample >= q.pivot - ||q|| * radius. The explicit float32 dot error
        allowance also covers the exact scorer's rounding. An inconclusive
        bound always falls back to every retained witness.
        """
        if not rows:return False
        if self.mask_penalty and any(quality(r).get('embedding_foreground')!=quality(row).get('embedding_foreground') for r in rows):
            return self.compare([row],rows)[1]>=threshold
        bank=self.bank(rows);state=self.banks[id(rows)]
        # Preserve the exact score cache for repeated identical features.
        from .body_numeric import vector_digest
        if len(bank)==1 and vector_digest(row['_vector']) in bank:
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
        if self.mask_penalty:
            from .body_numeric import domain
            left=[(a,domain(a)) for a in distinct_pixels(first)]
            right=[(b,domain(b)) for b in distinct_pixels(second)]
            total=0.;minimum=float('inf');count=0
            for a,da in left:
                for b,db in right:
                    score=self.face_similarity(a['_vector'],b['_vector'])
                    if da is not None and db is not None and da!=db:score-=self.mask_penalty
                    total+=score;minimum=min(minimum,score);count+=1
            return (total/count,minimum) if count else (-1.,-1.)
        bank=self.bank(second);total=0.;minimum=float('inf');count=0
        pixels=set()
        for row in first:
            pixel=(quality(row).get('pixel_digest') if 'quality' in row else None) or ('observation',row['id'])
            fresh=pixel not in pixels;pixels.add(pixel)
            value=row['_vector']
            feature=row.get('_score_key')
            if feature is None:
                from .body_numeric import vector_digest
                feature=row['_score_key']=vector_digest(value)
            for other,(sample,weight) in bank.items():
                key=(feature,other) if feature<=other else (other,feature)
                score=self.cache.get(key)
                if score is None:
                    from .body_numeric import dot
                    score=dot(value,sample,self._charge);self.cache[key]=score
                    if len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
                else:self.cache.move_to_end(key)
                total+=score*weight*fresh;count+=weight*fresh;minimum=min(minimum,score)
        return (total/count,minimum) if count else (-1.,-1.)


def _prepare_vectors(rows):
    from functools import lru_cache
    from .clustering import vector as face_vector

    @lru_cache(maxsize=16)
    def normalized(blob):
        result = vector({'embedding': blob})
        result.flags.writeable = False
        return result

    for row in rows:
        row['_vector'] = normalized(row['embedding'])
        row['_face_vector'] = (face_vector(dict(dimensions=row['face_dimensions'], embedding=row['face_embedding']))
            if row['face_embedding'] is not None and row['face_rejected'] is None else None)


def _load_scene(db,scan,config, *, prepare_vectors=True):
    settings=config.data['sapiensid']['identities'];expected=contract(config)
    record=db.execute('SELECT * FROM scans WHERE id=?',(scan,)).fetchone()
    completed=db.execute('SELECT COUNT(*) FROM body_cues WHERE scan=? AND contract=?',(scan,expected)).fetchone()[0]
    if record is None or record['status']!='complete' or completed!=record['total']:
        raise ValueError('Independent person input is incomplete; run recognition to collect all selected cues.')
    from .reference_roles import same_scene_face_sql, strict_sql
    from .pose_policy import anchor_sql
    rows=[];cost=0
    # The scene pass recomputes contact corrections; never reuse an old decision.
    for raw in db.execute('SELECT '+body_face_sql(settings=config.data['sapiensid']['identities'],respect_corrections=False)+',b.id,b.scan,b.cue,b.seconds,b.contract,b.face,b.geometry,b.quality,b.rejected,b.embedding,'
            'o.quality AS face_quality,o.dimensions AS face_dimensions,o.embedding AS face_embedding,o.rejected AS face_rejected,'
            'm.person AS face_person,'+strict_sql()+' AS strict_face,'+same_scene_face_sql()+' AS same_scene_face,'
            'CASE WHEN o.id IS NOT NULL THEN '+anchor_sql('o',config.data['clustering'])+
            ' ELSE 0 END AS face_quality_ok FROM body_observations b '
            'LEFT JOIN observations o ON o.id=b.face LEFT JOIN members m ON m.observation=o.id '
            'WHERE b.scan=? AND b.contract=? AND b.rejected IS NULL AND b.embedding IS NOT NULL ORDER BY b.seconds,b.id',
            (scan,expected)):
        row=dict(raw)
        row['face_anchor'] = bool(row.pop('strict_face') and row['face_quality_ok'])
        cost+=len(row['embedding'])*3+len(row['quality'])+len(row['geometry'])+len(row['anchor_face'] or '')+4096
        if cost>config.data['sapiensid']['execution']['reference_bytes']:
            raise ResourceLimit('Person scene references exceed sapiensid.execution.reference_bytes.')
        rows.append(row)
    if prepare_vectors:
        _prepare_vectors(rows)
    from .body_origins import origins_for_scene, ordered_owners
    roots={p:canonical(db,p) for p in {r['face_person'] for r in rows if r['face_person'] is not None}}
    origins=origins_for_scene(db,record,rows)
    record = dict(record)
    if turn_feature(settings,'turn_recovery') and origins:
        from .scan_heads import predicate
        ids = {p for values in origins.values() for p in values}
        record['protected_turn_owners'] = [r[0] for r in db.execute(
            "SELECT p.id FROM people p WHERE p.id IN (SELECT value FROM json_each(?)) AND "
            "(p.performer IS NOT NULL OR EXISTS (SELECT 1 FROM body_members m JOIN body_observations b "
            "ON b.id=m.observation JOIN scans s ON s.id=b.scan WHERE m.person=p.id AND m.status='accepted' "
            "AND s.status='complete' AND s.scene<>? AND s.scope=? AND s.namespace=? AND "+predicate()+"))",
            (encoded(sorted(ids)),record['scene'],record['scope'],record['namespace']))]
    shot_map = db.execute('SELECT value FROM metadata WHERE key=?',('cue-view-shots:'+scan,)).fetchone()
    record['cut_starts'] = tuple(start for start,shot in json.loads(shot_map[0]) if shot is not None) if shot_map else ()
    if not record['cut_starts']:
        record['cut_starts'] = tuple(sorted({q for r in rows if isinstance(q:=quality(r).get('view_shot'),int)}))
    from .cue_gaps import blockers
    record['gap_boxes'] = blockers(db,scan,expected)
    record['occupied_cues'] = sorted(record['gap_boxes'])
    motion = db.execute('SELECT value FROM metadata WHERE key=?',('cue-continuity:'+scan,)).fetchone()
    record['motion'] = motion[0] if motion else ''
    return record,rows,roots,origins,ordered_owners(db,origins)


def scene_plan(db,scan,config):
    return _plan(db,config,*_load_scene(db,scan,config))


def _plan(db,config,record,rows,roots,origins_by_key,owners,frozen_faces=None, *, capture_hints=None):
    exclusions = {}
    while True:
        retry = _plan_once(db,config,record,rows,roots,origins_by_key,owners,frozen_faces,capture_hints,exclusions)
        if isinstance(retry,tuple):
            return retry
        exclusions.update(retry)
        updated = []
        for row in rows:
            if row['id'] in retry:
                meta = json.loads(row['quality'])
                meta['quality'].update(query_only=True,query_only_reason='held_out_rival')
                row = dict(row,quality=encoded(meta))
            updated.append(row)
        rows = updated


def _plan_once(db,config,record,rows,roots,origins_by_key,owners,frozen_faces,capture_hints,exclusions):
    settings=config.data['sapiensid']['identities']
    from .body_guards import FaceGuard,resolve_claims
    scores=Scores(config);guard=FaceGuard(db,config,scores=scores,frozen=frozen_faces)
    graph = None
    if (settings.get('tracklet_matching') and settings.get('body_first_matching')
            or settings.get('correct_contact_face_links')):
        from .body_tracklets import paths
        graph = list(paths(rows, settings, scores, record.get('occupied_cues', ()), extended=turn_feature(settings,'extended_track_graph'), gap_boxes=record.get('gap_boxes'), motion=record.get('motion','')))
    from .contact_identity_proof import correct, rebind
    rows=correct(rows,graph,settings,scores)
    graph=rebind(graph,rows)
    seeds={};decisions={}
    from .contact_anchors import questionable_links
    contact_checks = {}
    uncertain = questionable_links(rows,settings,scores,graph,evidence=contact_checks)
    uncertain.difference_update(r['id'] for r in rows if r.get('_contact_correction'))
    for row in rows:
        if row.get('_contact_correction') and row['id'] in contact_checks:
            contact_checks[row['id']].update(status='corrected', reason='independent_clean_endpoints_corrected_link',
                witness_ids=row['_contact_correction']['witness_ids'])
    for row in rows:
        if row['face_person'] is None:continue
        person=roots[row['face_person']]
        if row['id'] in uncertain:
            decisions[row['id']] = dict(person=None,status='withheld',role='auxiliary',
                reason='unconfirmed_contact_face',track_blocker=dict(reason='unresolved_partner',cue=row['cue']))
            continue
        strong=bool(row['face_anchor']) and anchor(row,settings)
        decisions[row['id']]=dict(person=person,status='accepted',role='anchor' if strong else 'auxiliary',
                                  reason='measured_face_association')
        if row.get('_contact_correction'):
            decisions[row['id']].update(reason='corrected_contact_face_link',contact_correction=row['_contact_correction'])
        if strong:seeds.setdefault(person,[]).append(row)
    if graph:
        from .contact_anchors import bridge_faces
        bridge_faces(graph,decisions,settings,uncertain,guard)
    # Face-linked bodies are frozen before processing faceless queries.
    for row in rows:
        if quality(row).get('query_only_reason') == 'unresolved_partner' and row['id'] not in decisions:
            decisions[row['id']] = dict(person=None,status='withheld',role='auxiliary',
                reason='unresolved_partner',track_blocker=dict(reason='unresolved_partner',cue=row['cue']))
    queries=[r for r in rows if r['id'] not in decisions]
    from .body_seed_modes import FrozenSeeds
    review_faces = settings.get('tracklet_matching') and seeds and any(
        r.get('face_person') and not r.get('face_anchor') for r in rows)
    contextual = [r for r in rows if r.get('same_scene_face')]
    frozen=FrozenSeeds(seeds,scores,settings) if queries or review_faces or contextual else None
    if graph is None and turn_feature(settings,'extended_track_graph'):
        from .body_tracklets import paths
        graph = list(paths(rows,settings,scores,record.get('occupied_cues',()),extended=True, gap_boxes=record.get('gap_boxes'), motion=record.get('motion','')))
    anonymous = {}
    track_lineage = {}
    if settings.get('tracklet_matching') and (queries or review_faces):
        from .body_tracklets import plan, claim_fragments
        anonymous = plan(record,rows,seeds,decisions,settings,scores,guard,frozen,graph=graph)
        claim_fragments(rows,anonymous,decisions,frozen,guard,settings)
        track_lineage = {**{oid:d['anonymous_fragment'] for oid,d in decisions.items() if d.get('anonymous_fragment')}, **anonymous}
        queries = [r for r in queries if r['id'] not in decisions]
    if contextual:
        from .scene_body_anchors import corroborate
        corroborate(contextual, rows, decisions, frozen, guard)
    remaining=[]
    borderline=[]
    cue_choices={}
    for row in queries:
        ranked=frozen.rank(row,guard)
        if ranked and max(r[5] for r in ranked)>=settings['match_threshold']-settings['ambiguity_margin']:
            choices=[r for r in ranked if r[4]]
            mean,minimum,person,bank,admissible,_=(choices or ranked)[0]
            runner,rival=max(((r[5],r[2]) for r in ranked if r[2]!=person),default=(-1.,None))
            good=admissible and mean-runner>=settings['ambiguity_margin']
            decisions[row['id']]=dict(person=person if good else None,status='accepted' if good else 'withheld',
                role='auxiliary',reason='frozen_face_body_bank' if good else 'ambiguous_or_insufficient_face_body_bank',
                score=mean,minimum=minimum,runner=runner,rival=rival,query_owner=person,threshold=settings['match_threshold'],
                required_margin=settings['ambiguity_margin'], cohesion_threshold=settings['cohesion_threshold'],
                reference_count=len(bank), witness_ids=[w['id'] for w in bank[:8]])
            cue_choices[row['id']] = ranked
            if not good and anchor(row,settings):
                borderline.append(row)
        else:remaining.append(row)
    from .contact_anchors import joint_body_claims
    joint_body_claims(rows,decisions,cue_choices,settings,reflection_threshold=max(
        config.data['recognition']['match_threshold'],config.data['recognition']['cooccurrence_merge_threshold']))
    borderline = [r for r in borderline if decisions[r['id']]['status']=='withheld']
    groups=[];weak=[]
    from .body_seed_modes import paired_order
    if settings.get('body_first_matching'):
        remaining=paired_order(remaining,settings,scores,guard)
    for row in remaining:
        if not anchor(row,settings):weak.append(row);continue
        # With one possible owner there is no rival ordering to approximate.
        # A lower bound above both gates proves the same complete-link decision.
        if (len(groups)==1 and not copresent([row],groups[0]) and guard.allows(row,groups[0])
                and scores.certifies(row,groups[0],max(settings['match_threshold'],settings['cohesion_threshold']))):
            groups[0].append(row);continue
        ranked=[]
        for index,group in enumerate(groups):
            if copresent([row],group):
                continue
            mean,minimum=scores.compare([row],group)
            if guard.allows(row,group):ranked.append((mean,minimum,index))
        ranked.sort(key=lambda x:(-x[0],x[2]))
        if ranked and ranked[0][0]>=settings['match_threshold']:
            mean,minimum,index=ranked[0];runner=ranked[1][0] if len(ranked)>1 else -1.
            if minimum<settings['cohesion_threshold']:
                # A high mean cannot suppress a distinct complete-link core.
                # Keep it in the frozen competition pass before publishing.
                groups.append(SceneGroup(row))
            elif mean-runner>=settings['ambiguity_margin']:
                groups[index].append(row)
            else:decisions[row['id']]=dict(person=None,status='withheld',role='auxiliary',reason='body_group_ambiguous')
        else:groups.append(SceneGroup(row))
    # Later groups must compete with earlier assignments. Freeze all original
    # groups before pruning, so removing a query cannot improve another vote.
    retained=[]
    for index,group in enumerate(groups):
        keep=[]
        for row in group:
            rivals=[other for j,other in enumerate(groups) if j!=index
                    and not copresent([row],other) and guard.allows(row,other)]
            if rivals:
                mean,minimum=scores.compare([row],group)
                runner, rival_index=max((scores.compare([row],other)[0],i) for i,other in enumerate(rivals))
                # A dense ambiguous appearance cannot raise its own mean
                # enough to overrule the original group's weaker witnesses.
                if minimum-runner<settings['ambiguity_margin']:
                    decisions[row['id']]=dict(person=None,status='withheld',role='auxiliary',
                        reason='body_group_competing_identity',score=mean,minimum=minimum,runner=runner,
                        competing_witnesses=[next((r['id'] for r in group if r['id']!=row['id']),None),rivals[rival_index][0]['id']])
                    continue
            keep.append(row)
        if keep:retained.append(keep)
    groups=retained
    from .body_origins import physical_key, labels_for_groups
    labels=labels_for_groups(record,groups,origins_by_key,settings,owners)
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
        from .contact_identity_proof import birth_ready
        deferred=not birth_ready(group,settings,independent)
        if deferred:status='provisional'
        from .contact_identity_proof import pending_proof
        proof = pending_proof(group,settings,independent) if deferred else None
        for row in group:
            decisions[row['id']]=dict(person=person,status=status,role='auxiliary' if deferred else 'anchor',reason='contact_birth_pending' if deferred else 'body_scene_core',independent_support=support,
                required_support=settings['min_support'])
            if proof is not None:
                decisions[row['id']].update(contact_birth=proof, anonymous_fragment=person)
    # Weak observations may attach to frozen accepted cores but never train them.
    banks=[(decisions[g[0]['id']]['person'],g) for g in groups if decisions[g[0]['id']]['status']=='accepted']
    for row in weak:
        if (len(banks)==1 and not copresent([row],banks[0][1]) and guard.allows(row,banks[0][1])
                and scores.certifies(row,banks[0][1],max(settings['match_threshold'],settings['cohesion_threshold']))):
            decisions[row['id']]=dict(person=banks[0][0],status='accepted',role='auxiliary',reason='body_core_attachment',
                certified_floor=max(settings['match_threshold'], settings['cohesion_threshold']),
                reference_count=len(banks[0][1]), witness_ids=[w['id'] for w in banks[0][1][:8]])
            continue
        ranked=sorted([(scores.compare([row],g),p,g) for p,g in banks
            if not copresent([row],g) and guard.allows(row,g)],key=lambda x:(-x[0][0],x[1]))
        best=ranked[0] if ranked else None;runner=ranked[1][0][0] if len(ranked)>1 else -1.
        good=best and best[0][0]>=settings['match_threshold'] and best[0][1]>=settings['cohesion_threshold'] and best[0][0]-runner>=settings['ambiguity_margin']
        decisions[row['id']]=dict(person=best[1] if good else None,status='accepted' if good else 'withheld',role='auxiliary',
            reason='body_core_attachment' if good else pixel_gate(row['quality'],settings)['reason'] or 'body_anchor_quality')
        if good:
            (mean, minimum), _, witnesses = best
            decisions[row['id']].update(score=mean, minimum=minimum, threshold=settings['match_threshold'],
                cohesion_threshold=settings['cohesion_threshold'], required_margin=settings['ambiguity_margin'],
                reference_count=len(witnesses), witness_ids=[w['id'] for w in witnesses[:8]])
            if len(ranked)>1:
                decisions[row['id']].update(runner=runner, rival=ranked[1][1])
        rules=settings.get('anchor_keypoint_rules',{})
        if not good and (rules.get('all') or rules.get('any')):
            regional=body_gate(row,settings)['keypoint_rules']
            if not regional['eligible']:
                decisions[row['id']]['anchor_keypoint_rules']=regional
                if decisions[row['id']]['reason']=='body_anchor_quality':
                    decisions[row['id']]['reason']=regional['reason']
    # A near miss against a frozen face bank can retain its path while
    # the original competitors still participate in every subsequent replay.
    for oid,track in anonymous.items():
        d = decisions[oid]
        if d['status']!='accepted' and d.get('reason')!='different_face_root':
            d.update(person=track,role='auxiliary',status='provisional',reason='anonymous_body_track')
    from .scene_fragments import preserve_fragments, attach_with_scene_support
    preserve_fragments(record,borderline,decisions,settings,scores,guard)
    from .roster_presence import EntrantPaths
    starts = record.get('cut_starts',())
    entrants = EntrantPaths(rows,anonymous,settings,starts) if settings.get('scene_bonus_mode')=='evidence' and anonymous else None
    attach_with_scene_support(rows,groups,seeds,decisions,config,scores,guard,cut_starts=starts,entrants=entrants)
    from .body_perspectives import recover
    track_settings = dict(settings,face_threshold=config.data['recognition']['match_threshold'],
                          face_margin=config.data['recognition']['ambiguity_margin'],scene_context_bonus=scene_bonus(config.data),
                          roster_support=max(2,config.data['clustering']['same_scene_min_support']))
    recover(rows, decisions, seeds, track_settings, scores, guard, occupied=record.get('occupied_cues', ()),
            cut_starts=starts, entrants=entrants, graph=graph if turn_feature(settings,'extended_track_graph') else None,
            motion=record.get('motion',''))
    if turn_feature(settings,'turn_recovery'):
        from .scene_continuity import recover_turns
        recover_turns(rows, decisions, seeds, settings, scores, guard, record.get('protected_turn_owners', ()),
                      occupied_cues=record.get('occupied_cues',()),cuts=starts,scene_bonus=scene_bonus(config.data))
    if graph is not None:
        from bisect import bisect_right
        for chain,_ in graph:
            for first,last in zip(chain,chain[1:]):
                if (last['cue']>first['cue']+1 and
                        bisect_right(starts,first['cue'])==bisect_right(starts,last['cue'])):
                    decisions[last['id']]['gap_predecessor'] = first['id']
    for row in rows:
        blocker = decisions[row['id']].get('track_blocker')
        if blocker:
            blocker['scene'] = record['scene']
        q = quality(row)
        if q.get('query_only'):
            decisions[row['id']]['query_only_reason'] = q.get('query_only_reason','uncertain_body_geometry')
            decisions[row['id']]['retry_when'] = 'confirmed_core_changes'
    newly_excluded = {oid:d['view_exclusion'] for oid,d in decisions.items()
                      if oid not in exclusions and d.get('view_exclusion')}
    if newly_excluded:
        return newly_excluded
    for oid,proof in exclusions.items():
        decisions[oid]['view_exclusion'] = proof
    from .local_body_hints import observe
    if capture_hints is None:
        observe(rows, groups, seeds, scores, guard, config, frozen=frozen)
    else:
        capture_hints(groups, seeds, frozen)
    for oid,fragment in track_lineage.items():
        decisions[oid]['anonymous_fragment'] = fragment
    resolve_claims(rows,decisions,config,scores=scores)
    if settings['anchor_min_person_pixels'] is not None:
        for row in rows:decisions[row['id']]['anchor_pixel_gate']=pixel_gate(row['quality'],settings)
    for oid, proof in contact_checks.items():
        decisions[oid]['contact_confirmation'] = proof
    return record,rows,decisions


def rebuild(database,selected,config,result):
    if not selected or not available(database.connection):return result
    db=database.connection
    from .body_cache import cached,stamp as cache_stamp
    from .replay_intervention import controlled
    if not controlled() and cached(db,selected,config):
        from .staged_aliases import waiting
        if enabled(config) and waiting(db,selected,config):
            from .body_global import reconcile
            from .body_recovery import recover
            with db:
                if not db.in_transaction:
                    db.execute('BEGIN')
                reconcile(database,selected,config)
                recover(database,selected,config)
                cache_stamp(db,selected,config)
        database.body_rebuild_mode='warm' if enabled(config) else 'disabled'
        return results(db,selected,result) if getattr(database, 'collect_publication_results', True) else result
    database.body_rebuild_mode='cold' if enabled(config) else 'disabled'
    from .body_plan_spool import Plans
    with Plans(config.data['sapiensid']['execution']['plan_bytes'],config.data['execution']['replay_snapshot_bytes']) as plans:
        # Plan every selected scene before replacing any derived state.

        if enabled(config):
            with tracking(selected,'Build scene person identities',unit='scene') as items:
                from .body_scene_jobs import planned
                from .profiling import timed_stream
                for scan,rows,decisions in timed_stream('body_scene_planning',planned(db,items,config,ids_only=True)):
                    plans.append((scan,[dict(id=r['id']) for r in rows],decisions))
        from .assignment_delta import enqueue
        with db:
            from .body_aliases import retire_disabled
            retire_disabled(database,selected,config)
            changed=set()
            # Keep temporary DDL inside the same rollback boundary for callers
            # invoking the body stage directly, without the outer rebuild wrapper.
            if not db.in_transaction:db.execute('BEGIN')
            db.execute('CREATE TEMP TABLE desired_body_members(id TEXT PRIMARY KEY)')
            db.executemany('INSERT INTO desired_body_members VALUES (?)',((oid,) for oid in plans.identifiers()))
            for item in selected:
                scan=db.execute('SELECT * FROM scans WHERE id=?',(item['scan'],)).fetchone()
                removed=db.execute('DELETE FROM body_members WHERE observation IN (SELECT b.id FROM body_observations b '
                    'JOIN scans s ON s.id=b.scan WHERE s.scope=? AND s.namespace=? AND s.scene=?) '
                    'AND observation NOT IN (SELECT id FROM desired_body_members)',
                    (scan['scope'],scan['namespace'],scan['scene'])).rowcount
                if removed:changed.add(scan['scene'])
            db.execute('DROP TABLE desired_body_members')
            stamp=policy(config)
            installation = database.installation
            for scan, rows, decisions in plans:
                people = {}
                origins = []
                memberships = []
                for row in rows:
                    oid = row['id']
                    decision = decisions[oid]
                    person = decision['person']
                    if person is not None and person not in people:
                        people[person] = (person, scan['scope'], scan['namespace'], time.time(), None,
                                          f'stash-ai-recognition:{installation}:{person}')
                    if person is not None and decision['reason'] == 'body_scene_core':
                        origins.append((oid, person))
                    memberships.append((oid, person, decision['role'], decision['status'], stamp, encoded(decision)))
                db.executemany('INSERT OR IGNORE INTO people VALUES (?,?,?,?,?,?)', people.values())
                db.executemany('INSERT OR IGNORE INTO body_origins VALUES (?,?)', origins)
                updated = db.executemany('INSERT INTO body_members VALUES (?,?,?,?,?,?) ON CONFLICT(observation) DO UPDATE SET '
                        'person=excluded.person,role=excluded.role,status=excluded.status,policy=excluded.policy,proof=excluded.proof '
                        'WHERE body_members.person IS NOT excluded.person OR body_members.role IS NOT excluded.role '
                        'OR body_members.status IS NOT excluded.status OR body_members.policy IS NOT excluded.policy '
                        'OR body_members.proof IS NOT excluded.proof', memberships).rowcount
                if updated:
                    changed.add(scan['scene'])
            for item in selected:
                scan=db.execute('SELECT scope,namespace,scene FROM scans WHERE id=?',(item['scan'],)).fetchone()
                enqueue(db,scan['scope'],scan['namespace'],item,config,scan['scene'] in changed,'body_scene_rebuild')
            if plans:
                from .body_global import reconcile
                reconcile(database,selected,config)
                from .body_recovery import recover
                recover(database,selected,config)
                from .rear_recovery import recover as recover_rear
                recover_rear(db,selected,config)
            cache_stamp(db,selected,config)
        from .derived_cache import current
        store=current()
        if enabled(config) and store and 'body-scene' in store.hits:database.body_rebuild_mode='partial'
        return results(db,selected,result) if getattr(database, 'collect_publication_results', True) else result


def results(db,selected,result):
    for groups in result.values():
        for group in groups:
            if group['person'] is not None:group['person']=canonical(db,group['person'])
    from .rebuild import _batches
    for batch in _batches(selected):
        scans = [item['scan'] for item in batch]
        for scan in scans:
            result.setdefault(scan, [])
        rows = db.execute('SELECT b.scan,m.person,m.status,COUNT(*) AS observations FROM body_members m '
            'JOIN body_observations b ON b.id=m.observation WHERE b.scan IN (SELECT value FROM json_each(?)) '
            'GROUP BY b.scan,m.person,m.status ORDER BY b.scan,m.person,m.status', (encoded(scans),))
        for row in rows:
            result[row['scan']].append(dict(person=row['person'],
                status='accepted' if row['status']=='accepted' else 'withheld',
                observations=row['observations'], evidence='person'))
    return result


def publication_rows(db,scan):
    if not available(db):return []
    return db.execute('SELECT b.id,b.scan,b.cue,b.contract,b.rejected,b.embedding,m.person,m.status,m.role,m.policy,m.proof,'
        'p.scope,p.namespace,p.token FROM body_observations b JOIN body_members m ON m.observation=b.id '
        'LEFT JOIN people p ON p.id=m.person WHERE b.scan=? ORDER BY b.id',(scan,))

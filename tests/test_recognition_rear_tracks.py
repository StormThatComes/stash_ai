import json
import numpy as np
import pytest

from stash_ai.recognition.body_identity import rebuild
from stash_ai.recognition.database import FaceDatabase, encoded
from tests.test_recognition_body_global import configured, scene


@pytest.mark.parametrize('incremental,rejected_face,missing_root', [
    (False, False, False), (False, False, True), (True, False, True), (True, True, False)])
@pytest.mark.parametrize('perspective',['back','side'])
def test_external_rear_only_scene_requires_current_independent_roots(tmp_path, incremental, rejected_face, missing_root,perspective):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        front, back = [1.,0.], [.82,.5724]
        a, ids = scene(db, config, '1', [front,back,front,front,back,front],
                       times=[0,1,2,5,6,7])
        b, queries = scene(db, config, '2', [back]*3)
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'face-A'))
        for i,oid in enumerate(ids+queries):
            rooted = i in (0,2,3,5)
            color = (15 if i<3 else 180, 15,15)
            q = dict(pixel_digest=oid, keypoints=10, sharpness=100, orientation='front' if rooted else perspective,
                     frame_signature=(bytes(color)*16).hex())
            db.execute('UPDATE body_observations SET quality=?,geometry=? WHERE id=?',
                       (encoded(dict(quality=q)),encoded(dict(bounds=[0,0,80,100])),oid))
            if rooted:
                fid = 'face-'+oid
                quality = encoded(dict(pixels=100,sharpness=100,pose_deviation=0.,visible_face_fraction=1.))
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (fid,a['scan'],[0,1,2,5,6,7][i],float(i),'','[]','[]',quality,None,2,
                     np.array(front,dtype='<f4').tobytes(),None))
                db.execute('INSERT INTO members VALUES (?,?)',(fid,'A'))
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,oid))
        if rejected_face:
            for cue, oid in enumerate(queries):
                fid = 'rejected-' + oid
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           (fid,b['scan'],cue,float(cue),'','[]','[]','{}','pixels',2,None,None))
                db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,oid))
        db.commit()
        if incremental:
            rebuild(state,[b],config,{b['scan']:[]})
            rebuild(state,[a],config,{a['scan']:[]})
        else:
            rebuild(state,[a,b],config,{a['scan']:[],b['scan']:[]})
        def members():
            return [dict(r) for r in db.execute('SELECT * FROM body_members WHERE observation IN (?,?,?)',queries)]
        assert all(r['person']=='A' and r['role']=='auxiliary' and
                   json.loads(r['proof'])['reason']=='external_rear_view' for r in members())
        if not incremental and not missing_root:
            db.execute('CREATE TEMP TABLE touched_rear(observation)')
            db.execute('CREATE TEMP TRIGGER audit_rear AFTER UPDATE ON main.body_members '
                       'BEGIN INSERT INTO touched_rear VALUES(new.observation); END')
            unrelated,_ = scene(db,config,'3',[[0.,-1.]]*2)
            rebuild(state,[unrelated],config,{})
            assert not {r[0] for r in db.execute('SELECT observation FROM touched_rear')} & set(queries)
            new_rival,_ = scene(db,config,'4',[[.55,.835]]*3)
            rebuild(state,[new_rival],config,{})
            assert all(r['person']!='A' and json.loads(r['proof'])['rear_retry']['reason']=='competing_owner'
                       for r in members())
        if missing_root:
            db.execute('DELETE FROM members WHERE observation=?',('face-'+ids[0],))
            db.commit()
            rebuild(state,[a],config,{a['scan']:[]})
            assert all(r['person']!='A' for r in members())


def test_long_vtt_turn_allows_a_single_empty_cue_and_keeps_open_root():
    from stash_ai.recognition.body_perspectives import recover
    from stash_ai.recognition.body_identity import defaults, Scores
    from types import SimpleNamespace
    settings = dict(defaults(), min_appearance_distance=0)
    scorer = Scores(SimpleNamespace(data={'sapiensid':{'execution':{
        'max_witness_comparisons':2000,'reference_bytes':2**20}}}))
    rows, decisions, seeds = [], {}, {'A':[]}
    for cue in [0,1,2,4,5,6,7,8,11,12,13,17,18,19]:
        root = cue in (0,8,11,13,17)
        rgb = [15 if cue<=8 else 180 if cue<=13 else 60,15,15]
        row = dict(id=str(cue),cue=cue,scan='s',seconds=cue,face_person='A' if root else None,
                   geometry=encoded({'bounds':[0,0,80,100]}),_vector=np.array([1.,0.]),_face_vector=None,
                   quality=encoded({'quality':dict(orientation='front' if root else 'back',
                       frame_signature=(bytes(rgb)*16).hex(),pixel_digest=str(cue))}))
        rows.append(row)
        if root: seeds['A'].append(row)
        decisions[str(cue)] = dict(person='A' if root else None,status='accepted' if root else 'withheld',
                                   reason='measured_face_association' if root else 'body_anchor_quality')
    recover(rows,decisions,seeds,settings,scorer,SimpleNamespace(allows=lambda *args:True))
    assert decisions['19']['reason']=='face_rooted_body_view'
    sources = decisions['19']['source_tracks']
    assert {(start,end) for start,end,_ in sources}=={('0','8'),('11','13')}


def test_large_component_recovers_strong_pair_without_stealing_tied_partner():
    from stash_ai.recognition.frozen_subgroups import large_subgroups
    owners = [f'p{i:03}' for i in range(135)]
    edges = {p:set() for p in owners}
    pairs = {}
    for i,(a,b) in enumerate(zip(owners,owners[1:])):
        edges[a].add(b)
        edges[b].add(a)
        pairs[a,b] = {'score':.99 if i in (0,3,4) else .86}
    groups,reason = large_subgroups(owners,edges,pairs,.85,.05,2**20)
    assert groups == [tuple(owners[:2])]
    assert reason=='incomplete_body_component'
    assert large_subgroups(owners,edges,pairs,.85,.05,64) == ((), 'subgroup_budget')


@pytest.mark.parametrize('weak_has_face',[False,True])
def test_weak_satellite_can_retry_but_measured_face_vetoes_core_transfer(weak_has_face):
    from stash_ai.recognition.body_perspectives import recover
    from stash_ai.recognition.body_identity import defaults, Scores
    from types import SimpleNamespace
    settings = dict(defaults(), min_appearance_distance=0)
    scores = Scores(SimpleNamespace(data={'sapiensid':{'execution':{
        'max_witness_comparisons':1000,'reference_bytes':2**20}}}))
    rows, decisions, seeds = [], {}, {'A':[]}
    for cue in (0,1,2,5,6,7,11,15):
        face = cue in (0,2,5,7)
        weak = cue==15
        row = dict(id=str(cue),cue=cue,scan='scan',seconds=cue,
                   face_person='A' if face else 'B' if weak and weak_has_face else None,
                   geometry=encoded(dict(bounds=[0,0,80,100])),_face_vector=None,
                   _vector=np.array([0.,1.] if weak else [1.,0.]),
                   quality=encoded(dict(quality=dict(orientation='front' if face or weak else 'back',
                       frame_signature=(bytes([15 if cue<5 else 180,15,15])*16).hex(),pixel_digest=str(cue)))))
        rows.append(row)
        if face: seeds['A'].append(row)
        decisions[str(cue)] = dict(person='A' if face else 'B',status='accepted',
                                   role='auxiliary' if weak else 'anchor',
                                   reason='measured_face_association' if face or weak and weak_has_face
                                          else 'body_core_attachment' if weak else 'body_scene_core')
    recover(rows,decisions,seeds,settings,scores,SimpleNamespace(allows=lambda *args:True))
    if weak_has_face:
        assert decisions['11']['person']=='B'
    else:
        assert decisions['11']['person']=='A'
        assert decisions['15']['person'] is None and decisions['15']['retry_owner']=='A'


def test_duplicate_pixels_in_another_scene_cannot_confirm_the_query():
    from stash_ai.recognition.rear_gallery import Gallery
    query = np.array([1.,0.])
    rows = [dict(_vector=query,id=str(i),quality=encoded({'quality':{'pixel_digest':str(i)}})) for i in range(2)]
    gallery = Gallery([('source owner','source scan',rows)])
    settings = dict(match_threshold=.85,ambiguity_margin=.1,attach_min_support=2)
    [duplicate] = gallery.rank(query,'other scan',settings,'0')
    assert duplicate[0]==-1 and duplicate[5]==1
    [independent] = gallery.rank(query,'other scan',settings,'new')
    assert independent[0]==1 and len(independent[4])==2


def test_greatest_admissible_view_keeps_every_rival():
    from stash_ai.recognition.rear_gallery import winner
    settings = dict(match_threshold=.85, cohesion_threshold=.8, ambiguity_margin=.05)
    banks = [(.889,.78,'A','first',(),.889), (.87,.87,'A','second',(),.87)]
    assert winner(banks,settings) == banks[1]
    assert winner(banks+[(.84,.70,'B','third',(),.84)],settings) is None


def test_tiled_rival_decisions_match_each_scalar_query():
    from io import BytesIO
    from types import SimpleNamespace
    from stash_ai.recognition.rear_recovery import _global_rivals
    rng = np.random.default_rng(45)
    matrix = rng.normal(size=(11,4096))
    queries = rng.normal(size=(5,4096))
    search = SimpleNamespace(index=BytesIO(matrix.astype('<f8').tobytes()), entries=[(i,[]) for i in range(11)],
                             identities=list(range(11)), block=3)
    batch = [(i,q,{i},float(matrix[5]@q)+1e-6) for i,q in enumerate(queries)]
    expected = {}
    for i,q,ignored,floor in batch:
        for owner,value in enumerate(np.concatenate([matrix[s:s+3] @ q for s in range(0,len(matrix),3)])):
            if owner not in ignored and value+1e-6 >= floor:
                expected[i] = owner
                break
    assert _global_rivals(search,batch)==expected


def test_two_different_face_roots_veto_independent_local_gallery():
    from types import SimpleNamespace
    from stash_ai.recognition.body_identity import defaults, Scores
    from stash_ai.recognition.body_perspectives import recover
    rows, decisions, seeds = [], {}, {'A':[], 'B':[]}
    for cue, owner, colour in [(0,'A',15),(1,None,15),(2,'A',15),(5,'A',180),(6,None,180),
                               (7,'A',180),(10,'A',60),(11,None,60),(12,'B',60)]:
        body = [.82,np.sqrt(1-.82**2)] if owner=='B' else [1.,0.]
        quality = dict(pixel_digest=str(cue),orientation='front' if owner else 'back',
                       frame_signature=(bytes([colour,15,15])*16).hex())
        row = dict(id=str(cue),scan='s',cue=cue,seconds=cue,quality=encoded({'quality':quality}),
                   geometry=encoded(dict(bounds=[0,0,80,100])),face_person=owner,_vector=np.array(body))
        rows.append(row)
        if owner:
            seeds[owner].append(row)
        decisions[row['id']] = dict(person=owner,status='accepted' if owner else 'withheld',
                                    reason='measured_face_association' if owner else 'body_anchor_quality')
    scores = Scores(SimpleNamespace(data={'sapiensid':{'execution':{
        'max_witness_comparisons':2000,'reference_bytes':2**20}}}))
    recover(rows,decisions,seeds,defaults(),scores,SimpleNamespace(allows=lambda *args:True))
    assert decisions['11']['person'] is None
    assert decisions['11']['track_blocker']['reason']=='different_face_root'


def test_two_separately_rooted_scenes_build_one_external_view(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'A'))
        selected = []
        front, back = [1.,0.], [.82,.5724]
        for scene_id in ('1','2','3'):
            source = scene_id != '3'
            item, ids = scene(db,config,scene_id,[front,back,front] if source else [back]*3)
            selected.append(item)
            for cue, oid in enumerate(ids):
                rooted = source and cue != 1
                q = dict(pixel_digest=oid,keypoints=10,sharpness=100,orientation='front' if rooted else 'back',
                         frame_signature=(bytes([15,15,15])*16).hex())
                db.execute('UPDATE body_observations SET quality=?,geometry=? WHERE id=?',
                           (encoded(dict(quality=q)),encoded(dict(bounds=[0,0,80,100])),oid))
                if rooted:
                    fid = 'face-'+oid
                    db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                               (fid,item['scan'],cue,float(cue),'','[]','[]',encoded(dict(pixels=100,sharpness=100,
                                pose_deviation=0.,visible_face_fraction=1.)),None,2,np.array(front,dtype='<f4').tobytes(),None))
                    db.execute('INSERT INTO members VALUES (?,?)',(fid,'A'))
                    db.execute('UPDATE body_observations SET face=? WHERE id=?',(fid,oid))
        db.commit()
        rebuild(state,selected,config,{s['scan']:[] for s in selected})
        rows = db.execute('SELECT person,proof FROM body_members WHERE observation IN (?,?,?)',ids).fetchall()
        assert all(person=='A' and len(json.loads(proof)['source_scans'])==2 for person,proof in rows)


def test_pending_scan_only_wakes_when_a_reachable_gallery_or_recorded_rival_changes(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.rear_gallery import Gallery
    from stash_ai.recognition.rear_pending import Pending
    config=configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        item,ids=scene(db,config)
        stamp=policy(config)
        for oid in ids:
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(oid,None,'auxiliary','withheld',stamp,'{}'))
        search=SimpleNamespace(bank={},mean_versions={})
        axis=np.zeros(4096);axis[0]=1.
        def gallery(far):
            def bank(owner,value):
                return owner,'source', [dict(id=owner,roots=['start','end'],quality='{}',_vector=value)]
            return Gallery([bank('near',axis),bank('far',far)])
        def run(far, selected=()):
            pending=Pending(db,gallery(far),search,config.data['sapiensid']['identities'])
            return [s for s,_ in pending.candidates('scope','ns',set(selected),stamp)]
        assert run(-axis)==[item['scan']]
        assert run(-axis)==[]
        assert run(np.roll(axis,1))==[]
        assert run(axis)==[item['scan']]
        assert run(axis, [item['scan']]) == []
        db.execute("UPDATE body_members SET proof=? WHERE observation=?", ('{"reason":"different_face_root"}', ids[0]))
        assert run(axis, [item['scan']]) == [item['scan']]


def test_cross_scene_gallery_uses_only_external_witnesses():
    from stash_ai.recognition.rear_gallery import Gallery
    vector = np.array([1.,0.])
    rows = [dict(_vector=vector,scan=scan,id=scan,quality='{}') for scan in 'ABC']
    gallery = Gallery([('owner','A',rows)])
    [winner] = gallery.rank(vector,'A',dict(match_threshold=.85,ambiguity_margin=.05,attach_min_support=2),None)
    assert winner[0]==1 and [r['scan'] for r in winner[4]]==['B','C']


def test_weak_face_of_same_owner_allows_route_to_next_strict_face():
    from types import SimpleNamespace
    from stash_ai.recognition.body_perspectives import recover
    from stash_ai.recognition.body_identity import defaults
    rows, decisions = [], {}
    for cue, owner in enumerate(['A',None,'A','A']):
        row = dict(id=str(cue),cue=cue,seconds=float(cue),scan='s',face_person=owner,
                   geometry=encoded(dict(bounds=[0,0,80,100])),_vector=np.array([1.,0.]),
                   quality=encoded(dict(quality=dict(orientation='front' if owner else 'back',
                                                      frame_signature='11'*48,pixel_digest=str(cue)))))
        rows.append(row)
        decisions[row['id']] = dict(person=owner,status='accepted' if owner else 'withheld',
                                    reason='measured_face_association' if owner else 'body_anchor_quality')
    recover(rows,decisions,{'A':[rows[0],rows[3]]},defaults(),SimpleNamespace(face_similarity=np.dot),
            SimpleNamespace(allows=lambda *args:True))
    assert decisions['1']['rooted_track']['roots']==['0','3']
    assert decisions['2']['reason']=='measured_face_association'


def test_rival_bounds_exclude_blocks_without_changing_exact_results():
    from contextlib import ExitStack
    from io import BytesIO
    from types import SimpleNamespace
    from stash_ai.recognition.rear_recovery import _global_rivals
    matrix = np.zeros((12,4096))
    matrix[:9,0],matrix[9:,1] = 1.,1.
    batch = [('q',matrix[-1],set(),.99)]
    with ExitStack() as stack:
        search = SimpleNamespace(entries=[(i,[]) for i in range(12)],identities=list(range(12)),
                                 block=3,index=BytesIO(matrix.astype('<f8').tobytes()),stats={})
        exact = _global_rivals(search,batch)
        search.stack = stack
        assert _global_rivals(search,batch)==exact=={'q':9}
        assert search.stats['rear_rival_blocks_skipped']==3


def test_incremental_rear_does_not_rewrite_identical_claims(tmp_path):
    from stash_ai.recognition.rear_snapshot import frozen,apply
    from stash_ai.recognition.body_identity import policy
    cfg = configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        item,ids=scene(db,cfg)
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'A'))
        original=dict(person=None,status='withheld',role='auxiliary',reason='body_anchor_quality')
        proof=dict(person='A',status='accepted',role='auxiliary',reason='external_rear_view',previous=original)
        db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(ids[0],'A','auxiliary','accepted',policy(cfg),encoded(proof)))
        with frozen(db,'scope','ns') as active:
            assert active=={item['scan']:item['scene']}
            assert db.execute('SELECT person FROM body_members').fetchone()[0] is None
            changes=db.total_changes
            assert not apply(db,item['scan'],{ids[0]:proof}) and db.total_changes==changes
            assert apply(db,item['scan'],{})
        assert db.execute('SELECT person,proof FROM body_members').fetchone()[0] is None

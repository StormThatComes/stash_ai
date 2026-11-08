from stash_ai.recognition.crossing_tracks import crossings


def test_margin_depends_on_differing_edges_and_one_hidden_person_can_return():
    for count,hidden in ((3,None),(5,None),(6,2)):
        cues = {cue:[dict(id=f'{cue}-{owner}',seconds=cue) for owner in 'ab'] for cue in range(count)}
        if hidden is not None:
            cues[hidden] = [dict(id='shared',seconds=hidden)]
        roots = {row['id']:row['id'][-1] for cue in (0,count-1) for row in cues[cue]}
        decisions = {row['id']:{} for rows in cues.values() for row in rows}
        def transition(a,b):
            return .99 if a['id'][-1]==b['id'][-1] else .80
        claims = crossings(cues,roots,decisions,transition,.1,occluded={hidden})
        expected = {row['id']:row['id'][-1] for cue,rows in cues.items() if 0<cue<count-1 and cue!=hidden for row in rows}
        assert {key:value[0] for key,value in claims.items()}==expected
        if hidden is not None:
            assert decisions['shared']['person'] is None
            assert decisions['shared']['reason']=='unresolved_partner'


def test_confirmed_core_retry_preserves_face_contradiction(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.body_recovery import recover
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan,ids = scene(db,cfg,vectors=[[1.,0.]]*4)
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('A','scope','ns',1,None,'A'))
        for oid in ids[:-1]:
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(oid,'A','anchor','accepted',
                       policy(cfg),encoded(dict(reason='body_scene_core'))))
        proof = dict(reason='different_face_root',track_blocker=dict(reason='different_face_root'),
                     retry_when='confirmed_core_changes')
        db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(ids[-1],None,'auxiliary','withheld',policy(cfg),encoded(proof)))
        recover(state,[scan],cfg)
        assert tuple(db.execute('SELECT person,status FROM body_members WHERE observation=?',(ids[-1],)).fetchone())==(None,'withheld')


def test_subthreshold_partner_still_competes_with_whole_path():
    cues = {i:[dict(id=f'{i}{owner}',seconds=i) for owner in 'ab'] for i in range(3)}
    roots = {row['id']:row['id'][-1] for i in (0,2) for row in cues[i]}
    decisions = {row['id']:{} for pair in cues.values() for row in pair}
    def raw(a,b):
        return .801 if a['id'][-1]==b['id'][-1] else .799
    assert not crossings(cues,roots,decisions,lambda a,b:raw(a,b) if raw(a,b)>=.8 else None,.1,raw=raw)


def test_partial_face_pair_recovers_after_two_hidden_cues():
    cues = {i:[dict(id=f'{i}{owner}',seconds=i) for owner in ('x' if i in (1,2) else 'ab')] for i in range(5)}
    roots = {'0a':'A','4a':'A','4b':'B'}
    decisions = {row['id']:{} for pair in cues.values() for row in pair}
    result = crossings(cues,roots,decisions,lambda a,b: .99 if a['id'][-1]==b['id'][-1] else .68,
                       .1,occluded={1,2})
    assert {k:v[0] for k,v in result.items()}=={'3a':'A','3b':'B'}
    assert all(decisions[key]['reason']=='unresolved_partner' for key in ('1x','2x'))


def test_rooted_retry_keeps_identical_competing_core(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene
    from stash_ai.recognition.database import FaceDatabase,encoded
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.body_recovery import recover
    cfg = configuration(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        scan,ids = scene(db,cfg,vectors=[[1.,0.]]*7)
        for owner in 'AB':
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',1,None,owner))
        for oid,owner in zip(ids,'AAABBB'):
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(oid,owner,'anchor','accepted',policy(cfg),
                       encoded(dict(reason='body_scene_core'))))
        proof = dict(reason='body_anchor_quality',retry_when='confirmed_core_changes',open_face_root=dict(owner='A'))
        db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',(ids[-1],None,'auxiliary','withheld',policy(cfg),encoded(proof)))
        recover(state,[scan],cfg)
        assert db.execute('SELECT person FROM body_members WHERE observation=?',(ids[-1],)).fetchone()[0] is None


def test_complementary_endpoints_challenge_the_unconstrained_ending():
    cues = {i:[dict(id=p+str(i),seconds=i) for p in 'AB'] for i in range(3)}
    def run(roots,reverse=False):
        data = {i:rows[::-1] if reverse else rows for i,rows in cues.items()}
        decisions = {r['id']:{} for pair in data.values() for r in pair}
        return {k:v[0] for k,v in crossings(data,roots,decisions,
                lambda a,b:.95 if a['id'][0]==b['id'][0] else .2,.1).items()}
    assert run({'A0':'A','B2':'B'})==run({'A0':'A','B2':'B'},True)=={'A1':'A','B1':'B'}
    assert not run({'A0':'A','A2':'B'})


def test_pan_requires_one_unique_textured_offset():
    import numpy as np
    from stash_ai.recognition.perspective_shots import camera_shift,change
    left = np.random.default_rng(42).integers(0,256,(4,4,3)).astype(float)
    right = np.roll(left,1,axis=1)
    assert camera_shift(left,right)==(1,0)
    assert change(left.reshape(-1,3),right.reshape(-1,3),True)=='continuous'
    assert camera_shift(left*0,right*0) is None

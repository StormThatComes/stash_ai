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

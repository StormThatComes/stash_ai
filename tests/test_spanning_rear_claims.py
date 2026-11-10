from types import SimpleNamespace

from stash_ai.recognition.body_identity import policy
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rear_snapshot import frozen
from stash_ai.recognition.spanning_rear import SpanningClaims
from tests.test_recognition_body_global import configured, scene


def test_whole_donor_and_revoke_after_one_lost_witness(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        for owner in ('original','destination'):
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(owner,'scope','ns',0,None,owner))
        scans = {}
        for number in ('1','2'):
            item,ids = scene(db,config,number)
            scans[item['scan']] = ids
            for oid in ids:
                db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
                           (oid,'original','anchor','accepted',policy(config),'{}'))
        search = SimpleNamespace(bank={'original':{'cues':[(scan,0) for scan in scans]}},rear_revision='v')
        def verify(missing=None):
            with frozen(db,'scope','ns'):
                queue = SpanningClaims(db)
                queue.nominated.add('original')
                def revisit(scan):
                    for cue,oid in enumerate(scans[scan]):
                        if oid!=missing:
                            previous = dict(person='original',role='anchor',status='accepted')
                            queue.stage(scan,cue,oid,'original','destination',dict(person='destination',role='auxiliary',
                                        status='accepted',reason='external_rear_view',previous=previous),set())
                queue.verify(search,revisit,'scope','ns',config)
            return {r[0] for r in db.execute('SELECT person FROM body_members')}
        assert verify()=={'destination'}
        assert verify(next(iter(scans.values()))[0])=={'original'}


def test_proof_index_matches_full_scan_for_current_and_wrapped_proofs(tmp_path):
    import json
    from stash_ai.recognition.rear_gallery import proof_key
    cfg = configured(tmp_path)
    with FaceDatabase(cfg.database) as state:
        db = state.connection
        _,ids = scene(db,cfg,'1')
        for oid,proof in zip(ids,[{'rooted_track':{'perspective':'back'}},
                                  {'previous':{'rooted_view':{'perspective':'side'}}},{}]):
            db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
                       (oid,None,'auxiliary','withheld','p',json.dumps(proof)))
        where = " WHERE policy='p' AND "+proof_key('perspective')+" IN ('back','side')"
        query = "SELECT observation FROM body_members"
        assert set(db.execute(query+where))==set(db.execute(query+' NOT INDEXED'+where))
        assert any('body_rooted_proofs' in r[-1] for r in db.execute('EXPLAIN QUERY PLAN '+query+where))

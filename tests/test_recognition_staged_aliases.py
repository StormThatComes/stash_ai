import json

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.body_global import reconcile
from tests.test_recognition_body_integration import seeded


def test_retained_component_larger_than_batch_publishes_only_after_every_current_scene(tmp_path):
    cfg,scans,_,_ = seeded(tmp_path)
    cfg.data['sapiensid']['global_matching']['match_threshold']=.87
    with FaceDatabase(cfg.database) as state:
        db=state.connection
        from stash_ai.recognition.body_identity import rebuild
        for item in scans:
            rebuild(state,[item],cfg,{item['scan']:[]})
        cfg.data['sapiensid']['consolidation'].update(scope='retained',max_scenes=1)
        def replay():
            with db:
                db.execute('BEGIN')
                reconcile(state,scans[:1],cfg)
        def aliases():
            return list(db.execute('SELECT evidence FROM identity_merges'))
        replay()
        assert not aliases()
        assert db.execute("SELECT count(*) FROM evidence_fingerprints WHERE kind='retained-alias-stage'").fetchone()[0]==1
        db.execute("UPDATE scans SET status='partial' WHERE id=?",(scans[1]['scan'],))
        db.commit()
        replay()
        assert not aliases()
        db.execute("UPDATE scans SET status='complete' WHERE id=?",(scans[1]['scan'],))
        db.commit()
        replay()
        assert not aliases()  # Changing the head invalidates the earlier component certificate.
        for _ in range(3):
            if aliases():
                break
            rebuild(state,scans[:1],cfg,{scans[0]['scan']:[]})
        assert len(aliases())==1 and json.loads(aliases()[0][0])['staged_retained_validation']
        assert db.execute('SELECT count(*) FROM pending_body_merges').fetchone()[0]==0

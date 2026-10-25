import json
from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.hybrid_merge import apply
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.consolidation_queue import expand
from tests.test_recognition_hybrid_merge import prepared


def test_retained_queue_consumer_revalidates_and_drains_without_publishing(tmp_path):
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        apply(state,selected[:1],config)
        assert expand(db,selected[:1],config)==selected[:1]
        config.data['sapiensid']['consolidation']['scope']='retained'
        assert len(expand(db,selected[:1],config))==2
        result=rebuild_selected(state,selected[:1],config)
        assert set(result)=={s['scan'] for s in selected}
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==1
        assert not db.execute("SELECT 1 FROM metadata WHERE key LIKE 'hybrid-pending:%'").fetchone()
        assert db.execute('SELECT COUNT(*) FROM publications').fetchone()[0]==0
        assert db.execute("SELECT COUNT(*) FROM metadata WHERE key LIKE 'identity-publication:%'").fetchone()[0]==2


def test_queue_respects_scene_allowlist_limit_and_partial_scan(tmp_path):
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        apply(state,selected[:1],config)
        policy=config.data['sapiensid']['consolidation'];policy.update(scope='retained',max_scenes=1)
        assert expand(db,selected[:1],config)==selected[:1]
        policy.update(max_scenes=10,scene_ids=[selected[0]['scene']])
        assert expand(db,selected[:1],config)==selected[:1]
        policy['scene_ids']=[]
        db.execute("UPDATE scans SET status='partial' WHERE id=?",(selected[1]['scan'],))
        assert expand(db,selected[:1],config)==selected[:1]


def test_stale_queue_never_executes_the_stored_merge(tmp_path):
    config,people=prepared(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        apply(state,selected[:1],config)
        config.data['sapiensid']['consolidation']['scope']='retained'
        # Current body evidence contradicts the old pending nomination.
        db.execute('UPDATE person_evidence SET status=\'unavailable\',embedding=NULL WHERE observation IN '
            '(SELECT observation FROM members WHERE person=?)',(people[1],));db.commit()
        rebuild_selected(state,selected[:1],config)
        assert db.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0]==0

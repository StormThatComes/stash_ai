import json

from stash_ai.recognition.assignment_explanation import body
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_report import export
from stash_ai.recognition.models import namespace
from tests.test_recognition_identity_report import seed


def test_export_explains_body_merge_and_only_follows_member_chain(tmp_path):
    config, person, _ = seed(tmp_path, 1)
    with FaceDatabase(config.database) as state:
        db = state.connection
        for source in ('original', 'unrelated'):
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       (source, 'scope', namespace(config), 0, None, source))
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                       (source, person, 1, '{"route":"body_global"}', 'local'))
        db.execute("UPDATE candidates SET payload=json_set(payload, '$.reason','body_global_identity',"
                   "'$.body_merge_origin',json(?))",
                   (json.dumps(dict(person='original', reason='candidate_group')),))
        db.commit()
    sample = export(config, tmp_path/'audit', identity_id=person)['samples'][0]
    assert sample['assignment']['role'] == 'attachment'
    assert not sample['current_anchor_eligible']
    step, = sample['explanation']['decisions']
    assert step['reason'] == 'candidate_group'
    assert step['merges'] == [dict(source='original', target=person, route='body_global',
                                   proof_location='identity_merges')]


def test_body_track_references_keep_source_order_and_kind():
    proof = dict(reason='face_rooted_body_view', source_tracks=[['a','c','b'], ['d','f','e']],
                 score=.91, runner=-1)
    explanation = body(proof, linked_face=None, role='auxiliary', status='accepted',
                       policy='old', current=False, merges={}, target='A')
    step, = explanation['decisions']
    assert explanation['evidence_status'] == 'recorded_other_policy'
    assert step['metrics'] == {'score': .91}
    assert [(ref['observation'], ref['kind']) for ref in step['references']] == [
        (c, 'body') for c in 'abcdef']


def test_strict_matches_record_the_pre_decision_reference_set(tmp_path):
    from stash_ai.recognition.clustering import cluster_scene
    from stash_ai.recognition.decision_contexts import unpack
    from tests.test_recognition_clustering import setup, scan
    config = setup(tmp_path, {'max_samples_per_person': 2, 'strict_appearance_matching': False})
    with FaceDatabase(config.database) as state:
        first, _ = scan(state, config, '1')
        cluster_scene(state, first, config)
        rows = list(state.connection.execute("SELECT o.id,c.payload FROM observations o JOIN candidates c "
            "ON o.id=json_extract(c.payload,'$.observation') ORDER BY o.cue"))
        ids = [r[0] for r in rows]
        proofs = [unpack(state.connection,r[1]) for r in rows]
        assert proofs[0]['face_comparison']['stage'] == 'seed'
        for i in (1,2):
            assert proofs[i]['compared_references'] == ids[:i]
            assert proofs[i]['face_comparison']['minimum'] > .99
            assert 'runner' not in proofs[i]['face_comparison']


def test_shared_merge_proofs_survive_prune_until_last_reference(tmp_path):
    from stash_ai.recognition.decision_contexts import Packer, unpack, collect
    config, _, _ = seed(tmp_path, 2)
    with FaceDatabase(config.database) as state:
        db = state.connection
        pack = Packer(db)
        ids, links = [], []
        for row in db.execute('SELECT id,payload FROM candidates').fetchall():
            payload = json.loads(row['payload'])
            payload.update(compared_references=[row['id']], same_scene_merge={'support_cues':[1,2,3]})
            packed = pack(json.dumps(payload))
            assert unpack(db, packed) == payload
            ids.append(row['id'])
            links.append(json.loads(packed)['merge_context'])
            db.execute('UPDATE candidates SET payload=? WHERE id=?',(packed,row['id']))
        assert links[0] == links[1]
        for i, identifier in enumerate(ids):
            db.execute('DELETE FROM candidates WHERE id=?',(identifier,))
            collect(db)
            assert bool(db.execute('SELECT 1 FROM decision_contexts WHERE id=?',(links[0],)).fetchone()) == (i==0)


def test_replay_alias_chain_survives_label_binding(tmp_path):
    from stash_ai.recognition.rebuild import _mapped_candidates
    config, person, _ = seed(tmp_path, 1)
    with FaceDatabase(config.database) as state:
        db = state.connection
        for source, target in [('middle',person), ('original','middle')]:
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', (source,'scope',namespace(config),0,None,source))
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                       (source,target,1,'{"support_cues":[]}', 'local'))
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.first_merge_source','original',"
                   "'$.merged_from','middle')")
        policy = db.execute('SELECT policy FROM candidates').fetchone()[0]
        record, = _mapped_candidates(db, {person:'published'}, policy, '')
        assert record[3]=='published'
        chain = json.loads(record[5])['replay_merge_chain']
        assert [(edge['source'],edge['target']) for edge in chain] == [('original','middle'),('middle',person)]
        assert all(edge['route']=='strict_merge' for edge in chain)

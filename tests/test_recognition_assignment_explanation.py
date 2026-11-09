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

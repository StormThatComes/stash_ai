from stash_ai.recognition.solver_validation import _metrics
from stash_ai.recognition.crowd_labels import detection_counts,validate


def test_joint_id_cannot_recover_turn_and_duplicate_projection_does_not_add_a_person():
    labels,locations,samples,results = {},{},{},{}
    for cue in range(3):
        for truth in 'AB':
            for projection in (['face','body'] if cue!=1 else ['body']):
                oid = f'{truth}-{cue}-{projection}'
                labels[oid] = truth
                locations[oid] = '1',cue
                samples[oid] = dict(slices=['back','entangled'] if cue==1 else [])
                results[oid] = dict(accepted_identity='X',evidence_kind='person' if projection=='body' else 'face',
                                    evidence=dict(reason='known_identity'))
    values = _metrics(labels,samples,locations,results)
    assert values['physical']['observations']==6
    assert values['sequences']['one_id_turns']==2
    assert values['sequences']['own_id_turns']==0 and values['sequences']['anchored_turns']==2
    results['A-0-body']['accepted_identity']='foreign'
    assert _metrics(labels,samples,locations,results)['physical']['projection_conflicts']==1


def test_visible_partner_can_be_missing_without_an_observation_id(tmp_path):
    from tests.test_recognition_body_identity import configuration,scene
    from tests.test_recognition_crowded_tracks import pose
    from stash_ai.recognition.database import FaceDatabase,encoded
    cfg = configuration(tmp_path)
    visible = {'1':{'0':[dict(truth='A',point=[40,45]),dict(truth='B',point=[110,45])]}}
    validate(visible,['1'])
    with FaceDatabase(cfg.database) as state:
        item,ids = scene(state.connection,cfg)
        state.connection.execute('UPDATE body_observations SET geometry=? WHERE id=?',(encoded(pose(40)),ids[0]))
        values = detection_counts(state.connection,[dict(scene='1',id=item['scan'],total=3)],visible)
        assert (values['expected_people'],values['missing_poses'],values['duplicate_poses'])==(2,1,0)

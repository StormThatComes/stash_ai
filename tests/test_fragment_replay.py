from stash_ai.recognition.merge_solver_tighten import TRACK, route_parameters
from stash_ai.recognition.solver_validation import holdout_verdict
from stash_ai.recognition.sequence_validation import sequence_metrics


def test_local_track_is_credited_without_a_scene_bonus():
    counts = route_parameters([dict(reason='local_body_track',status='accepted',evidence_kind='person')])
    assert counts[TRACK] == 1
    assert 'sapiensid.identities.scene_match_bonus' not in counts


def test_contact_exit_loss_is_a_regression_even_with_correct_interior():
    labels, positions, samples = {}, {}, {}
    for p in 'AB':
        for i in range(3):
            oid = p+str(i)
            labels[oid], positions[oid] = p, ('1',i)
            samples[oid] = dict(slices=['entangled'] if i==1 else [])
    def measures(lost):
        decisions = {o:dict(accepted_identity=None if lost and o=='A2' else p,
                            evidence_kind='person') for o,p in labels.items()}
        return dict(sequences=sequence_metrics(labels,samples,positions,decisions),
                    partners=dict(false_joins=0,partner_pairs=2),
                    visible=dict(split_identities=0,reference_identities=2))
    verdict = holdout_verdict({'baseline/joint':measures(False),'winner/joint':measures(True)})
    assert verdict['status']=='regressed'
    assert verdict['measures']['lost_person_endpoints']['winner']==1


def test_rear_scene_with_tracks_keeps_core_timestamps(tmp_path):
    from types import SimpleNamespace
    from stash_ai.recognition import rear_recovery
    from stash_ai.recognition.body_identity import rebuild, policy
    from stash_ai.recognition.database import FaceDatabase
    from tests.test_recognition_body_identity import configuration, scene
    config = configuration(tmp_path)
    config.data['sapiensid']['identities']['tracklet_matching'] = True
    with FaceDatabase(config.database) as state:
        item, _ = scene(state.connection,config)
        state.connection.commit()
        rebuild(state,[item],config,{item['scan']:[]})
        decisions = rear_recovery._scene(state.connection,item['scan'],policy(config),
                                         SimpleNamespace(rank=lambda *args:[]),SimpleNamespace(block=8),config)
        assert decisions == {}


def test_export_mixed_holdout_counts(tmp_path):
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.merge_goal_output import export
    from stash_ai.recognition.merge_report_policy import proposal
    config, results = fixture(tmp_path,'query')
    results.update(best_result=proposal(config,{}),simulation={},
                   holdout_validation=dict(verdict=dict(status='regressed',measures={
                       'lost': dict(baseline=0,winner=1,status='regressed'),
                       'returns': dict(baseline=1,winner=0,baseline_total=1,winner_total=1,status='regressed')})))
    dest = tmp_path/'proposal.yml'
    export(config,results,tmp_path/'result.json',dest,yaml_mode='diff')
    assert '0 -> 1 (regressed)' in dest.read_text() and '1/1 -> 0/1' in dest.read_text()

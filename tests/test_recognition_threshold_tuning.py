import json

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.merge_report import build
from stash_ai.recognition.replay_simulation import impact, replay
from stash_ai.recognition.threshold_tuning import configurations, execute, truth_split, tune
from tests.test_recognition_offline_evaluation import fixture
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_rebuild import changed


def test_partition_delta_counts_pairs_without_uuid_or_withheld_merges():
    before = dict(a='x', b='x', c='y', d=None, e='z')
    after = dict(a='p', b='q', c='p', d='q', e=None)
    delta = impact(before, after)
    assert delta['newly_joined_pairs'] == delta['split_pairs'] == 1
    assert delta['coverage_gained'] == delta['coverage_lost'] == 1
    assert impact(before, dict(a='1', b='1', c='2', d=None, e='3'))['split_pairs'] == 0


def test_unlabelled_search_is_read_only_and_does_not_invent_truth(tmp_path, monkeypatch):
    config, labels = fixture(tmp_path)
    before = config.database.read_bytes()
    monkeypatch.setattr(FaceDatabase, '__init__', lambda *a, **k: pytest.fail('writer opened'))
    from stash_ai.recognition import models
    monkeypatch.setattr(models, 'load_backend', lambda *a, **k: pytest.fail('model loaded'))
    result = tune(config, {'recognition.match_threshold': [.999]})
    assert result['status'] == 'unlabelled_comparison' and result['recommendation'] is None
    assert result['trials']['trial-1']['versus_baseline']['split_pairs'] > 0
    assert config.database.read_bytes() == before
    assert not list(tmp_path.glob('*.rebuild*'))
    assert tune(config, {'recognition.match_threshold': [.999]}, labels=labels)['status'] == 'insufficient_independent_truth'


def test_grid_budget_invalid_combinations_and_contracts(tmp_path):
    config, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='max_trials'):
        configurations(config, {'recognition.match_threshold': [.8, .9]}, 2)
    with pytest.raises(ValueError, match='supported'):
        configurations(config, {'models.device': ['cpu']}, 2)
    variants, _, skipped = configurations(config, {'recognition.match_threshold': [.1, .999]}, 3)
    assert len(variants) == len(skipped) == 1
    with pytest.raises(ValueError, match='contracts'):
        replay(config, {'bad': changed(config, detection={'min_face_pixels': 80})})
    with pytest.raises(ValueError, match='selected latest'):
        tune(config, {'recognition.match_threshold': [.999]}, labels={'missing': 'one'})


def test_truth_split_keeps_whole_people_together():
    labels = {str(i): str(i//3) for i in range(18)}
    train, holdout = truth_split(labels)
    assert set(train.values()).isdisjoint(holdout.values())
    assert train.keys() | holdout.keys() == labels.keys()
    assert truth_split(dict(reversed(list(labels.items())))) == (train, holdout)


def test_real_replays_can_validate_a_threshold_proposal(tmp_path):
    import numpy as np
    vectors = []
    for index in range(4):
        vector = np.zeros(8)
        vector[index] = 1.
        other = vector * .99
        other[index+4] = (1-.99**2)**.5
        vectors.extend([vector.tolist(), other.tolist()])
    config, _ = seeded(tmp_path, vectors=vectors, recognition={'match_threshold': .999})
    with FaceDatabase(config.database) as db:
        labels = {r[0]: str((int(r[1])-1)//2) for r in db.connection.execute(
            'SELECT o.id,s.scene FROM observations o JOIN scans s ON s.id=o.scan')}
    before = config.database.read_bytes()
    result = tune(config, {'recognition.match_threshold': [.98]}, labels=labels)
    assert result['status'] == 'validated_proposal'
    assert result['recommendation']['yaml_patch'] == {'recognition': {'match_threshold': .98}}
    assert result['selected_labelled_metrics']['false_merge_pairs'] == 0
    assert result['holdout_metrics']['pair_recall'] > result['baseline_holdout_metrics']['pair_recall']
    assert config.database.read_bytes() == before


def test_full_report_simulation_checks_pair_and_additional_identity(tmp_path):
    config, people = seeded(tmp_path, vectors=[[1.,0,0,0], [.7,.71414284285,0,0], [0,0,1.,0]])
    before = config.database.read_bytes()
    report = build(config, *people[:2], simulate=True)
    trials = report['simulation']['trials']
    assert report['simulation']['selection'] == ['1', '2', '3']
    assert not trials['baseline']['requested_pair']['fully_merged']
    assert trials['baseline']['requested_pair']['observation_scope'] == 'selected_latest_scans'
    assert all(m['outside_latest_selection'] == 0 for m in trials['baseline']['requested_pair']['members'])
    assert any(trials[p['simulation_trial']]['requested_pair']['fully_merged'] for p in report['recommendations'])
    assert all(trials[p['simulation_trial']]['versus_baseline']['identities'] == 2 for p in report['recommendations'])
    assert config.database.read_bytes() == before
    text = json.dumps(report)
    assert 'private-crop' not in text and 'user:secret' not in text


def test_cli_exclusive_outputs_and_no_unvalidated_yaml(tmp_path):
    config, _ = fixture(tmp_path)
    grid, output, yaml = tmp_path/'grid.json', tmp_path/'result.json', tmp_path/'patch.yml'
    grid.write_text(json.dumps({'recognition.match_threshold': [.999]}))
    args = parser().parse_args(['tune', '--all-scenes', '--grid', str(grid), '--output', str(output), '--yaml-output', str(yaml)])
    assert execute(args, config) == 0
    assert not yaml.exists() and not json.loads(output.read_text())['yaml_exported']
    with pytest.raises(FileExistsError):
        execute(args, config)
    assert parser().parse_args(['report-merge', '--identity-id', 'a', '--identity-id', 'b', '--simulate']).simulate


@pytest.mark.parametrize('reject_holdout', [False, True])
def test_holdout_rejection_never_reranks_candidates(tmp_path, monkeypatch, reject_holdout):
    config, _ = fixture(tmp_path)
    labels = {str(i): str(i//2) for i in range(8)}
    train, holdout = truth_split(labels)
    from stash_ai.recognition import threshold_tuning as module
    def simulate(config, variants, *, consume, **kwargs):
        trials = {}
        for index, name in enumerate(['baseline', *variants]):
            predictions = {k: (k if k in train and index == 0 else v) for k, v in labels.items()}
            if reject_holdout and index:
                predictions.update({k: 'merged' for k in holdout})
            trials[name] = consume(name, dict.fromkeys(labels, '1'), {}, predictions)
        return dict(trials=trials, limitations=[])
    monkeypatch.setattr(module, 'replay', simulate)
    result = tune(config, {'recognition.match_threshold': [.98, .99]}, labels=labels)
    assert result['selected_on_training'] == 'trial-1'
    assert result['status'] == ('holdout_rejected' if reject_holdout else 'validated_proposal')
    assert bool(result['recommendation']) is not reject_holdout

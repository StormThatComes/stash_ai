"""The default constrained search and the explicit legacy solver share replay contracts."""
from contextlib import closing
from pathlib import Path

import pytest
import yaml

from stash_ai.recognition import cli
from stash_ai.recognition.config import load
from stash_ai.recognition.config_check import report
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_goals import resolve
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.merge_solver import solve
from stash_ai.recognition.solver_config_policy import StrictPolicy
from tests.test_recognition_config_check import profile
from tests.test_recognition_merge_report import seeded


def clean(config):
    return not any(r['solver_constraint'] and r['status'] != 'ok' for r in report(config)['checks'])


def test_both_shipped_profiles_start_without_repair():
    root = Path(__file__).resolve().parents[1]
    for name in ('facial_recognition.yml', 'facial_recognition_solver.yml'):
        config = load(root / name)
        assert config.data['execution']['solver_strict'] is True
        policy = StrictPolicy(config)
        assert policy.input_issues == []
        assert policy.repairs(config, {}, initial=True) == [{}]


@pytest.mark.parametrize('track', [.845, None])
def test_only_adjacent_cues_require_a_score_gap(tmp_path, track):
    c = load(profile(tmp_path, {'matching.body.identities.track_similarity_threshold': track,
                                'matching.body.identities.cohesion_threshold': .845}))
    policy = StrictPolicy(c)
    fixes = policy.repairs(c, {}, initial=True)
    assert len(fixes) == 2
    assert all(clean(patched(c, patch)) for patch in fixes)
    if track is None:
        assert all('sapiensid.identities.track_similarity_threshold' not in patch for patch in fixes)
    valid = patched(c, fixes[0])
    edits = {'recognition.ambiguity_margin': 0.,
             'clustering.view_match_threshold': valid.data['recognition']['match_threshold']}
    assert policy.repairs(valid, edits) == [edits]
    disabled = patched(valid, {'sapiensid.identities.endpoint_matching': False})
    assert 'sapiensid.identities.endpoint_matching' in policy.violations(disabled)
    assert policy.violations(disabled, diagnostic=True) == []


def test_joint_face_proposals_and_body_normalization_allow_independent_cutoffs():
    from stash_ai.recognition.merge_solver import _face_neighbours
    from stash_ai.recognition.merge_solver_domains import normalize
    config = load(Path(__file__).resolve().parents[1] / 'facial_recognition_solver.yml')
    options = {'recognition.match_threshold': [.71]}
    joint = next(patch for patch in _face_neighbours(config, {}, options)
                 if 'clustering.view_match_threshold' in patch)
    assert joint['clustering.view_match_threshold'] == joint['recognition.match_threshold']
    body = {'sapiensid.identities.cohesion_threshold': .90}
    assert normalize(config, body) == body


@pytest.mark.parametrize('mode', ['full', 'diff'])
def test_default_replays_from_equal_floors_and_preserves_portraits_in_exports(tmp_path, monkeypatch, mode):
    import json
    from stash_ai.recognition import replay_simulation
    config, _ = seeded(tmp_path, vectors=[[1., 0.], [.83, (1-.83**2)**.5]])
    # Exercise a real pre-existing v1 YAML with no mode key.
    document = yaml.safe_load(config.source)
    document.pop('execution')
    document['detection'] = dict(min_face_pixels=config.data['detection']['min_face_pixels'])
    document['clustering']['min_auxiliary_face_pixels'] = document['detection']['min_face_pixels']
    document['clustering']['min_anchor_face_pixels'] = document['detection']['min_face_pixels']
    document['portraits'] = dict(anchor_only=True, min_face_pixels=document['detection']['min_face_pixels'],
                                 weights=dict(resolution=0.))
    # Guarantee that the requested export includes a genuine, independently required support-cap repair.
    document['recognition']['min_merge_support'] = 9
    config.source_path.write_text(yaml.safe_dump(document))
    config = load(config.source_path, database=config.database)
    original = config.source_path.read_bytes()
    seen = []
    rebuild = replay_simulation.rebuild_selected
    def traced(state, scans, variant):
        seen.append(clean(variant))
        return rebuild(state, scans, variant)
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', traced)
    output = tmp_path / 'winner.yml'
    result_path = tmp_path / 'report.json'
    args = ['report-merge', '--face-config', str(config.source_path), '--faces-db', str(config.database),
            '--performer-set', '900,901', '--solve', '--max-trials', '6', '--progress-style', 'off',
            '--solver-report', 'full', '--yaml-mode', mode, '--yaml-output', str(output), '--output', str(result_path)]
    assert cli.main(args) == 0
    result = json.loads(result_path.read_text())
    assert all(seen) and 2 <= len(seen) <= 6
    assert result['config_policy']['mode'] == 'strict'
    assert result['config_policy']['input_control_replays'] == 1
    assert result['search']['attempted'] == len(seen)
    assert {r['parameter'] for r in result['config_policy']['baseline_repair']['changes']} == {'recognition.min_merge_support'}
    for trial in result['simulation']['trials'].values():
        config_patch = {v['parameter']: v['proposed'] for v in trial['changes']}
        assert clean(patched(config, config_patch))
        assert all(not key.startswith('portraits.') for key in config_patch)
    if mode == 'diff':
        assert 'portraits' not in yaml.safe_load(output.read_text())
        for section, edits in yaml.safe_load(output.read_text()).items():
            document.setdefault(section, {}).update(edits)
        output.write_text(yaml.safe_dump(document))
    winner = load(output)
    assert winner.data['portraits'] == config.data['portraits']
    assert clean(winner) and config.source_path.read_bytes() == original


@pytest.mark.parametrize('path', ['pose.refinement.min_presence', 'pose.point_quality.min_presence'])
def test_scan_only_warning_is_reported_and_does_not_require_repair(tmp_path, path):
    changes = {path: 1.}
    if 'point_quality' in path:
        changes['pose.refinement.min_presence'] = None
    config = load(profile(tmp_path, changes))
    row = next(r for r in report(config)['checks'] if r['rule'] == 'pose.refinement.min_presence-ceiling')
    assert row['status'] == 'warning' and path in row['settings']
    assert not row['solver_constraint']
    assert StrictPolicy(config).repairs(config, {}, initial=True) == [{}]


def test_strict_budget_checkpoint_and_escape_hatch(tmp_path):
    c, _ = seeded(tmp_path)
    c = patched(c, {'execution.solver_strict': True, 'recognition.min_merge_support': 9})
    path = tmp_path / 'strict.json'
    with closing(_open_readonly(c.database)) as db:
        goals = resolve(db, c, [['900','901']])
        limited = solve(c, goals, db, maximum=1)
        assert limited['status'] == 'strict_config_budget' and limited['recommendation'] is None
        result = solve(c, goals, db, maximum=5, checkpoint=path)
        resumed = solve(c, goals, db, maximum=5, checkpoint=path, resume=True)
        assert resumed['best_result'] == result['best_result']
        assert resumed['simulation']['input_control'] == result['simulation']['input_control']
        with pytest.raises(ValueError, match='[Cc]heckpoint'):
            solve(c, goals, db, maximum=5, checkpoint=path, resume=True, strict=False)
        assert solve(c, goals, db, maximum=1, strict=False)['selected_trial'] == 'baseline'
    assert cli.parser().parse_args(['report-merge', '--solve', '--no-strict', '--performer-set', '900,901']).strict is False


def test_portrait_and_sample_preferences_are_valid(tmp_path):
    config = load(profile(tmp_path, {'portraits.weights.resolution': 0., 'portraits.selection.anchor_only': True,
                                    'portraits.quality.min_sharpness': 25., 'matching.face.anchors.quality.min_sharpness': 25.,
                                    'matching.face.samples.minimum.min_face_pixels': 32, 'face.quality.min_face_pixels': 32}))
    rows = {r['rule']: r for r in report(config)['checks']}
    for rule in ('portrait-weight-resolution', 'portrait-anchor-min_sharpness', 'face-query-min_face_pixels'):
        assert rows[rule]['status'] == 'ok' and not rows[rule]['solver_constraint']
    policy = StrictPolicy(config)
    assert policy.repairs(config, {}, initial=True) == [{}]
    assert policy.repairs(config, {'portraits.min_sharpness': 25.02}) == []


def test_retained_body_replay_repairs_only_the_adjacent_cue_band(tmp_path):
    from tests.test_recognition_solver_feasibility import body_handoff
    config, _, _ = body_handoff(tmp_path)
    config = patched(config, {'sapiensid.identities.track_similarity_threshold': .845})
    with closing(_open_readonly(config.database)) as db:
        goals = resolve(db, config, [['900']])
        result = solve(config, goals, db, maximum=4, strict=True)
    assert result['config_policy']['input_control_replays'] == 1
    assert [r['rule'] for r in result['config_policy']['input_issues']] == ['body-adjacent-cues']
    for row in result['simulation']['trials'].values():
        assert clean(patched(config, {c['parameter']: c['proposed'] for c in row['changes']}))
    assert result['search']['failed_trials'] == 0


def test_parallel_search_only_returns_compliant_trials(tmp_path, monkeypatch):
    from benchmarks.benchmark_merge_solver import fixture
    monkeypatch.setattr('stash_ai.recognition.solver_execution.os.process_cpu_count', lambda: 4, raising=False)
    config, goals = fixture(tmp_path / 'input', 'disjoint')
    execution = dict(cpus=4, trials=2, scheduling='batch', memory_bytes=128 * 1024**3)
    with closing(_open_readonly(config.database)) as db:
        result = solve(config, goals, db, maximum=7, strict=True, execution=execution)
    trials = result['simulation']['trials']
    assert result['search']['execution']['trials'] == 2
    assert result['config_policy']['input_control_replays'] == 0
    assert 2 < result['search']['executed'] <= 7 and result['search']['failed_trials'] == 0
    for row in trials.values():
        assert clean(patched(config, {c['parameter']: c['proposed'] for c in row.get('changes', [])}))

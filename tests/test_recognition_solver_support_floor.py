"""Solver support bounds apply to every route, baseline and exported winner."""
from types import SimpleNamespace

import pytest
import yaml

from stash_ai.recognition.config import defaults, load, plain
from stash_ai.recognition.merge_report_policy import parameter, patched
from stash_ai.recognition.merge_solver_domains import normalize, registry, support_baseline_changes
from stash_ai.recognition.merge_solver_options import compact


def body_config():
    config = SimpleNamespace(data=defaults())
    body = config.data['sapiensid']
    body['enabled'] = body['collection']['enabled'] = body['identities']['enabled'] = True
    body['global_matching']['enabled'] = True
    return config


def support_domains(config):
    return {key: domain for key, domain in registry(config).items() if key.endswith('_support')}


@pytest.mark.parametrize('proposed', [0, 1, 2, 3, 8, 1000])
def test_all_support_domains_bound_numeric_proposals_and_option_pools(proposed):
    config = body_config()
    domains = support_domains(config)
    assert len(domains) == 10
    for key, domain in domains.items():
        assert domain.low == 2
        expected = min(domain.high, max(2, proposed))
        patch = normalize(config, {key: proposed})
        assert patch.get(key, parameter(config.data, key)) == expected
        assert compact(config, {key: [0, 1]}, {key: [proposed]})[key] == sorted({2, expected}, reverse=True)
    # Other integer minima describe different quantities and remain independently tunable.
    patch = normalize(config, {'sapiensid.identities.anchor_min_keypoints': 1,
                               'sapiensid.identities.min_appearance_distance': 0})
    assert patch.get('sapiensid.identities.anchor_min_keypoints',
                     parameter(config.data, 'sapiensid.identities.anchor_min_keypoints')) == 1
    assert registry(config)['sapiensid.identities.min_appearance_distance'].low == 0


def test_baseline_repairs_both_bounds_without_touching_disabled_body_routes():
    config = body_config()
    for key, domain in support_domains(config).items():
        section, leaf = key.rsplit('.', 1)
        parameter(config.data, section)[leaf] = 1
    config.data['recognition']['min_merge_support'] = 1000
    repairs = support_baseline_changes(config)
    assert repairs == {key: 8 if key == 'recognition.min_merge_support' else 2
                       for key in support_domains(config)}
    config.data['sapiensid']['enabled'] = False
    assert not any(key.startswith('sapiensid.') for key in support_baseline_changes(config))
    assert config.data['sapiensid']['matching']['min_support'] == 1


@pytest.mark.parametrize('inherited', [1, 1000])
def test_enabling_global_body_route_projects_inherited_support(inherited):
    config = body_config()
    config.data['sapiensid']['global_matching'].update(enabled=False, min_support=inherited, max_references=4)
    key = 'sapiensid.global_matching.min_support'
    assert key not in support_baseline_changes(config)
    patch = normalize(config, {'sapiensid.global_matching.enabled': True})
    assert patch[key] == (2 if inherited == 1 else 4)
    assert normalize(config, dict(patch, **{key: 1}))[key] == 2


def test_single_global_reference_cannot_bypass_support_floor():
    config = body_config()
    config.data['sapiensid']['global_matching'].update(max_references=1, min_support=1)
    with pytest.raises(ValueError, match=r'max_references >= 2'):
        support_baseline_changes(config)
    config.data['sapiensid']['global_matching']['enabled'] = False
    assert 'sapiensid.global_matching.min_support' not in registry(config)
    with pytest.raises(ValueError, match=r'max_references >= 2'):
        normalize(config, {'sapiensid.global_matching.enabled': True})


def test_solver_cannot_select_a_solution_requiring_single_sample_support(tmp_path):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import evaluate, resolve
    from stash_ai.recognition.merge_solver import solve
    from tests.test_recognition_merge_report import seeded

    config, _ = seeded(tmp_path, count=1,
        recognition=dict(match_threshold=.3, min_identity_support=1, min_merge_support=1),
        clustering=dict(cohesion_threshold=.2))
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900', '901']], policy='strict')
        assert evaluate(config, goals, db)['trials']['baseline']['goal_result']['fully_solved']
        result = solve(config, goals, db, maximum=8, strict=False)
    assert not result['best_result']['fully_solved']
    assert all(not trial['goal_result']['fully_solved'] for trial in result['simulation']['trials'].values())


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('version', [1, 2])
def test_replays_and_exports_raise_single_support_input(tmp_path, monkeypatch, strict, version):
    from stash_ai.recognition import replay_simulation
    from stash_ai.recognition.config_layout import v2_document
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.merge_solver import solve
    from tests.test_recognition_merge_report import seeded

    config, _ = seeded(tmp_path, recognition={'min_identity_support': 1, 'min_merge_support': 1},
        clustering={'auxiliary_min_support': 1, 'same_scene_min_support': 1, 'view_attach_min_support': 1})
    if version == 2:
        config.source_path.write_text(yaml.safe_dump(v2_document(plain(config.data))))
        config = load(config.source_path, database=config.database)
    source, data = config.source_path.read_bytes(), plain(config.data)
    seen = []
    rebuild = replay_simulation.rebuild_selected
    def traced(state, scans, variant):
        seen.append(variant)
        return rebuild(state, scans, variant)
    monkeypatch.setattr(replay_simulation, 'rebuild_selected', traced)
    with _open_readonly(config.database) as db:
        goals = resolve(db, config, [['900', '901']])
        result = solve(config, goals, db, maximum=8, strict=strict)
    # Strict mode can replay the original YAML for comparison, never as a candidate.
    candidates = seen[1:] if strict else seen
    assert len(candidates) >= 2
    for variant in candidates:
        assert all(2 <= parameter(variant.data, key) <= domain.high
                   for key, domain in support_domains(variant).items())
    repairs = support_baseline_changes(config)
    baseline = result['simulation']['trials']['baseline']
    assert {row['parameter']: row['current'] for row in baseline['changes_from_previous']} == repairs
    for trial in result['simulation']['trials'].values():
        variant = patched(config, {row['parameter']: row['proposed'] for row in trial['changes']})
        assert all(parameter(variant.data, key) >= 2 for key in support_domains(variant))
    winner = result['best_result']
    assert winner and winner['changed']
    for mode in ('full', 'diff'):
        text = configuration_document(config, winner, {}, mode=mode)
        document = yaml.safe_load(text)
        if mode == 'diff':
            from stash_ai.recognition.config import _merge
            base = yaml.safe_load(config.source)
            _merge(base, document)
            document = base
        destination = tmp_path / (mode + '.yml')
        destination.write_text(yaml.safe_dump(document))
        exported = load(destination)
        assert all(parameter(exported.data, key) >= 2 for key in support_domains(exported))
    assert config.source_path.read_bytes() == source and plain(config.data) == data

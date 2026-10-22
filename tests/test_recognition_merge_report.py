"""Two-identity proposals must pass real consolidation, with all rivals retained."""
from dataclasses import replace
import json
import math
import uuid

import numpy as np
import pytest
import yaml

from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.clustering import _groups, matching_policy
from stash_ai.recognition.config import load, plain, validate
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.global_views import consolidation_plan
from stash_ai.recognition.merge_report import build
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.models import namespace
from stash_ai.recognition.reference_usage import ReferenceUsage
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.strict_partition import resolve, summary_views


def seeded(tmp_path, vectors=None, *, count=3, recognition=None, clustering=None):
    path = tmp_path / 'faces.yml'
    settings = dict(version=1, recognition=dict(match_threshold=.9),
        clustering=dict(cohesion_threshold=.85, max_anchor_pose_deviation=.2, anchor_view_matching=True))
    settings['recognition'].update(recognition or {})
    settings['clustering'].update(clustering or {})
    path.write_text(yaml.safe_dump(settings))
    config = load(path, database=tmp_path / 'faces.sqlite3')
    vectors = vectors or [[1., 0., 0., 0.], [.7, math.sqrt(1-.7**2), 0., 0.]]
    people = [str(uuid.uuid4()) for _ in vectors]
    with FaceDatabase(config.database) as state:
        db = state.connection
        for index, (person, sample) in enumerate(zip(people, vectors)):
            scan = str(uuid.uuid4())
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       (person, 'scope', namespace(config), index, str(900+index), 'secret-'+person))
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (scan, 'scope', str(index+1), namespace(config), 'input', scan_identity(config, namespace(config)),
                        '{}', '{}', count, count, 'complete'))
            for cue in range(count):
                observation = str(uuid.uuid4())
                vector = np.asarray(sample, dtype='<f4')
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (observation, scan, cue, float(cue), 'http://user:secret@stash/private', '[]', '[]',
                     json.dumps(dict(pose_deviation=.1, pixels=80, sharpness=30)), None, len(vector), vector.tobytes(), b'private-crop'))
                db.execute('INSERT INTO members VALUES (?,?)', (observation, person))
                db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                    (str(uuid.uuid4()), scan, matching_policy(config), person, 'accepted',
                     json.dumps(dict(observation=observation, reason='candidate_group'))))
        db.commit()
    return config, people


def changes(item):
    return {row['parameter']: row['proposed'] for row in item['changes']}


def test_joint_threshold_patch_is_valid_and_actual_strict_partition_merges(tmp_path):
    config, people = seeded(tmp_path)
    before = config.database.read_bytes()
    report = build(config, *people)
    assert report['status'] == 'verified_parameter_changes'
    assert report['guarantees_full_rebuild_merge'] is False
    strict = next(item for item in report['recommendations'] if item['route'] == 'strict_core')
    patch = changes(strict)
    assert patch['recognition.match_threshold'] <= .7
    assert patch['clustering.cohesion_threshold'] <= patch['recognition.match_threshold']
    assert 'recognition.min_merge_support' not in patch
    assert strict['additional_identity_ids'] == []
    proposed = patched(config, patch)
    validate(plain(proposed.data))
    assert yaml.safe_load(strict['yaml']) == strict['yaml_patch']
    assert config.database.read_bytes() == before
    text = json.dumps(report, allow_nan=False)
    assert 'private-crop' not in text and 'user:secret' not in text and 'secret-' not in text
    with FaceDatabase(config.database) as state:
        selected = [dict(scene=r['scene'], scan=r['id']) for r in state.connection.execute('SELECT id,scene FROM scans')]
        resolve(state, selected, proposed)
        assert state.connection.execute('SELECT COUNT(DISTINCT person) FROM members').fetchone()[0] == 1


def test_copresence_requires_its_own_joint_threshold_change(tmp_path):
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        first = db.execute('SELECT o.scan FROM members m JOIN observations o ON o.id=m.observation WHERE m.person=?', (people[0],)).fetchone()[0]
        second = db.execute('SELECT observation FROM members WHERE person=? LIMIT 1', (people[1],)).fetchone()[0]
        db.execute('UPDATE observations SET scan=?,cue=0 WHERE id=?', (first, second))
        db.commit()
    report = build(config, *people)
    strict = next(item for item in report['recommendations'] if item['route'] == 'strict_core')
    assert changes(strict)['recognition.cooccurrence_merge_threshold'] <= .7
    assert strict['verification']['pair_evidence'][0]['cooccurring']


def test_outsider_ambiguity_margin_is_measured_and_rechecked(tmp_path):
    vectors = [[1., 0, 0, 0], [.95, math.sqrt(1-.95**2), 0, 0], [.85, 0, math.sqrt(1-.85**2), 0]]
    config, people = seeded(tmp_path, vectors, recognition=dict(ambiguity_margin=.15), clustering=dict(cohesion_threshold=.8))
    report = build(config, *people[:2])
    current = report['paths'][0]['current']
    assert 'competing_view' in current['blockers']
    strict = next(item for item in report['recommendations'] if item['route'] == 'strict_core')
    assert 0 <= changes(strict)['recognition.ambiguity_margin'] < .15
    assert strict['verification']['can_merge_in_frozen_catalogue']
    assert people[2] not in strict['merged_identity_ids']


def test_incomplete_chain_is_not_reported_as_a_verified_pair_merge(tmp_path):
    vectors = [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0, 0] for a in (0, 20, 40)]
    config, people = seeded(tmp_path, vectors, clustering=dict(cohesion_threshold=.8))
    report = build(config, *people[:2])
    assert report['status'] == 'no_verified_proposal'
    assert 'incomplete_strong_component' in report['paths'][0]['current']['blockers']
    assert all(not a['verification']['can_merge_in_frozen_catalogue'] for p in report['paths'] for a in p['attempts'])


def test_other_identities_in_a_successful_component_are_explicit(tmp_path):
    vectors = [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0, 0] for a in (0, 10, 20)]
    config, people = seeded(tmp_path, vectors)
    report = build(config, *people[:2])
    assert report['status'] == 'eligible_under_current_parameters'
    assert report['recommendations'][0]['additional_identity_ids'] == [people[2]]
    assert set(report['recommendations'][0]['merged_identity_ids']) == set(people)


def test_view_support_and_both_view_and_global_gates_are_tuned(tmp_path):
    config, people = seeded(tmp_path, count=2)
    report = build(config, *people)
    view = next(item for item in report['recommendations'] if item['route'] == 'confirmed_views')
    patch = changes(view)
    assert patch['clustering.view_min_support'] == 2
    assert patch['recognition.min_merge_support'] == 2
    assert patch['clustering.view_match_threshold'] <= .7
    assert patch['recognition.match_threshold'] <= .7
    assert patch['clustering.view_cohesion_threshold'] <= .7
    validate(plain(patched(config, patch).data))


@pytest.mark.parametrize('reason', ['auxiliary_identity', 'same_scene_identity', 'global_view_identity'])
def test_attachment_members_never_become_strict_merge_proof(tmp_path, reason):
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.reason',?) WHERE person=?", (reason, people[1]))
        state.connection.commit()
    report = build(config, *people)
    assert report['status'] == 'no_verified_proposal'
    assert report['identities'][1]['members'] == 3
    assert report['identities'][1]['selected_strict_references'] == 0
    assert report['evidence_actions'][0]['identity_id'] == people[1]


def test_superseded_scan_and_incompatible_namespace_do_not_supply_proof(tmp_path):
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        old = db.execute('SELECT s.* FROM scans s JOIN observations o ON o.scan=s.id JOIN members m ON m.observation=o.id WHERE m.person=? LIMIT 1', (people[1],)).fetchone()
        row = list(old);row[0] = str(uuid.uuid4());row[4] = 'new-input';row[-1] = 'partial'
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)', row)
        db.commit()
    report = build(config, *people)
    assert report['status'] == 'no_verified_proposal'
    assert report['identities'][1]['selected_strict_references'] == 0
    data = plain(config.data);data['models']['recognizer'] = 'auraface'
    with pytest.raises(ValueError, match='No identity'):
        build(replace(config, data=data), *people)


def test_disabled_feature_is_proposed_only_when_its_anchor_prerequisite_exists(tmp_path):
    config, people = seeded(tmp_path, clustering=dict(anchor_view_matching=False))
    report = build(config, *people)
    assert any(changes(p).get('clustering.anchor_view_matching') is True for p in report['recommendations'])
    data = plain(config.data);data['clustering']['max_anchor_pose_deviation'] = None
    report = build(replace(config, data=data), *people)
    assert report['status'] == 'no_verified_proposal'
    assert 'anchor_restriction_required' in report['paths'][0]['current']['blockers']


def test_negative_scores_cannot_be_solved_by_illegal_negative_yaml_thresholds(tmp_path):
    config, people = seeded(tmp_path, [[1., 0, 0, 0], [-1., 0, 0, 0]])
    report = build(config, *people)
    assert report['status'] == 'no_verified_proposal'
    assert report['recommendations'] == []
    assert 'pair_requires_a_negative_threshold_outside_the_yaml_range' in report['paths'][0]['unresolved_reasons']


def test_aliases_are_resolved_and_cross_scope_or_missing_ids_fail(tmp_path):
    config, people = seeded(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)', (people[1], people[0], 1., '{}', 'confirmed'))
        db.commit()
    report = build(config, *people)
    assert report['status'] == 'already_merged'
    assert report['identity_ids'] == [people[0], people[0]]
    with pytest.raises(ValueError, match='No identity'):
        build(config, people[0], 'missing')
    with FaceDatabase(config.database) as state:
        state.connection.execute('DELETE FROM identity_merges')
        state.connection.execute('UPDATE people SET scope=? WHERE id=?', ('other', people[1]))
        state.connection.commit()
    with pytest.raises(ValueError, match='same Stash scope'):
        build(config, *people)


@pytest.mark.parametrize('focus_positions', [(0, 1), (1, 2), (2, 0)])
@pytest.mark.parametrize('outsiders', [0, 17])
def test_focused_planner_matches_full_components_and_outsider_competition(tmp_path, focus_positions, outsiders):
    vectors = [[math.cos(math.radians(a)), math.sin(math.radians(a)), 0, 0] for a in (0, 20, 40)]
    vectors += [[0., 0., 1., 0.]] * outsiders
    config, people = seeded(tmp_path, vectors, clustering=dict(cohesion_threshold=.8))
    with FaceDatabase(config.database) as state:
        groups = _groups(state.connection, 'scope', namespace(config), config.data['clustering'])
        views = summary_views(groups, ReferenceUsage(config.data['clustering']['max_reference_bytes']))
        focus = [people[i] for i in focus_positions]
        full = consolidation_plan(state.connection, views, config)
        focused = consolidation_plan(state.connection, views, config, focus=focus)
        assert focused == tuple(p for p in full if set(focus).intersection(p['people']))


def test_cli_json_export_is_offline_readonly_and_exclusive(tmp_path, monkeypatch, capsys):
    config, people = seeded(tmp_path)
    import stash_ai.recognition.run as run
    monkeypatch.setattr(run, 'execute', lambda *a, **k: pytest.fail('must remain offline'))
    before = config.database.read_bytes()
    args = ['report-merge', '--identity-id', people[0], '--identity-id', people[1],
            '--face-config', str(tmp_path/'faces.yml'), '--faces-db', str(config.database)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['kind'] == 'recognition_identity_merge'
    output = tmp_path/'merge.json'
    assert main(args + ['--output', str(output)]) == 0
    assert json.loads(output.read_text())['recommendations']
    assert config.database.read_bytes() == before
    with pytest.raises(FileExistsError):
        main(args + ['--output', str(output)])
    with pytest.raises(ValueError, match='overwrite'):
        main(args + ['--output', str(config.database)])
    for count in (0, 1, 3):
        with pytest.raises(SystemExit):
            parser().parse_args(['report-merge'] + [part for _ in range(count) for part in ('--identity-id', people[0])])

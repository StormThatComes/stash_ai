"""Read-only scene fragmentation reporting must never run inference or change SQLite."""
import json
import math
import sqlite3
import uuid

import numpy as np
import pytest

from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import build
from stash_ai.recognition.models import namespace


def seeded(tmp_path, settings=''):
    (tmp_path / 'facial_recognition.yml').write_text('version: 1\n' + settings)
    config = load(root=tmp_path)
    scan = str(uuid.uuid4())
    policy, ns = matching_policy(config), namespace(config)
    a, b, c = [str(uuid.uuid4()) for _ in range(3)]
    vectors = {
        a: np.asarray([1., 0., 0., 0.], dtype=np.float32),
        b: np.asarray([.7, math.sqrt(1 - .7 ** 2), 0., 0.], dtype=np.float32),
        c: np.asarray([0., 0., 1., 0.], dtype=np.float32),
    }
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   (scan, 'scope', '123', ns, 'input', 'detection', '{}', '{}', 3, 3, 'complete'))
        for cue, (identity, face) in enumerate(vectors.items()):
            obs = str(uuid.uuid4())
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       (identity, 'scope', ns, float(cue), str(cue + 900), 'token-' + identity))
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (obs, scan, cue, float(cue), 'vtt', '{}', str(cue),
                        '{}', None, 4, face.tobytes(), None))
            db.execute('INSERT INTO members VALUES (?,?)', (obs, identity))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                       (str(uuid.uuid4()), scan, policy, identity, 'accepted',
                        json.dumps({'observation': obs, 'reason': 'candidate_group'})))
        withheld = str(uuid.uuid4())
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                   (withheld, scan, 3, 3., 'vtt', '{}', 'withheld-geometry',
                    '{}', None, 4, vectors[a].tobytes(), None))
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                   (str(uuid.uuid4()), scan, policy, None, 'withheld',
                    json.dumps({'observation': withheld,
                                'reason': 'ambiguous_identity', 'scores': [0.66, 0.64]})))
        db.commit()
    return config


def test_report_ranks_pairs_and_explains_cross_sample_blockers(tmp_path):
    config = seeded(tmp_path)
    result = build(config, '123')
    assert result['format'] == 1
    assert result['summary']['identities'] == 3
    assert result['summary']['pairs_checked'] == 3
    assert result['summary']['pairs_above_min_mean'] == 1
    pair = result['potential_fragments'][0]
    assert pair['cross_mean'] == pytest.approx(.7, abs=1e-5)
    assert pair['cross_minimum'] == pytest.approx(.7, abs=1e-5)
    assert pair['blockers'] == []
    assert pair['recorded_merge_support']['validated_nominations'] == 0
    assert pair['recorded_merge_support']['complete_replay_proof'] is False
    assert pair['can_merge'] is None
    assert pair['worst_representatives']['left']['scene'] == '123'
    assert result['decisions'][0]['count'] == 3
    assert result['withheld_examples'][0]['reason'] == 'ambiguous_identity'
    assert result['withheld_examples'][0]['seconds'] == 3.
    assert result['withheld_examples'][0]['scores'] == [.66, .64]
    assert result['summary']['observations'] == 4
    assert result['observation_quality']['retained']['metrics']['pixels']['missing'] == 4
    assert result['detection_gates']['min_face_pixels'] == 32
    assert result['detection_gates']['max_pose_deviation'] == .5
    assert result['rejected_examples'] == {}
    assert result['thresholds']['same_scene_min_support'] == 3
    assert result['thresholds']['local_match_threshold'] == pytest.approx(.62)
    assert result['thresholds']['local_cohesion_threshold'] == pytest.approx(.57)
    assert pair['same_scene_merge']['strict_accepted_scene_cues'] == {'left': 1, 'right': 1}
    assert 'no_possible_strict_local_core_by_cue_count' in pair['same_scene_merge']['numeric_and_count_blockers']
    assert pair['same_scene_merge']['can_merge'] is None
    assert pair['score_diagnostics']['selected_scene_only']['comparisons'] == 1
    assert len(pair['score_diagnostics']['worst_pairs']) == 1
    assert pair['recorded_merge_support']['stored_nominations'] == 0
    assert pair['same_scene_merge']['recorded_merge_support']['stored_nominations'] == 0


def test_report_thresholds_limits_and_config_mismatch(tmp_path):
    config = seeded(tmp_path)
    result = build(config, '123', min_mean=-1, limit=2)
    assert result['summary']['pairs_above_min_mean'] == 3
    assert len(result['potential_fragments']) == 2
    assert any('cross_mean_below_match_threshold' in pair['blockers']
               for pair in result['potential_fragments'])
    with pytest.raises(ValueError, match='Report limit'):
        build(config, '123', limit=0)
    with pytest.raises(ValueError, match='Report min-mean'):
        build(config, '123', min_mean=float('nan'))
    changed = tmp_path / 'different.yml'
    changed.write_text('version: 1\nrecognition: {match_threshold: 0.8}\n')
    with pytest.raises(ValueError, match='current face configuration'):
        build(load(changed, database=config.database), '123')


def test_cli_writes_json_without_modifying_database_and_does_not_call_run(tmp_path, monkeypatch):
    config = seeded(tmp_path)
    output = tmp_path / 'fragmentation.json'
    with sqlite3.connect(config.database) as db:
        before = tuple(db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                       for table in ('members', 'candidates', 'runs', 'identity_merges'))
    import stash_ai.recognition.run as run_module
    monkeypatch.setattr(run_module, 'execute', lambda *args, **kwargs:
                        pytest.fail('Report must not enter recognition run'))
    assert main(['report', '--scene-id', '123',
                  '--face-config', str(tmp_path / 'facial_recognition.yml'),
                  '--faces-db', str(config.database),
                 '--output', str(output)]) == 0
    data = json.loads(output.read_text())
    assert data['scene_id'] == '123'
    with sqlite3.connect(config.database) as db:
        after = tuple(db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                      for table in ('members', 'candidates', 'runs', 'identity_merges'))
    assert before == after
    with pytest.raises(FileExistsError):
        main(['report', '--scene-id', '123',
                  '--face-config', str(tmp_path / 'facial_recognition.yml'),
                  '--faces-db', str(config.database),
              '--output', str(output)])


def test_report_rejects_non_database_and_parser_requires_scene(tmp_path):
    config = load(root=tmp_path)
    with pytest.raises(ValueError, match='does not exist'):
        build(config, '123')
    assert not config.database.exists()
    with pytest.raises(SystemExit):
        parser().parse_args(['report'])


def test_report_local_bonus_and_recorded_local_nominations(tmp_path):
    config = seeded(tmp_path,
        'recognition: {match_threshold: 0.75, min_merge_support: 1}\n'
        'clustering: {cohesion_threshold: 0.60, same_scene_match_bonus: 0.10, '
        'same_scene_min_support: 1}\n')
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.identity_merges import _merge_support_policy
    with FaceDatabase(config.database) as state:
        db = state.connection
        scan = db.execute("SELECT id FROM scans WHERE scene='123'").fetchone()[0]
        people = list(db.execute('SELECT id FROM people ORDER BY created'))
        local_data = plain(config.data)
        local_data['recognition']['match_threshold'] -= .1
        local_data['clustering']['cohesion_threshold'] -= .1
        local = replace(config, data=local_data)
        sample = db.execute('SELECT observation FROM members WHERE person=?', (people[0][0],)).fetchone()[0]
        db.execute('INSERT INTO merge_support VALUES (?,?,?,?)',
                   (people[0][0], people[1][0], sample, _merge_support_policy(local, scene_scan=scan)))
        db.commit()
    result = build(config, '123')
    pair = result['potential_fragments'][0]
    assert pair['blockers'] == ['cross_mean_below_match_threshold']
    local = pair['same_scene_merge']
    assert local['effective_match_threshold'] == pytest.approx(.65)
    assert local['effective_cohesion_threshold'] == pytest.approx(.5)
    assert local['numeric_and_count_blockers'] == []
    assert local['strict_accepted_scene_cues'] == {'left': 1, 'right': 1}
    assert local['strict_core_revalidation_complete'] is False
    assert local['recorded_merge_support']['stored_nominations'] == 1
    assert local['recorded_merge_support']['validated_nominations'] == 1
    assert pair['recorded_merge_support']['stored_nominations'] == 0
    assert pair['can_merge'] is None


def test_report_quality_representative_provenance_and_cap(tmp_path):
    config = seeded(tmp_path, 'clustering: {max_samples_per_person: 2}\n')
    with FaceDatabase(config.database) as state:
        db = state.connection
        scan = db.execute("SELECT id FROM scans WHERE scene='123'").fetchone()[0]
        person = db.execute("SELECT id FROM people ORDER BY created LIMIT 1").fetchone()[0]
        sample = db.execute(
            'SELECT o.id,o.embedding FROM observations o JOIN members m ON m.observation=o.id '
            'WHERE m.person=?', (person,)).fetchone()
        db.execute('UPDATE observations SET quality=? WHERE id=?',
                   (json.dumps({'pixels': 64, 'pose_deviation': .12, 'sharpness': 30}), sample['id']))
        for cue in (4, 5):
            observation = str(uuid.uuid4())
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (observation, scan, cue, float(cue), 'vtt', '{}', str(cue),
                        json.dumps({'pixels': 48, 'pose_deviation': .19, 'sharpness': 12}),
                        None, 4, sample['embedding'], None))
            db.execute('INSERT INTO members VALUES (?,?)', (observation, person))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                       (str(uuid.uuid4()), scan, matching_policy(config), person, 'accepted',
                        json.dumps({'observation': observation, 'reason': 'known_identity'})))
        db.commit()
    result = build(config, '123')
    identity = next(item for item in result['identities'] if item['identity_id'] == person)
    refs = identity['representatives']
    assert (refs['available'], refs['selected'], refs['limit'], refs['truncated']) == (3, 2, 2, True)
    assert len(refs['samples']) == 2
    assert refs['samples'][0]['quality']['pixels'] == 64.
    assert refs['quality']['pixels']['median'] == 56.
    assert refs['quality']['visible_face_fraction']['missing'] == 2
    assert result['observation_quality']['retained']['metrics']['pixels']['count'] == 3
    assert identity['strict_accepted_scene_cues'] == 3
    pair = next(p for p in result['potential_fragments']
                if p['left']['identity_id'] == person or p['right']['identity_id'] == person)
    assert pair['score_diagnostics']['cross_scores']['comparisons'] == 2
    assert pair['score_diagnostics']['selected_scene_only']['comparisons'] == 2
    assert pair['score_diagnostics']['worst_pairs'][0]['left']['quality']['pixels'] is not None or (
        pair['score_diagnostics']['worst_pairs'][0]['right']['quality']['pixels'] is not None)

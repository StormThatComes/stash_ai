"""Identity exports preserve all memberships, recorded evidence and decoded pixels."""
from io import BytesIO
import json
import os
import sqlite3
import uuid

import pytest
from PIL import Image

from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_report import export
from stash_ai.recognition.models import namespace
from tests.helpers.recognition import FACE


def seed(tmp_path, count=5):
    path = tmp_path / 'faces.yml'
    path.write_text('version: 1\nclustering: {max_samples_per_person: 2, max_anchor_pose_deviation: 0.2}\n')
    config = load(path, database=tmp_path / 'faces.sqlite3')
    person = str(uuid.uuid4())
    output = BytesIO()
    with Image.new('RGB', (37, 43), (121, 34, 67)) as image:
        image.save(output, 'JPEG')
    blob = output.getvalue()
    policy = matching_policy(config)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', (person, 'scope', namespace(config), 1., '900', 'private-token'))
        scans = {}
        for scene in ('2', '10'):
            scan = scans[scene] = str(uuid.uuid4())
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (scan, 'scope', scene, namespace(config), 'input', 'detection', '{}', '{}', count, count, 'complete'))
        for index in range(count):
            scene, cue = ('2' if index < (count + 1) // 2 else '10'), index // 2
            observation = str(uuid.uuid4())
            quality = {'pixels': 32, 'sharpness': 5, 'pose_deviation': .1 if index % 2 == 0 else .3,
                       'crop_encoding': {'pixels': [37, 43], 'downscaled': True},
                       'crop_bounds': [0, 0, 37, 43]}
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (observation, scans[scene], cue, float(cue) * 10, 'http://user:secret@stash/sprite?token=private',
                        '[0,0,160,160]', json.dumps(FACE.row), json.dumps(quality), None, 4, b'\x00' * 16, blob))
            db.execute('INSERT INTO members VALUES (?,?)', (observation, person))
            reason = 'known_identity' if index % 2 == 0 else 'auxiliary_identity'
            payload = {'observation': observation, 'reason': reason, 'scores': [.87, .61],
                       'policy': policy, 'representative_limit': 2,
                       'embedding': 'raw-embedding-must-not-export', 'image': 'raw-image-must-not-export'}
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                       (str(uuid.uuid4()), scans[scene], policy, person, 'accepted', json.dumps(payload)))
        db.commit()
    return config, person, blob


def test_all_faces_export_beyond_reference_and_128_limits_with_exact_pixels(tmp_path):
    config, person, blob = seed(tmp_path, count=137)
    before = config.database.read_bytes()
    target = tmp_path / 'identity'
    result = export(config, target, identity_id=person)
    assert result['summary'] == {'members': 137, 'pngs_exported': 137, 'missing_crops': 0,
                                 'invalid_crops': 0, 'roles': {'anchor': 69, 'auxiliary': 68, 'unknown': 0},
                                 'complete': True}
    assert len(list(target.glob('*.png'))) == 411  # Clean, annotated and threshold copies.
    assert sum(item['current_reference_selected'] for item in result['samples']) == 2
    assert result['cue_index_base'] == 0
    assert result['samples'][0]['scene_id'] == '2' and result['samples'][-1]['scene_id'] == '10'
    # Even samples below today's sharpness gate remain part of this identity.
    assert result['samples'][0]['quality']['sharpness'] < config.data['detection']['min_sharpness']
    names = []
    with Image.open(BytesIO(blob)) as original:
        for sample in result['samples']:
            name = sample['image']['file']
            names.append(name)
            assert f'scene-{sample["scene_id"]}_cue-{sample["cue_index"]:06d}_' in name
            assert sample['assignment']['role'] in name
            with Image.open(target / name) as png:
                assert png.format == 'PNG' and png.size == (37, 43)
                assert png.tobytes() == original.tobytes()
    assert len(set(names)) == 137
    text = (target / 'manifest.json').read_text()
    assert json.loads(text) == result
    for secret in ('private-token', 'raw-embedding-must-not-export', 'raw-image-must-not-export', 'user:secret', 'token=private'):
        assert secret not in text
    assert config.database.read_bytes() == before
    if os.name == 'posix':
        assert target.stat().st_mode & 0o777 == 0o700
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in target.iterdir())


def test_recorded_context_view_merge_and_origins_explain_assignment(tmp_path):
    config, person, _ = seed(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        sample = db.execute('SELECT * FROM candidates LIMIT 1').fetchone()
        payload = json.loads(sample['payload'])
        predecessor = str(uuid.uuid4())
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                   (predecessor, 'scope', namespace(config), 0., '899', 'old-token'))
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                   (predecessor, person, 2., json.dumps({'support_seconds': [10, 20], 'scores': {'minimum_cross_similarity': .81}}), 'confirmed'))
        payload.update(reason='same_scene_identity', merged_from=predecessor,
                       same_scene_context={'threshold': .62, 'cohesion_threshold': .57,
                                           'support': [{'seconds': 10., 'observation': 'supporting-face'}]})
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (json.dumps(payload), sample['id']))
        db.execute('INSERT INTO identity_origins VALUES (?,?,?)', ('key', predecessor, payload['observation']))
        # A separate historical decision is preserved without overriding current evidence.
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                   ('old-decision', sample['scan'], 'old-policy', predecessor, 'accepted',
                    json.dumps({'observation': payload['observation'], 'reason': 'view_identity',
                                'view_prototype': {'medoid': 'face', 'match_score': .91, 'threshold': .82}})))
        db.commit()
    result = export(config, tmp_path / 'identity', performer_id='899')
    assert result['requested_identity']['id'] == predecessor
    assert result['identity']['id'] == person
    assert result['identity_merges'][0]['evidence']['support_seconds'] == [10, 20]
    sample = next(row for row in result['samples'] if row['observation_id'] == payload['observation'])
    assert sample['assignment']['category'] == 'anchor-same-scene'
    assert sample['assignment']['evidence_status'] == 'recorded_current_policy'
    assert sample['original_identity_ids'] == [predecessor]
    current = next(item for item in sample['decisions'] if item['matches_current_policy'])
    assert current['recorded']['scores'] == [.87, .61]
    assert current['recorded']['same_scene_context']['threshold'] == .62
    old = next(item for item in sample['decisions'] if not item['matches_current_policy'])
    assert old['supports_membership'] and old['recorded']['view_prototype']['match_score'] == .91


def test_missing_truncated_and_invalid_crops_are_counted_and_cli_returns_incomplete(tmp_path):
    config, person, blob = seed(tmp_path)
    with sqlite3.connect(config.database) as db:
        ids = [row[0] for row in db.execute('SELECT id FROM observations ORDER BY rowid')]
        for observation, data in zip(ids, (None, b'not an image', blob[:100])):
            db.execute('UPDATE observations SET crop=? WHERE id=?', (data, observation))
    target = tmp_path / 'identity'
    assert main(['report-identity', '--performer-id', '900', '--face-config', str(tmp_path / 'faces.yml'),
                 '--faces-db', str(config.database), '--output-dir', str(target)]) == 1
    result = json.loads((target / 'manifest.json').read_text())
    assert result['summary']['members'] == 5 and result['summary']['pngs_exported'] == 2
    assert result['summary']['missing_crops'] == 1 and result['summary']['invalid_crops'] == 2
    assert not result['summary']['complete']
    assert sum(item['image']['file'] is None for item in result['samples']) == 3


def test_no_decision_and_old_policy_are_explicit_never_inferred_from_scores(tmp_path):
    config, person, _ = seed(tmp_path)
    with sqlite3.connect(config.database) as db:
        candidates = db.execute('SELECT id,payload FROM candidates ORDER BY rowid').fetchall()
        first = json.loads(candidates[0][1])['observation']
        second = json.loads(candidates[1][1])['observation']
        db.execute('DELETE FROM candidates WHERE id=?', (candidates[0][0],))
        db.execute("UPDATE candidates SET policy='old' WHERE id=?", (candidates[1][0],))
    result = export(config, tmp_path / 'identity', identity_id=person)
    missing = next(row for row in result['samples'] if row['observation_id'] == first)
    assert missing['assignment']['role'] == 'unknown'
    assert missing['assignment']['evidence_status'] == 'membership_only'
    assert not missing['decisions'] and '_unknown_' in missing['image']['file']
    historical = next(row for row in result['samples'] if row['observation_id'] == second)
    assert historical['assignment']['evidence_status'] == 'recorded_other_policy'
    assert historical['assignment']['role'] == 'auxiliary'


def test_unaccepted_or_other_person_decisions_do_not_explain_membership(tmp_path):
    config, person, _ = seed(tmp_path)
    with sqlite3.connect(config.database) as db:
        db.execute("UPDATE candidates SET status='withheld'")
        db.execute("UPDATE candidates SET status='accepted',person='someone-else' WHERE rowid=1")
    result = export(config, tmp_path / 'identity', identity_id=person)
    assert result['summary']['roles']['unknown'] == 5
    assert all(not decision['supports_membership'] for row in result['samples'] for decision in row['decisions'])


@pytest.mark.parametrize('reason,category', [
    ('candidate_group', 'anchor'), ('known_identity', 'anchor'),
    ('same_scene_identity', 'anchor-same-scene'), ('anchor_view_identity', 'anchor-view'),
    ('cross_scene_identity', 'anchor-cross-scene'), ('cross_scene_provisional', 'anchor-cross-scene'),
    ('auxiliary_identity', 'auxiliary'), ('view_identity', 'auxiliary-view'),
])
def test_recorded_assignment_category_is_not_relabelled_by_current_quality(tmp_path, reason, category):
    config, person, _ = seed(tmp_path, count=1)
    with sqlite3.connect(config.database) as db:
        candidate, raw = db.execute('SELECT id,payload FROM candidates').fetchone()
        payload = json.loads(raw)
        payload['reason'] = reason
        db.execute('UPDATE candidates SET payload=? WHERE id=?', (json.dumps(payload), candidate))
        # Current quality marks this sample as an anchor even if the original
        # accepted path was auxiliary; the two concepts must remain distinct.
        db.execute("UPDATE observations SET quality=json_set(quality,'$.pose_deviation',0.1)")
    result = export(config, tmp_path / 'identity', identity_id=person)
    sample = result['samples'][0]
    assert sample['assignment']['category'] == category
    assert f'_{category}_' in sample['image']['file']
    assert sample['current_anchor_eligible']
    assert sample['decisions'][0]['explanation']


def test_selection_disambiguates_scope_and_filters_namespace_before_creating_output(tmp_path):
    config, person, _ = seed(tmp_path)
    with sqlite3.connect(config.database) as db:
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                   ('other-scope', 'second-scope', namespace(config), 1., '900', 'other-token'))
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                   ('other-model', 'scope', 'other-namespace', 1., '900', 'other-model-token'))
    with pytest.raises(ValueError, match='ambiguous'):
        export(config, tmp_path / 'ambiguous', performer_id='900')
    assert not (tmp_path / 'ambiguous').exists()
    result = export(config, tmp_path / 'selected', performer_id='900', scope='scope')
    assert result['identity']['id'] == person
    with pytest.raises(ValueError, match='No identity'):
        export(config, tmp_path / 'wrong-model', identity_id='other-model')
    with pytest.raises(ValueError, match='No identity'):
        export(config, tmp_path / 'wrong-scope', identity_id=person, scope='second-scope')


def test_cli_offline_dispatch_never_enters_run_stash_or_inference(tmp_path, monkeypatch):
    config, person, _ = seed(tmp_path)
    from stash_ai.recognition import run, models
    def forbidden(*args, **kwargs):
        pytest.fail('Identity report must be offline and read-only')
    monkeypatch.setattr(run, 'execute', forbidden)
    monkeypatch.setattr(run, 'Services', forbidden)
    monkeypatch.setattr(models, 'ensure_assets', forbidden)
    with FaceDatabase(config.database) as live_writer:
        live_writer.start_run('scope', ['2'], {})
        assert main(['report-identity', '--identity-id', person, '--face-config', str(tmp_path / 'faces.yml'),
                     '--faces-db', str(config.database), '--output-dir', str(tmp_path / 'offline')]) == 0
        assert live_writer.connection.execute('SELECT status FROM runs').fetchone()[0] == 'running'


def test_export_consumes_real_scanning_and_cold_rebuild_records(tmp_path, monkeypatch):
    from tests.test_recognition_run import setup, run
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    before = config.database.read_bytes()
    result = export(config, tmp_path / 'identity', performer_id=stash.links['2'][-1])
    assert result['summary']['members'] == result['summary']['pngs_exported'] == 6
    assert {sample['scene_id'] for sample in result['samples']} == {'1', '2'}
    assert all(sample['detection_policy_matches_current'] for sample in result['samples'])
    assert all(sample['assignment']['evidence_status'] == 'recorded_current_policy' for sample in result['samples'])
    assert all(decision['recorded']['identity_basis'] == 'cold_selected_replay'
               for sample in result['samples'] for decision in sample['decisions'])
    assert config.database.read_bytes() == before


def test_output_refuses_existing_directory_and_write_failure_is_fatal(tmp_path, monkeypatch):
    config, person, _ = seed(tmp_path)
    target = tmp_path / 'existing'
    target.mkdir()
    (target / 'keep.txt').write_text('preserve')
    with pytest.raises(FileExistsError):
        export(config, target, identity_id=person)
    assert (target / 'keep.txt').read_text() == 'preserve'
    from stash_ai.recognition import identity_report
    def fail(*args):
        raise OSError('synthetic disk full')
    monkeypatch.setattr(identity_report, '_png', fail)
    with pytest.raises(OSError, match='disk full'):
        export(config, tmp_path / 'failure', identity_id=person)
    assert not (tmp_path / 'failure' / 'manifest.json').exists()


def test_missing_database_or_invalid_selection_never_creates_outputs(tmp_path):
    config = load(root=tmp_path)
    with pytest.raises(ValueError, match='does not exist'):
        export(config, tmp_path / 'missing', identity_id='x')
    assert not config.database.exists() and not (tmp_path / 'missing').exists()
    for args in (['report-identity'], ['report-identity', '--output-dir', 'out'],
                 ['report-identity', '--identity-id', 'x', '--performer-id', '1', '--output-dir', 'out']):
        with pytest.raises(SystemExit):
            parser().parse_args(args)


@pytest.mark.parametrize('change', ['cycle', 'scope', 'membership'])
def test_incompatible_merge_history_and_members_fail_before_writing(tmp_path, change):
    config, person, _ = seed(tmp_path)
    with sqlite3.connect(config.database) as db:
        if change == 'membership':
            db.execute("UPDATE scans SET scope='other-scope'")
        else:
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       ('other', 'other-scope' if change == 'scope' else 'scope', namespace(config), 0., None, 'other'))
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)', (person, 'other', 1., '{}', 'confirmed'))
            if change == 'cycle':
                db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)', ('other', person, 1., '{}', 'confirmed'))
    with pytest.raises(ValueError, match='cycle|scope|incompatible'):
        export(config, tmp_path / 'invalid', identity_id=person)
    assert not (tmp_path / 'invalid').exists()

"""Portrait refresh must never replay identities or publish scene/name changes."""
import json

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.portrait_refresh import retained_targets
from stash_ai.recognition.run import execute, instance_scope
from tests.helpers.recognition import Backend, Visibility
from tests.test_recognition_image_retry import lose_images
from tests.test_recognition_metadata import setup as video_setup, Decoder
from tests.test_recognition_portrait_removal import change, last_counts
from tests.test_recognition_run import run
from tests.test_recognition_vtt_portraits import setup, image_size, FACE


def refresh(config, services, *options):
    args = parser().parse_args(['portraits', '--progress-style', 'off', *options])
    return execute(args, config, services=services)


def forbidden(*args, **kwargs):
    pytest.fail('portrait-only refresh must skip this recognition/name operation')


def forbid_recognition(monkeypatch, services, stash):
    for name in ('scan_scene', 'ScenePrefetch', 'rebuild_selected', '_publish_selection',
                 'reconcile_legacy_merges', 'reconcile_unlinks', 'starter'):
        monkeypatch.setattr('stash_ai.recognition.run.' + name, forbidden)
    monkeypatch.setattr('stash_ai.recognition.metadata.sync_name', forbidden)
    services.metadata_config = services.metadata_plan = forbidden
    stash.scene_ids = forbidden


def identity_state(config):
    with FaceDatabase(config.database) as state:
        result = {table: [tuple(row) for row in state.connection.execute('SELECT * FROM ' + table)]
                  for table in ('models', 'scans', 'people', 'members', 'candidates', 'identity_merges',
                                'identity_origins', 'merge_support')}
        result['observations'] = [tuple(row) for row in state.connection.execute(
            'SELECT id,scan,cue,seconds,source,rect,geometry,rejected,dimensions,embedding,crop FROM observations')]
        return result


@pytest.mark.parametrize('options', [[], ['--scene-id', '0001'], ['--all-scenes'], ['--retry-unresolved-images'],
    ['--performer-id', '00123'], ['--performer-id', '00123', '--scene-id', '0001'],
    ['--performer-id', '00123', '--all-scenes']])
def test_parser_accepts_portrait_refresh_with_optional_scope(options):
    args = parser().parse_args(['portraits', *options])
    assert args.command == 'portraits'
    if args.scene_id:
        assert args.scene_id == ['1']
    if args.performer_id:
        assert args.performer_id == ['123']


@pytest.mark.parametrize('identifier', ['0', '-1', '1.2', '+1', 'abc', '\u0661', ''])
def test_portraits_rejects_invalid_performer_ids(identifier):
    with pytest.raises(SystemExit):
        parser().parse_args(['portraits', '--performer-id', identifier])


@pytest.mark.parametrize('argv', [
    ['run', '--scene-id', '1', '--portraits-only'],
    ['run', '--scene-id', '1', '--performer-id', '123'],
    ['portraits', '--portraits-only'],
    ['portraits', '--scene-id', '1', '--all-scenes'],
])
def test_portrait_selection_options_belong_to_the_portraits_action(argv):
    with pytest.raises(SystemExit):
        parser().parse_args(argv)


def test_refresh_reframes_owned_vtt_image_without_recognition_models_or_names(tmp_path, monkeypatch, capsys):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    original = identity_state(config)
    profiles = {key: dict(value) for key, value in stash.profiles.items()}
    links = {key: list(value) for key, value in stash.links.items()}
    writes = len(stash.writes)
    forbid_recognition(monkeypatch, services, stash)
    services.assets = services.backend = services.decoder = forbidden
    config = change(config, margin=.5, aspect_ratio=.8, jpeg_quality=75)
    assert refresh(config, services) == 0
    assert image_size(stash.images[performer]) == (200, 250)
    assert identity_state(config) == original
    assert stash.profiles == profiles and stash.links == links
    assert all(w[0] == 'metadata' and set(w[2]) == {'image_path'} for w in stash.writes[writes:])
    assert last_counts(config)['portraits_updated'] == 1
    with FaceDatabase(config.database) as state:
        history = state.connection.execute('SELECT * FROM runs ORDER BY started DESC LIMIT 1').fetchone()
        assert json.loads(history['config'])['invocation'] == {'command': 'portraits'}
        assert json.loads(history['selection']) == ['1'] and history['status'] == 'success'
        phases = {r[0] for r in state.connection.execute('SELECT phase FROM progress WHERE run=?', (history['id'],))}
        assert 'portrait_selection' in phases and not phases & {'scan', 'rebuild', 'matching', 'names', 'name_starter', 'backend'}
    assert 'portrait_people_selected=1' in capsys.readouterr().out
    writes = list(stash.writes)
    assert refresh(config, services, '--all-scenes') == 0
    assert stash.writes == writes


def test_selection_deduplicates_people_and_limits_evidence_to_requested_scenes(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    with FaceDatabase(config.database) as state:
        targets = retained_targets(state, config, instance_scope(stash.url))
        assert len(targets) == 1
        assert [s['scene'] for t in targets.values() for s in t['scans']] == ['1', '2']
    forbid_recognition(monkeypatch, services, stash)
    captured = []
    services.metadata = lambda *a, **kw: captured.append(a[6]) or {}
    assert refresh(config, services, '--scene-id', '2', '--scene-id', '2') == 0
    assert len(captured) == 1 and [s['scene'] for s in captured[0]['scans']] == ['2']
    assert refresh(config, services, '--scene-id', '999') == 0
    assert len(captured) == 1 and last_counts(config)['portrait_people_selected'] == 0


@pytest.mark.parametrize('scene_ids', [None, ['2'], ['999']])
def test_performer_filter_refreshes_only_requested_person_with_intersected_scene_evidence(tmp_path, monkeypatch, scene_ids):
    from stash_ai.recognition.models import checked_detection
    import numpy as np

    config, stash, services, _ = setup(tmp_path, monkeypatch)
    geometry = list(FACE.row)
    for index in (0, 4, 6, 8, 10, 12):
        geometry[index] += 150
    other = checked_detection(geometry)
    def backend(*args):
        value = Backend()
        value.detect = lambda image: [FACE, other]
        value.embed = lambda image, face: np.asarray(
            [1., 0., 0., 0.] if face.box[0] < 300 else [0., 1., 0., 0.], dtype='float32')
        return value
    services.backend = backend
    assert run(config, services, '1', '2') == 0
    selected, untouched = stash.links['1'][-2:]
    assert set(stash.links['2']) == {selected, untouched}
    original = identity_state(config)
    images = dict(stash.images)
    writes = len(stash.writes)
    config = change(config, margin=.5)
    forbid_recognition(monkeypatch, services, stash)
    services.assets = services.backend = forbidden
    options = ['--performer-id', '000' + selected, '--performer-id', selected]
    for scene in scene_ids or []:
        options += ['--scene-id', scene]
    assert refresh(config, services, *options) == 0
    expected = 0 if scene_ids == ['999'] else 1
    assert last_counts(config)['portrait_people_selected'] == expected
    assert last_counts(config).get('portraits_updated', 0) == expected
    assert stash.images[untouched] == images[untouched]
    assert all(write[0] == 'metadata' and write[1] == selected for write in stash.writes[writes:])
    assert identity_state(config) == original
    with FaceDatabase(config.database) as state:
        history = state.connection.execute('SELECT * FROM runs ORDER BY started DESC LIMIT 1').fetchone()
        assert json.loads(history['config'])['invocation'] == {'command': 'portraits', 'performer_ids': [selected]}
        assert json.loads(history['selection']) == (scene_ids or ['1', '2'])
        targets = json.loads(state.connection.execute(
            "SELECT data FROM progress WHERE run=? AND phase='portrait_selection'", (history['id'],)).fetchone()[0])
        assert targets['target_ids'] == ([selected] if expected else [])
        expected_scenes = (scene_ids or ['1', '2']) if expected else []
        assert [s['scene'] for s in targets['selection']] == expected_scenes


def test_repeated_performer_ids_select_each_person_once_and_unknown_ids_never_broaden_selection(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    first = stash.links['1'][-1]
    original = services.backend
    def other(*args):
        backend = original(*args)
        backend.vector = (0., 1., 0., 0.)
        return backend
    services.backend = other
    assert run(config, services, '2') == 0
    second = stash.links['2'][-1]
    assert first != second
    forbid_recognition(monkeypatch, services, stash)
    captured = []
    services.metadata = lambda *a, **kw: captured.append(a[6]['performer']) or {}
    assert refresh(config, services, '--performer-id', first, '--performer-id', second,
                   '--performer-id', first, '--performer-id', '999') == 0
    assert sorted(captured) == sorted([first, second])
    captured.clear()
    assert refresh(config, services, '--performer-id', '999', '--scene-id', '1') == 0
    assert captured == [] and last_counts(config)['portrait_people_selected'] == 0


@pytest.mark.parametrize('filtered', [False, True])
@pytest.mark.parametrize('excluded', ['scope', 'namespace', 'unpublished', 'partial', 'rejected', 'historical', 'alias', 'pending_merge'])
def test_only_complete_current_owned_identity_memberships_are_targets(tmp_path, monkeypatch, excluded, filtered):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    with FaceDatabase(config.database) as state:
        db = state.connection
        person = db.execute('SELECT * FROM people').fetchone()
        if excluded == 'scope':
            db.execute("UPDATE people SET scope='other'")
        elif excluded == 'namespace':
            db.execute("UPDATE people SET namespace='other'")
        elif excluded == 'unpublished':
            db.execute('UPDATE people SET performer=NULL')
        elif excluded == 'partial':
            db.execute("UPDATE scans SET status='partial'")
        elif excluded == 'rejected':
            db.execute("UPDATE observations SET rejected='quality'")
        elif excluded == 'historical':
            db.execute('DELETE FROM members')
        else:
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       ('other', person['scope'], person['namespace'], person['created'] + 1, '999', 'other-token'))
            source, target = (person['id'], 'other') if excluded == 'alias' else ('other', person['id'])
            db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
                       (source, target, 1., '{}', 'confirmed' if excluded == 'alias' else 'pending'))
        db.commit()
    forbid_recognition(monkeypatch, services, stash)
    services.metadata = services.assets = services.backend = forbidden
    writes = list(stash.writes)
    assert refresh(config, services, *(['--performer-id', performer] if filtered else [])) == (1 if excluded == 'pending_merge' else 0)
    if excluded == 'pending_merge':
        assert last_counts(config)['metadata_withheld'] == 1
    assert last_counts(config)['portrait_people_selected'] == 0 and stash.writes == writes


def test_empty_database_is_a_recorded_noop_without_models_or_scene_enumeration(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    forbid_recognition(monkeypatch, services, stash)
    services.assets = services.backend = services.source = forbidden
    assert refresh(config, services) == 0
    assert last_counts(config)['portrait_people_selected'] == 0
    assert not stash.writes and not stash.reads and stash.closed


def test_disabling_portraits_removes_only_owned_images_without_loading_assets(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    config = change(config, enabled=False)
    forbid_recognition(monkeypatch, services, stash)
    services.assets = services.backend = services.source = forbidden
    assert refresh(config, services) == 0
    assert performer not in stash.images and last_counts(config)['portraits_removed'] == 1
    assert stash.links['1'] == ['77', performer]


@pytest.mark.parametrize('manual', [False, True])
def test_refresh_preserves_manual_images_and_rejects_changed_ownership(tmp_path, monkeypatch, manual):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    performer = stash.links['1'][-1]
    if manual:
        stash.images[performer] = b'manual replacement'
    else:
        stash.profiles[performer]['details'] = 'ownership removed'
    writes, images = list(stash.writes), dict(stash.images)
    config = change(config, margin=.5)
    forbid_recognition(monkeypatch, services, stash)
    assert refresh(config, services) == (0 if manual else 1)
    assert stash.writes == writes and stash.images == images


def test_refresh_retries_unresolved_image_only_with_explicit_flag(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    original, attempts = lose_images(stash)
    assert run(config, services, '1') == 1
    forbid_recognition(monkeypatch, services, stash)
    services.assets = services.backend = services.decoder = forbidden
    assert refresh(config, services) == 1 and len(attempts) == 1
    stash.update_person = original
    assert refresh(config, services, '--retry-unresolved-images') == 0
    assert last_counts(config)['portraits_retried'] == 1
    assert stash.images[stash.links['1'][-1]] == attempts[0][1]


def test_video_refresh_uses_portrait_decoding_and_closes_shared_models(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    config = change(config, time_offsets=[.5])
    forbid_recognition(monkeypatch, services, stash)
    backends = []
    def backend(*args):
        value = Backend()
        backends.append(value)
        return value
    services.backend = backend
    Decoder.instances.clear()
    assert refresh(config, services) == 0
    assert len(backends) == 1 and backends[0].closed
    assert Decoder.instances and all(d.closed for d in Decoder.instances)
    assert all(d.calls == [.5, 2.5] for d in Decoder.instances)


def test_new_quality_measurement_loads_assets_once_without_face_recognition(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch, enabled=False)
    assert run(config, services, '1') == 0
    config = change(config, enabled=True)
    forbid_recognition(monkeypatch, services, stash)
    services.backend = forbidden
    assets, measurements = [], []
    def prepare(*args, **kwargs):
        assets.append(True)
        return {'faceocc': 'verified-quality-path'}
    class Quality(Visibility):
        def __init__(self, config, paths):
            assert paths['faceocc'] == 'verified-quality-path'
            assert paths['faceocc'] == 'verified-quality-path'
            super().__init__()
            measurements.append(self)
    services.assets, services.visibility = prepare, Quality
    assert refresh(config, services) == 0
    assert assets == [True] and len(measurements) == 1
    assert measurements[0].calls == 3 and measurements[0].closed
    assert last_counts(config)['portraits_filled'] == 1


def test_one_failed_identity_does_not_stop_other_portraits(tmp_path, monkeypatch):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    original = services.backend
    def other(*args):
        backend = original(*args)
        backend.vector = (0., 1., 0., 0.)
        return backend
    services.backend = other
    assert run(config, services, '2') == 0
    with FaceDatabase(config.database) as state:
        first = state.connection.execute('SELECT performer FROM people ORDER BY id LIMIT 1').fetchone()[0]
    stash.profiles[first]['details'] = 'ownership removed'
    config = change(config, margin=.5)
    forbid_recognition(monkeypatch, services, stash)
    services.backend = services.assets = forbidden
    assert refresh(config, services) == 1
    counts = last_counts(config)
    assert counts['portrait_people_selected'] == 2 and counts['portraits_failed'] == 1
    assert counts['portraits_updated'] == 1


def test_interruption_closes_models_and_records_history(tmp_path, monkeypatch):
    config, stash, services = video_setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    config = change(config, time_offsets=[.5])
    forbid_recognition(monkeypatch, services, stash)
    model = Backend()
    services.backend = lambda *a: model
    model.detect = lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        refresh(config, services)
    assert model.closed and stash.closed
    with FaceDatabase(config.database) as state:
        assert state.connection.execute('SELECT status FROM runs ORDER BY started DESC LIMIT 1').fetchone()[0] == 'interrupted'


def test_portrait_progress_has_people_counter_without_scan_or_rebuild_phases(tmp_path, monkeypatch, capsys):
    config, stash, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    capsys.readouterr()
    forbid_recognition(monkeypatch, services, stash)
    args = parser().parse_args(['portraits', '--progress-style', 'compact'])
    assert execute(args, config, services=services) == 0
    output = capsys.readouterr()
    assert 'Portraits: inspected 1/1 person' in output.err
    assert 'Recognition VTT' not in output.err and 'identity rebuild' not in output.err

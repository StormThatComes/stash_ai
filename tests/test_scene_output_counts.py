"""Full scene output and offline retirement of historical count caps."""
from copy import deepcopy
from dataclasses import asdict
import sqlite3

import numpy as np
import pytest

from stash_ai.db.checks import check_database
from stash_ai.db.update import update_database
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.inference import sprites
from stash_ai.models import SOURCES, Vocabulary
from stash_ai.storage.codec import decode, encode
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from stash_ai.storage.snapshots import FORMAT, FORMAT_KEY, decode as decode_snapshot
from stash_ai.storage.store import StateStore, scope_for
from tests.helpers.evidence import paired_records, empty_policies
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_merge')


def old_envelope(snapshot):
    """Mimic the actual v6 writer, outside the current snapshot contract."""
    saved = deepcopy(snapshot)
    def capped(item):
        options = item.get('build_options')
        if options is not None:
            if 'settings' in options:
                options['settings']['MAX_MARKERS_PER_SCENE'] = 2
            else:
                options['max_tags'] = 2
        for hit in item.get('scene_evidence', []):
            capped(hit)
    for item in saved['contributions']:
        capped(item)
    return encode({FORMAT_KEY: 6, 'snapshot': saved, 'proofs': []})


@pytest.mark.parametrize('version', [23, 24])
def test_upgrade_removes_caps_from_all_publication_states_and_scopes(pipeline, version):
    p = pipeline
    assert app.main([*p.args, '--db-init']) == 0
    scope = scope_for(app.STASH_URL)
    with StateStore(p.path, scope) as state:
        snapshots = {kind: state.snapshot('1', kind)[0] for kind in ('scene', 'markers')}
        records = {kind: state.records('1', kind)['wd'] for kind in ('scene', 'markers')}
    # Include a shared-scene contract nested inside a marker publication.
    hit = snapshots['scene']['contributions'][0]
    hit['build_options'] = dict(policy=asdict(app.TEMPORAL_POLICY), scene_thresholds={},
                               scene_exclusions={source: [] for source in SOURCES})
    snapshots['markers']['contributions'][0]['scene_evidence'] = [deepcopy(hit)]
    with sqlite3.connect(p.path) as db:
        init_before = db.execute('SELECT * FROM db_init_scenes').fetchall()
    for target in (scope, 'second-stash'):
        with StateStore(p.path, target) as state:
            if target != scope:
                state.put('1', 'wd', records)
            with state.connection as db:
                for kind, snapshot in snapshots.items():
                    blob = old_envelope(snapshot)
                    db.execute('INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?)',
                               (target, '1', kind, blob, blob))
                    db.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)',
                               (target, '1', kind, 'saved-operation', blob))
    with sqlite3.connect(p.path) as db:
        db.execute(f'PRAGMA user_version={version}')
        evidence = db.execute('SELECT * FROM results ORDER BY scope,kind').fetchall()
    before = p.path.read_bytes(), p.path.stat().st_mtime_ns
    assert update_database(p.path, dry_run=True) == (version, LATEST_SCHEMA_VERSION)
    assert (p.path.read_bytes(), p.path.stat().st_mtime_ns) == before
    assert update_database(p.path) == (version, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(p.path) as db:
        assert db.execute('SELECT * FROM results ORDER BY scope,kind').fetchall() == evidence
        assert db.execute('SELECT * FROM db_init_scenes').fetchall() == init_before
        for kind, published, pending in db.execute('SELECT kind,published,pending FROM snapshots'):
            for blob in (published, pending):
                assert decode(blob)[FORMAT_KEY] == FORMAT
                assert decode_snapshot(blob) == snapshots[kind]
        for kind, blob in db.execute('SELECT kind,data FROM recovery_publications'):
            assert decode_snapshot(blob) == snapshots[kind]
    assert check_database(p.path, scope)['ok']


def test_cached_merge_republishes_all_tags_and_ranges_after_old_caps(pipeline, monkeypatch):
    p = pipeline
    labels = [f'test_label_{i:02}' for i in range(35)]
    p.vocab = Vocabulary(labels, {label: 'General' for label in labels})
    monkeypatch.setattr(app.services, 'load_vocabulary', lambda _: p.vocab)
    from tests.helpers.inference import IndexedTagger
    def load(settings, *args):
        model = IndexedTagger(settings, p.scores['wd'], labels)
        p.loaded.append(model)
        return model
    monkeypatch.setattr(app.services, 'load_tagger', load)
    p.scores['wd'] = np.zeros((10, len(labels)), np.float32)
    p.scores['wd'][[0, 1, 4, 5, 8, 9], :] = .9
    assert app.main(p.args) == 0
    scope = scope_for(app.STASH_URL)
    with StateStore(p.path, scope) as state:
        snapshots = {kind: state.snapshot('1', kind)[0] for kind in ('scene', 'markers')}
    # Recreate a previous publication limited to two tags and two ranges.
    p.stash.scene['tags'] = [tag for tag in p.stash.scene['tags'] if tag['name'] in {'ai:' + label for label in labels[:2]}]
    p.stash.scene['scene_markers'] = p.stash.scene['scene_markers'][:2]
    for kind, snapshot in snapshots.items():
        snapshot['contributions'] = snapshot['contributions'][:2]
        if kind == 'scene':
            snapshot['owned_ids'] = [tag['id'] for tag in p.stash.scene['tags']]
    with sqlite3.connect(p.path) as db:
        for kind, snapshot in snapshots.items():
            db.execute('UPDATE snapshots SET published=? WHERE kind=?', (old_envelope(snapshot), kind))
        db.execute('PRAGMA user_version=24')
        evidence = db.execute('SELECT data FROM results ORDER BY kind').fetchall()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('count-cap retirement loaded a model'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *args: pytest.fail('cached merge read media'))
    assert update_database(p.path) == (24, LATEST_SCHEMA_VERSION)
    assert app.main([*p.args, '--merge-only']) == 0
    assert {tag['name'] for tag in p.stash.scene['tags']} == {'ai:' + label for label in labels}
    assert len(p.stash.scene['scene_markers']) == 105
    for label in labels:
        assert [(m['seconds'], m['end_seconds']) for m in p.stash.scene['scene_markers']
                if m['primary_tag']['name'] == 'ai:' + label] == [(0, 2), (4, 6), (8, 10)]
    with sqlite3.connect(p.path) as db:
        assert db.execute('SELECT data FROM results ORDER BY kind').fetchall() == evidence


def test_shared_scene_qualification_keeps_all_labels_and_occurrence_points(tmp_path, monkeypatch):
    from stash_ai.evidence.pipeline import evidence
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', TemporalPolicy(shared_marker_frames=False))
    scenes, markers = paired_records({'wd': [.85, 0], 'dtq': [0, .86]})
    labels = [f'shared_label_{i:02}' for i in range(30)]
    with StateStore(tmp_path/'state.db', 's') as state:
        for source in scenes:
            scene, marker = deepcopy(scenes[source]), deepcopy(markers[source])
            scene['items'] = [{**scene['items'][0], 'tag': label} for label in labels]
            for row in scene['frames']:
                row['tags'] = [(label, score) for _, score in row['tags'] for label in labels]
            for row in marker['items']:
                row['tags'] = [(label, score) for _, score in row['tags'] for label in labels]
            state.put('1', source, {'scene': scene, 'markers': marker})
        hits, points = evidence(app, state, '1', empty_policies(), empty_policies())
    assert {hit['tag'] for hit in hits} == set(labels)
    assert all(hit['derived'] == 'shared_scene_frames' for hit in hits)
    assert len(points) == len(labels)
    assert all(point['derived'] == 'occurrence' for point in points)


@pytest.mark.parametrize('kind', ['scene', 'markers'])
@pytest.mark.parametrize('invalid', [-1, True, '2', None])
def test_invalid_historical_count_cap_rolls_back_upgrade(tmp_path, kind, invalid):
    from stash_ai.evidence.temporal.policy import build_options
    path = tmp_path/'state.db'
    with StateStore(path, 's'):
        pass
    policies = {source: [] for source in SOURCES}
    options = (dict(policy=asdict(app.TEMPORAL_POLICY), max_tags=invalid,
                    scene_thresholds={}, scene_exclusions=policies) if kind == 'scene' else
               {**build_options(app), 'core_exclusions': policies, 'tag_exclusions': policies})
    if kind == 'markers':
        options['settings']['MAX_MARKERS_PER_SCENE'] = invalid
    snapshot = {'contributions': [{'build_options': options}], 'owned_ids': ['saved-owner']}
    blob = encode({FORMAT_KEY: 6, 'snapshot': snapshot, 'proofs': []})
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO snapshots VALUES (?,?,?,?,?)', ('s', '1', kind, blob, blob))
        db.execute('PRAGMA user_version=24')
    with sqlite3.connect(path) as db:
        before = db.execute('SELECT * FROM snapshots').fetchall()
    with pytest.raises(ValueError, match='historical scene .* limit'):
        update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT * FROM snapshots').fetchall() == before
        assert db.execute('PRAGMA user_version').fetchone() == (24,)

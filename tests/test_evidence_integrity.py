"""Stricter validation repairs provable duplicates without another inference pass."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import empty_policies, segment, inference_record
from stash_ai.storage.store import StateStore
from stash_ai.storage.snapshots import validate_policy
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.formats import validate_current
from stash_ai.db.update import update_database
from stash_ai.db.checks import check_database

pytestmark = pytest.mark.usefixtures('reset_merge')


@pytest.mark.parametrize('name,value', [
    ('MARKER_MIN_HITS', '2'), ('MARKER_MIN_HITS', True), ('MARKER_MIN_HITS', 0),
    ('MARKER_MAX_TAGS_PER_FRAME', -1), ('MARKER_MAX_TAGS_PER_FRAME', .5),
    ('MARKER_MERGE_GAP_SECONDS', float('nan')), ('MARKER_MERGE_GAP_SECONDS', float('inf')),
    ('MARKER_MERGE_GAP_SECONDS', -1), ('MARKER_MERGE_GAP_SECONDS', True)])
def test_frozen_numeric_values_are_validated_at_all_boundaries(name, value):
    from stash_ai.evidence.source_markers import marker_contributions
    from stash_ai.db.validation.publications import validate_snapshot
    record = segment('wd', [[('core', .95)], [('core', .95)]])
    spans = marker_contributions(app, {'wd': record}, empty_policies(), tag_policies=empty_policies())
    spans[0]['build_options']['settings'][name] = value
    with pytest.raises(ValueError, match='Invalid frozen marker setting'):
        validate_policy(spans[0])
    with pytest.raises(ValueError, match='Invalid frozen marker setting'):
        validate_snapshot({'contributions': spans}, 'markers')


@pytest.mark.parametrize('threshold', [True, '0.8', float('nan'), float('inf'), -.1, 1.1])
def test_frozen_thresholds_are_probabilities(threshold):
    from stash_ai.evidence.temporal.policy import build_options
    options = {**build_options(app), 'core_exclusions': {s: [] for s in empty_policies()},
               'tag_exclusions': {s: [] for s in empty_policies()}}
    options['marker_thresholds']['signature'] = threshold
    with pytest.raises(ValueError, match='Invalid frozen thresholds'):
        validate_policy({'build_options': options})


def record_with_duplicate():
    record = inference_record(mode='scene', items=[dict(tag='core', score=.95)])
    record['items'].append(deepcopy(record['items'][0]))
    return record


def test_duplicate_aggregates_are_rejected_on_active_writes(tmp_path):
    record = record_with_duplicate()
    with pytest.raises(ValueError, match='Duplicate scene aggregates'):
        validate_current(record)
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        with pytest.raises(ValueError, match='Duplicate scene aggregates'):
            store.put('1', 'wd', {'scene': record})


def test_offline_duplicate_repair_preserves_all_pinned_publication_states(tmp_path):
    path = tmp_path/'state.sqlite3'
    record = record_with_duplicate()
    old = marker_segment_signature(record)
    with StateStore(path, 'scope') as store:
        clean = {**record, 'items': record['items'][:1]}
        store.put('1', 'wd', {'scene': clean})
        store.connection.execute("UPDATE results SET data=? WHERE kind='scene'", (store.pack(record),))
        store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
                                 ('scope', '1', 'wd', 'scene', old, store.pack(record)))
        hit = {**record['items'][0], 'source': 'wd', 'meta': record['meta'],
               'evidence_signature': old, 'scene_context': {'wd': old},
               'build_options': dict(policy=asdict(app.TEMPORAL_POLICY), scene_thresholds={old: .4},
                                     scene_exclusions={s: [] for s in empty_policies()})}
        snapshot = {'contributions': [hit], 'owned_ids': ['17']}
        store.publish('1', {'scene': snapshot})
        store.prepare('1', {'scene': snapshot})
        store.connection.execute('INSERT INTO recovery_publications VALUES (?,?,?,?,?)',
                                 ('scope', '1', 'scene', 1, store.pack_snapshot(snapshot)))
        store.connection.commit()
    update_database(path)
    with StateStore(path, 'scope') as store:
        clean = store.model_record('1', 'scene', 'wd')
        assert len(clean['items']) == 1
        new = marker_segment_signature(clean)
        assert clean['meta']['inference_signature'] == record['meta']['inference_signature']
        published, pending = store.snapshot('1', 'scene')
        from stash_ai.storage.recovery import publications
        for snapshot in (published, pending, *publications(store, '1', 'scene')):
            hit = snapshot['contributions'][0]
            assert hit['evidence_signature'] == new
            assert hit['scene_context'] == {'wd': new}
            assert hit['build_options']['scene_thresholds'] == {new: .4}
            assert snapshot['owned_ids'] == ['17']
    assert check_database(path, 'scope')['ok']


def test_conflicting_duplicate_repair_does_not_guess_scores(tmp_path):
    path = tmp_path/'state.sqlite3'
    record = record_with_duplicate()
    record['items'][1]['score'] = .5
    with StateStore(path, 'scope') as store:
        clean = {**record, 'items': record['items'][:1]}
        store.put('1', 'wd', {'scene': clean})
        store.connection.execute('UPDATE results SET data=?', (store.pack(record),))
        store.connection.commit()
    with sqlite3.connect(path) as db:
        before = db.execute('SELECT data FROM results').fetchall()
    with pytest.raises(ValueError, match='Conflicting scene aggregates'):
        update_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT data FROM results').fetchall() == before


def test_checkpoint_geometry_reset_is_scoped_to_partial_predictions(tmp_path):
    from stash_ai.inference.checkpoints import Checkpoint
    from stash_ai.inference.prediction import _restore_prediction_state
    from stash_ai.domain import Cue, FramePrediction
    from stash_ai.runtime.settings import RuntimeSettings
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    context = RuntimeSettings(APPLY_SCENE_TAGS=False, TEMPORAL_POLICY=TemporalPolicy(occurrence_markers=False))
    settings = SimpleNamespace(scene_threshold=.8, marker_threshold=.8, scene_cache_floor=None, marker_cache_floor=None)
    with StateStore(tmp_path/'state.sqlite3', 'scope') as store:
        store.put('2', 'dtq', {'markers': segment('dtq', [[('core', .9)]])})
        before = store.connection.execute('SELECT data FROM results').fetchone()[0]
        cp = Checkpoint(store, '1', 'wd', 'signature', 1, 1, 1)
        cp.record(1, np.array([[.95]], np.float32),
                  [FramePrediction(Cue(100, 101, '', (0,0,1,1)), (('core', .95),))])
        state = _restore_prediction_state(context, SimpleNamespace(settings=settings, labels=['core']),
                                          [Cue(0, 1, '', (0,0,1,1))], checkpoint=cp, report=None)
        assert state.processed == 0 and state.frame_predictions == []
        assert not store.connection.execute('SELECT * FROM checkpoints').fetchall()
        assert store.connection.execute('SELECT data FROM results').fetchone()[0] == before

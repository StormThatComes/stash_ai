"""Offline checks reject evidence that cannot be resumed or replayed."""
import sqlite3

import numpy as np
import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import segment, span, inject_records, empty_policies, inference_record
from stash_ai.storage.store import StateStore
from stash_ai.db.checks import check_database
from stash_ai.evidence import pipeline
from stash_ai.evidence.temporal.statistics import SceneAccumulator
from stash_ai.domain import Cue, FramePrediction


pytestmark = pytest.mark.usefixtures('reset_temporal')


def codes(path):
    return {issue['code'] for issue in check_database(path, 's')['errors']}


@pytest.mark.parametrize('damage', ['duplicate_label', 'duplicate_time', 'overlap'])
def test_writer_and_offline_check_reject_inflated_frame_evidence(tmp_path, damage):
    path = tmp_path/'state.db'
    record = segment('wd', [[('a', .95)], [('a', .95)]])
    if damage == 'duplicate_label':
        record['items'][0]['tags'].append(('a', .95))
    else:
        record['items'][1]['start'] = 0 if damage == 'duplicate_time' else .5
    with StateStore(path, 's') as state:
        with pytest.raises(ValueError):
            state.put('1', 'wd', {'markers': record})
        inject_records(state, '1', 'wd', {'markers': record})
    assert 'result_payload_invalid' in codes(path)


@pytest.mark.parametrize('damage', ['shape', 'processed', 'label', 'counter', 'nan'])
def test_invalid_sparse_checkpoint_analysis_is_diagnosed(tmp_path, damage):
    from stash_ai.inference.checkpoints import Checkpoint
    path = tmp_path/'state.db'
    cue = Cue(0, 1, '', (0, 0, 1, 1))
    accumulator = SceneAccumulator(.4, app.TEMPORAL_POLICY)
    accumulator.add_sparse([(0, .9)], cue)
    analysis = accumulator.state()
    if damage == 'shape': analysis = {}
    elif damage == 'processed': analysis['frames'] = 2
    elif damage == 'label': analysis['items'][1] = analysis['items'].pop(0)
    elif damage == 'counter': analysis['items'][0]['hits'] = -1
    else: analysis['covered'] = float('nan')
    with StateStore(path, 's') as state:
        # JSON correctly refuses NaN; inject it outside the normal writer.
        if damage == 'nan':
            analysis['covered'] = 0
        cp = Checkpoint(state, '1', 'wd', 'sig', 2, 1, 1)
        cp.record(1, np.array([[.9]], np.float32), [FramePrediction(cue, (('a', .9),))], analysis=analysis)
        if damage == 'nan':
            import json, zlib
            analysis['covered'] = float('nan')
            with state.connection:
                state.connection.execute('UPDATE checkpoint_analysis SET data=?',
                    (zlib.compress(json.dumps(analysis).encode()),))
    assert 'checkpoint_analysis_progress' in codes(path)
    with StateStore(path, 's') as state:
        with pytest.raises(ValueError):
            Checkpoint(state, '1', 'wd', 'sig', 2, 1, 1).restore()


@pytest.mark.parametrize('kind,contribution', [
    ('markers', {'unexpected': 1}),
    ('markers', {**span('wd', 0, 2, 'a'), 'end': -1}),
    ('scene', {'tag': 'a', 'source': 'invalid', 'score': .9, 'meta': {}}),
])
def test_unusable_publication_contributions_are_diagnosed(tmp_path, kind, contribution):
    path = tmp_path/'state.db'
    with StateStore(path, 's') as state:
        state.publish('1', {kind: {'contributions': [contribution]}})
    assert 'state_payload_invalid' in codes(path)


def test_missing_proof_and_wrong_scope_are_diagnosed_but_archived_proof_is_usable(tmp_path):
    path = tmp_path/'state.db'
    record = segment('wd', [[('a', .95)], [('a', .95)]])
    with StateStore(path, 's') as state:
        state.put('1', 'wd', {'markers': record})
        _, spans = pipeline.evidence(app, state, '1', empty_policies(), empty_policies())
        state.publish('1', {'markers': {'contributions': spans}})
    assert not codes(path)
    with StateStore(path, 's') as state:
        state.put('1', 'wd', {'markers': segment('wd', [[('b', .95)]], identity='new')})
    assert not codes(path)  # The publication still pins the archived old proof.
    with sqlite3.connect(path) as db:
        db.execute("UPDATE inference_archive SET scope='elsewhere'")
    assert 'snapshot_proof_missing' in codes(path)


def test_db_update_repairs_valid_shaped_mismatched_metadata(tmp_path):
    from stash_ai.db.update import update_database
    from stash_ai.storage.metadata import encode, summarize
    path = tmp_path/'state.db'
    record = inference_record(mode='scene', items=[{'tag': 'a', 'score': .9}], meta={'scene_score_floor': .1})
    with StateStore(path, 's') as state:
        state.put('1', 'wd', {'scene': record})
        metadata = summarize(record)
        metadata['meta']['scene_score_floor'] = .2
        with state.connection:
            state.connection.execute('UPDATE result_metadata SET data=?', (encode(metadata),))
    assert 'result_metadata_mismatch' in codes(path)
    update_database(path)
    assert not codes(path)

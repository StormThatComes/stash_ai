"""Compression compatibility, checkpoint profiling and atomic failure recovery."""
import json
import sqlite3
import zlib

import numpy as np
import pytest

from stash_ai.domain import Cue, FramePrediction
from stash_ai.inference import checkpoints, prediction
from stash_ai.models import ModelSettings, WD
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.storage import codec
from stash_ai.storage.store import StateStore
from tests.helpers.inference import IndexedTagger
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_progress')


@pytest.mark.parametrize('level', [6, 9])
def test_codec_reads_existing_zlib_blobs_and_preserves_decoded_values(level):
    value = dict(tags=[['unicode-ą', .123456789]], frames=[dict(start=0., end=1.)])
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                     sort_keys=True, separators=(',', ':')).encode('utf-8')
    assert codec.decode(zlib.compress(raw, level)) == value
    encoded = codec.encode(value)
    assert encoded == zlib.compress(raw, level=1)
    assert zlib.decompress(encoded) == raw


def frame(index):
    return FramePrediction(Cue(index, index + 1, 'sprite', (0, 0, 1, 1)), [('tag', .5)])


@pytest.mark.parametrize('failure', ['encode', 'sqlite'])
def test_checkpoint_failure_preserves_durable_prefix_and_pending_retry(tmp_path, failure):
    with StateStore(tmp_path / 'state.sqlite3', 'fixture') as store:
        checkpoint = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 4, 1)
        scores = np.array([[.5]], dtype=np.float32)
        checkpoint.record(1, scores, [frame(0)])
        checkpoint.flush()
        checkpoint.record(2, scores, [frame(1)])
        if failure == 'encode':
            checkpoint.analysis = {'invalid': float('nan')}
            error = ValueError
        else:
            store.connection.execute('''CREATE TRIGGER reject_checkpoint_frame
                BEFORE INSERT ON checkpoint_frames BEGIN
                SELECT RAISE(ABORT, 'checkpoint write failed'); END''')
            error = sqlite3.IntegrityError
        with pytest.raises(error):
            checkpoint.flush()
        assert checkpoint.saved_processed == 1
        assert len(checkpoint.pending) == 1
        restored = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 4, 1)
        assert restored.restore()[0] == 1
        assert not store.connection.in_transaction
        if failure == 'encode':
            checkpoint.analysis = None
        else:
            store.connection.execute('DROP TRIGGER reject_checkpoint_frame')
        checkpoint.flush()
        processed, top, frames = restored.restore()
        assert processed == 2 and len(frames) == 2
        np.testing.assert_array_equal(top, scores)


@pytest.mark.parametrize('old_level', [6, 9])
def test_checkpoint_restores_old_compression_and_profiles_all_payloads(pipeline, tmp_path, old_level):
    report = RunReport(write=lambda _: None)
    report.profile = ProfileMetrics()
    config = ModelSettings(WD, tmp_path / 'model', .4, .65, 4)
    model = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
    with StateStore(tmp_path / 'state.sqlite3', 'fixture') as store:
        store.report = report
        checkpoint = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 10, 2)
        with report.model('wd'):
            report.work('fresh')
            expected = prediction.predict_scene(app, model, pipeline.stash, {}, pipeline.cues,
                                                 checkpoint=checkpoint, prefetch=1, report=report)
        metrics = {row['name']: row for row in report.profile.export()}
        for name in ('checkpoint_write', 'checkpoint_prepare', 'checkpoint_json_encode',
                     'checkpoint_compress', 'checkpoint_sqlite_write'):
            assert metrics[name]['calls'] > 0
            assert metrics[name]['model'] == 'wd' and metrics[name]['work'] == 'fresh'
        assert metrics['checkpoint_compress']['calls'] == 3
        assert metrics['checkpoint_json_encode']['calls'] == 2
        assert 'state_compress' not in metrics
        with store.connection:
            for table, column in (('checkpoints', 'top_scores'), ('checkpoint_frames', 'data'),
                                  ('checkpoint_analysis', 'data')):
                blob = store.connection.execute(f'SELECT {column} FROM {table}').fetchone()[0]
                store.connection.execute(f'UPDATE {table} SET {column}=?',
                    (zlib.compress(zlib.decompress(blob), old_level),))
        restored = checkpoints.Checkpoint(store, '1', 'wd', 'signature', 10, 2)
        resumed = IndexedTagger(config, pipeline.scores['wd'], pipeline.vocab.labels)
        actual = prediction.predict_scene(app, resumed, pipeline.stash, {}, pipeline.cues,
                                          checkpoint=restored, prefetch=1)
        assert actual == expected and resumed.seen == []

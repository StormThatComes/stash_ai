"""Ready evidence, original failures and complete checkpoint prefixes survive faults."""
from copy import deepcopy
from threading import BoundedSemaphore, Event, Thread
from time import monotonic
import sqlite3
import zlib

import numpy as np
import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.cleanup import FastStash
from tests.test_merged_model import model_run, run_model, assert_inference_workers_closed
from stash_ai.storage.store import scope_for

pytestmark = pytest.mark.usefixtures('reset_progress')


def test_load_failure_preserves_later_ready_scene_without_predict_or_retry(pipeline, monkeypatch):
    stash = FastStash()
    stash.scenes['2'] = dict(deepcopy(stash.scene), id='2')
    monkeypatch.setattr(app.services, 'create_stash_client', lambda _: stash)
    args = ['--models', 'wd', '--no-hierarchy', '--state-db', str(pipeline.path)]
    assert app.main([*args, '--scene-id', '2']) == 0
    with sqlite3.connect(pipeline.path) as db:
        before = db.execute("SELECT data FROM results WHERE scene='2' ORDER BY kind").fetchall()
    loads = []
    def fail(*args):
        loads.append(1)
        raise RuntimeError('missing backend')
    monkeypatch.setattr(app.services, 'load_tagger', fail)
    assert app.main(args) == 1
    with sqlite3.connect(pipeline.path) as db:
        assert not db.execute("SELECT 1 FROM refresh_requests WHERE scene='2'").fetchone()
        assert db.execute("SELECT data FROM results WHERE scene='2' ORDER BY kind").fetchall() == before
    assert app.main([*args, '--scene-id', '2']) == 0
    assert loads == [1]


@pytest.mark.parametrize('prefetch', [0, 2])
def test_cleanup_failure_keeps_original_interrupt_and_closes_owner(model_run, monkeypatch, prefetch):
    context = model_run
    context.args.prefetch_batches = prefetch
    load = app.services.load_tagger
    def failing_backend(*args):
        model = load(*args)
        def predict(images):
            raise KeyboardInterrupt('original interruption')
        def close():
            model.closed = True
            raise RuntimeError('secondary close failure')
        model.predict_batch, model.close = predict, close
        return model
    monkeypatch.setattr(app.services, 'load_tagger', failing_backend)
    with pytest.raises(KeyboardInterrupt, match='original interruption') as caught:
        run_model(context)
    assert any('secondary close failure' in note for note in caught.value.__notes__)
    assert context.pipeline.loaded[0].closed
    assert_inference_workers_closed()


@pytest.mark.parametrize('failure', ['construct', 'start'])
def test_failed_read_thread_returns_slot(monkeypatch, failure):
    from stash_ai.stash_api import deadlines
    slots = BoundedSemaphore(1)
    monkeypatch.setattr(deadlines, '_slots', slots)
    class BrokenThread:
        def __init__(self, *args, **kwargs):
            if failure == 'construct':
                raise RuntimeError('thread failure')
        def start(self):
            raise RuntimeError('thread failure')
    monkeypatch.setattr(deadlines, 'Thread', BrokenThread)
    with pytest.raises(RuntimeError, match='thread failure'):
        deadlines.call_with_deadline(lambda: None, monotonic() + 1)
    assert slots.acquire(blocking=False)
    slots.release()


def test_interrupted_start_retains_capacity_until_reader_finishes(monkeypatch):
    from stash_ai.stash_api import deadlines
    slots, finish = BoundedSemaphore(1), Event()
    monkeypatch.setattr(deadlines, '_slots', slots)
    workers = []
    class InterruptedStart(Thread):
        def start(self):
            super().start()
            workers.append(self)
            raise KeyboardInterrupt('started')
    monkeypatch.setattr(deadlines, 'Thread', InterruptedStart)
    try:
        with pytest.raises(KeyboardInterrupt, match='started'):
            deadlines.call_with_deadline(lambda: finish.wait(2), monotonic() + 2)
        assert not slots.acquire(blocking=False)
    finally:
        finish.set()
        for worker in workers:
            worker.join(2)
    assert slots.acquire(blocking=False)
    slots.release()


@pytest.mark.parametrize('aggregation,keep,rows', [('max', 1, 1), ('topk_mean', 2, 2)])
def test_checkpoint_contract_checks_aggregate_and_chunk_prefix(tmp_path, aggregation, keep, rows):
    from stash_ai.inference.checkpoints import Checkpoint
    from stash_ai.storage.store import StateStore
    from stash_ai.storage.codec import encode
    with StateStore(tmp_path/'state.sqlite3', 'fixture') as store:
        db = store.connection
        db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)',
                   ('fixture', '1', 'wd', 'sig', 2, 2, rows,
                    zlib.compress(np.array([[.8]] * rows, np.float32).tobytes())))
        frame = lambda n: dict(start=n, end=n+1, tags=[])
        db.execute('INSERT INTO checkpoint_frames VALUES (?,?,?,?,?)',
                   ('fixture', '1', 'wd', 2, encode([frame(0), frame(1)])))
        db.commit()
        cp = Checkpoint(store, '1', 'wd', 'sig', 2, 1, keep=keep)
        assert cp.restore()[0] == 2
        db.execute('UPDATE checkpoint_frames SET processed=1')
        db.commit()
        with pytest.raises(ValueError, match='chunk progress'):
            cp.restore()
        db.execute('UPDATE checkpoint_frames SET processed=2')
        db.execute('UPDATE checkpoints SET top_rows=0,top_scores=?', (zlib.compress(b''),))
        db.commit()
        with pytest.raises(ValueError, match='aggregate scores'):
            cp.restore()


def test_offline_check_uses_registered_aggregation_and_warns_when_missing(pipeline):
    from stash_ai.db.checks import check_database
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    with sqlite3.connect(pipeline.path) as db:
        row = db.execute('SELECT top_scores FROM checkpoints').fetchone()
        raw = np.frombuffer(zlib.decompress(row[0]), dtype='<f4').reshape(2, 2)
        db.execute('UPDATE checkpoints SET top_rows=1,top_scores=?',
                   (zlib.compress(raw[:1].tobytes()),))
    report = check_database(pipeline.path, scope_for(app.STASH_URL))
    assert {r['code']: r['count'] for r in report['errors']}['checkpoint_aggregate_incomplete'] == 1
    with sqlite3.connect(pipeline.path) as db:
        db.execute('DELETE FROM inference_identities')
    report = check_database(pipeline.path, scope_for(app.STASH_URL))
    assert any(row['code'] == 'checkpoint_aggregation_unverified' for row in report['warnings'])

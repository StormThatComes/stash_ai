"""Model lifetime, low-memory passes and diagnostics without GPU hardware."""
from dataclasses import replace
import json
import os
import sys
from types import SimpleNamespace

import pytest

from stash_ai.recognition import gpu_memory
from stash_ai.recognition.backend import LazyBackend
from stash_ai.recognition.config import plain, defaults, validate
from stash_ai.recognition.database import FaceDatabase
from stash_ai.runtime import shutdown
from stash_ai.runtime.report import RunReport
from tests.helpers.recognition import Backend, Visibility
from tests.test_recognition_run import setup, run
from tests.test_recognition_person_collection import Person


def configured(config, policy):
    data = plain(config.data)
    data['execution']['gpu_model_policy'] = policy
    return replace(config, data=data)


def test_unload_invalidates_old_lease_and_reload_stays_lazy():
    models = []
    def factory(*args):
        models.append(Backend())
        return models[-1]
    owner = LazyBackend(None, {}, factory)
    first = owner.borrow()
    owner.unload()
    owner.unload()
    assert models[0].closed and len(models) == 1 and not owner.closed
    with pytest.raises(RuntimeError, match='no longer active'):
        first.detect(None)
    second = owner.borrow()
    first.close()
    assert len(models) == 2 and second.active and not models[1].closed
    owner.close()
    assert models[1].closed
    with pytest.raises(RuntimeError, match='closed'):
        owner.borrow()


@pytest.mark.parametrize('policy', ['phase', 'sequential', 'retain'])
def test_face_models_and_visibility_lifetime_at_rebuild(tmp_path, monkeypatch, policy):
    import stash_ai.recognition.run as runner
    config, _, services, models = setup(tmp_path, monkeypatch)
    config = configured(config, policy)
    config.data['detection']['min_visible_face_fraction'] = .5
    visibility = []
    def load_visibility(*args):
        visibility.append(Visibility())
        return visibility[-1]
    services.visibility = load_visibility
    rebuild = runner.rebuild_selected
    def checked(*args, **kwargs):
        assert models and visibility
        assert all(m.closed for m in models + visibility) == (policy != 'retain')
        return rebuild(*args, **kwargs)
    monkeypatch.setattr(runner, 'rebuild_selected', checked)
    assert run(config, services, '1', '2') == 0
    assert len(models) == 1 and all(m.closed for m in models + visibility)
    with FaceDatabase(config.database) as state:
        row = state.connection.execute('SELECT id,summary FROM runs').fetchone()
        summary = json.loads(row['summary'])['gpu_memory']
        recorded = state.connection.execute("SELECT data FROM progress WHERE run=? AND phase='gpu_memory'", (row['id'],)).fetchone()
        assert json.loads(recorded[0]) == summary
        assert 'final:after_release' in summary['latest']
    # A completely cached run must never load inference models just to unload.
    monkeypatch.setattr(runner, 'rebuild_selected', rebuild)
    assert run(config, services, '1', '2') == 0
    assert len(models) == 1


@pytest.mark.parametrize('batch_size',[1,4])
@pytest.mark.parametrize('failure', [None, 'error', 'interrupt'])
def test_sequential_passes_never_overlap_and_resume_body_evidence(tmp_path, monkeypatch, failure,batch_size):
    config, _, services, faces = setup(tmp_path, monkeypatch)
    config = configured(config, 'sequential')
    config.data['sapiensid']['enabled'] = True
    config.data['execution']['sequential_scene_batch_size']=batch_size
    people = []
    face_factory = services.backend
    def load_face(*args):
        assert all(model.closed for model in people)
        return face_factory(*args)
    services.backend = load_face
    class Body(Person):
        def embed(self, image, face):
            assert all(model.closed for model in faces)
            if len(people) == 1 and self.calls == 1 and failure:
                self.calls += 1
                raise KeyboardInterrupt() if failure == 'interrupt' else RuntimeError('body failed')
            return super().embed(image, face)
    def load_body(*args, **kwargs):
        assert all(model.closed for model in faces)
        people.append(Body())
        return people[-1]
    monkeypatch.setattr('stash_ai.recognition.sapiensid_backend.SapiensIDFaces', load_body)
    if failure == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            run(config, services, '1')
    else:
        assert run(config, services, '1', '2') == (1 if failure else 0)
    assert all(m.closed for m in faces + people)
    if failure is None and batch_size>1:assert len(faces)==len(people)==1
    with FaceDatabase(config.database) as state:
        before = [tuple(r) for r in state.connection.execute('SELECT id,embedding FROM observations ORDER BY id')]
    loaded = len(faces)
    assert run(config, services, '1') == 0
    assert len(faces) == loaded
    with FaceDatabase(config.database) as state:
        after = [tuple(r) for r in state.connection.execute('SELECT id,embedding FROM observations ORDER BY id')]
        assert after == before
        assert state.connection.execute("SELECT COUNT(*) FROM person_evidence p JOIN observations o ON o.id=p.observation "
                                        "JOIN scans s ON s.id=o.scan WHERE s.scene='1'").fetchone()[0] == 3
        if failure is None:
            summary = json.loads(state.connection.execute('SELECT summary FROM runs ORDER BY started DESC').fetchone()[0])
            assert not any(key.startswith(('scene:', 'faces:')) for key in summary['gpu_memory']['latest'])


def monitor(monkeypatch, *, enabled=True):
    monkeypatch.setattr(gpu_memory.shutil, 'which', lambda _: None)
    data = defaults()
    data['execution']['log_vram'] = enabled
    saved, messages = [], []
    report = RunReport(write=messages.append)
    db = SimpleNamespace(progress=lambda *args: saved.append(args))
    return gpu_memory.GpuMemory(SimpleNamespace(data=data), report, db, 'run'), saved, messages


def test_cleanup_drops_models_before_gc_and_cache_without_cuda_initialization(monkeypatch):
    events = []
    cuda = SimpleNamespace(is_initialized=lambda: True, empty_cache=lambda: events.append('cache'))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(gpu_memory.gc, 'collect', lambda: events.append('gc'))
    gpu, _, _ = monitor(monkeypatch, enabled=False)
    gpu.release('vtt', [('close models', lambda: events.append('models'))])
    assert events == ['models', 'gc', 'cache']
    cuda.is_initialized = lambda: False
    gpu.reclaim()
    assert events[-1] == 'gc'
    assert gpu_memory.torch_memory() is None


def test_cleanup_continues_and_preserves_primary_error(monkeypatch):
    events = []
    gpu, _, _ = monitor(monkeypatch, enabled=False)
    monkeypatch.setattr(gpu_memory, 'reclaim', lambda: events.append('cache'))
    def fail():
        raise OSError('model close failed')
    with pytest.raises(ValueError, match='primary') as caught:
        try:
            raise ValueError('primary')
        finally:
            gpu.release('final', [('bad model', fail), ('other model', lambda: events.append('other'))])
    assert events == ['other', 'cache']
    assert any('model close failed' in note for note in caught.value.__notes__)


def test_quarantined_worker_defers_models_and_cache_together(monkeypatch):
    gpu, _, _ = monitor(monkeypatch, enabled=False)
    events, done = [], [False]
    monkeypatch.setattr(gpu_memory, 'reclaim', lambda: events.append('cache'))
    owner = object()
    with pytest.raises(shutdown.ShutdownIncomplete):
        shutdown.quarantine(lambda: done[0], owner)
    try:
        gpu.release('final', [('model', lambda: events.append('model'))])
        assert events == []
    finally:
        done[0] = True
        shutdown.require_idle()
    assert events == ['model', 'cache']


def test_snapshots_are_bounded_and_unknown_process_usage_is_not_zero(monkeypatch):
    gpu, saved, messages = monitor(monkeypatch)
    clock = iter(range(0, 1000, 10))
    monkeypatch.setattr(gpu_memory, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(gpu_memory, 'torch_memory', lambda: dict(device=0, allocated_bytes=100, reserved_bytes=200))
    for index in range(100):
        gpu.record('scene', 'after_release', scene=str(index))
    summary = gpu.summary()
    assert summary['samples'] == 100 and len(summary['latest']) == 1
    assert summary['latest']['scene:after_release']['process_bytes'] is None
    assert summary['sampled_peak_bytes'] == dict(allocated_bytes=100, reserved_bytes=200)
    assert saved[-1][2] == summary
    gpu.record('rebuild', 'before')
    assert 'process n/a' in gpu.display and 'torch reserved' in gpu.display
    assert not messages


def test_fast_scenes_skip_monitoring_but_keep_paired_boundary_samples(monkeypatch):
    gpu, saved, _ = monitor(monkeypatch)
    monkeypatch.setattr(gpu_memory, 'monotonic', lambda: 0)
    for scene in ('1', '2', '3'):
        for point in ('before_release', 'after_release'):
            gpu.record('scene', point, scene=scene)
    assert len(saved) == 2
    gpu.record('rebuild', 'before')
    assert len(saved) == 3


@pytest.mark.parametrize('output,expected', [
    ('{pid}, 100\n123456789, 200\n{pid}, 50\n', 150 * 1048576),
    ('{pid}, N/A\n', None), ('', None)])
def test_process_totals_include_only_our_pid_on_all_devices(monkeypatch, output, expected):
    def query(args, **kwargs):
        assert kwargs['timeout'] == 2 and kwargs['check']
        return SimpleNamespace(stdout=output.format(pid=os.getpid()))
    monkeypatch.setattr(gpu_memory.subprocess, 'run', query)
    assert gpu_memory.process_memory('nvidia-smi') == expected


def test_diagnostic_failure_cannot_break_recognition_and_disable_skips_queries(monkeypatch):
    gpu, saved, _ = monitor(monkeypatch)
    def fail():
        raise OSError('GPU monitoring unavailable')
    monkeypatch.setattr(gpu_memory, 'torch_memory', fail)
    gpu.record('vtt', 'before')
    assert saved[-1][2]['latest']['vtt:before']['torch'] is None
    gpu.enabled = False
    monkeypatch.setattr(gpu, 'snapshot', lambda: pytest.fail('disabled monitoring'))
    gpu.record('vtt', 'after')
    assert len(saved) == 1


def test_diagnostics_never_commit_an_active_evidence_transaction(monkeypatch):
    import sqlite3
    gpu, saved, _ = monitor(monkeypatch)
    db = sqlite3.connect(':memory:')
    gpu.database.connection = db
    try:
        db.execute('CREATE TABLE evidence (value)')
        db.execute('INSERT INTO evidence VALUES (1)')
        gpu.record('vtt', 'after')
        assert db.in_transaction and not saved
        db.rollback()
        assert db.execute('SELECT COUNT(*) FROM evidence').fetchone()[0] == 0
        gpu.record('final', 'after_release')
        assert saved[-1][2]['samples'] == 2
    finally:
        db.close()


@pytest.mark.parametrize('key,value', [('gpu_model_policy', 'all'), ('gpu_model_policy', True), ('log_vram', 'yes')])
def test_invalid_gpu_configuration_is_rejected(key, value):
    data = defaults()
    data['execution'][key] = value
    with pytest.raises(ValueError, match='execution.' + key):
        validate(data)

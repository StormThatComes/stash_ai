"""Publication invalidation, sparse previews and ordered CPU preparation, offline."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from stash_ai.domain import Cue, TagHierarchy
from stash_ai.inference import checkpoints, prediction, preparation, sprites
from stash_ai import cli as app
from stash_ai import merge
from stash_ai import performance
from stash_ai import progress
from stash_ai.runtime import RamCache, retained_size


pytestmark = pytest.mark.usefixtures("reset_progress")


def fail(*args, **kwargs):
    pytest.fail('Unchanged data was unnecessarily recomputed or published')


@pytest.mark.parametrize('filename', [
    'tags.py', 'domain.py', 'state.py', 'state_db.py', 'state_results.py', 'state_thresholds.py', 'state_publications.py',
    'models.py', 'vocabulary.py', 'onnx_tagger.py', 'prediction.py', 'sprites.py', 'preparation.py', 'progress.py', 'inputs.py', 'identity.py', 'checkpoints.py', 'signatures.py',
    'merge_run.py', 'merge_publish.py', 'merge_parallel.py', 'merge_sync.py', 'separate_sync.py',
    'config.py',
])
def test_publication_code_tracks_extracted_modules(monkeypatch, filename):
    before = performance.publication_code()
    read_bytes = performance.Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return data + b'\n# changed' if path.name == filename else data

    monkeypatch.setattr(performance.Path, 'read_bytes', changed)
    assert performance.publication_code() != before


@pytest.mark.parametrize('filename', [
    'client.py', 'transport.py', 'vtt.py', 'cli_args.py', 'cli_modes.py',
    'merge_cleanup.py', 'recovery.py', 'cleanup_sql.py', 'cleanup_sql_session.py',
])
def test_publication_code_ignores_runtime_and_maintenance_modules(monkeypatch, filename):
    before = performance.publication_code()
    read_bytes = performance.Path.read_bytes

    def changed(path):
        data = read_bytes(path)
        return data + b'\n# changed' if path.name == filename else data

    monkeypatch.setattr(performance.Path, 'read_bytes', changed)
    assert performance.publication_code() == before


def test_repeated_merge_avoids_decoding_evidence_and_publication(pipeline, monkeypatch, capsys):
    p = pipeline
    assert app.main(p.args) == 0
    reads = []
    find = p.stash.find_scene
    monkeypatch.setattr(p.stash, 'find_scene', lambda i: (reads.append(i), find(i))[1])
    monkeypatch.setattr(merge, 'evidence', fail)
    monkeypatch.setattr(merge.StateStore, 'prepare', fail)
    monkeypatch.setattr(merge.StateStore, 'unpack', fail)
    p.stash.writes.clear()
    assert app.main(p.args) == 0
    assert reads == ['1'] and p.stash.writes == []
    assert 'merge_reused=1' in capsys.readouterr().out


@pytest.mark.parametrize('change', ['marker', 'policy', 'result', 'pending'])
def test_merge_check_invalidates_real_changes(pipeline, monkeypatch, tmp_path, change):
    p = pipeline
    assert app.main(p.args) == 0
    args = list(p.args)
    if change == 'marker':
        p.stash.scene['scene_markers'][0]['title'] = 'edited'
    elif change == 'policy':
        path = tmp_path / 'exclude.yml'
        path.write_text('version: 1\nexcluded_scene_tags: [common]\n')
        args.extend(['--scene-tag-exclusions', str(path)])
    else:
        with merge.StateStore(p.path, merge.scope_for(app.STASH_URL)) as state:
            if change == 'result':
                # Even an older writer using direct SQL invalidates the check.
                with state.connection:
                    state.connection.execute('UPDATE results SET data=data')
            else:
                state.prepare('1', {'markers': state.snapshot('1', 'markers')[0]})
    calls = []
    original = merge.evidence
    monkeypatch.setattr(merge, 'evidence', lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    monkeypatch.setattr(app, 'load_tagger', fail)
    assert app.main(args) == 0 and calls == [1]


def test_derived_invalidation_rolls_back_with_the_transaction(tmp_path):
    with merge.StateStore(tmp_path/'state.db', 's') as state:
        state.put('1', 'wd', {'scene': dict(meta={}, items=[])})
        with state.connection:
            performance.record_check(state, '1', 'p', 'f')
        with pytest.raises(RuntimeError):
            with state.connection:
                state.connection.execute('UPDATE results SET data=data')
                assert not performance.checked(state, '1', 'p', 'f')
                raise RuntimeError('rollback')
        assert performance.checked(state, '1', 'p', 'f')
        with state.connection:
            state.connection.execute('DELETE FROM results')
        assert not performance.checked(state, '1', 'p', 'f')


def test_manual_hierarchy_change_invalidates_unchanged_scene(pipeline, monkeypatch):
    p = pipeline
    hierarchy = TagHierarchy({'common': ('ai:category:General',), 'extra': ('ai:category:General',)},
        {'ai:category:General': ()}, 'ai:category:', merge.AI.hierarchy_description)
    monkeypatch.setattr(merge, 'combined_hierarchy', lambda *args: hierarchy)
    assert app.main(p.args) == 0
    original = merge.evidence
    monkeypatch.setattr(merge, 'evidence', fail)
    assert app.main(p.args) == 0
    fingerprint = performance.scene_fingerprint(p.stash.scene)
    p.stash.tags['ai:common']['parents'] = []
    assert performance.scene_fingerprint(p.stash.scene) == fingerprint
    calls = []
    monkeypatch.setattr(merge, 'evidence', lambda *a, **kw: (calls.append(1), original(*a, **kw))[1])
    assert app.main(p.args) == 0 and calls == [1]
    assert {p['name'] for p in p.stash.tags['ai:common']['parents']} == {'ai:category:General'}


def test_cleanup_generation_invalidates_only_its_scope(tmp_path):
    from stash_ai.cleanup import save_state
    with merge.StateStore(tmp_path/'state.db', 'a') as state:
        with state.connection:
            performance.record_check(state, '1', 'p', 'f')
        with state.connection:
            state.connection.execute("INSERT INTO merge_checks VALUES ('b','1','p','f')")
        state.cleanup_generation = 1
        save_state(state)
        assert not performance.checked(state, '1', 'p', 'f')
        assert state.connection.execute("SELECT COUNT(*) FROM merge_checks WHERE scope='b'").fetchone() == (1,)


def test_commit_promotes_exact_pending_bytes_and_is_atomic(tmp_path, monkeypatch):
    with merge.StateStore(tmp_path/'state.db', 's') as state:
        state.publish('1', {'scene': {'old': True}})
        state.prepare('1', {'scene': {'new': True}})
        pending = state.connection.execute('SELECT pending FROM snapshots').fetchone()[0]
        with pytest.raises(RuntimeError, match='intent disappeared'):
            state.commit_prepared('1', ['scene', 'markers'])
        assert state.snapshot('1', 'scene') == ({'old': True}, {'new': True})
        monkeypatch.setattr(state, 'pack', fail)
        state.commit_prepared('1', ['scene'])
        assert state.connection.execute('SELECT published,pending FROM snapshots').fetchone() == (pending, None)


def test_preview_reads_and_shadows_without_copying_large_unrelated_blobs(tmp_path):
    path = tmp_path/'state.db'
    with merge.StateStore(path, 's') as state:
        state.put('1', 'wd', {'scene': dict(meta={}, items=[['old', .9]])})
        state.publish('1', {'scene': {'old': True}})
        with state.connection:
            state.connection.execute("INSERT INTO results VALUES ('s','unrelated','wd','scene',zeroblob(8000000))")
    before = hashlib.sha256(path.read_bytes()).digest()
    with merge.StateStore(path, 's', True, cache_bytes=1000000) as preview:
        assert preview.preview_source
        assert preview.connection.execute('SELECT COUNT(*) FROM main.results').fetchone() == (0,)
        assert preview.records('1', 'scene')['wd'][0]['items'] == [['old', .9]]
        preview.put('1', 'wd', {'scene': dict(meta={}, items=[['new', .8]])})
        assert preview.records('1', 'scene')['wd'][0]['items'] == [['new', .8]]
        preview.prepare('1', {'scene': {'new': True}})
        assert preview.snapshot('1', 'scene') == ({'old': True}, {'new': True})
        preview.commit_prepared('1', ['scene'])
        assert preview.snapshot('1', 'scene') == ({'new': True}, None)
        assert preview.connection.execute('SELECT COUNT(*) FROM main.results').fetchone() == (1,)
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_preview_resumes_checkpoint_and_does_not_resurrect_deleted_source(pipeline):
    p = pipeline
    p.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(p.args)
    before = p.path.read_bytes()
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as preview:
        assert preview.connection.execute('SELECT processed FROM ' + preview.table('checkpoints')).fetchone() == (4,)
        progress.request_refresh(preview, ['1'], ['wd'], ['scene', 'markers'])
        checkpoint = checkpoints.Checkpoint(preview, '1', 'wd', 'new', 10, 2)
        assert checkpoint.restore()[0] == 0
        assert not preview.connection.execute('SELECT 1 FROM ' + preview.table('checkpoint_frames')).fetchall()
        checkpoint.record(2, np.zeros((2, 2), np.float32), [])
        checkpoint.flush()
        assert checkpoints.Checkpoint(preview, '1', 'wd', 'new', 10, 2).restore()[0] == 2
        checkpoints.finish(preview, '1', 'wd', {k: dict(meta={'origin':'inference', 'inference_signature':'new'}, items=[]) for k in merge.KINDS}, {})
        assert progress.completed(preview, '1', 'wd', merge.KINDS, 'new')
        assert not preview.connection.execute('SELECT 1 FROM ' + preview.table('checkpoints')).fetchall()
    assert p.path.read_bytes() == before


def test_preview_detects_external_source_commit_even_for_fast_skip(tmp_path):
    path = tmp_path/'state.db'
    with merge.StateStore(path, 's') as state:
        with state.connection:
            performance.record_check(state, '1', 'p', 'f')
    with merge.StateStore(path, 's', True) as preview:
        assert performance.checked(preview, '1', 'p', 'f')
        with sqlite3.connect(path) as writer:
            writer.execute('DELETE FROM merge_checks')
        with pytest.raises(RuntimeError, match='changed during dry-run'):
            performance.checked(preview, '1', 'p', 'f')


@pytest.mark.parametrize('prefetch', [0, 2])
def test_cpu_transforms_are_parallel_ordered_and_own_their_images(monkeypatch, prefetch):
    rendezvous = threading.Barrier(4)
    threads, copies, originals = set(), [], []
    def frames(*args):
        for i in range(8):
            frame = Image.new('RGB', (2, 2), (i, 0, 0))
            originals.append(frame)
            try:
                yield Cue(i, i+1, 'sprite', (0,0,2,2)), frame
            finally:
                frame.close()
    def prepare(image):
        threads.add(threading.get_ident())
        copies.append(image)
        rendezvous.wait(timeout=5)
        return image.getpixel((0, 0))[0]
    monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    batches = list(preparation.prepared_batches(app, SimpleNamespace(batch_size=4, prepare=prepare), None, {}, [], prefetch, workers=4))
    assert [v for batch, _ in batches for v in batch] == list(range(8))
    assert len(threads) == 4 and threading.get_ident() not in threads
    assert all(a is not b for a in copies for b in originals)
    for image in copies + originals:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    assert not any(t.name.startswith(('prepare-frames', 'transform-frames')) for t in threading.enumerate())


def test_cpu_preparation_failure_closes_workers_and_owned_images(monkeypatch):
    copies = []
    def frames(*args):
        for i in range(8):
            image = Image.new('RGB', (2, 2))
            try:
                yield Cue(i, i+1, 'sprite', (0,0,2,2)), image
            finally:
                image.close()
    def prepare(image):
        copies.append(image)
        raise ValueError('bad frame')
    monkeypatch.setattr(sprites, 'iter_sprite_frames', frames)
    with pytest.raises(ValueError, match='bad frame'):
        list(preparation.prepared_batches(app, SimpleNamespace(batch_size=4, prepare=prepare), None, {}, [], 2, workers=4))
    assert copies
    for image in copies:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))
    assert not any(t.name.startswith(('prepare-frames', 'transform-frames')) for t in threading.enumerate())


@pytest.mark.parametrize('threshold', [.35, .65, float(np.float32(.65)), float(np.nextafter(np.float32(.65), np.float32(1)))])
def test_vectorized_score_filter_preserves_float_boundaries_and_order(threshold):
    scores = np.random.default_rng(15).random(1024).astype(np.float32)
    scores[:4] = [.65, np.nextafter(np.float32(.65), np.float32(0)), .65, .35]
    labels = [f'tag_{i}' for i in range(len(scores))]
    expected = sorted(((label, score) for label, score in zip(labels, scores.tolist())
                       if score >= threshold), key=lambda item: (-item[1], item[0]))
    for limit in [0, 7]:
        assert prediction.select_scores(scores, labels, threshold, limit) == (expected[:limit] if limit else expected)


def test_cache_accounts_cue_attributes_and_evicts_under_process_pressure(monkeypatch):
    cue = Cue(0, 1, 's'*10000, (0,0,1,1))
    assert retained_size(cue) > 10000
    monkeypatch.setattr('stash_ai.runtime.process_rss', lambda: 1000)
    cache = RamCache(20000, rss_budget=40000)
    for i in range(10):
        cache.put(i, bytes(900))
    before = cache.size
    monkeypatch.setattr('stash_ai.runtime.process_rss', lambda: 39999)
    cache.pressure(force=True)
    assert cache.size < before and cache.stats['pressure_trims'] == 1
    assert cache.rss_peak == 39999


def test_shared_cache_stays_bounded_with_concurrent_media_and_result_access():
    cache = RamCache(10000)
    def work(worker):
        for i in range(50):
            key = (worker, i)
            cache.put(key, bytes(100))
            cache.get(key)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(work, range(4)))
    assert cache.size == sum(cost for _, cost in cache.entries.values()) <= cache.limit


def test_detached_records_cannot_reuse_a_stale_segment_digest(tmp_path):
    with merge.StateStore(tmp_path/'state.db', 's', cache_bytes=1000000) as state:
        state.put('1', 'wd', {'markers': dict(meta={}, items=[])})
        borrowed = state.records('1', 'markers', borrow=True)['wd'][0]
        original = performance.segment_digest(borrowed)
        detached = state.records('1', 'markers')['wd'][0]
        detached['items'].append({'new': True})
        assert performance.segment_digest(detached) != original
        assert performance.segment_digest(borrowed) == original and borrowed['items'] == []

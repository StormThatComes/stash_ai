"""Execution changes reuse proven scores; missing or changed evidence cannot."""
from dataclasses import replace
import shutil
import sqlite3

import pytest

from stash_ai.db.update import update_database
from stash_ai.storage.codec import encode, decode
from tests.helpers.runtime import runtime as app
from tests.test_reuse_diagnostics import settings, decisions

pytestmark = pytest.mark.usefixtures('reset_progress')


def raw_rows(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute('SELECT * FROM ' + table + ' ORDER BY 1,2,3').fetchall()
                for table in ('results', 'snapshots', 'inference_archive')}


@pytest.mark.parametrize('change', ['batch', 'location', 'retention', 'combined'])
def test_execution_and_easier_retention_preserve_raw_records_and_snapshots(pipeline, monkeypatch, tmp_path, change):
    selected = settings(monkeypatch, tmp_path, scene_cache_floor=.01, marker_cache_floor=.01)
    assert app.main(pipeline.args) == 0
    before = raw_rows(pipeline.path)
    changes = {}
    if change in ('batch', 'combined'):
        changes['batch_size'] = 8
    if change in ('retention', 'combined'):
        changes.update(scene_cache_floor=.02, marker_cache_floor=.02)
    if change in ('location', 'combined'):
        moved = tmp_path / 'moved'
        shutil.copytree(selected[0].model_dir, moved)
        changes['model_dir'] = moved
    selected[0] = replace(selected[0], **changes)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('compatible cache was reinferred'))
    assert app.main(pipeline.args) == 0
    assert app.main(pipeline.args) == 0
    assert raw_rows(pipeline.path) == before
    assert decisions(pipeline.path)[0]['action'] == 'cache_reused'


def test_v26_upgrade_and_matching_read_register_legacy_identity_without_rewriting_scores(pipeline, monkeypatch, tmp_path):
    selected = settings(monkeypatch, tmp_path, scene_cache_floor=.01, marker_cache_floor=.01)
    assert app.main([*pipeline.args, '--no-run-history']) == 0
    with sqlite3.connect(pipeline.path) as db:
        db.execute('DROP TABLE inference_identities')
        db.execute('PRAGMA user_version=26')
    before = raw_rows(pipeline.path)
    assert update_database(pipeline.path) == (26, 27)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('upgrade required inference'))
    assert app.main([*pipeline.args, '--no-run-history']) == 0
    selected[0] = replace(selected[0], batch_size=8, scene_cache_floor=.02)
    assert app.main([*pipeline.args, '--no-run-history']) == 0
    assert raw_rows(pipeline.path) == before


@pytest.mark.parametrize('damage', ['weights', 'input', 'provenance', 'missing_provenance'])
def test_changed_or_unproven_identity_requires_only_local_inference(pipeline, monkeypatch, tmp_path, damage):
    from stash_ai.backends.vocabulary import weight_files
    from stash_ai.domain import Cue
    selected = settings(monkeypatch, tmp_path)
    assert app.main(pipeline.args) == 0
    loaded = len(pipeline.loaded)
    if damage == 'weights':
        (selected[0].model_dir / weight_files(selected[0])[0]).write_bytes(b'changed weights')
    elif damage == 'input':
        pipeline.cues = [Cue(c.start, c.end - .25, c.image, c.rect) for c in pipeline.cues]
    else:
        with sqlite3.connect(pipeline.path) as db:
            if damage == 'missing_provenance':
                db.execute('DELETE FROM inference_identities')
            else:
                data = db.execute('SELECT data FROM inference_identities LIMIT 1').fetchone()[0]
                value = decode(data)
                value['batch_size'] = 8
                db.execute('UPDATE inference_identities SET data=?', (encode(value),))
        selected[0] = replace(selected[0], batch_size=8)
    extra = ['--force-fetch-vtt'] if damage == 'input' else []
    assert app.main([*pipeline.args, *extra]) == 0
    assert len(pipeline.loaded) == loaded + 1
    assert decisions(pipeline.path)[0]['action'] == 'infer'
    with sqlite3.connect(pipeline.path) as db:
        assert db.execute('SELECT count(*) FROM inference_identities').fetchone()[0] == 1


def test_batch_change_resumes_proven_checkpoint_frames(pipeline, monkeypatch, tmp_path):
    selected = settings(monkeypatch, tmp_path)
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    pipeline.fail.clear()
    selected[0] = replace(selected[0], batch_size=8)
    assert app.main(pipeline.args) == 0
    assert pipeline.loaded[-1].seen == list(range(4, 10))
    assert decisions(pipeline.path)[0]['details']['checkpoint']['resumed_frames'] == 4


def test_timeline_reset_drops_resume_count_and_reports_real_reason(pipeline):
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    with sqlite3.connect(pipeline.path) as db:
        data = db.execute('SELECT data FROM checkpoint_frames').fetchone()[0]
        rows = decode(data)
        rows[0]['start'] = .1
        db.execute('UPDATE checkpoint_frames SET data=?', (encode(rows),))
    pipeline.fail.clear()
    assert app.main(pipeline.args) == 0
    assert pipeline.loaded[-1].seen == list(range(10))
    decision = decisions(pipeline.path)[0]
    assert decision['action'] == 'infer'
    assert decision['details']['checkpoint']['resumed_frames'] == 0
    assert decision['details']['checkpoint']['reason'] == 'timeline_inconsistent'
    assert 'checkpoint_compatible' not in decision['reasons']

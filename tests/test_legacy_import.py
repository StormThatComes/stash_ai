"""Explicit legacy conversion keeps source objects, unknown evidence and retries safe."""
from copy import deepcopy
import sqlite3

import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.stash import Stash
from stash_ai.commands import legacy
from stash_ai.models import WD
from stash_ai.storage.store import StateStore, scope_for
from stash_ai.storage import progress

pytestmark = pytest.mark.usefixtures('reset_progress')


def command(path, *extra):
    return ['--import-legacy-only', '--state-db', str(path), '--scene-id', '1', *extra]


def seed_stash():
    stash = Stash()
    stash.scene['tags'] = [stash.tag('wd:common'), stash.tag('dtq:unique'), stash.tag('human')]
    marker = stash.marker(WD, 'common', 'source')
    marker.update(seconds=12, end_seconds=None)
    marker['tags'].append(stash.tag('manual extra'))
    stash.marker(WD, 'common', 'manual', True)
    return stash


def disable_assets(monkeypatch, stash):
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *_: stash)
    for name in ('load_model_settings', 'load_vocabulary', 'load_tagger', 'ensure_files', 'load_exclusion_policy'):
        monkeypatch.setattr(app.services, name, lambda *a, **k: pytest.fail('legacy import accessed model/policy assets'))
    monkeypatch.setattr(stash, 'get_media', lambda *a: pytest.fail('legacy import downloaded media'))


def state_rows(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute(f'SELECT * FROM {table} ORDER BY 1,2,3').fetchall()
                for table in ('results', 'legacy_imports', 'result_index', 'snapshots')}


def test_import_only_is_idempotent_and_preserves_unknown_scores_ends_and_manual_objects(tmp_path, monkeypatch):
    path, stash = tmp_path / 'state.sqlite3', seed_stash()
    disable_assets(monkeypatch, stash)
    original = deepcopy(stash.scene)
    assert app.main(command(path)) == 0
    with StateStore(path, scope_for(app.STASH_URL)) as state:
        records = state.records('1', 'scene')
        assert set(records) == {'wd', 'dtq'}  # No selected-model dependency.
        assert all(item['score'] is None for segments in records.values() for segment in segments for item in segment['items'])
        marker = state.records('1', 'markers')['wd'][0]['items'][0]
        assert marker['start'] == 12 and marker['end'] is None and marker['score'] is None and marker['hits'] is None
        assert len(state.imported_legacy('1', 'markers')) == 1
        assert not state.has_pending_publications()
        assert not progress.completed(state, '1', 'wd', ['scene', 'markers'], 'inference-signature')
    first = state_rows(path)
    assert app.main(command(path)) == 0
    assert state_rows(path) == first
    assert stash.scene == original and not stash.writes


@pytest.mark.parametrize('existing', [False, True])
def test_import_dry_run_never_changes_stash_or_source_database(tmp_path, monkeypatch, existing):
    path, stash = tmp_path / 'state.sqlite3', seed_stash()
    disable_assets(monkeypatch, stash)
    if existing:
        with StateStore(path, scope_for(app.STASH_URL)):
            pass
    before = (path.read_bytes(), path.stat().st_mtime_ns) if existing else None
    original = deepcopy(stash.scene)
    assert app.main(command(path, '--dry-run')) == 0
    assert ((path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None) == before
    assert stash.scene == original and not stash.writes


def test_import_rejects_pending_plans_before_stash_access(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    with StateStore(path, scope_for(app.STASH_URL)) as state:
        state.prepare('1', {'markers': {'contributions': []}})
    before = state_rows(path)
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *_: pytest.fail('pending import accessed Stash'))
    with pytest.raises(RuntimeError, match='Pending publication plans'):
        app.main(command(path))
    assert state_rows(path) == before


@pytest.mark.parametrize('flag,kind', [('--markers-only', 'markers'), ('--scene-tags-only', 'scene')])
def test_import_can_select_one_output_without_stash_mutations(tmp_path, monkeypatch, flag, kind):
    path, stash = tmp_path / 'state.sqlite3', seed_stash()
    disable_assets(monkeypatch, stash)
    assert app.main(command(path, flag)) == 0
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT DISTINCT kind FROM results').fetchall() == [(kind,)]
        assert db.execute('SELECT DISTINCT kind FROM legacy_imports').fetchall() == [(kind,)]
    assert not stash.writes


def test_publication_preserves_uncaptured_objects_added_after_the_source_scan(tmp_path):
    from stash_ai.evidence import pipeline
    from stash_ai.publication import merged
    from stash_ai.evidence.labels import AI
    from tests.helpers.evidence import empty_policies
    path, stash = tmp_path / 'state.sqlite3', seed_stash()
    with StateStore(path, scope_for(app.STASH_URL)) as state:
        legacy.import_scene(app, state, stash.find_scene('1'))
        late_tag = stash.tag('wd:later')
        stash.scene['tags'].append(late_tag)
        late_marker = stash.marker(WD, 'later', 'late-source')
        hits, spans = pipeline.evidence(app, state, '1', empty_policies(), empty_policies())
        merged.sync_merged(app, stash, app.services.TagResolver(stash, source=AI), state,
                           stash.find_scene('1'), hits, spans)
        assert any(tag['id'] == late_tag['id'] for tag in stash.scene['tags'])
        assert any(marker['id'] == late_marker['id'] for marker in stash.scene['scene_markers'])


def test_import_failure_rolls_back_evidence_and_ledger_for_both_outputs(tmp_path, monkeypatch):
    path, stash = tmp_path / 'state.sqlite3', seed_stash()
    with StateStore(path, scope_for(app.STASH_URL)) as state:
        remember = state.legacy.remember_in_transaction
        def fail_markers(scene, kind, data):
            if kind == 'markers':
                raise RuntimeError('interrupted ledger write')
            remember(scene, kind, data)
        with monkeypatch.context() as patch:
            patch.setattr(state.legacy, 'remember_in_transaction', fail_markers)
            with pytest.raises(RuntimeError, match='interrupted ledger'):
                legacy.import_scene(app, state, stash.find_scene('1'))
        assert state.records('1', 'scene') == state.records('1', 'markers') == {}
        assert not state.imported_legacy('1', 'scene') and not state.imported_legacy('1', 'markers')
        assert legacy.import_scene(app, state, stash.find_scene('1')) == {
            'legacy_scene_tags_imported': 2, 'legacy_markers_imported': 1}


@pytest.mark.parametrize('extra', [
    ['--models', 'wd'], ['--offline'], ['--resume'], ['--restart'], ['--output-mode', 'separate'],
    ['--abandon-pending'], ['--force-merge'], ['--prune-empty-tags'], ['--merge-only'],
])
def test_import_rejects_execution_and_publication_options_before_state_access(tmp_path, extra):
    path = tmp_path / 'state.sqlite3'
    with pytest.raises(SystemExit) as error:
        app.main(command(path, *extra))
    assert error.value.code == 2 and not path.exists()


def test_separate_outputs_require_explicit_import_before_merged_publication(pipeline, monkeypatch, capsys):
    p = pipeline
    p.stash.scene['tags'].append(p.stash.tag('wd:common'))
    p.stash.marker(WD, 'common', 'historical')
    source_scene = deepcopy(p.stash.scene)
    assert not p.path.exists()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *a: pytest.fail('conversion loaded weights'))
    assert app.main([*p.args, '--merge-only']) == 1
    assert '--import-legacy-only' in capsys.readouterr().out
    assert p.stash.scene == source_scene
    with StateStore(p.path, scope_for(app.STASH_URL)) as state:
        assert not state.records('1', 'scene') and not state.imported_legacy('1', 'scene')
    assert app.main(command(p.path)) == 0
    assert p.stash.scene == source_scene
    assert app.main([*p.args, '--merge-only']) == 0
    assert any(tag['name'].startswith('ai:') for tag in p.stash.scene['tags'])
    assert not any(tag['name'].startswith('wd:') for tag in p.stash.scene['tags'])

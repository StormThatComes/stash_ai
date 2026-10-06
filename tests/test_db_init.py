"""Fresh-state adoption, per-scene readiness and recovery without the old DB."""
from copy import deepcopy
import json
import sqlite3

import pytest

from stash_ai.commands.arguments import parse_args
from stash_ai.db.checks import check_database
from stash_ai.db.inspection import inspect_database
from stash_ai.db.update import update_database
from stash_ai.evidence.labels import AI
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG
from stash_ai.storage.store import StateStore, scope_for
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_merge')


def seed_outputs(stash):
    stash.scene['tags'] = [stash.tag(name) for name in (
        'manual', 'ai:stale', 'ai:common', 'ai:category:Keep', AI.owner_tag, OCCURRENCE_TAG)]
    stash.marker(AI, 'stale', 'old-ai')
    stash.marker(AI, 'foreign', 'manual-marker', manual=True)


def init_args(p, models=None):
    args = [*p.args, '--db-init']
    return args if models is None else [*args, '--models', *models]


def resume_args(args):
    return [value for value in args if value != '--db-init']


def state_rows(path, table):
    with sqlite3.connect(path) as db:
        return db.execute(f'SELECT * FROM {table} ORDER BY 1,2').fetchall()


def collection(p, monkeypatch):
    from tests.helpers.cleanup import FastStash
    stash = FastStash()
    seed_outputs(stash)
    second = deepcopy(stash.scene)
    second['id'] = '2'
    for marker in second['scene_markers']:
        marker['id'] += '-2'
    stash.scenes['2'] = second
    p.stash = stash
    monkeypatch.setattr(app.services, 'create_stash_client', lambda _: stash)
    return [value for value in p.args if value not in ('--scene-id', '1')]


def test_fresh_db_adopts_existing_ai_outputs_and_records_real_evidence(pipeline):
    p = pipeline
    seed_outputs(p.stash)
    assert app.main(init_args(p)) == 0
    names = {tag['name'] for tag in p.stash.scene['tags']}
    assert names == {'manual', 'ai:common', 'ai:extra', 'ai:category:Keep', AI.owner_tag, OCCURRENCE_TAG}
    assert 'old-ai' not in {marker['id'] for marker in p.stash.scene['scene_markers']}
    assert 'manual-marker' in {marker['id'] for marker in p.stash.scene['scene_markers']}
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.initialization.active
        assert store.initialization.scenes == {'1': True}
        assert set(store.records('1', 'scene')) == {'wd'}
        published, pending = store.snapshot('1', 'scene')
        assert pending is None and published['contributions']
        assert set(published['owned_ids']) == {p.stash.tags[name]['id'] for name in ('ai:common', 'ai:extra')}
    assert check_database(p.path, scope_for(app.STASH_URL))['ok']
    report = inspect_database(p.path, scope_for(app.STASH_URL))
    assert report['db_init_completed_scenes'] == 1 and report['db_init_pending_scenes'] == 0


def test_ordinary_fresh_db_retains_missing_source_guard(pipeline, monkeypatch):
    p = pipeline
    seed_outputs(p.stash)
    before = deepcopy(p.stash.scene)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('inferred without init authorization'))
    assert app.main(p.args) == 1
    assert p.stash.scene == before and not p.stash.writes
    assert state_rows(p.path, 'db_initializations') == []


@pytest.mark.parametrize('dry', [False, True])
def test_init_refuses_populated_database_without_modifying_it(pipeline, dry):
    p = pipeline
    assert app.main(p.args) == 0
    before = p.path.read_bytes(), p.path.stat().st_mtime_ns, deepcopy(p.stash.scene)
    p.stash.writes.clear()
    args = init_args(p) + (['--dry-run'] if dry else [])
    with pytest.raises(RuntimeError, match='new or empty'):
        app.main(args)
    assert (p.path.read_bytes(), p.path.stat().st_mtime_ns, p.stash.scene) == before
    assert not p.stash.writes


@pytest.mark.parametrize('dry', [False, True])
def test_init_refuses_state_in_another_stash_scope(pipeline, dry):
    p = pipeline
    with StateStore(p.path, 'other-stash') as store:
        store.connection.execute('INSERT INTO refresh_requests VALUES (?,?,?,?)', ('other-stash', '8', 'wd', 'scene'))
        store.connection.commit()
    before = p.path.read_bytes()
    with pytest.raises(RuntimeError, match='new or empty'):
        app.main(init_args(p) + (['--dry-run'] if dry else []))
    assert p.path.read_bytes() == before and not p.stash.writes


def test_schema_only_database_can_be_initialized(pipeline):
    p = pipeline
    with StateStore(p.path, scope_for(app.STASH_URL)):
        pass
    seed_outputs(p.stash)
    assert app.main(init_args(p)) == 0


def test_one_model_failure_keeps_outputs_until_all_selected_models_finish(pipeline, monkeypatch):
    p = pipeline
    seed_outputs(p.stash)
    before = deepcopy(p.stash.scene)
    original = app.services.load_tagger
    def load(config, *args):
        if config.source.key == 'dtq':
            raise RuntimeError('GPU model unavailable')
        return original(config, *args)
    monkeypatch.setattr(app.services, 'load_tagger', load)
    args = init_args(p, ['wd', 'dtq'])
    assert app.main(args) == 1
    assert p.stash.scene == before and not p.stash.writes
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.initialization.active
        assert set(store.records('1', 'scene')) == {'wd'}
        assert store.snapshot('1', 'scene') == (None, None)
        with pytest.raises(RuntimeError, match='awaits complete inference'):
            store.initialization.require_ready('1')
    monkeypatch.setattr(app.services, 'load_tagger', original)
    assert app.main(resume_args(args)) == 0
    assert [model.settings.source.key for model in p.loaded] == ['wd', 'dtq']
    assert state_rows(p.path, 'db_init_scenes')[0][-1] == 1


@pytest.mark.parametrize('missing', ['sprite', 'vtt', 'empty_vtt'])
def test_missing_inputs_keep_existing_outputs_and_can_resume(pipeline, monkeypatch, missing):
    from stash_ai.inference import sprites
    p = pipeline
    seed_outputs(p.stash)
    before = deepcopy(p.stash.scene)
    paths = dict(p.stash.scene['paths'])
    original = sprites.fetch_cues
    if missing == 'empty_vtt':
        monkeypatch.setattr(sprites, 'fetch_cues', lambda *args: ([], 0))
    else:
        p.stash.scene['paths'].pop(missing)
        before = deepcopy(p.stash.scene)
    assert app.main(init_args(p)) == 1
    assert p.stash.scene == before and not p.stash.writes
    assert state_rows(p.path, 'db_init_scenes')[0][-1] == 0
    p.stash.scene['paths'] = paths
    monkeypatch.setattr(sprites, 'fetch_cues', original)
    assert app.main(resume_args(init_args(p))) == 0


def test_successful_zero_detections_remove_old_ai_outputs(pipeline):
    p = pipeline
    seed_outputs(p.stash)
    p.scores['wd'][:] = 0
    assert app.main(init_args(p)) == 0
    assert {tag['name'] for tag in p.stash.scene['tags']} == {'manual', 'ai:category:Keep', AI.owner_tag, OCCURRENCE_TAG}
    assert [marker['id'] for marker in p.stash.scene['scene_markers']] == ['manual-marker']
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.initialization.scenes == {'1': True}
        assert not store.records('1', 'scene')['wd']['items']
        assert all(not item['tags'] for item in store.records('1', 'markers')['wd']['items'])


@pytest.mark.parametrize('repeat_flag', [False, True])
def test_interrupted_init_resumes_checkpoint_without_resetting_it(pipeline, repeat_flag):
    p = pipeline
    seed_outputs(p.stash)
    before = deepcopy(p.stash.scene)
    p.fail['wd'] = 4
    args = init_args(p)
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    assert p.stash.scene == before and not p.stash.writes
    assert state_rows(p.path, 'checkpoints')[0][4] == 4
    p.fail.clear()
    assert app.main(args if repeat_flag else resume_args(args)) == 0
    assert p.loaded[-1].seen == list(range(4, 10))
    assert state_rows(p.path, 'checkpoints') == []


def test_changed_vtt_invalidates_finished_model_and_partial_checkpoint(pipeline):
    p = pipeline
    seed_outputs(p.stash)
    p.fail['dtq'] = 4
    args = init_args(p, ['wd', 'dtq'])
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    p.fail.clear()
    p.cues = p.cues[:-1]
    assert app.main(resume_args(args)) == 0
    assert [model.settings.source.key for model in p.loaded] == ['wd', 'dtq', 'wd', 'dtq']
    assert all(model.seen == list(range(9)) for model in p.loaded[-2:])


def test_failed_partial_publication_resumes_offline_and_keeps_ownership(pipeline, monkeypatch):
    p = pipeline
    seed_outputs(p.stash)
    original = p.stash.update_scene_tags
    def update_then_fail(scene, ids):
        original(scene, ids)
        raise RuntimeError('connection lost after scene update')
    monkeypatch.setattr(p.stash, 'update_scene_tags', update_then_fail)
    args = init_args(p)
    assert app.main(args) == 1
    assert state_rows(p.path, 'db_init_scenes')[0][-1] == 0
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.has_pending_publications()
    monkeypatch.setattr(p.stash, 'update_scene_tags', original)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('publication retry inferred again'))
    monkeypatch.setattr(app.services, 'load_vocabulary', lambda *args: pytest.fail('offline retry accessed vocabulary'))
    assert app.main([*resume_args(args), '--merge-only']) == 0
    assert 'old-ai' not in {marker['id'] for marker in p.stash.scene['scene_markers']}
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.initialization.active and not store.has_pending_publications()
        assert set(store.snapshot('1', 'scene')[0]['owned_ids']) == {p.stash.tags[name]['id'] for name in ('ai:common', 'ai:extra')}


def test_init_adopts_ai_tags_added_after_initial_scan(pipeline, monkeypatch):
    p = pipeline
    seed_outputs(p.stash)
    original = p.stash.find_scene_for_merge
    def refreshed(scene):
        p.stash.scene['tags'].append(p.stash.tag('ai:late_stale'))
        return original(scene)
    monkeypatch.setattr(p.stash, 'find_scene_for_merge', refreshed)
    assert app.main(init_args(p)) == 0
    assert 'ai:late_stale' not in {tag['name'] for tag in p.stash.scene['tags']}


@pytest.mark.parametrize('workers', [1, 2])
def test_collection_init_and_completed_manifest_do_not_adopt_later_scenes(pipeline, monkeypatch, workers):
    p = pipeline
    args = [*collection(p, monkeypatch), '--merge-workers', str(workers)]
    assert app.main([*args, '--db-init']) == 0
    for scene in p.stash.scenes.values():
        assert 'ai:stale' not in {tag['name'] for tag in scene['tags']}
        assert len(scene['scene_markers']) > 1
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.initialization.scenes == {'1': True, '2': True}
    third = deepcopy(p.stash.scenes['1'])
    third['id'] = '3'
    p.stash.scenes['3'] = third
    before = deepcopy(third)
    assert app.main([*args, '--resume']) == 1
    assert p.stash.scenes['3'] == before
    assert len(state_rows(p.path, 'db_init_scenes')) == 2


def test_scene_selection_leaves_other_scenes_untouched(pipeline, monkeypatch):
    p = pipeline
    collection(p, monkeypatch)
    before = deepcopy(p.stash.scenes['2'])
    assert app.main(init_args(p)) == 0
    assert p.stash.scenes['2'] == before
    assert state_rows(p.path, 'db_init_scenes')[0][1] == '1'
    assert len(state_rows(p.path, 'db_init_scenes')) == 1


def test_interrupted_collection_scan_resumes_enrollment(pipeline, monkeypatch):
    p = pipeline
    args = [*collection(p, monkeypatch), '--db-init', '--merge-workers', '1']
    original = p.stash.iter_scenes
    def interrupted(identifiers=None):
        yield p.stash.find_scene('1')
        raise KeyboardInterrupt
    monkeypatch.setattr(p.stash, 'iter_scenes', interrupted)
    before = deepcopy(p.stash.scenes)
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    assert p.stash.scenes == before and not p.stash.writes
    assert state_rows(p.path, 'db_initializations')[0][-1] == 0
    monkeypatch.setattr(p.stash, 'iter_scenes', original)
    assert app.main(resume_args(args)) == 0
    assert all(row[-1] == 1 for row in state_rows(p.path, 'db_init_scenes'))
    assert len(state_rows(p.path, 'db_init_scenes')) == 2


def test_failed_commit_does_not_mark_scene_initialized(pipeline, monkeypatch):
    from stash_ai.storage import merge_checks
    p = pipeline
    seed_outputs(p.stash)
    original = merge_checks.record_check
    def fail(*args):
        raise RuntimeError('commit failed')
    monkeypatch.setattr(merge_checks, 'record_check', fail)
    args = init_args(p)
    assert app.main(args) == 1
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.initialization.pending('1')
        assert store.snapshot('1', 'scene')[0] is None
        assert store.has_pending_publications()
    monkeypatch.setattr(merge_checks, 'record_check', original)
    assert app.main([*resume_args(args), '--merge-only']) == 0
    assert state_rows(p.path, 'db_init_scenes')[0][-1] == 1


def test_missing_result_blocks_init_even_without_refresh_request(pipeline):
    p = pipeline
    seed_outputs(p.stash)
    p.fail['wd'] = 4
    args = init_args(p)
    with pytest.raises(KeyboardInterrupt):
        app.main(args)
    with sqlite3.connect(p.path) as db:
        db.execute('DELETE FROM refresh_requests')
    before = deepcopy(p.stash.scene)
    assert app.main([*resume_args(args), '--merge-only']) == 1
    assert p.stash.scene == before and not p.stash.writes


def test_repeated_finished_init_reuses_results_without_resetting_progress(pipeline, monkeypatch):
    p = pipeline
    seed_outputs(p.stash)
    args = init_args(p)
    assert app.main(args) == 0
    rows = state_rows(p.path, 'db_init_scenes')
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('repeated init reset inference'))
    assert app.main(args) == 0
    assert state_rows(p.path, 'db_init_scenes') == rows


@pytest.mark.parametrize('change', ['models', 'selection', 'partial', 'restart', 'rethreshold'])
def test_pending_init_requires_original_models_scope_and_both_outputs(pipeline, change):
    p = pipeline
    seed_outputs(p.stash)
    p.stash.scene['paths'].pop('vtt')
    args = init_args(p)
    assert app.main(args) == 1
    changed = resume_args(args)
    changed += {'models': ['--models', 'dtq'], 'selection': ['--scene-id', '2'],
                'partial': ['--scene-tags-only'], 'restart': ['--restart'],
                'rethreshold': ['--merge-only', '--rethreshold-scenes']}[change]
    before = p.path.read_bytes(), deepcopy(p.stash.scene)
    with pytest.raises(RuntimeError, match='DB initialization|Resume DB initialization'):
        app.main(changed)
    assert (p.path.read_bytes(), p.stash.scene) == before


@pytest.mark.parametrize('existing', ['missing', 'empty', 'interrupted'])
def test_init_dry_run_preserves_source_database_and_stash(pipeline, existing):
    p = pipeline
    seed_outputs(p.stash)
    args = init_args(p)
    if existing == 'empty':
        with StateStore(p.path, scope_for(app.STASH_URL)):
            pass
    elif existing == 'interrupted':
        p.fail['wd'] = 4
        with pytest.raises(KeyboardInterrupt):
            app.main(args)
        p.fail.clear()
    before = (p.path.read_bytes(), p.path.stat().st_mtime_ns) if p.path.exists() else None
    scene = deepcopy(p.stash.scene)
    p.stash.dry_run = True
    assert app.main([*args, '--dry-run']) == 0
    after = (p.path.read_bytes(), p.path.stat().st_mtime_ns) if p.path.exists() else None
    assert before == after and p.stash.scene == scene and not p.stash.writes


@pytest.mark.parametrize('options', [
    ['--merge-only'], ['--markers-only'], ['--scene-tags-only'], ['--restart'],
    ['--abandon-pending'], ['--cleanup-only'], ['--validate-hierarchy'], ['--download-models'],
    ['--db-report'], ['--db-check'], ['--db-update'], ['--db-compact'],
    ['--model-report', 'models.json'], ['--tag-review-report', 'review.json'], ['--cache-report', 'cache.json'],
])
def test_db_init_rejects_incompatible_commands(options):
    with pytest.raises(SystemExit):
        parse_args(app, ['--db-init', *options])


def test_v23_upgrade_adds_init_state_without_creating_authorization(pipeline):
    p = pipeline
    assert app.main(p.args) == 0
    before = state_rows(p.path, 'results'), state_rows(p.path, 'snapshots')
    with sqlite3.connect(p.path) as db:
        for event in ('insert', 'update', 'delete'):
            db.execute('DROP TRIGGER perf_db_init_scenes_' + event)
        db.execute('DROP TABLE db_init_scenes')
        db.execute('DROP TABLE db_initializations')
        db.execute('PRAGMA user_version=23')
    assert update_database(p.path) == (23, 24)
    assert (state_rows(p.path, 'results'), state_rows(p.path, 'snapshots')) == before
    assert state_rows(p.path, 'db_initializations') == []
    assert check_database(p.path, scope_for(app.STASH_URL))['ok']


@pytest.mark.parametrize('damage', ['models', 'owned_ids', 'orphan'])
def test_db_check_rejects_invalid_initialization_authorization(pipeline, damage):
    p = pipeline
    p.stash.scene['paths'].pop('vtt')
    assert app.main(init_args(p)) == 1
    with sqlite3.connect(p.path) as db:
        if damage == 'models':
            db.execute("UPDATE db_initializations SET models='[\"unknown\"]'")
        elif damage == 'owned_ids':
            db.execute("UPDATE db_init_scenes SET owned_ids='[false]'")
        else:
            db.execute('DELETE FROM db_initializations')
    report = check_database(p.path, scope_for(app.STASH_URL))
    assert not report['ok'] and 'db_init_invalid' in {issue['code'] for issue in report['errors']}

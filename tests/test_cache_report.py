"""Coverage counts and advisory plans must never mistake absence for negative detections."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from stash_ai import cache_report, cli as app, merge
from stash_ai.models import ModelSettings, SOURCES
from tests.test_model_report import inference_pair

pytestmark = pytest.mark.usefixtures('reset_temporal')


def config(model='wd', **changes):
    return ModelSettings(source=SOURCES[model], model_dir=app.INSTALL_ROOT / 'models' / model,
                         scene_threshold=.4, marker_threshold=.8, batch_size=4, **changes)


def pair(label='common', **kwargs):
    outputs = inference_pair(label, **kwargs)
    outputs['scene']['meta'].update(scene_frames_format=1, scene_score_floor=.1,
                                    scene_support_floor=.34)
    outputs['scene']['frames'] = deepcopy(outputs['markers']['items'])
    return outputs


def seed(path, scene='1', model='wd', outputs=None, history=False):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        store.put(scene, model, pair() if outputs is None else outputs,
                  {'scene': False, 'markers': False} if history else None)


def report(path, scenes=None, settings=None):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True, allow_cleanup=True) as store:
        return cache_report.build_report(app, store, cache_report.scene_ids(store, scenes),
                                         [config()] if settings is None else settings)


def issue(data, scene='1', model='wd'):
    return next(row for row in data['issues'] if row['scene_id'] == scene and row['model'] == model)


def actions(data):
    return [row for row in data['plan'] if row['uses_inference']]


def test_missing_pairs_count_selected_models_and_explicit_unseen_scenes(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    data = report(path, ['1', '1', '404'], [config('wd'), config('dtq')])
    assert data['totals']['scenes'] == 2 and data['totals']['scene_model_pairs'] == 4
    assert data['totals']['reusable_pairs'] == 1 and data['totals']['inference_pairs'] == 3
    assert issue(data, '404')['outputs']['scene']['status'] == 'missing'
    assert {row['model']: row['scene_ids'] for row in actions(data)} == {'wd': ['404'], 'dtq': ['1', '404']}
    assert all('--restart' not in row['argv'] for row in actions(data))


def test_empty_detections_are_completed_inference_and_need_no_plan(tmp_path):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    outputs['scene']['items'] = outputs['scene']['frames'] = outputs['markers']['items'] = []
    seed(path, outputs=outputs)
    data = report(path)
    assert data['totals']['fully_ready_scenes'] == data['totals']['evidence_ready_pairs'] == 1
    assert not data['issues'] and not data['plan']


def test_rethreshold_uses_sql_overrides_and_has_no_gpu_action(tmp_path):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    outputs['markers']['meta']['marker_score_floor'] = .2
    seed(path, outputs=outputs)
    settings = [replace(config(), scene_threshold=.5, marker_threshold=.6)]
    first = report(path, settings=settings)
    assert first['totals']['rethreshold_pairs'] == 1 and not actions(first)
    assert first['plan'][0]['action'] == 'rethreshold'
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        from stash_ai import scene_cache
        updates = scene_cache.plan(app, store, '1', settings[0], markers=True, metadata_only=True)
        scene_cache.apply(store, '1', 'wd', updates, metadata_only=True)
    before = path.read_bytes()
    assert report(path, settings=settings)['totals']['fully_ready_scenes'] == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize('kind', ['scene', 'markers'])
def test_required_retention_checks_explicit_yaml_floor_even_at_same_threshold(tmp_path, kind):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    outputs[kind]['meta']['scene_score_floor' if kind == 'scene' else 'marker_score_floor'] = .3
    seed(path, outputs=outputs)
    settings = [config(**{'scene_cache_floor' if kind == 'scene' else 'marker_cache_floor': .2})]
    data = report(path, settings=settings)
    assert issue(data)['outputs'][kind]['status'] == 'insufficient'
    assert '--restart' in actions(data)[0]['argv']


@pytest.mark.parametrize('kind', ['scene', 'markers'])
def test_missing_output_is_a_gap_while_other_completed_output_stays_ready(tmp_path, kind):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    del outputs[kind]
    seed(path, outputs=outputs)
    data = report(path)
    assert issue(data)['outputs'][kind]['status'] == 'missing'
    assert issue(data)['outputs']['markers' if kind == 'scene' else 'scene']['status'] == 'ready'
    assert data['totals']['inference_pairs'] == 1


def test_imported_history_is_not_completed_inference(tmp_path):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    for segment in outputs.values():
        segment['meta']['origin'] = 'legacy'
    seed(path, outputs=outputs)
    data = report(path)
    assert {row['status'] for row in issue(data)['outputs'].values()} == {'legacy_only'}
    assert actions(data)[0]['action'] == 'restart_inference'


def test_old_history_is_validated_even_when_last_segment_has_enough_retention(tmp_path):
    path = tmp_path / 'state.sqlite3'
    old = pair()
    old['markers']['meta']['marker_score_floor'] = .95
    seed(path, outputs=old)
    seed(path, outputs=pair('later'), history=True)
    assert issue(report(path))['outputs']['markers']['status'] == 'insufficient'


def test_removed_frame_blocklist_labels_require_reinference(tmp_path):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    outputs['markers']['meta']['marker_tag_blocklist'] = ['formerly_blocked']
    seed(path, outputs=outputs)
    assert issue(report(path))['outputs']['markers']['status'] == 'insufficient'


def test_index_without_results_is_inconsistent_and_not_ready(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.connection.execute("DELETE FROM results WHERE kind='scene'")
    data = report(path)
    assert issue(data)['outputs']['scene']['status'] == 'inconsistent'
    assert actions(data)[0]['action'] == 'restart_inference'


def test_wrong_index_signature_is_detected_without_decoding_predictions(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.connection.execute("UPDATE result_index SET signature='wrong' WHERE kind='scene'")
    assert issue(report(path))['outputs']['scene']['status'] == 'inconsistent'


def test_working_state_only_scene_and_checkpoint_progress_are_included(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.connection.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)',
                                     (store.scope, '2', 'wd', 'partial', 3, 10, 2, b'not decoded'))
    data = report(path)
    assert data['totals']['scenes'] == 2
    assert issue(data, '2')['checkpoint'] == dict(processed=3, total=10, compatibility='unverified')
    assert issue(data, '2')['outputs']['scene']['status'] == 'partial'
    assert actions(data)[0]['action'] == 'resume_or_infer'


def test_refresh_keeps_checkpoint_resume_instead_of_restart(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.connection.execute('INSERT INTO refresh_requests VALUES (?,?,?,?)', (store.scope, '1', 'wd', 'scene'))
            store.connection.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?)',
                                     (store.scope, '1', 'wd', 'partial', 3, 10, 2, b'not decoded'))
    data = report(path)
    assert issue(data)['outputs']['scene']['status'] == 'refresh_required'
    assert '--restart' not in actions(data)[0]['argv']


@pytest.mark.parametrize('blocker', ['publication', 'outside_refresh', 'replacement'])
def test_blocked_pairs_are_counted_and_receive_no_executable_plan(tmp_path, monkeypatch, blocker):
    path = tmp_path / 'state.sqlite3'
    outputs = pair()
    if blocker == 'replacement':
        outputs['markers']['meta']['marker_score_floor'] = .99
        monkeypatch.setattr(app, 'REPLACE_MARKERS', False)
    seed(path, outputs=outputs)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        if blocker == 'publication':
            store.prepare('1', {'scene': {'owned_ids': [], 'contributions': []}})
        elif blocker == 'outside_refresh':
            with store.connection:
                store.connection.execute('INSERT INTO refresh_requests VALUES (?,?,?,?)',
                                         (store.scope, '1', 'dtq', 'markers'))
    data = report(path)
    assert data['totals']['blocked_pairs'] == 1 and data['totals']['fully_ready_scenes'] == 0
    assert issue(data)['blockers'] and not data['plan']


@pytest.mark.parametrize('problem', ['floor', 'historical_floor', 'old_floor', 'blocklist',
                                    'scene_projection', 'unreadable'])
def test_unselected_publication_dependencies_block_plan_without_expanding_targets(
        tmp_path, monkeypatch, problem):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    peer = pair()
    if problem in ('floor', 'historical_floor'):
        peer['markers']['meta']['marker_score_floor'] = .8
    elif problem == 'old_floor':
        peer['markers']['meta'].pop('marker_score_floor')
    elif problem == 'blocklist':
        peer['markers']['meta']['marker_tag_blocklist'] = ['unblocked']
    elif problem == 'scene_projection':
        peer['scene']['meta'].update(scene_support_floor=.4, scene_score_floor=.4)
    seed(path, model='dtq', outputs=peer)
    if problem == 'historical_floor':
        seed(path, model='dtq', history=True)
    elif problem == 'unreadable':
        with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
            with state.connection:
                state.connection.execute("UPDATE results SET data=? WHERE model='dtq' AND kind='markers'",
                                         (b'broken',))
    before = path.read_bytes()
    if problem != 'unreadable':
        monkeypatch.setattr(merge.StateStore, 'unpack', lambda *args: pytest.fail('prediction BLOB decoded'))
    data = report(path, settings=[replace(config(), marker_threshold=.9)])
    assert [row['model'] for row in data['models']] == ['wd']
    assert data['totals']['scene_model_pairs'] == data['totals']['blocked_pairs'] == 1
    assert data['totals']['fully_ready_scenes'] == 0 and not data['plan']
    row = issue(data)
    assert 'unusable_evidence_for_unselected_model' in row['blockers']
    dependency, = row['publication_dependencies']
    assert dependency['model'] == 'dtq'
    assert dependency['kind'] == ('scene' if problem == 'scene_projection' else 'markers')
    assert dependency['reason'] and '--models' in dependency['resolution']
    assert path.read_bytes() == before


@pytest.mark.parametrize('blocker', ['replacement', 'unreadable'])
@pytest.mark.parametrize('order', [('wd', 'dtq'), ('dtq', 'wd')])
def test_blocked_selected_pair_blocks_every_command_for_its_scene(
        tmp_path, monkeypatch, blocker, order):
    path = tmp_path / 'state.sqlite3'
    broken = pair()
    if blocker == 'replacement':
        broken['markers']['meta']['marker_score_floor'] = .8
        monkeypatch.setattr(app, 'REPLACE_MARKERS', False)
    seed(path, outputs=broken)
    if blocker == 'unreadable':
        with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
            with state.connection:
                state.connection.execute("UPDATE results SET data=? WHERE model='wd' AND kind='markers'",
                                         (b'broken',))
    before = path.read_bytes()
    data = report(path, ['1', 'safe'], [config(model) for model in order])
    assert data['totals']['blocked_pairs'] == 2
    assert 'blocked_selected_model' in issue(data, model='dtq')['blockers']
    assert issue(data, model='dtq')['blocking_models'] == ['wd']
    assert all(row['scene_ids'] == ['safe'] for row in data['plan'])
    assert {row['model'] for row in actions(data)} == {'wd', 'dtq'}
    assert path.read_bytes() == before


def test_blocked_scene_is_not_added_back_by_publication_recovery(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    broken = pair()
    broken['markers']['meta']['marker_score_floor'] = .8
    seed(path, outputs=broken)
    monkeypatch.setattr(app, 'REPLACE_MARKERS', False)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as state:
        state.prepare('1', {'scene': {'owned_ids': [], 'contributions': []}})
        from stash_ai.recovery import abandon
        abandon(state, report=lambda _: None)
    data = report(path, settings=[config('wd'), config('dtq')])
    assert data['publication_recovery_scene_ids'] == ['1']
    assert not data['plan']


@pytest.mark.parametrize('peer_kind', ['no_scene', 'no_markers', 'imported_spans',
                                     'old_scene', 'different_threshold', 'higher_override'])
def test_usable_unselected_evidence_does_not_expand_or_block_selected_plan(tmp_path, peer_kind):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    peer = pair()
    if peer_kind == 'no_scene':
        del peer['scene']
    elif peer_kind == 'no_markers':
        del peer['markers']
    elif peer_kind == 'imported_spans':
        peer['markers'] = dict(mode='spans', meta=dict(origin='legacy'),
            items=[dict(tag='imported', start=0., end=2., score=.9, hits=2, extra_tags=[])])
    elif peer_kind == 'old_scene':
        peer['scene'].pop('frames')
        for field in ('scene_frames_format', 'scene_score_floor', 'scene_support_floor'):
            peer['scene']['meta'].pop(field)
    elif peer_kind == 'different_threshold':
        peer['markers']['meta'].update(marker_threshold=.5, marker_score_floor=.4)
    elif peer_kind == 'higher_override':
        peer['markers']['meta']['marker_score_floor'] = .75
        peer['markers']['marker_threshold_override'] = .9
    seed(path, model='dtq', outputs=peer)
    data = report(path, settings=[replace(config(), marker_threshold=.9)])
    assert data['totals']['rethreshold_pairs'] == 1 and data['totals']['blocked_pairs'] == 0
    assert [row['model'] for row in data['plan'] if 'model' in row] == ['wd']
    assert data['plan'][-1]['action'] == 'merge_after_backfill'


def test_unselected_dependency_checks_respect_scene_and_db_scope(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    peer = pair()
    peer['markers']['meta']['marker_score_floor'] = .8
    seed(path, scene='other', model='dtq', outputs=peer)
    with merge.StateStore(path, merge.scope_for('http://other.example')) as state:
        state.put('1', 'dtq', peer)
    data = report(path, ['1'], [replace(config(), marker_threshold=.9)])
    assert data['totals']['blocked_pairs'] == 0
    assert all(row['scene_ids'] == ['1'] for row in data['plan'])


def test_cli_reports_unselected_dependency_without_models_or_stash(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    peer = pair()
    peer['markers']['meta']['marker_score_floor'] = .8
    seed(path, model='dtq', outputs=peer)
    destination = tmp_path / 'cache.json'
    for name in ('load_tagger', 'load_vocabulary', 'ensure_files', 'create_stash_client'):
        monkeypatch.setattr(app, name, lambda *args: pytest.fail('model/Stash access'))
    before = path.read_bytes()
    assert app.main(['--cache-report', str(destination), '--state-db', str(path), '--models', 'wd']) == 0
    data = json.loads(destination.read_text())
    assert not data['plan'] and data['totals']['blocked_pairs'] == 1
    assert issue(data)['publication_dependencies'][0]['model'] == 'dtq'
    assert 'Unselected cached models block publication' in capsys.readouterr().out
    assert path.read_bytes() == before


def test_including_unselected_dependency_produces_a_plan_that_completes(pipeline, monkeypatch, tmp_path):
    import yaml
    p = pipeline
    wd = pair()
    wd['markers']['meta']['marker_score_floor'] = .2
    seed(p.path, outputs=wd)
    dtq = pair()
    dtq['markers']['meta']['marker_score_floor'] = .8
    seed(p.path, model='dtq', outputs=dtq)
    config_data = app.load_unique_yaml(app.MODELS_CONFIG_FILE, 'test')
    for model in ('wd', 'dtq'):
        config_data[model].update(scene_threshold=.4, marker_threshold=.9 if model == 'wd' else .8,
                                  scene_cache_floor=.1, marker_cache_floor=.2)
    yaml_path = tmp_path / 'models.yml'
    yaml_path.write_text(yaml.safe_dump(config_data))
    context = ['--state-db', str(p.path), '--model-config', str(yaml_path)]
    selected = app.load_model_settings(yaml_path, ['wd'], None, True)
    before = p.path.read_bytes()
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        blocked = cache_report.build_report(app, state, ['1'], selected, context)
    assert not blocked['plan'] and blocked['totals']['blocked_pairs'] == 1
    assert p.path.read_bytes() == before and not p.stash.writes
    settings = app.load_model_settings(yaml_path, ['wd', 'dtq'], None, True)
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as state:
        data = cache_report.build_report(app, state, ['1'], settings, context)
    assert data['totals']['blocked_pairs'] == 0
    assert [row['model'] for row in actions(data)] == ['dtq']
    assert [row['action'] for row in data['plan']] == [
        'restart_inference', 'rethreshold', 'merge_after_backfill']
    original = app.load_tagger
    loaded = []

    def load(config, *args):
        assert config.source.key == 'dtq'
        loaded.append(config.source.key)
        return original(config, *args)

    monkeypatch.setattr(app, 'load_tagger', load)
    assert [app.main([*row['argv'][2:], '--no-hierarchy']) for row in data['plan']] == [0, 0, 0]
    assert loaded == ['dtq']
    completed = report(p.path, settings=settings)
    assert completed['totals']['fully_ready_scenes'] == 1 and not completed['plan']


def test_pending_global_cleanup_blocks_backfill_without_changing_source(tmp_path):
    from stash_ai.cleanup import begin
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        begin(store, 'interrupted', {}, {}, {})
    before = path.read_bytes()
    data = report(path, ['1', '404'])
    assert data['totals']['blocked_pairs'] == 2 and not data['plan']
    assert issue(data)['blockers'] == ['pending_global_cleanup']
    assert path.read_bytes() == before


def test_archived_recovery_does_not_block_complete_cache_and_plans_reconciliation(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        store.prepare('1', {'scene': {'owned_ids': [], 'contributions': []}})
        from stash_ai.recovery import abandon
        abandon(store, report=lambda _: None)
    before = path.read_bytes()
    data = report(path)
    assert data['totals']['fully_ready_scenes'] == 1 and not data['issues']
    assert data['publication_recovery_scene_ids'] == ['1']
    assert data['plan'][0]['action'] == 'merge_after_backfill' and not actions(data)
    assert path.read_bytes() == before


def test_unreadable_payload_is_reported_as_blocked_not_replacement(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.connection.execute("UPDATE results SET data=? WHERE kind='scene'", (b'broken',))
    data = report(path)
    assert issue(data)['outputs']['scene']['unreadable']
    assert data['totals']['blocked_pairs'] == 1 and not data['plan']


def test_compact_metadata_path_does_not_read_prediction_blobs(tmp_path, monkeypatch):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    monkeypatch.setattr(merge.StateStore, 'unpack', lambda *args: pytest.fail('prediction BLOB decoded'))
    assert report(path)['totals']['fully_ready_scenes'] == 1


def test_legacy_metadata_backfill_and_filter_do_not_change_source(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    seed(path, scene='2')
    with merge.StateStore(path, merge.scope_for(app.STASH_URL)) as store:
        with store.connection:
            store.connection.execute('DELETE FROM result_metadata')
            store.connection.execute("UPDATE results SET data=? WHERE scene='2'", (b'broken',))
    before = path.read_bytes()
    assert report(path, ['1'])['totals']['fully_ready_scenes'] == 1
    assert path.read_bytes() == before


def test_scope_separation_and_empty_scope_are_explicit(tmp_path):
    path = tmp_path / 'state.sqlite3'
    with merge.StateStore(path, merge.scope_for('http://other.example')) as store:
        store.put('foreign', 'wd', pair())
    data = report(path)
    assert data['totals']['scenes'] == data['totals']['scene_model_pairs'] == 0
    assert data['models'][0]['reusable_pairs'] == 0 and not data['plan']


def test_plan_batches_are_bounded_and_argv_preserves_unusual_ids(tmp_path):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    scenes = [str(value) for value in range(101)] + ['odd id; $(literal)']
    data = report(path, scenes)
    assert all(len(row['scene_ids']) <= 50 for row in data['plan'])
    assert sum(len(row['scene_ids']) for row in actions(data)) == 101
    assert any('odd id; $(literal)' in row['argv'] for row in actions(data))


def test_cli_is_offline_and_preserves_settings_in_plan(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'state.sqlite3'
    seed(path)
    destination = tmp_path / 'cache.json'
    for name in ('load_tagger', 'load_vocabulary', 'ensure_files', 'create_stash_client'):
        monkeypatch.setattr(app, name, lambda *args: pytest.fail('model/Stash access'))
    before = path.read_bytes()
    assert app.main(['--cache-report', str(destination), '--state-db', str(path),
                     '--models', 'wd', '--scene-id', '404', '--no-temporal']) == 0
    data = json.loads(destination.read_text())
    assert data['targets'][0]['model'] == 'wd'
    assert '--no-temporal' in data['plan'][0]['argv']
    assert str(path.resolve()) in data['plan'][0]['argv']
    assert path.read_bytes() == before
    assert 'freshness is unverified' in capsys.readouterr().out


def test_cli_missing_database_does_not_create_database_or_report(tmp_path):
    path, destination = tmp_path / 'absent.sqlite3', tmp_path / 'cache.json'
    assert app.main(['--cache-report', str(destination), '--state-db', str(path)]) == 1
    assert not path.exists() and not destination.exists()


@pytest.mark.parametrize('extra', [
    ['--restart'], ['--resume'], ['--dry-run'], ['--device', 'cpu'], ['--offline'],
    ['--markers-only'], ['--scene-tags-only'], ['--output-mode', 'separate'],
    ['--abandon-pending'], ['--merge-only'], ['--tag-review-margin', '.1'],
])
def test_cli_rejects_incompatible_modifiers(tmp_path, extra):
    with pytest.raises(SystemExit):
        app.main(['--cache-report', str(tmp_path / 'cache.json'), *extra])


def test_cli_refuses_existing_report(tmp_path):
    destination = tmp_path / 'cache.json'
    destination.write_text('keep')
    with pytest.raises(SystemExit):
        app.main(['--cache-report', str(destination)])
    assert destination.read_text() == 'keep'


@pytest.mark.usefixtures('reset_progress')
def test_generated_backfill_resumes_actual_inference_checkpoint(pipeline, monkeypatch):
    from stash_ai.temporal import TemporalPolicy
    monkeypatch.setattr(app, 'DEFAULT_TEMPORAL', False)
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', TemporalPolicy(enabled=False))
    p = pipeline
    p.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(p.args)
    settings = app.load_model_settings(app.MODELS_CONFIG_FILE, ['wd'], None, True)
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as store:
        data = cache_report.build_report(app, store, ['1'], settings,
                                         ['--state-db', str(p.path), '--no-temporal'])
    assert actions(data)[0]['action'] == 'resume_or_infer'
    assert issue(data)['checkpoint']['processed'] == 4
    p.fail.clear()
    assert app.main([*actions(data)[0]['argv'][2:], '--no-hierarchy']) == 0
    assert p.loaded[-1].seen == list(range(4, 10))
    assert report(p.path, settings=settings)['totals']['fully_ready_scenes'] == 1


@pytest.mark.usefixtures('reset_progress')
def test_generated_rethreshold_applies_yaml_without_loading_a_model(pipeline, monkeypatch, tmp_path):
    import yaml
    from stash_ai.temporal import TemporalPolicy
    monkeypatch.setattr(app, 'DEFAULT_TEMPORAL', False)
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', TemporalPolicy(enabled=False))
    p = pipeline
    assert app.main(p.args) == 0
    data = app.load_unique_yaml(app.MODELS_CONFIG_FILE, 'test')
    data['wd'].update(scene_threshold=.5, marker_threshold=.6)
    yaml_path = tmp_path / 'thresholds.yml'
    yaml_path.write_text(yaml.safe_dump(data))
    settings = app.load_model_settings(yaml_path, ['wd'], None, True)
    with merge.StateStore(p.path, merge.scope_for(app.STASH_URL), True) as store:
        result = cache_report.build_report(app, store, ['1'], settings,
            ['--state-db', str(p.path), '--model-config', str(yaml_path), '--no-temporal'])
    assert not actions(result) and result['plan'][0]['action'] == 'rethreshold'
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('model loaded for rethreshold'))
    assert app.main([*result['plan'][0]['argv'][2:], '--no-hierarchy']) == 0
    assert report(p.path, settings=settings)['totals']['fully_ready_scenes'] == 1

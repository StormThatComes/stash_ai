"""Global exclusions across cached outputs, shared proofs and Stash cleanup."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from stash_ai.inference import sprites
from stash_ai import merge_cleanup
from stash_ai import separate_sync
from stash_ai import cli as app
from stash_ai import merge
from stash_ai.exclusions import policies_or_empty
from stash_ai.models import SOURCES, WD
from tests.helpers.stash import Stash
from tests.helpers.stash import sync
from tests.helpers.evidence import segment


pytestmark = pytest.mark.usefixtures("reset_temporal")


def policy_file(tmp_path, labels=(), by_model=None, key='excluded_scene_tags'):
    import yaml
    path = tmp_path/'global.yml'
    path.write_text(yaml.safe_dump({'version': 1, key: list(labels), 'by_model': by_model or {}}))
    return path


def names_in_scene(stash):
    return {tag['name'] for tag in stash.scene['tags']} | {
        tag['name'] for marker in stash.scene['scene_markers']
        for tag in [marker['primary_tag'], *marker['tags']]}


def raw_results(path):
    with merge.StateStore(path, merge.scope_for(app.STASH_URL), True) as store:
        return store.connection.execute('SELECT model,kind,data FROM results ORDER BY model,kind').fetchall()


@pytest.mark.parametrize('temporal', [False, True])
def test_global_exclusions_reuse_raw_results_and_remove_all_roles(pipeline, monkeypatch, tmp_path, temporal):
    p = pipeline
    args = ['--models', 'all', *p.args[2:]] + ([] if temporal else ['--no-temporal'])
    assert app.main(args) == 0
    assert 'ai:common' in names_in_scene(p.stash)
    before = raw_results(p.path)
    path = policy_file(tmp_path, ['common'])
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('blacklist reran inference'))
    assert app.main([*args, '--tag-exclusions', str(path)]) == 0
    assert 'ai:common' not in names_in_scene(p.stash)
    assert 'ai:common' not in p.stash.tags
    assert raw_results(p.path) == before
    # Removing the entry restores from raw evidence, including a pruned tag.
    path = policy_file(tmp_path)
    assert app.main([*args, '--merge-only', '--tag-exclusions', str(path)]) == 0
    assert 'ai:common' in names_in_scene(p.stash)


@pytest.mark.parametrize('dry', [False, True])
def test_cleanup_removes_shared_and_legacy_extras_without_images_or_models(pipeline, monkeypatch, tmp_path, dry):
    p = pipeline
    p.scores['wd'][:] = 0; p.scores['dtq'][:] = 0
    p.scores['wd'][:, 0] = .95; p.scores['dtq'][:, 1] = .94
    core = tmp_path/'core.yml'; core.write_text('version: 1\nexcluded_core_tags: [extra]\n')
    args = ['--models', 'wd', 'dtq', *p.args[2:], '--marker-core-exclusions', str(core)]
    assert app.main(args) == 0
    marker = next(m for m in p.stash.scene['scene_markers'] if separate_sync.owned_marker(m, merge.AI))
    assert 'ai:extra' in {t['name'] for t in marker['tags']}
    marker['tags'].append(p.stash.tag('op:extra'))
    manual = p.stash.marker(WD, 'keep', 'manual', manual=True)
    manual['tags'] = [p.stash.tag('dtq:extra'), p.stash.tag('manual')]
    before_scene, before_db = deepcopy(p.stash.scene), p.path.read_bytes()
    before_raw = raw_results(p.path)
    before_marker = marker['id'], marker['seconds'], marker['end_seconds'], marker['title']
    path = policy_file(tmp_path, ['extra'])
    for name in ('load_tagger', 'load_vocabulary'):
        monkeypatch.setattr(app, name, lambda *args: pytest.fail('cleanup accessed model/input assets'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *args: pytest.fail('cleanup accessed model/input assets'))
    p.stash.dry_run = dry
    cleanup = ['--models', 'all', *p.args[2:], '--cleanup-only', '--tag-exclusions', str(path),
               '--marker-core-exclusions', str(core)] + (['--dry-run'] if dry else [])
    assert app.main(cleanup) == 0
    if dry:
        assert p.stash.scene == before_scene and p.path.read_bytes() == before_db
        return
    assert not {'ai:extra', 'op:extra', 'dtq:extra'} & names_in_scene(p.stash)
    assert not {'ai:extra', 'op:extra', 'dtq:extra'} & set(p.stash.tags)
    marker = next(m for m in p.stash.scene['scene_markers'] if m['id'] == before_marker[0])
    assert (marker['id'], marker['seconds'], marker['end_seconds'], marker['title']) == before_marker
    assert next(m for m in p.stash.scene['scene_markers'] if m['id'] == 'manual')['tags'][0]['name'] == 'manual'
    assert raw_results(p.path) == before_raw
    p.stash.writes.clear()
    assert app.main(cleanup) == 0 and not p.stash.writes


@pytest.mark.parametrize('old_snapshot', [False, True])
def test_source_exclusion_rebuilds_extra_proofs_without_resurrecting_revoked_sources(tmp_path, old_snapshot):
    records = {
        'wd': [segment('wd', [[('core', .95), ('detail', .91)]] * 3)],
        'dtq': [segment('dtq', [[('detail', .92)]] * 3)],
    }
    cores = {source: frozenset({'detail'}) for source in SOURCES}
    empty = policies_or_empty()
    with merge.StateStore(tmp_path/'state.sqlite3', 'fixture') as store:
        for source, items in records.items(): store.put('1', source, {'markers': items[0]})
        _, spans = merge.evidence(app, store, '1', cores, empty)
        if old_snapshot:
            for span in spans: span['build_options'].pop('tag_exclusions', None)
        tags = policies_or_empty({'wd': ['detail']})
        cleaned = merge_cleanup.cleanup_support(app, store, '1', spans, {'wd'}, cores, tags)
        marker = merge.union_intervals(cleaned)[0]
        assert marker['extra_tags'] == ['detail']
        assert set(marker['extra_evidence']['detail']['source_evidence']) == {'dtq'}
        # The current file no longer bans WD. Cleanup must still retain that
        # earlier removal, using the frozen global exclusions from publication.
        tags = policies_or_empty({'dtq': ['detail']})
        cleaned = merge_cleanup.cleanup_support(app, store, '1', cleaned, {'dtq'}, cores, tags)
        marker = merge.union_intervals(cleaned)[0]
        assert marker['extra_tags'] == []
        assert (marker['start'], marker['end']) == (0, 3)


def test_global_exclusion_revokes_consensus_before_qualification():
    records = {'wd': [segment('wd', [[('core', .9)], [], [('core', .9)], []])],
               'dtq': [segment('dtq', [[], [('core', .9)], [], [('core', .9)]])]}
    assert merge.marker_contributions(app, records, policies_or_empty())
    assert merge.marker_contributions(app, records, policies_or_empty(),
                                     tag_policies=policies_or_empty({'wd': ['core']})) == []


@pytest.mark.parametrize('mode', ['--markers-only', '--cleanup-markers-only'])
def test_markers_only_always_loads_the_global_list(pipeline, monkeypatch, tmp_path, mode):
    p = pipeline
    assert app.main(p.args) == 0
    before_tags = deepcopy(p.stash.scene['tags'])
    path = policy_file(tmp_path, ['extra'])
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('partial cleanup reran model'))
    options = ['--merge-only', mode] if mode == '--markers-only' else [mode]
    assert app.main([*p.args, *options, '--scene-tag-exclusions', str(path)]) == 0
    assert p.stash.scene['tags'] == before_tags
    assert all(t['name'] != 'ai:extra' for m in p.stash.scene['scene_markers'] for t in [m['primary_tag'], *m['tags']])


@pytest.mark.parametrize('source', list(SOURCES.values()))
def test_separate_sync_removes_extras_from_retained_markers_and_scoped_manual_assignments(source, monkeypatch):
    monkeypatch.setattr(app, 'REPLACE_MARKERS', False)
    stash = Stash()
    retained = stash.marker(source, 'keep', 'retained')
    retained['tags'].extend([stash.tag(source.prefix+'blocked'), stash.tag('manual')])
    manual = stash.marker(source, 'manual_core', 'manual', manual=True)
    manual['tags'].append(stash.tag(source.prefix+'blocked'))
    removed = stash.marker(source, 'blocked', 'blocked', manual=True)
    before = retained['seconds'], retained['end_seconds'], retained['title']
    sync(stash, source, scenes=frozenset({'blocked'}))
    kept = next(m for m in stash.scene['scene_markers'] if m['id'] == 'retained')
    assert (kept['seconds'], kept['end_seconds'], kept['title']) == before
    assert source.prefix+'blocked' not in names_in_scene(stash)
    assert all(m['id'] != removed['id'] for m in stash.scene['scene_markers'])
    assert 'manual' in {t['name'] for t in kept['tags']}


@pytest.mark.parametrize('output_mode', ['merged', 'separate'])
def test_hierarchy_create_all_does_not_recreate_excluded_tags(pipeline, monkeypatch, tmp_path, output_mode):
    p = pipeline
    path = policy_file(tmp_path, ['common'], key='excluded_tags')
    args = ['--models', 'wd', '--output-mode', output_mode, '--sync-hierarchy-only', '--create-all-tags',
            '--tag-exclusions', str(path), '--state-db', str(p.path)]
    # Metadata hierarchy validation itself is covered separately; this exercise
    # targets the actual hierarchy command's full-vocabulary creation loop.
    monkeypatch.setattr(app, 'load_tag_hierarchy', lambda *args: None)
    monkeypatch.setattr(merge, 'combined_hierarchy', lambda *args: None)
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('hierarchy loaded a model'))
    assert app.main(args) == 0
    prefix = 'ai:' if output_mode == 'merged' else 'wd:'
    assert prefix+'extra' in p.stash.tags and prefix+'common' not in p.stash.tags


def test_combined_hierarchy_filters_each_source_before_union():
    root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(no_hierarchy=False, hierarchy=root/'joytag_hierarchy.yml',
        **{source+'_hierarchy': root/f'{source}_hierarchy.yml' for source in ('wd', 'dtq', 'oppai')})
    policies = {source: frozenset({'blue_hair'}) for source in SOURCES}
    hierarchy = merge.combined_hierarchy(app, args, policies)
    assert 'blue_hair' not in hierarchy.leaf_parents
    policies['wd'] = frozenset()
    assert 'blue_hair' in merge.combined_hierarchy(app, args, policies).leaf_parents


def test_exclusion_key_alias_is_backward_compatible_and_unambiguous(tmp_path):
    legacy = policy_file(tmp_path, ['common'])
    expected = app.load_exclusion_policy(legacy, 'excluded_scene_tags')
    current = policy_file(tmp_path, ['common'], key='excluded_tags')
    assert app.load_exclusion_policy(current, 'excluded_scene_tags') == expected
    current.write_text('version: 1\nexcluded_scene_tags: []\nexcluded_tags: [common]\n')
    with pytest.raises(ValueError, match='not both'):
        app.load_exclusion_policy(current, 'excluded_scene_tags')


def test_pruned_legacy_extra_id_cannot_reappear_or_break_the_next_merge(pipeline, tmp_path, monkeypatch):
    p = pipeline
    old = p.stash.marker(WD, 'legacy', 'legacy')
    old['tags'].append(p.stash.tag('ai:blocked'))
    path = policy_file(tmp_path, ['blocked'])
    args = [*p.args, '--merge-only', '--tag-exclusions', str(path)]
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('legacy merge loaded model'))
    assert app.main(args) == 0
    assert 'ai:blocked' not in p.stash.tags
    p.stash.writes.clear()
    assert app.main(args) == 0 and not p.stash.writes
    assert 'ai:blocked' not in names_in_scene(p.stash)


def test_cleanup_pruning_flag_is_supported_and_does_not_load_assets(pipeline, monkeypatch):
    p = pipeline
    p.stash.tag('op:unused')
    monkeypatch.setattr(app, 'load_tagger', lambda *args: pytest.fail('cleanup loaded model'))
    assert app.main([*p.args, '--cleanup-only', '--prune-empty-tags']) == 0
    assert 'op:unused' not in p.stash.tags


def test_failed_cleanup_does_not_start_definition_pruning(pipeline, monkeypatch, tmp_path):
    p = pipeline
    assert app.main(p.args) == 0
    path = policy_file(tmp_path, ['common'])
    monkeypatch.setattr(p.stash, 'delete_marker', lambda *args: (_ for _ in ()).throw(RuntimeError('write failed')))
    from stash_ai import prune as stash_prune
    monkeypatch.setattr(stash_prune, 'prune_tags', lambda *args, **kwargs: pytest.fail('pruning after failed cleanup'))
    assert app.main([*p.args, '--cleanup-only', '--tag-exclusions', str(path)]) == 1
    assert 'ai:common' in p.stash.tags

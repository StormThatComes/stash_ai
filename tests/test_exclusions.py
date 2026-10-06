"""Global exclusions across cached outputs, shared proofs and Stash cleanup."""
from tests.helpers.evidence import empty_policies as empty_tag_policies

from tests.helpers.temporal import policy_args

from stash_ai.publication import markers as marker_ops
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from stash_ai.commands import hierarchy as hierarchy_commands
from stash_ai.inference import sprites
from stash_ai.cleanup import evidence as cleanup_evidence
from tests.helpers.runtime import runtime as app
from stash_ai.evidence import pipeline as merge
import stash_ai.evidence.intervals as merge_intervals
import stash_ai.evidence.labels as merge_labels
import stash_ai.evidence.source_markers as merge_source_markers
import stash_ai.storage.store as merge_store
from tests.helpers.evidence import empty_policies
from stash_ai.models import SOURCES, WD
from tests.helpers.stash import Stash
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
    with merge_store.StateStore(path, merge_store.scope_for(app.STASH_URL), True) as store:
        return store.connection.execute('SELECT model,kind,data FROM results ORDER BY model,kind').fetchall()


@pytest.mark.parametrize('temporal', [False, True])
def test_global_exclusions_reuse_raw_results_and_remove_all_roles(pipeline, monkeypatch, tmp_path, temporal):
    p = pipeline
    args = ['--models', 'all', *p.args[2:]] + policy_args(p.path.parent, temporal)
    assert app.main(args) == 0
    assert 'ai:common' in names_in_scene(p.stash)
    before = raw_results(p.path)
    path = policy_file(tmp_path, ['common'])
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('blacklist reran inference'))
    assert app.main([*args, '--scene-tag-exclusions', str(path)]) == 0
    assert 'ai:common' not in names_in_scene(p.stash)
    assert 'ai:common' not in p.stash.tags
    assert raw_results(p.path) == before
    # Removing the entry restores from raw evidence, including a pruned tag.
    path = policy_file(tmp_path)
    assert app.main([*args, '--merge-only', '--scene-tag-exclusions', str(path)]) == 0
    assert 'ai:common' in names_in_scene(p.stash)


@pytest.mark.parametrize('dry', [False, True])
def test_cleanup_removes_ai_extras_and_preserves_foreign_tags_without_assets(pipeline, monkeypatch, tmp_path, dry):
    p = pipeline
    p.scores['wd'][:] = 0; p.scores['dtq'][:] = 0
    p.scores['wd'][:, 0] = .95; p.scores['dtq'][:, 1] = .94
    core = tmp_path/'core.yml'; core.write_text('version: 1\nexcluded_core_tags: [extra]\n')
    args = ['--models', 'wd', 'dtq', *p.args[2:], '--marker-core-exclusions', str(core)]
    assert app.main(args) == 0
    marker = next(m for m in p.stash.scene['scene_markers'] if marker_ops.owned_marker(m, merge_labels.AI))
    assert 'ai:extra' in {t['name'] for t in marker['tags']}
    marker['tags'].append(p.stash.tag('op:extra'))
    manual = p.stash.marker(WD, 'keep', 'manual', manual=True)
    manual['tags'] = [p.stash.tag('dtq:extra'), p.stash.tag('manual')]
    before_scene, before_db = deepcopy(p.stash.scene), p.path.read_bytes()
    before_raw = raw_results(p.path)
    before_marker = marker['id'], marker['seconds'], marker['end_seconds'], marker['title']
    path = policy_file(tmp_path, ['extra'])
    for name in ('load_tagger', 'load_vocabulary'):
        monkeypatch.setattr(app.services, name, lambda *args: pytest.fail('cleanup accessed model/input assets'))
    monkeypatch.setattr(sprites, 'fetch_cues', lambda *args: pytest.fail('cleanup accessed model/input assets'))
    p.stash.dry_run = dry
    cleanup = ['--models', 'all', *p.args[2:], '--cleanup-only', '--scene-tag-exclusions', str(path),
               '--marker-core-exclusions', str(core)] + (['--dry-run'] if dry else [])
    assert app.main(cleanup) == 0
    if dry:
        assert p.stash.scene == before_scene and p.path.read_bytes() == before_db
        return
    assert 'ai:extra' not in names_in_scene(p.stash)
    assert {'op:extra', 'dtq:extra'} <= names_in_scene(p.stash)
    assert 'ai:extra' not in p.stash.tags
    assert {'op:extra', 'dtq:extra'} <= set(p.stash.tags)
    marker = next(m for m in p.stash.scene['scene_markers'] if m['id'] == before_marker[0])
    assert (marker['id'], marker['seconds'], marker['end_seconds'], marker['title']) == before_marker
    assert {t['name'] for m in p.stash.scene['scene_markers'] if m['id'] == 'manual' for t in m['tags']} == {'dtq:extra', 'manual'}
    assert raw_results(p.path) == before_raw
    p.stash.writes.clear()
    assert app.main(cleanup) == 0 and not p.stash.writes


@pytest.mark.parametrize('old_snapshot', [False, True])
def test_source_exclusion_rebuilds_extra_proofs_without_resurrecting_revoked_sources(tmp_path, old_snapshot):
    records = {
        'wd': segment('wd', [[('core', .95), ('detail', .91)]] * 3),
        'dtq': segment('dtq', [[('detail', .92)]] * 3),
    }
    cores = {source: frozenset({'detail'}) for source in SOURCES}
    empty = empty_policies()
    with merge_store.StateStore(tmp_path/'state.sqlite3', 'fixture') as store:
        for source, items in records.items(): store.put('1', source, {'markers': items})
        _, spans = merge.evidence(app, store, '1', cores, empty)
        if old_snapshot:
            for span in spans: span['build_options'].pop('tag_exclusions', None)
            from stash_ai.db.snapshot_migration import normalize
            spans = normalize({'contributions': spans})['contributions']
        tags = {**empty_policies(), 'wd': frozenset({'detail'})}
        cleaned = cleanup_evidence.cleanup_support(app, store, '1', spans, {'wd'}, cores, tags)
        marker = merge_intervals.union_intervals(cleaned)[0]
        assert marker['extra_tags'] == ['detail']
        assert set(marker['extra_evidence']['detail']['source_evidence']) == {'dtq'}
        # The current file no longer bans WD. Cleanup must still retain that
        # earlier removal, using the frozen global exclusions from publication.
        tags = {**empty_policies(), 'dtq': frozenset({'detail'})}
        cleaned = cleanup_evidence.cleanup_support(app, store, '1', cleaned, {'dtq'}, cores, tags)
        marker = merge_intervals.union_intervals(cleaned)[0]
        assert marker['extra_tags'] == []
        assert (marker['start'], marker['end']) == (0, 3)


def test_global_exclusion_revokes_consensus_before_qualification():
    records = {'wd': segment('wd', [[('core', .9)], [], [('core', .9)], []]),
               'dtq': segment('dtq', [[], [('core', .9)], [], [('core', .9)]])}
    assert merge_source_markers.marker_contributions(app, records, empty_policies(), tag_policies=empty_tag_policies())
    assert merge_source_markers.marker_contributions(app, records, empty_policies(),
                                     tag_policies={**empty_policies(), 'wd': frozenset({'core'})}) == []


@pytest.mark.parametrize('mode', ['--markers-only', '--cleanup-markers-only'])
def test_markers_only_always_loads_the_global_list(pipeline, monkeypatch, tmp_path, mode):
    p = pipeline
    assert app.main(p.args) == 0
    before_tags = deepcopy(p.stash.scene['tags'])
    path = policy_file(tmp_path, ['extra'])
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('partial cleanup reran model'))
    options = ['--merge-only', mode] if mode == '--markers-only' else [mode]
    assert app.main([*p.args, *options, '--scene-tag-exclusions', str(path)]) == 0
    assert p.stash.scene['tags'] == before_tags
    assert all(t['name'] != 'ai:extra' for m in p.stash.scene['scene_markers'] for t in [m['primary_tag'], *m['tags']])


def test_current_cleanup_removes_extras_from_retained_manual_markers(monkeypatch, tmp_path):
    from stash_ai.evidence.labels import AI as source
    from tests.helpers.cleanup import options, install
    stash = Stash()
    retained = stash.marker(source, 'keep', 'retained', manual=True)
    retained['tags'].extend([stash.tag(source.prefix+'blocked'), stash.tag('manual')])
    manual = stash.marker(source, 'manual_core', 'manual', manual=True)
    manual['tags'].append(stash.tag(source.prefix+'blocked'))
    removed = stash.marker(source, 'blocked', 'blocked', manual=True)
    before = retained['seconds'], retained['end_seconds'], retained['title']
    install(monkeypatch, stash)
    assert app.main([*options(tmp_path, ['blocked']), '--cleanup-strategy', 'scenes']) == 0
    kept = next(m for m in stash.scene['scene_markers'] if m['id'] == 'retained')
    assert (kept['seconds'], kept['end_seconds'], kept['title']) == before
    assert source.prefix+'blocked' not in names_in_scene(stash)
    assert all(m['id'] != removed['id'] for m in stash.scene['scene_markers'])
    assert 'manual' in {t['name'] for t in kept['tags']}


def test_hierarchy_create_all_does_not_recreate_excluded_tags(pipeline, monkeypatch, tmp_path):
    p = pipeline
    path = policy_file(tmp_path, ['common'], key='excluded_scene_tags')
    args = ['--models', 'wd', '--sync-hierarchy-only', '--create-all-tags',
            '--scene-tag-exclusions', str(path), '--state-db', str(p.path)]
    # Metadata hierarchy validation itself is covered separately; this exercise
    # targets the actual hierarchy command's full-vocabulary creation loop.
    monkeypatch.setattr(app.services, 'load_tag_hierarchy', lambda *args: None)
    monkeypatch.setattr(hierarchy_commands, 'combined_hierarchy', lambda *args: None)
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('hierarchy loaded a model'))
    assert app.main(args) == 0
    prefix = 'ai:'
    assert prefix+'extra' in p.stash.tags and prefix+'common' not in p.stash.tags


def test_combined_hierarchy_filters_each_source_before_union():
    root = Path(__file__).resolve().parents[1]
    args = SimpleNamespace(no_hierarchy=False, hierarchy=root/'joytag_hierarchy.yml',
        **{source+'_hierarchy': root/f'{source}_hierarchy.yml' for source in ('wd', 'dtq', 'oppai')})
    policies = {source: frozenset({'blue_hair'}) for source in SOURCES}
    hierarchy = hierarchy_commands.combined_hierarchy(app, args, policies)
    assert 'blue_hair' not in hierarchy.leaf_parents
    policies['wd'] = frozenset()
    assert 'blue_hair' in hierarchy_commands.combined_hierarchy(app, args, policies).leaf_parents


@pytest.mark.parametrize('contents', [
    'version: 1\nexcluded_tags: [common]\n',
    'version: 1\nexcluded_scene_tags: []\nexcluded_tags: [common]\n',
])
def test_retired_exclusion_key_is_rejected(tmp_path, contents):
    path = tmp_path / 'policy.yml'
    path.write_text(contents)
    with pytest.raises(ValueError, match='excluded_scene_tags'):
        app.services.load_exclusion_policy(path, 'excluded_scene_tags')


@pytest.mark.parametrize('argv', [
    ['--model', 'wd'], ['--tag-exclusions', 'policy.yml'],
    ['--sync-marker-core-only'], ['--scene-tag-excl', 'policy.yml'],
])
def test_retired_cli_aliases_and_abbreviations_fail_before_stash_access(monkeypatch, argv):
    monkeypatch.setattr(app.services, 'create_stash_client', lambda *_: pytest.fail('invalid CLI accessed Stash'))
    with pytest.raises(SystemExit) as error:
        app.main(argv)
    assert error.value.code == 2


def test_missing_canonical_marker_policy_is_not_redirected_to_an_old_filename(tmp_path):
    from stash_ai.runtime.settings import RuntimeSettings
    current = tmp_path / 'marker_core_exclusions.yml'
    old = tmp_path / 'joytag_marker_core_exclusions.yml'
    old.write_text('version: 1\nexcluded_core_tags: [common]\n')
    runtime = RuntimeSettings(MARKER_CORE_EXCLUSIONS_FILE=current)
    assert runtime.MARKER_CORE_EXCLUSIONS_FILE == current
    with pytest.raises(FileNotFoundError, match='marker_core_exclusions.yml'):
        app.services.load_exclusion_policy(runtime.MARKER_CORE_EXCLUSIONS_FILE, 'excluded_core_tags')
    assert app.services.load_exclusion_policy(old, 'excluded_core_tags')['wd'] == frozenset({'common'})


def test_cleanup_pruning_flag_is_supported_and_does_not_load_assets(pipeline, monkeypatch):
    p = pipeline
    p.stash.tag('ai:unused')
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('cleanup loaded model'))
    assert app.main([*p.args, '--cleanup-only', '--prune-empty-tags']) == 0
    assert 'ai:unused' not in p.stash.tags


def test_failed_cleanup_does_not_start_definition_pruning(pipeline, monkeypatch, tmp_path):
    p = pipeline
    assert app.main(p.args) == 0
    path = policy_file(tmp_path, ['common'])
    monkeypatch.setattr(p.stash, 'delete_marker', lambda *args: (_ for _ in ()).throw(RuntimeError('write failed')))
    from stash_ai.cleanup import prune as stash_prune
    monkeypatch.setattr(stash_prune, 'prune_tags', lambda *args, **kwargs: pytest.fail('pruning after failed cleanup'))
    assert app.main([*p.args, '--cleanup-only', '--scene-tag-exclusions', str(path)]) == 1
    assert 'ai:common' in p.stash.tags

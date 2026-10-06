"""Shared taxonomy coverage, exclusion semantics and cache reuse contracts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from stash_ai.cli import main
from stash_ai.commands import bootstrap, hierarchy as hierarchy_commands
from stash_ai.commands.arguments import build_parser
from stash_ai.config import load_tag_hierarchy, validate_hierarchy_vocabulary, vocabulary_digest
from stash_ai.models import SOURCES, Vocabulary
from stash_ai.runtime.context import RuntimeContext
from tests.helpers.stash import Stash


def write_hierarchy(tmp_path, groups, *, metadata=False):
    labels = sorted({label for values in groups.values() for label in values})
    data = dict(version=2, groups=groups)
    if metadata:
        data['source'] = dict(models=list(SOURCES), tag_count=len(labels),
                              vocabulary_sha256=vocabulary_digest(labels))
    path = tmp_path/'hierarchy.yml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


def empty_policies():
    return {source: frozenset() for source in SOURCES}


def test_bundled_union_preserves_every_original_label_and_assignment():
    path = Path(__file__).resolve().parents[1]/'tag_hierarchy.yml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    parents = defaultdict(list)
    for group, labels in data['groups'].items():
        for label in labels:
            parents[label].append(group)
    assert len(parents) == 21154 and len(data['groups']) == 92
    # Independently computed from all four pre-unification YAMLs. Pinning both
    # digests catches lost model-specific labels and changes to any assignment.
    assert vocabulary_digest(sorted(parents)) == '3cd42350c5b0abac48eef1d24a1a3d2c9b988954fe40fcfc646f38fb82a7336e'
    canonical = {label: sorted(values) for label, values in sorted(parents.items())}
    mapping_digest = hashlib.sha256(json.dumps(canonical, ensure_ascii=False,
                                   separators=(',', ':')).encode()).hexdigest()
    assert mapping_digest == 'bad0581f651e776b9e90fda31ffce82c558c4a44455fe54a71bb7e0363ffbe6e'
    assert data['source']['models'] == list(SOURCES)
    assert len(load_tag_hierarchy(path).leaf_parents) == 21154


@pytest.mark.parametrize('source', SOURCES.values())
def test_shared_metadata_validates_union_and_accepts_model_subsets(tmp_path, source):
    path = write_hierarchy(tmp_path, {'First': ['common'], 'Second': ['other_model']}, metadata=True)
    hierarchy = load_tag_hierarchy(path)
    validate_hierarchy_vocabulary(hierarchy, ['common'], source)
    assert set(hierarchy.leaf_parents) == {'common', 'other_model'}
    with pytest.raises(ValueError, match=f'Hierarchy/{source.key}.*1 unmapped labels.*missing'):
        validate_hierarchy_vocabulary(hierarchy, ['common', 'missing'], source)


@pytest.mark.parametrize('field,value', [('tag_count', 1), ('vocabulary_sha256', vocabulary_digest(['common']))])
def test_shared_metadata_cannot_describe_only_one_model(tmp_path, field, value):
    path = write_hierarchy(tmp_path, {'First': ['common', 'other_model']}, metadata=True)
    data = yaml.safe_load(path.read_text())
    data['source'][field] = value
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=f'source.{field}.*shared vocabulary'):
        load_tag_hierarchy(path)


def test_extra_labels_are_validated_even_when_selected_model_does_not_predict_them(tmp_path):
    # Reserved category names are rejected even outside the selected vocabulary.
    path = write_hierarchy(tmp_path, {'Example': ['common', 'category:Example']})
    with pytest.raises(ValueError, match='reserved merged label'):
        load_tag_hierarchy(path)


def test_shared_file_preserves_explicit_multiple_parents(tmp_path):
    path = write_hierarchy(tmp_path, {'First': ['common'], 'Second': ['common']}, metadata=True)
    assert load_tag_hierarchy(path).leaf_parents['common'] == ('ai:category:First', 'ai:category:Second')


def test_bootstrap_reads_hierarchy_once_and_validates_excluded_model_labels(tmp_path):
    app = RuntimeContext()
    path = write_hierarchy(tmp_path, {'First': ['common', 'other_model']}, metadata=True)
    args = SimpleNamespace(no_hierarchy=False, hierarchy=path)
    settings = app.services.load_model_settings(app.MODELS_CONFIG_FILE, ['all'], None)
    reads = []
    app.services.load_tag_hierarchy = lambda path: (reads.append(path), load_tag_hierarchy(path))[1]
    app.services.load_vocabulary = lambda config: Vocabulary(['common'] if config.source.key == 'joytag' else ['other_model'], {})
    policies = {source: {'common'} for source in SOURCES}  # Mutable sets are valid too.
    catalogs, hierarchy = bootstrap.catalogs(app, settings, args, policies)
    assert reads == [path] and set(catalogs) == set(SOURCES)
    assert set(hierarchy.leaf_parents) == {'other_model'}
    app.services.load_vocabulary = lambda config: Vocabulary(['missing'], {})
    policies = {source: {'missing'} for source in SOURCES}
    with pytest.raises(ValueError, match='unmapped labels.*missing'):
        bootstrap.catalogs(app, settings, args, policies)


@pytest.mark.parametrize('models', ['joytag', 'wd', 'dtq', 'oppai', 'all'])
def test_custom_hierarchy_flag_applies_to_any_selection_without_stash_or_weights(tmp_path, models):
    app = RuntimeContext()
    path = write_hierarchy(tmp_path, {'First': ['common', 'other_model']}, metadata=True)
    app.services.load_vocabulary = lambda config: Vocabulary(['common'], {})
    app.services.create_stash_client = lambda *args: pytest.fail('validation contacted Stash')
    app.services.load_tagger = lambda *args: pytest.fail('validation loaded weights')
    assert main(['--models', models, '--validate-hierarchy', '--hierarchy', str(path), '--offline'], runtime=app) == 0


def test_hierarchy_sync_creates_only_selected_allowed_labels(tmp_path):
    app = RuntimeContext()
    path = write_hierarchy(tmp_path, {'First': ['common', 'blocked'], 'Second': ['other_model']})
    policy = tmp_path/'exclusions.yml'
    policy.write_text('version: 1\nexcluded_scene_tags: []\nby_model: {wd: [blocked]}\n')
    stash = Stash()
    app.services.load_vocabulary = lambda config: Vocabulary(['common', 'blocked'], {})
    app.services.create_stash_client = lambda *args: stash
    app.services.load_tagger = lambda *args: pytest.fail('hierarchy sync loaded weights')
    assert main(['--models', 'wd', '--sync-hierarchy-only', '--create-all-tags',
                 '--hierarchy', str(path), '--scene-tag-exclusions', str(policy), '--offline'], runtime=app) == 0
    assert 'ai:common' in stash.tags
    assert 'ai:blocked' not in stash.tags and 'ai:other_model' not in stash.tags


def test_offline_hierarchy_load_does_not_access_model_vocabularies(tmp_path):
    app = RuntimeContext()
    path = write_hierarchy(tmp_path, {'First': ['common']}, metadata=True)
    app.services.load_vocabulary = lambda *args: pytest.fail('offline merge accessed model assets')
    args = SimpleNamespace(no_hierarchy=False, hierarchy=path)
    assert hierarchy_commands.combined_hierarchy(app, args, empty_policies()).leaf_parents['common'] == ('ai:category:First',)


@pytest.mark.parametrize('flag', ['--wd-hierarchy', '--dtq-hierarchy', '--oppai-hierarchy'])
def test_removed_model_hierarchy_flags_are_rejected(flag):
    with pytest.raises(SystemExit):
        build_parser(RuntimeContext()).parse_args([flag, 'old.yml'])


def test_no_hierarchy_does_not_require_a_shared_file(tmp_path):
    app = RuntimeContext()
    app.services.load_tag_hierarchy = lambda *args: pytest.fail('disabled hierarchy read YAML')
    args = SimpleNamespace(no_hierarchy=True, hierarchy=tmp_path/'missing.yml')
    assert hierarchy_commands.combined_hierarchy(app, args, empty_policies()) is None


@pytest.mark.usefixtures('reset_merge')
def test_hierarchy_edit_reuses_inference_and_reparents_existing_tags(pipeline, monkeypatch, tmp_path):
    from tests.helpers.runtime import runtime as app
    path = write_hierarchy(tmp_path, {'First': ['common', 'extra']}, metadata=True)
    args = [value for value in pipeline.args if value != '--no-hierarchy'] + ['--hierarchy', str(path), '--resume']
    assert app.main(args) == 0
    assert len(pipeline.loaded) == 1
    assert {parent['name'] for parent in pipeline.stash.tags['ai:common']['parents']} == {'ai:category:First'}
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('hierarchy edit reran inference'))
    data = yaml.safe_load(path.read_text())
    data['groups'] = {'Second': ['common', 'extra']}
    path.write_text(yaml.safe_dump(data))  # The union metadata stays valid.
    assert app.main(args) == 0
    assert {parent['name'] for parent in pipeline.stash.tags['ai:common']['parents']} == {'ai:category:Second'}
    assert 'ai:category:First' in pipeline.stash.tags

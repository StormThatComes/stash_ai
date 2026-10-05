"""Explicit hierarchy migration preserves taxonomy and rejects partial conversions."""
from copy import deepcopy
import yaml
import pytest

from stash_ai.commands.hierarchy_migration import convert, migrate
from stash_ai.config import load_tag_hierarchy, vocabulary_digest
from stash_ai.evidence.labels import AI
from stash_ai.models import WD


def document():
    return dict(version=1, root='WD', category_prefix='wd:category:',
        source=dict(tag_count=2, vocabulary_sha256=vocabulary_digest(['core', 'extra'])),
        groups={'Appearance / Hair': ['core'], 'General': ['extra', 'core']})


def test_runtime_requires_canonical_hierarchy_and_explicit_migration(tmp_path):
    path = tmp_path/'hierarchy.yml'
    original = yaml.safe_dump(document()).encode()
    path.write_bytes(original)
    with pytest.raises(ValueError, match='hierarchy_migration'):
        load_tag_hierarchy(path, ['core', 'extra'], WD)
    assert migrate([path], dry_run=True) == [path]
    assert path.read_bytes() == original and not (tmp_path/'hierarchy.v1.yml').exists()
    assert migrate([path]) == [path]
    assert (tmp_path/'hierarchy.v1.yml').read_bytes() == original
    hierarchy = load_tag_hierarchy(path, ['core', 'extra'], WD)
    assert hierarchy.category_parents['ai:category:Appearance / Hair'] == ('ai:category:Appearance',)
    assert hierarchy.leaf_parents['core'] == ('ai:category:Appearance / Hair', 'ai:category:General')
    assert hierarchy.description_prefix == AI.hierarchy_description
    assert yaml.safe_load(path.read_text())['groups'] == document()['groups']
    assert migrate([path]) == []


def test_all_inputs_are_validated_before_any_file_is_changed(tmp_path):
    good, bad = tmp_path/'good.yml', tmp_path/'bad.yml'
    good.write_text(yaml.safe_dump(document()))
    invalid = document()
    invalid['source']['vocabulary_sha256'] = 'wrong'
    bad.write_text(yaml.safe_dump(invalid))
    before = good.read_bytes(), bad.read_bytes()
    with pytest.raises(ValueError, match='vocabulary_sha256'):
        migrate([good, bad])
    assert (good.read_bytes(), bad.read_bytes()) == before
    assert not list(tmp_path.glob('*.v1.yml'))


def test_migration_rejects_duplicate_paths_and_preserves_existing_backups(tmp_path):
    path = tmp_path/'hierarchy.yml'
    path.write_text(yaml.safe_dump(document()))
    with pytest.raises(ValueError, match='Duplicate'):
        migrate([path, path])
    backup = tmp_path/'hierarchy.v1.yml'
    backup.write_text('manual backup')
    with pytest.raises(ValueError, match='backup differs'):
        migrate([path])
    assert yaml.safe_load(path.read_text())['version'] == 1
    assert backup.read_text() == 'manual backup'

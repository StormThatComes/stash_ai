"""Compatibility boundary for extracted tagger vocabulary helpers."""
from stash_ai import taggers
from stash_ai import vocabulary


def test_taggers_reexports_vocabulary_helpers():
    assert taggers.metadata_files is vocabulary.metadata_files
    assert taggers.weight_files is vocabulary.weight_files
    assert taggers.ensure_files is vocabulary.ensure_files
    assert taggers.validate_labels is vocabulary.validate_labels
    assert taggers.load_vocabulary is vocabulary.load_vocabulary
    assert taggers.load_oppai_vocabulary is vocabulary.load_oppai_vocabulary
    assert taggers.CATEGORY_NAMES is vocabulary.CATEGORY_NAMES

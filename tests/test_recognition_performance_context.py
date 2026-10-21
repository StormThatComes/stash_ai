from dataclasses import replace

import pytest

from stash_ai.recognition.clustering import Group, matching_policy
from stash_ai.recognition.config import digest, load, plain
from stash_ai.recognition.context import policy_scope, portrait_reference_scope, portrait_references
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.merge_publication import canonical_targets


def test_policy_scope_preserves_keys_and_does_not_reuse_mutated_or_later_config(tmp_path, monkeypatch):
    from stash_ai.recognition import context
    config = load(root=tmp_path)
    expected = digest({'format': 24, 'recognition': config.data['recognition'],
                       'clustering': {k:v for k,v in config.data['clustering'].items()
                                      if k not in ('max_reference_bytes','max_people_per_lookup')}})
    calls, original = [], context.digest
    def counted(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(context, 'digest', counted)
    with policy_scope():
        assert matching_policy(config) == matching_policy(config) == expected
        assert len(calls) == 1
        mutable = replace(config, data=plain(config.data))
        assert matching_policy(mutable) == expected
        mutable.data['recognition']['match_threshold'] = .8
        assert matching_policy(mutable) != expected
    with policy_scope():
        assert matching_policy(config) == expected
    assert len(calls) == 4


def test_portrait_references_are_shared_only_inside_one_finalized_phase(tmp_path):
    import numpy as np
    config, db, calls = load(root=tmp_path), object(), []
    def build():
        calls.append(1)
        return [Group('p', [np.array([1., 0.], dtype='float32')], True)]
    with portrait_reference_scope():
        first = portrait_references(db, 'scope', 'namespace', config, build)
        assert portrait_references(db, 'scope', 'namespace', config, build)[0] is first[0]
        portrait_references(db, 'other-scope', 'namespace', config, build)
    with portrait_reference_scope():
        second = portrait_references(db, 'scope', 'namespace', config, build)
        assert first[0] is not second[0] and first[1] == second[1]
    assert len(calls) == 3


def test_canonical_target_snapshot_deduplicates_scans_and_blocks_pending_descendants(tmp_path):
    with FaceDatabase(tmp_path / 'faces.sqlite') as database:
        db = database.connection
        for index, person in enumerate(('old', 'middle', 'young')):
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       (person, 'scope', 'ns', index, '1', person))
        db.execute("INSERT INTO identity_merges VALUES ('young','middle',1,'{}','local')")
        db.execute("INSERT INTO identity_merges VALUES ('middle','old',1,'{}','confirmed')")
        scan = {'scene': '2', 'scan': 's2'}
        ready = {'young': {'person': 'young', 'scans': [scan, dict(scan)]},
                 'middle': {'person': 'middle', 'scans': [scan, {'scene': '3', 'scan': 's3'}]}}
        result = canonical_targets(database, ready)
        assert result == [{'person': 'old', 'performer': '1', 'scans': [scan, {'scene': '3', 'scan': 's3'}]}]
        db.execute("UPDATE identity_merges SET status='pending' WHERE source='young'")
        from stash_ai.recognition.publication import PendingRecognition
        with pytest.raises(PendingRecognition, match='metadata withheld'):
            canonical_targets(database, ready)
        db.execute("UPDATE identity_merges SET target='young' WHERE source='middle'")
        with pytest.raises(ValueError, match='cycle'):
            canonical_targets(database, ready)

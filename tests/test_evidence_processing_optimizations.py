"""Streaming prefix validation and exact, selective frozen proof replay."""
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from stash_ai.storage.checkpoint_contract import validate_chunks
from stash_ai.storage.store import StateStore
from stash_ai.storage import snapshots
from stash_ai.evidence.frozen_replay import PublishedRecords
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.temporal.policy import TemporalPolicy, frozen_policy, validation_scope
from tests.helpers.evidence import span
from tests.test_result_thresholds import outputs


def frame(i):
    return dict(start=float(i),end=float(i+1),tags=[['core',.9]],scene_tags=[])


def test_chunks_are_consumed_and_validated_once_without_rebuilding_prior_prefix(monkeypatch):
    from stash_ai.db import payloads
    valid, calls = payloads.valid_frames, []
    def checked(rows, **kwargs):
        calls.append(len(rows))
        return valid(rows, **kwargs)
    monkeypatch.setattr(payloads,'valid_frames',checked)
    def chunks():
        for i in range(0,20,2):
            assert len(calls)==i//2  # No eager decoding of later chunks.
            yield i+2,[frame(i),frame(i+1)]
    assert validate_chunks(chunks(),20)==[frame(i) for i in range(20)]
    assert calls==[2]*10


@pytest.mark.parametrize('chunks,processed', [
    ([(1,[frame(0)])],2),
    ([(2,[frame(0)]),(3,[frame(2)])],3),
    ([(1,[frame(1)]),(2,[frame(0)])],2),
    ([(1,[frame(0)]),(2,[dict(start=.5,end=1.5,tags=[])])],2),
    ([(1,[dict(start=0,end=1,tags=[['core',float('nan')]])])],1),
    ([(1,[frame(0)]),(1,[frame(1)])],2),
])
def test_invalid_chunk_progress_and_cross_chunk_timeline_are_rejected(chunks,processed):
    with pytest.raises(ValueError,match='checkpoint frame'):
        validate_chunks(iter(chunks),processed)


def test_frozen_replay_decodes_only_requested_proof_and_reuses_it(tmp_path,monkeypatch):
    with StateStore(tmp_path/'state.db','scope') as store:
        store.put('1','wd',outputs(signature='active'))
        records=[outputs(signature=f'archive-{i}')['markers'] for i in range(100)]
        for record in records:
            store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
                (store.scope,'1','wd','markers',marker_segment_signature(record),store.pack(record)))
        # An unrelated corrupt archive must not prevent exact valid replay.
        store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
            (store.scope,'1','wd','markers','unrelated',b'invalid'))
        store.connection.commit()
        original,calls=store.unpack,[]
        def unpack(blob):
            calls.append(1)
            return original(blob)
        monkeypatch.setattr(store,'unpack',unpack)
        published=PublishedRecords(store,'1')
        signature=marker_segment_signature(records[13])
        assert published.matching('markers',{'wd':signature},error='missing')['wd']==records[13]
        assert len(calls)==2
        published.matching('markers',{'wd':signature},error='missing')
        assert len(calls)==2
        with pytest.raises(RuntimeError,match='missing'):
            published.matching('markers',{'wd':'absent'},error='missing')
        store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',
            (store.scope,'1','wd','markers','wrong-key',store.pack(records[0])))
        with pytest.raises(ValueError,match='signature mismatch'):
            published.matching('markers',{'wd':'wrong-key'},error='missing')


def test_repeated_options_validate_policy_once_per_operation_and_each_item(monkeypatch):
    calls=[]
    original=TemporalPolicy.validate
    def validate(policy):
        calls.append(1)
        return original(policy)
    monkeypatch.setattr(TemporalPolicy,'validate',validate)
    item=span('wd',0,2)
    snapshot=dict(contributions=[dict(item) for _ in range(100)])
    calls.clear()
    snapshots.compact(snapshot)
    assert len(calls)==1
    snapshots.compact(snapshot)
    assert len(calls)==2
    broken=deepcopy(snapshot)
    broken['contributions'][-1]['frame_context']={'wd':['invalid']}
    with pytest.raises(ValueError,match='one signature'):
        snapshots.compact(broken)


def test_type_changes_and_distinct_frozen_policies_never_reuse_wrong_context():
    options=span('wd',0,2)['build_options']
    with validation_scope():
        first=frozen_policy(options)
        with pytest.raises(FrozenInstanceError): first.marker_continue_ratio=.4
        changed=deepcopy(options)
        changed['policy']['marker_continue_ratio']=.7
        assert frozen_policy(changed).marker_continue_ratio==.7
        changed['policy']['scene_min_hits']=float(first.scene_min_hits)
        with pytest.raises(ValueError,match='integer'):
            frozen_policy(changed)
        from stash_ai.evidence.temporal.policy import validate_marker_settings
        settings=dict(options['settings'])
        validate_marker_settings(settings)
        settings['MARKER_MIN_HITS']=float(settings['MARKER_MIN_HITS'])
        with pytest.raises(ValueError,match='marker setting'):
            validate_marker_settings(settings)

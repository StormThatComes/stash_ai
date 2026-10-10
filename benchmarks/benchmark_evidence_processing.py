"""Controlled checkpoint, exact archive replay and shared-option validation costs."""
import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

from stash_ai.storage.codec import decode,encode
from stash_ai.storage.checkpoint_contract import validate_chunks
from stash_ai.db.payloads import valid_frames
from stash_ai.storage.store import StateStore
from stash_ai.storage.archive import records as archived_records
from stash_ai.evidence.frozen_replay import PublishedRecords
from stash_ai.evidence.identity import marker_segment_signature
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.storage.snapshots import compact,validate_policy
from stash_ai.runtime.context import RuntimeContext
from benchmarks.fixtures import record as proof_record


def measure(callback):
    tracemalloc.start();start=time.perf_counter();value=callback()
    elapsed=time.perf_counter()-start;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return dict(seconds=elapsed,python_peak_bytes=peak),value


def checkpoints(frames,chunks):
    encoded=[]
    size=max(1,frames//chunks)
    for start in range(0,frames,size):
        rows=[dict(start=float(i),end=float(i+1),tags=[['tag',.9]]) for i in range(start,min(start+size,frames))]
        encoded.append((start+len(rows),encode(rows)))
    def reference():
        decoded=[(end,decode(blob)) for end,blob in encoded]
        restored=[];previous=0
        for end,chunk in decoded:
            assert valid_frames(chunk,checkpoint=True) and end==previous+len(chunk)
            restored.extend(chunk);previous=end
        assert previous==frames and valid_frames(restored,checkpoint=True)
        assert valid_frames(restored,checkpoint=True)
        return restored
    ref,a=measure(reference)
    optimized,b=measure(lambda:validate_chunks(((end,decode(blob)) for end,blob in encoded),frames))
    assert a==b
    return dict(frames=frames,chunks=len(encoded),reference=ref,optimized=optimized)


def archives(count):
    # The shared benchmark fixture builder yields valid current frame evidence.
    def proof(identity):
        return proof_record("wd", [dict(start=0.,end=1.,tags=[["tag",.9]])], identity=identity)
    with tempfile.TemporaryDirectory(prefix='stash-replay-bench-') as directory:
        with StateStore(Path(directory)/'state.db','scope') as store:
            store.put('1','wd',{'markers':proof('active')})
            wanted=None
            for i in range(count):
                record=proof(f'archive-{i}');signature=marker_segment_signature(record)
                store.connection.execute('INSERT INTO inference_archive VALUES (?,?,?,?,?,?)',(store.scope,'1','wd','markers',signature,store.pack(record)))
                if i==count//2:wanted=signature
            store.connection.commit()
            unpack=store.unpack;calls=[]
            def decode_counted(blob):calls.append(1);return unpack(blob)
            store.unpack=decode_counted
            def reference():
                archived=archived_records(store,'1','markers')['wd']
                active=store.model_record('1','markers','wd',borrow=True)
                return {marker_segment_signature(r):r for r in [*archived,active]}[wanted]
            ref,a=measure(reference);ref['decodes']=len(calls);calls.clear()
            optimized,b=measure(lambda:PublishedRecords(store,'1').matching('markers',{'wd':wanted},error='missing')['wd'])
            optimized['decodes']=len(calls);assert a==b
            return dict(archives=count,reference=ref,optimized=optimized)


def policies(count):
    from stash_ai.evidence.temporal.policy import build_options
    from stash_ai.models import SOURCES
    app=RuntimeContext(TEMPORAL_POLICY=TemporalPolicy())
    options={**build_options(app),'core_exclusions':{k:[] for k in SOURCES},'tag_exclusions':{k:[] for k in SOURCES}}
    snapshot=dict(contributions=[dict(build_options=options) for _ in range(count)])
    original=TemporalPolicy.validate;calls=[]
    def counted(policy):calls.append(1);return original(policy)
    TemporalPolicy.validate=counted
    try:
        ref,_=measure(lambda:[validate_policy(item) for item in snapshot['contributions']]);ref['policy_validations']=len(calls);calls.clear()
        optimized,_=measure(lambda:compact(snapshot));optimized['policy_validations']=len(calls)
    finally:TemporalPolicy.validate=original
    return dict(contributions=count,reference=ref,optimized=optimized)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=int,default=5000);p.add_argument('--chunks',type=int,default=100);p.add_argument('--archives',type=int,default=200);p.add_argument('--contributions',type=int,default=500)
    args=p.parse_args();print(json.dumps(dict(checkpoint=checkpoints(args.frames,args.chunks),archives=archives(args.archives),options=policies(args.contributions)),indent=2))


if __name__=='__main__':main()

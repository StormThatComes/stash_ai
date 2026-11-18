import json
import sqlite3
import numpy as np
from stash_ai.recognition.config import defaults
from stash_ai.recognition.face_evidence_review import review, collapse
from stash_ai.recognition.independent_evidence import count


def rows():
    geometry=[10,10,80,80,30,30,60,30,45,45,35,65,55,65,.99]
    return [dict(id=str(i),scan='s',cue=i,seconds=i,geometry=json.dumps(geometry),dimensions=2,
                 embedding=np.asarray([0,1] if i==2 else [1,0],dtype='<f4').tobytes(),
                 quality=json.dumps(dict(view_shot=0,transition_verified=True,alignment_valid=True,
                                         evidence_pixel_sha256=f'{i:064x}'))) for i in range(5)]


def test_anchor_demotion_requires_independent_bracketing_and_is_reversible():
    source=rows(); grouping=defaults()['clustering']
    assert set(review(source,grouping))=={'2'}
    assert all('query_only' not in json.loads(r['quality']) for r in source)
    source[2]['embedding']=source[0]['embedding']
    assert not review(source,grouping)
    source=rows()
    for r in source:r['quality']=json.dumps(dict(view_shot=0,transition_verified=False))
    assert not review(source,grouping)


def test_near_duplicate_votes_use_original_witness_without_transitive_drift():
    source=rows()[:3]
    pixels=np.tile(np.arange(32,dtype=np.uint8)*6,(32,3,1)).transpose(0,2,1)
    for i,r in enumerate(source):
        r['embedding']=source[0]['embedding']
        q=json.loads(r['quality']);q.update(evidence_thumbnail=(pixels+i*2).tobytes().hex(),evidence_aspect=1.)
        r['quality']=json.dumps(q)
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE observations(id TEXT,quality TEXT)')
        db.executemany('INSERT INTO observations VALUES (?,?)',[(r['id'],r['quality']) for r in source])
        collapse(db,source,dict(near_duplicate_pixel_rms=2.1,near_duplicate_similarity=.995))
        assert count(source)==2
        assert 'evidence_independence_unit' not in json.loads(source[2]['quality'])

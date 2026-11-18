import json
from types import SimpleNamespace
import numpy as np
from stash_ai.recognition.config import defaults
from stash_ai.recognition.body_identity import Scores
from stash_ai.recognition.body_numeric import body_similarity, vector_digest


def settings():return SimpleNamespace(data=defaults())


def row(i,masked):
    return dict(id=str(i),_vector=np.frombuffer(np.array([1.,0.],np.float32).tobytes(),np.float32),
                quality=json.dumps({'quality':{'embedding_foreground':masked}}))


def test_mask_penalty_is_shared_by_banks_tracks_and_contact_proofs():
    config=settings();config.data['sapiensid']['identities']['mixed_mask_score_penalty']=.1
    scores=Scores(config);a,b=row(1,True),row(2,False)
    assert scores.compare([a],[b])==(.9,.9)
    assert body_similarity(scores,a,b)==.9
    assert scores.face_similarity(a['_vector'],b['_vector'])==1.
    assert body_similarity(scores,a,row(3,None))==1.


def test_immutable_vector_digest_is_computed_once_without_run_cache(monkeypatch):
    import stash_ai.recognition.body_numeric as numeric
    original=numeric.vector_digest;calls=[]
    monkeypatch.setattr(numeric,'vector_digest',lambda a:(calls.append(a),original(a))[1])
    scores=Scores(settings());a,b=row(1,True),row(2,False)
    for _ in range(10):scores.face_similarity(a['_vector'],b['_vector'])
    assert len(calls)==2


def test_immutable_slices_do_not_alias_and_mutable_vectors_are_rehashed():
    from stash_ai.recognition.run_cache import scope
    source=np.frombuffer(np.arange(8,dtype=np.float32).tobytes(),np.float32)
    with scope(settings()):
        assert vector_digest(source[:4])!=vector_digest(source[4:])
        mutable=source.copy();first=vector_digest(mutable);mutable[0]=10
        assert vector_digest(mutable)!=first

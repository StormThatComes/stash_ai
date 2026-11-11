from contextlib import nullcontext
from types import SimpleNamespace
import numpy as np
import pytest
from stash_ai.recognition.config import defaults,validate
from stash_ai.recognition import inference_precision


def test_precision_invalidates_only_dependent_body_and_pose_contracts():
    from stash_ai.recognition.inference_contracts import face,detector,pose,body_encoder
    from stash_ai.recognition.person_signal import contract
    from stash_ai.recognition.body_observations import contract as body_contract
    from stash_ai.recognition.sapiensid_backend import identity_contract
    data=defaults();cfg=SimpleNamespace(data=data)
    data['sapiensid']['pose']['backend']='pmpose'
    data['sapiensid']['pose']['refinement']['enabled']=True
    signatures=lambda:[f(cfg) for f in (face,detector,pose,body_encoder,contract,body_contract,identity_contract)]
    before=signatures()
    for key in ('pmpose_precision','sam_precision'):data['execution'].pop(key)
    assert signatures()==before
    for key in ('pmpose_precision','sam_precision'):
        data['execution'][key]='bf16'
        after=signatures()
        assert [a!=b for a,b in zip(before,after)]==[False,False,True,False,True,True,True]
        before=after
    from stash_ai.recognition.cached_pose import RawPMPose
    model=object()
    assert RawPMPose(model,'cuda').contract!=RawPMPose(model,'cuda',precision='bf16').contract


@pytest.mark.parametrize('key',['pmpose_precision','sam_precision'])
def test_optional_precision_rejects_cpu_and_inactive_models(key):
    data=defaults();data['execution'][key]='fp16'
    with pytest.raises(ValueError,match='requires.*backend'):validate(data)
    data['sapiensid']['pose']['backend']='pmpose'
    data['sapiensid']['pose']['refinement']['enabled']=True
    data['models']['device']='cpu'
    with pytest.raises(ValueError,match='requires CUDA'):validate(data)
    data['models']['device']='cuda';validate(data)
    data['execution'][key]='int8'
    with pytest.raises(ValueError,match='fp32, fp16 or bf16'):validate(data)


def test_pmpose_converts_reduced_outputs_before_averaging_and_decode(monkeypatch):
    torch=pytest.importorskip('torch')
    from stash_ai.recognition.pmpose_pose import PMPose
    observed=[]
    monkeypatch.setattr(inference_precision,'autocast',lambda device,precision:observed.append(precision) or nullcontext())
    def model(inputs):
        return [torch.zeros(len(inputs),17,64,48,dtype=torch.bfloat16)]+[torch.ones(len(inputs),17,dtype=torch.bfloat16) for _ in range(3)]
    pose=PMPose(model,device='cpu',batch_size=1)
    pose.precision='bf16'  # Simulated model output; no GPU numerical claim.
    outputs=pose._forward(torch.zeros(2,3,256,192))
    assert observed==['bf16'] and all(v.dtype==torch.float32 for v in outputs)


def test_sam_scalar_casts_bf16_scores_before_numpy():
    torch=pytest.importorskip('torch')
    from stash_ai.recognition.sam_pose2seg import Segmenter
    class Predictor:
        def _prep_prompts(self,*args):return None,None,None,None
        def _predict(self,*args,**kwargs):
            return torch.ones(1,1,4,5,dtype=torch.bool),torch.full((1,1),.75,dtype=torch.bfloat16),object()
    segmenter=Segmenter(Predictor(),'cpu');segmenter.precision='bf16'
    mask,score=segmenter.predict(np.array([[1,2]]))
    assert mask.shape==(4,5) and score==.75


def test_cuda_events_are_bounded_nonblocking_and_disabled_on_cpu(monkeypatch):
    torch=pytest.importorskip('torch')
    from stash_ai.recognition.profiling import pipeline_scope,pipeline_snapshot,pipeline_cuda_timing
    ready=False;events=[]
    class Event:
        def __init__(self,**kwargs):events.append(self)
        def record(self):pass
        def query(self):return ready
        def elapsed_time(self,other):return 2.5
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'device',lambda device:nullcontext())
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:100)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:200)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:pytest.fail('forced synchronization'))
    with pipeline_scope(True,cuda_sample_every=1):
        for _ in range(70):
            with pipeline_cuda_timing('forward','cuda'):pass
        stats=pipeline_snapshot()['cuda']
        assert stats['pending_samples']==64 and stats['dropped_samples']==6
        assert stats['stages']['forward']['samples']==0 and len(events)==128
        ready=True
        stats=pipeline_snapshot()['cuda']
        assert stats['pending_samples']==0 and stats['stages']['forward']['samples']==64
        assert stats['stages']['forward']['seconds']==pytest.approx(.16)
        assert stats['stages']['forward']['allocated_bytes_max']==100
    with pipeline_scope(True,cuda_sample_every=10):
        with pipeline_cuda_timing('cpu','cpu'):pass
        assert pipeline_snapshot()['cuda']['stages']=={}
    with pipeline_scope(True):
        with pipeline_cuda_timing('disabled','cuda'):pass
        assert 'cuda' not in pipeline_snapshot()

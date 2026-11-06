from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from stash_ai.recognition.config import defaults,validate
from stash_ai.recognition.person_signal import contract
from stash_ai.recognition.body_observations import contract as body_contract
from stash_ai.recognition.sapiensid_backend import identity_contract
from stash_ai.recognition import inference_precision


def test_fp32_contracts_unchanged_and_precision_separates_body_inputs():
    data=defaults();pose=data['sapiensid']['pose']
    pose.update(detector='rfdetr_xl',backend='vitpose_plus_large')
    config=SimpleNamespace(data=data)
    signatures=lambda:(contract(config),body_contract(config),identity_contract(config))
    before=signatures()
    for key in ('rfdetr_precision','vitpose_precision'):data['execution'].pop(key)
    assert signatures()==before
    data['execution']['rfdetr_precision']='fp16'
    reduced=signatures();assert all(a!=b for a,b in zip(before,reduced))
    data['execution']['vitpose_precision']='bf16'
    assert all(a!=b for a,b in zip(reduced,signatures()))


@pytest.mark.parametrize('key',['rfdetr_precision','vitpose_precision'])
def test_precision_requires_corresponding_backend_and_cuda(key):
    data=defaults();data['execution'][key]='fp16'
    with pytest.raises(ValueError,match='requires.*backend'):validate(data)
    data['sapiensid']['pose'].update(detector='rfdetr_xl',backend='vitpose_plus_large')
    data['models']['device']='cpu'
    with pytest.raises(ValueError,match='requires CUDA'):validate(data)
    with pytest.raises(ValueError,match='requires a CUDA'):inference_precision.validate('cpu','fp16')


@pytest.mark.parametrize('precision',['fp32','fp16','bf16'])
def test_explicit_autocast_dtype_and_no_silent_fallback(monkeypatch,precision):
    import torch
    calls=[]
    @contextmanager
    def capture(**kwargs):calls.append(kwargs);yield
    monkeypatch.setattr(torch,'autocast',capture)
    with inference_precision.autocast('cuda:0',precision):pass
    assert calls==[dict(device_type='cuda',dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(precision),enabled=precision!='fp32')]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    if precision=='fp32':inference_precision.validate('cuda',precision)
    else:
        with pytest.raises(ValueError,match='requires a CUDA'):inference_precision.validate('cuda',precision)


def test_bf16_checks_the_selected_device(monkeypatch):
    import torch
    from contextlib import nullcontext
    devices=[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device',lambda device:devices.append(device) or nullcontext())
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:False)
    with pytest.raises(ValueError,match='does not support bf16'):inference_precision.validate('cuda:1','bf16')
    assert devices==['cuda:1']

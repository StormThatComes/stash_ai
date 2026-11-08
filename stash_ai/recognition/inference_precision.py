"""Explicit optional CUDA autocast, with numerical policy in evidence contracts."""


def contract(config):
    settings=config.data.get('execution',{});pose=config.data['sapiensid']['pose']
    result={}
    for name,active in (('rfdetr_precision',pose.get('detector')=='rfdetr_xl'),
                        ('vitpose_precision',pose.get('backend')=='vitpose_plus_large')):
        value=settings.get(name,'fp32')
        if active and value!='fp32':result[name]=value
    if pose.get('backend')=='vitpose_plus_large' and settings.get('vitpose_device_decode'):
        result['vitpose_device_decode'] = 'compact-dark-v1'
    return {'inference_precision':result} if result else {}


def validate(device,precision):
    if precision not in ('fp32','fp16','bf16'):raise ValueError('Inference precision must be fp32, fp16 or bf16.')
    if precision=='fp32':return
    import torch
    if torch.device(device).type!='cuda' or not torch.cuda.is_available():
        raise ValueError('Optional RF-DETR/ViTPose mixed precision requires a CUDA device; select fp32 for CPU.')
    if precision=='bf16':
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():raise ValueError('Selected CUDA device does not support bf16 inference.')


def autocast(device,precision):
    import torch
    dtype={'fp16':torch.float16,'bf16':torch.bfloat16}.get(precision)
    return torch.autocast(device_type=torch.device(device).type,dtype=dtype,enabled=precision!='fp32')

"""Stage dependencies exclude downstream quality and identity decisions."""
from .config import plain


def face(config):
    from .models import ASSETS
    model=config.data['models']['detector']
    return ['face-detector-v1',model,ASSETS[model].sha256,
            {k:config.data['detection'][k] for k in ('score_threshold','nms_threshold','max_faces_per_frame')},
            config.data['models']['device']]


def detector(config):
    from .sapiensid_registry import POSE_SHA256,POSE_INFERENCE
    from .person_detector import contract
    p=config.data['sapiensid']['pose'];model=p['detector']
    return ['person-detector-v1',contract(model) if model!='yolo' else [p['sha256'] or POSE_SHA256,POSE_INFERENCE],
            {k:p[k] for k in ('score_threshold','nms_threshold','max_people_per_frame')},
            p['backend']=='yolo',config.data['models']['device'],config.data['execution']['rfdetr_precision']]


def pose(config):
    p=config.data['sapiensid']['pose'];backend=p['backend']
    if backend=='vitpose_plus_large':
        from .vitpose_registry import contract
        model=contract()
        if config.data['execution'].get('vitpose_device_decode'):
            model = [model,'compact-dark-v1']
    elif backend=='sapiens':
        from .sapiens_pose_registry import contract
        model=contract(p['sapiens_variant'])
    else:
        from .pmpose_registry import contract
        model=contract(p)
        for gate in ('presence_threshold','visibility_threshold'):model.pop(gate,None)
        if p.get('refinement',{}).get('enabled'):
            # Older cached refined poses did not retain a quality mask.
            # Raw PMPose/SAM caches remain valid and can rebuild this output.
            model['quality_mask'] = 'crop-bits-v1'
        from .inference_precision import contract as precision_contract
        numerical=precision_contract(config).get('inference_precision',{})
        active={key:value for key,value in numerical.items() if key in ('pmpose_precision','sam_precision')}
        if active:model['inference_precision']=active
    return ['pose-estimator-v1',model,config.data['models']['device'],
            config.data['execution']['vitpose_precision'] if backend=='vitpose_plus_large' else 'fp32']


def body_encoder(config):
    from .sapiensid_registry import MODELS
    from .sapiensid_v1 import PREPROCESSING
    p=config.data['sapiensid']['model']
    return ['person-encoder-v1',MODELS[p['variant']]['sha256'],p['model_yaml_sha256'],PREPROCESSING,config.data['models']['device']]

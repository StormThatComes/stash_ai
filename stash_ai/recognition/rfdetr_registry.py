"""Pinned RF-DETR-XL COCO detection weights and inference contract; no ML imports."""

VERSION = '1.11.2'
PLUS_VERSION = '1.1.0'
NAME = 'rf-detr-xlarge.pth'
GENERATION = '1768355998186184'
URL = ('https://storage.googleapis.com/rfdetr/platform-licensed/' + NAME + '?generation=' + GENERATION)
SIZE = 505646472
SHA256 = 'ca40619d5d80105c702cc0d436f12e43b6c44f59688be21b81d39938b056dbce'
INFERENCE = 'rfdetr-xl-coco91-rgb700-bilinear-no-antialias-fp32-person-first-cpu-nms-v1'
PERSON_LABEL = 1  # Raw COCO category IDs, not D-FINE's contiguous COCO80 indices.
RESOLUTION = 700
MEAN = (.485, .456, .406)
STD = (.229, .224, .225)


def contract(detector='rfdetr_xl'):
    if detector != 'rfdetr_xl':
        raise ValueError('Unsupported RF-DETR detector.')
    return dict(detector=detector, rfdetr=VERSION, rfdetr_plus=PLUS_VERSION,
                generation=GENERATION, sha256=SHA256, inference=INFERENCE,
                person_label=PERSON_LABEL, resolution=RESOLUTION, mean=list(MEAN), std=list(STD))

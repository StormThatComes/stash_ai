"""Pinned PMPose-B 1.0.0 assets and bbox-only COCO17 inference contract."""

REPOSITORY = 'vrg-prague/BBoxMaskPose'
REVISION = 'cfc89052cf16655a52889f1f4e1b866e74a6c19c'
SOURCE_REVISION = '49a070a6f8396147323b1c4959474077dbe2ce8e'
NAME = 'PMPose-b-1.0.0.pth'
SIZE = 617696116
SHA256 = '23b0c1896a3ba5fd802a1cef635a3dad598e85960de57a29815a5bb3dd3ce5f7'
URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/PMPose/{NAME}'
INFERENCE = 'pmpose-b-1.0.0-bbox-coco17-fp32-flip-udp-v1'
INPUT_SIZE = (192, 256)
HEATMAP_SIZE = (48, 64)
PADDING = 1.25
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
FLIP = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


def contract(pose):
    result = dict(repository=REPOSITORY, revision=REVISION, source_revision=SOURCE_REVISION,
                inference=INFERENCE, sha256=SHA256, score='expected_oks',
                presence_threshold=pose['pmpose_presence_threshold'],
                visibility_threshold=pose['pmpose_visibility_threshold'])

    if pose.get('refinement', {}).get('enabled', False):
        from .pose_refinement_config import contract as refinement_contract
        result['refinement'] = refinement_contract(pose['refinement'])
    return result

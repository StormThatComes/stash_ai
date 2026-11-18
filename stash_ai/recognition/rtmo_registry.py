"""Pinned official OpenMMLab RTMO-S body7 end-to-end ONNX export."""
URL=('https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/'
     'rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip')
SHA256='d0703d40d19f3921da51ae725402d5fdae4d2478c7442072d3101bd396f370d8'
SIZE=39617685
ARCHIVE_SIZE=36442532
BYTE_RANGE=(783,36440719)


def contract(settings=None):
    result=dict(version='rtmo-s-body7-onnx-coco17-v1',sha256=SHA256,input_size=[640,640],
                color='BGR',padding='bottom-right-114',normalization='raw-0-255',visibility='confidence-proxy')
    if settings is not None:
        result.update(torso_nms=settings.get('rtmo_torso_nms',False),
                      keypoint_score_threshold=settings['keypoint_score_threshold'])
    return result

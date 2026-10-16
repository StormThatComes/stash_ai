"""CUDA-capable MediaPipe mesh/blendshape exports on the application ORT."""
import math

from .models import ASSETS, verified_bytes
from .runtime import runtime, session

# MediaPipe Authors, Apache-2.0. Subset/order from face_blendshapes_graph.cc;
# attribution and the original/converted model provenance are in the docs.
LANDMARKS_SUBSET = (
    0,1,4,5,6,7,8,10,13,14,17,21,33,37,39,40,46,52,53,54,55,58,61,63,65,66,67,70,78,80,
    81,82,84,87,88,91,93,95,103,105,107,109,127,132,133,136,144,145,146,148,149,150,152,153,
    154,155,157,158,159,160,161,162,163,168,172,173,176,178,181,185,191,195,197,234,246,249,
    251,263,267,269,270,276,282,283,284,285,288,291,293,295,296,297,300,308,310,311,312,314,
    317,318,321,323,324,332,334,336,338,356,361,362,365,373,374,375,377,378,379,380,381,
    382,384,385,386,387,388,389,390,397,398,400,402,405,409,415,454,466,468,469,470,471,472,
    473,474,475,476,477)
BLENDSHAPE_NAMES = (
    '_neutral','browDownLeft','browDownRight','browInnerUp','browOuterUpLeft','browOuterUpRight',
    'cheekPuff','cheekSquintLeft','cheekSquintRight','eyeBlinkLeft','eyeBlinkRight','eyeLookDownLeft',
    'eyeLookDownRight','eyeLookInLeft','eyeLookInRight','eyeLookOutLeft','eyeLookOutRight',
    'eyeLookUpLeft','eyeLookUpRight','eyeSquintLeft','eyeSquintRight','eyeWideLeft','eyeWideRight',
    'jawForward','jawLeft','jawOpen','jawRight','mouthClose','mouthDimpleLeft','mouthDimpleRight',
    'mouthFrownLeft','mouthFrownRight','mouthFunnel','mouthLeft','mouthLowerDownLeft',
    'mouthLowerDownRight','mouthPressLeft','mouthPressRight','mouthPucker','mouthRight','mouthRollLower',
    'mouthRollUpper','mouthShrugLower','mouthShrugUpper','mouthSmileLeft','mouthSmileRight',
    'mouthStretchLeft','mouthStretchRight','mouthUpperUpLeft','mouthUpperUpRight','noseSneerLeft','noseSneerRight')


def prepare(image, face):
    """RGB NHWC square crop, eyes level; return the inverse map to source pixels."""
    import numpy as np
    from PIL import Image
    x, y, width, height = face.box
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        from .expression import ExpressionGeometryError
        raise ExpressionGeometryError('Expression face extends beyond the real source.')
    eyes = sorted(face.landmarks[:2], key=lambda point: point[0])
    if math.dist(*eyes) <= 1:
        from .expression import ExpressionGeometryError
        raise ExpressionGeometryError('Expression target eye landmarks are invalid.')
    angle = math.atan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0])
    scale = max(width, height) * 1.5 / 256
    a, b = scale * math.cos(angle), -scale * math.sin(angle)
    d, e = -b, a
    c, f = x + width / 2 - 128 * (a+b), y + height / 2 - 128 * (d+e)
    transform = np.array([[a,b,c], [d,e,f]], dtype=np.float32)
    with image.convert('RGB') as rgb, rgb.transform((256,256), Image.Transform.AFFINE,
            tuple(transform.ravel()), resample=Image.Resampling.BILINEAR) as crop:
        blob = np.array(crop,dtype=np.float32,copy=True)[None]
        np.divide(blob,255.,out=blob)
    return np.ascontiguousarray(blob), transform


def decode_mesh(points, presence, transform):
    import numpy as np
    values, confidence = np.asarray(points), np.asarray(presence)
    if (values.shape != (1,1,1,1434) or confidence.shape != (1,1,1,1) or
            not np.isfinite(values).all() or not np.isfinite(confidence).all()):
        raise ValueError('Expression mesh outputs have invalid shapes/values.')
    if float(confidence.ravel()[0]) < 0:
        from .expression import ExpressionGeometryError
        raise ExpressionGeometryError('Expression mesh does not confirm a face in the verified crop.')
    coordinates = values.reshape(478,3)[:,:2]
    return coordinates @ transform[:,:2].T + transform[:,2]


def check_io(value, input_name, input_shape, outputs):
    def compatible(actual, wanted):
        return len(actual) == len(wanted) and all(
            expected is None and (type(dim) is int and dim == 1 or isinstance(dim,str)) or dim == expected
            for dim,expected in zip(actual,wanted))
    inputs = value.get_inputs()
    results = {item.name:item for item in value.get_outputs()}
    if (len(inputs) != 1 or inputs[0].name != input_name or inputs[0].type != 'tensor(float)' or
            not compatible(inputs[0].shape,input_shape) or set(results) != set(outputs) or
            any(results[key].type != 'tensor(float)' or not compatible(results[key].shape,shape)
                for key,shape in outputs.items())):
        raise ValueError('Expression ONNX input/output declarations are incompatible.')


class ExpressionBackend:
    def __init__(self, config, paths):
        self.mesh = self.blendshapes = None
        blobs = {key: verified_bytes(paths[key], ASSETS[key])
                 for key in ('expression_mesh', 'expression_blendshapes')}
        info = runtime(config)
        try:
            from .portrait_threads import native_threads
            self.mesh, mesh_device = session(config,blobs['expression_mesh'], runtime_info=info, cpu_fallback=False, cpu_threads=native_threads(config))
            check_io(self.mesh,'input_12',(None,256,256,3),
                     {'Identity':(None,1,1,1434),'Identity_1':(None,1,1,1),'Identity_2':(None,1)})
            self.blendshapes, blend_device = session(config,blobs['expression_blendshapes'], runtime_info=info, cpu_fallback=False, cpu_threads=native_threads(config))
            check_io(self.blendshapes,'serving_default_input_points:0',(1,146,2),{'StatefulPartitionedCall:0':(52,)})
            self.device = mesh_device if mesh_device == blend_device else 'mixed'
            self.provenance = {'ort':info[0].__version__, 'mesh_device':mesh_device, 'blendshape_device':blend_device,
                               'mesh_providers':self.mesh.get_providers(),
                               'blendshape_providers':self.blendshapes.get_providers()}
        except BaseException:
            self.close()
            raise

    def measure(self, image, face):
        import numpy as np
        from .expression import associated
        blob, transform = prepare(image,face)
        points,presence = self.mesh.run(['Identity','Identity_1'],{'input_12':blob})
        coordinates = decode_mesh(points,presence,transform)
        associated(coordinates,face)
        subset = np.ascontiguousarray(coordinates[list(LANDMARKS_SUBSET)][None],dtype=np.float32)
        scores = np.asarray(self.blendshapes.run(['StatefulPartitionedCall:0'],
                            {'serving_default_input_points:0':subset})[0])
        if scores.shape != (52,) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError('Expression blendshape outputs have invalid shapes/values.')
        return dict(zip(BLENDSHAPE_NAMES,map(float,scores))), transform.tolist()

    def close(self):
        self.mesh = self.blendshapes = None

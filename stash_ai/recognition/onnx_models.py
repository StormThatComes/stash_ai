"""ORT neural inference with the pinned OpenCV YuNet/SFace contracts."""
import hashlib
from stash_ai.runtime.cleanup import cleanup

from .models import ASSETS, verified_bytes, checked_detection, normalized_embedding
from .runtime import session


def prepared_model(path, *, detector=False):
    """Adapt declarations in memory only; node/initializer values stay identical.

    OpenCV overrides YuNet's exported 640px declaration and pads to divisor 32.
    ORT needs dynamic spatial/anchor declarations to execute the same pixels.
    Legacy SFace initializer-inputs are made constant, exactly as OpenCV's loader.
    The verified upstream weights are never rewritten or stored as sidecars.
    """
    import onnx
    model = onnx.load_model_from_string(verified_bytes(path, ASSETS['yunet' if detector else 'sface']))
    initializers = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    model.ir_version = max(4, model.ir_version)
    if detector:
        if len(inputs) != 1 or [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] != [1, 3, 640, 640]:
            raise ValueError('Pinned YuNet spatial input contract changed.')
        dimensions = model.graph.input[0].type.tensor_type.shape.dim
        dimensions[2].dim_param, dimensions[3].dim_param = 'height', 'width'
        for output in model.graph.output:
            output.type.tensor_type.shape.dim[1].dim_param = 'anchors'
        del model.graph.value_info[:]
    blob = model.SerializeToString()
    return blob, hashlib.sha256(blob).hexdigest()


def sface_blob(aligned_bgr):
    import numpy as np
    value = np.asarray(aligned_bgr)
    if value.shape != (112,112,3) or value.dtype != np.uint8:
        raise ValueError('SFace requires a 112x112 uint8 BGR aligned crop.')
    # SFace already contains (pixel-127.5)/128 normalization in its graph.
    return np.ascontiguousarray(value[:, :, ::-1].transpose(2,0,1)[None], dtype=np.float32)


class OnnxYuNet:
    names = tuple(f'{kind}_{stride}' for kind in ('cls', 'obj', 'bbox', 'kps') for stride in (8,16,32))

    def __init__(self, config, path, cv, runtime_info):
        self.session = None
        try:
            self.config, self.cv = config, cv
            blob, sha = prepared_model(path, detector=True)
            self.session, self.device = session(config, blob, runtime_info=runtime_info)
            inputs = self.session.get_inputs()
            if len(inputs) != 1 or inputs[0].type != 'tensor(float)' or list(inputs[0].shape[:2]) != [1,3]:
                raise ValueError('YuNet ORT input contract changed.')
            outputs = self.session.get_outputs()
            if {o.name for o in outputs} != set(self.names) or any(o.type != 'tensor(float)' for o in outputs):
                raise ValueError('YuNet ORT output contract changed.')
            self.input = inputs[0].name
            self.provenance = {'onnxruntime': runtime_info[0].__version__, 'providers': self.session.get_providers(),
                               'transform': 'opencv-pad32-dynamic-declarations-v1', 'prepared_sha256': sha}
        except BaseException:
            cleanup([('close failed ONNX adapter', self.close)])
            raise

    def detect(self, data):
        import numpy as np
        height, width = data.shape[:2]
        padded_height, padded_width = (height+31)//32*32, (width+31)//32*32
        padded = np.zeros((padded_height,padded_width,3), dtype=np.float32)
        padded[:height,:width] = data
        blob = np.ascontiguousarray(padded.transpose(2,0,1)[None])  # raw BGR, no normalization/swap
        outputs = dict(zip(self.names, self.session.run(list(self.names), {self.input: blob})))
        return self.decode(outputs, padded_width, padded_height)

    def decode(self, outputs, width, height):
        import numpy as np
        settings = self.config.data['detection']
        detections = []
        for stride in (8,16,32):
            columns, rows = width//stride, height//stride
            count = columns * rows
            for kind, channels in (('cls',1), ('obj',1), ('bbox',4), ('kps',10)):
                value = np.asarray(outputs[f'{kind}_{stride}'])
                if value.shape != (1,count,channels) or not np.isfinite(value).all():
                    raise ValueError('YuNet returned malformed/non-finite predictions.')
            scores = np.sqrt(np.clip(outputs[f'cls_{stride}'][0,:,0],0,1) *
                             np.clip(outputs[f'obj_{stride}'][0,:,0],0,1))
            indices = np.flatnonzero(scores > settings['score_threshold'])
            maximum = settings['max_faces_per_frame']
            if len(indices) > maximum:
                # A global top-K candidate must be in its stride's top K.
                # Preserve original anchor order for equal-score NMS ties.
                ranked = np.argsort(-scores[indices], kind='stable')[:maximum]
                indices = np.sort(indices[ranked])
            for index in indices:
                row, column = int(index)//columns, int(index)%columns
                box = outputs[f'bbox_{stride}'][0,index]
                w, h = np.exp(box[2:4]) * stride
                anchor = np.asarray([column,row], dtype=np.float32)
                cx, cy = (box[:2] + anchor) * stride
                landmarks = (outputs[f'kps_{stride}'][0,index].reshape(5,2) + anchor) * stride
                detections.append(checked_detection([cx-w/2, cy-h/2, w, h, *landmarks.ravel(), scores[index]]))
        if len(detections) > 1:
            # OpenCV uses integer truncation and top_k before NMS, not a
            # float-box approximation or an unbounded score sort/matrix.
            boxes = [[int(v) for v in face.box] for face in detections]
            keep = self.cv.dnn.NMSBoxes(boxes, [face.score for face in detections],
                    settings['score_threshold'], settings['nms_threshold'], top_k=settings['max_faces_per_frame'])
            detections = [detections[int(i)] for i in np.asarray(keep).reshape(-1)]
        return detections

    def close(self):
        self.session = None


class OnnxSFace:
    def __init__(self, config, path, runtime_info):
        self.session = None
        try:
            blob, sha = prepared_model(path)
            self.session, self.device = session(config, blob, runtime_info=runtime_info)
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if (len(inputs) != 1 or inputs[0].type != 'tensor(float)' or list(inputs[0].shape) != [1,3,112,112]
                    or len(outputs) != 1 or outputs[0].type != 'tensor(float)' or list(outputs[0].shape) != [1,128]):
                raise ValueError('Pinned SFace ORT input/output contract changed.')
            self.input, self.output = inputs[0].name, outputs[0].name
            self.provenance = {'onnxruntime': runtime_info[0].__version__, 'providers': self.session.get_providers(),
                               'transform': 'opencv-aligncrop-raw-rgb-constant-initializers-v1', 'prepared_sha256': sha}
        except BaseException:
            cleanup([('close failed ONNX adapter', self.close)])
            raise

    def embed(self, aligned_bgr):
        result = self.session.run([self.output], {self.input: sface_blob(aligned_bgr)})
        if len(result) != 1 or result[0].shape != (1,128):
            raise ValueError('SFace returned an incompatible output batch.')
        return normalized_embedding(result[0],128)

    def close(self):
        self.session = None

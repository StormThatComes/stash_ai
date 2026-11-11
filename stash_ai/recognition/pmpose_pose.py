"""Optional bbox-only PMPose-B adapter with separate OKS/presence/visibility."""
from .profiling import pipeline_stage, pipeline_timing, pipeline_bind
from . import pmpose_registry as registry


def assets(config):
    from .sapiensid_assets import _require_download
    from .sapiensid_downloads import install
    from .sapiensid_loading import local_path
    value = config.data['sapiensid']['pose']['pmpose_directory']
    directory = local_path(config, value) if value else config.model_dir / 'pmpose_b' / registry.REVISION
    path = directory / registry.NAME
    if not path.exists():
        _require_download(config, path)
        install(registry.URL, path, registry.SIZE, registry.SHA256)
    return path  # Hash and deserialize the same descriptor in load_verified_weights.


class _TrainingHistory:
    """Inert container for checkpoint metadata; no upstream code is imported."""


def _history_attribute(obj, name):
    # The pinned checkpoint pickles four HistoryBuffer methods in training
    # metadata. Never allow unrestricted getattr during deserialization.
    if obj is not _TrainingHistory or name not in ('min', 'max', 'current', 'mean'):
        raise ValueError('Unexpected PMPose training metadata attribute.')
    return None


def load(config, device):
    from .resident_models import current,Stage
    if current() is None:return _pipeline(_load_model(config,device),config,device)
    # A miss in downstream refinement need not reload weights for a cached raw pass.
    pose=Stage('pmpose_raw',lambda:_load_model(config,device),device)
    pose.batch_size=config.data['sapiensid']['pose']['pmpose_batch_size']
    pose.supports_masked_batches=True
    return _pipeline(pose,config,device)


def _load_model(config, device):
    from .inference_precision import validate
    validate(device,config.data['execution']['pmpose_precision'])
    try:
        import cv2
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError('PMPose requires PyTorch, NumPy and OpenCV in the CPU/GPU environment.') from error
    from .pmpose_network import PMPoseB
    from .sapiensid_inference import load_verified_weights
    path = assets(config)
    model = PMPoseB()
    # Restricted weights-only loading of this hash-pinned training checkpoint.
    # NumPy arrays and inert HistoryBuffer metadata are discarded by the loader.
    from numpy._core import multiarray
    allowed = [(multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               (multiarray.scalar, 'numpy.core.multiarray.scalar'), np.ndarray, np.dtype,
               *(type(np.dtype(t)) for t in ('float32', 'float64', 'uint8', 'int64')),
               (_TrainingHistory, 'mmengine.logging.history_buffer.HistoryBuffer'),
               (_history_attribute, 'builtins.getattr')]
    with torch.serialization.safe_globals(allowed):
        load_verified_weights(model, path, expected_size=registry.SIZE, expected_sha256=registry.SHA256)
    model = model.float().to(device).eval()
    pose = PMPose(model, device=device, batch_size=config.data['sapiensid']['pose']['pmpose_batch_size'],
                  fused_flip=config.data['execution']['pmpose_fused_flip'],
                  precision=config.data['execution']['pmpose_precision'])
    return pose


def _pipeline(pose, config, device):
    """Assemble optional stages around one shared first-pass pose estimator."""
    from .cached_pose import RawPMPose
    settings = config.data['sapiensid']['pose'].get('refinement', {})
    from .inference_cache import current
    if current() is not None:
        pose=RawPMPose(pose,device,precision=config.data['execution']['pmpose_precision'])
    refined = None
    if settings.get('enabled', False):
        from .sam_pose2seg import load as load_segmenter
        from .pose_refinement import RefinedPMPose
        from .resident_models import current as model_pool,Stage
        segmenter=(Stage('pmpose_segmenter',lambda:load_segmenter(config,device),device)
                   if model_pool() is not None else load_segmenter(config,device))
        refined = RefinedPMPose(pose,segmenter,settings,
            frame_batch_size=config.data['execution']['sam_frame_batch_size'],
            queue_bytes=config.data['execution']['refinement_queue_bytes'])
    deduplication = config.data['sapiensid']['pose'].get('deduplication', {})
    if deduplication.get('enabled', False):
        from .pose_deduplication import DeduplicatedPMPose
        return DeduplicatedPMPose(pose, deduplication, refined)
    return refined or pose


class PMPose:
    supports_masked_batches = True
    def __init__(self, model, *, device, batch_size, fused_flip=True, precision='fp32'):
        from .adaptive_gpu import Batches
        self.model, self.device, self.batch_size = model, device, batch_size
        self.batches=Batches(batch_size,'PMPose')
        from .inference_precision import validate
        validate(device,precision)
        self.fused_flip=fused_flip
        self.precision=precision

    def resident_modules(self):return [self.model]

    def predict(self, image, bounds, masks=None):
        return self.predict_many([(image, bounds, masks)])[0]

    def predict_many(self, frames):
        import numpy as np
        import torch
        from .pmpose_geometry import prepare, decode
        frames = list(frames)
        result, jobs = [], []
        for frame, item in enumerate(frames):
            image, bounds = item[:2]
            masks = item[2] if len(item) == 3 else None
            if masks is not None and len(masks) != len(bounds):
                raise ValueError('PMPose requires exactly one mask per person box.')
            count = len(bounds)
            result.append((np.empty((count, 17, 2), dtype=np.float32),
                           np.empty((count, 17), dtype=np.float32),
                           {key: np.empty((count, 17), dtype=np.float32) for key in ('presence', 'visibility')}))
            if not count:
                continue
            boxes = np.asarray(bounds, dtype=np.float32)
            if boxes.shape != (count, 4) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError('PMPose requires finite, non-empty person boxes.')
            jobs.extend((frame, index, box, None if masks is None else masks[index])
                        for index, box in enumerate(boxes))
        from contextlib import nullcontext
        from concurrent.futures import ThreadPoolExecutor
        from .gpu_transfers import to_device
        cuda=str(self.device).split(':')[0]=='cuda'

        @pipeline_stage('pmpose_prepare')
        def prepare_batch(batch):
            return [(job,prepare(frames[job[0]][0],job[2],job[3])) for job in batch]

        @pipeline_stage('pmpose_inference')
        def infer(prepared):
            inputs=to_device(torch.from_numpy(np.stack([crop for _,(crop,_,_) in prepared])),self.device)
            if self.fused_flip:
                doubled=self._forward(torch.cat((inputs,inputs.flip(-1))))
                output=[v[:len(inputs)] for v in doubled]
                flipped=[v[len(inputs):] for v in doubled]
            else:
                output=self._forward(inputs);flipped=self._forward(inputs.flip(-1))
            values=[(a+(b[:,registry.FLIP].flip(-1) if i==0 else b[:,registry.FLIP]))*.5
                    for i,(a,b) in enumerate(zip(output,flipped))]
            centers=np.stack([c for _,(_,c,_) in prepared]);scales=np.stack([s for _,(_,_,s) in prepared])
            if cuda:
                from .pmpose_decode_batch import decode_batch
                points=decode_batch(values[0],centers,scales)
            else:
                points=[decode(h,c,s) for h,c,s in zip(values[0].cpu().numpy(),centers,scales)]
            with pipeline_timing('pmpose_probability_readback'):
                scores,presence,visibility=torch.stack(values[1:]).cpu().numpy()
            return list(zip(points,scores,presence,visibility))

        # A single future owns at most one additional batch of CPU crops.
        if not jobs:return result
        with torch.inference_mode(), (ThreadPoolExecutor(max_workers=1) if cuda else nullcontext()) as pool:
            position=0;pending=None
            while position<len(jobs):
                batch=jobs[position:position+self.batches.maximum] if pending is None else queued
                if pending is None:
                    prepared=prepare_batch(batch)
                else:
                    with pipeline_timing('pmpose_prepare_wait'):
                        prepared=pending.result()
                position+=len(batch)
                queued=jobs[position:position+self.batches.maximum]
                pending=pool.submit(pipeline_bind(prepare_batch),queued) if pool is not None and queued else None
                outputs=self.batches.run(prepared,infer,self.device)
                for ((frame,index,_,_),_), (points,scores,presence,visibility) in zip(prepared,outputs):
                    result[frame][0][index]=points;result[frame][1][index]=scores
                    result[frame][2]['presence'][index]=presence;result[frame][2]['visibility'][index]=visibility
        return result

    def _forward(self, inputs):
        import torch
        from .inference_precision import autocast
        from .profiling import pipeline_cuda_timing,pipeline_count
        with autocast(self.device,self.precision), pipeline_cuda_timing('pmpose_forward',self.device):
            output = self.model(inputs)
        pipeline_count('pmpose_'+self.precision+'_forwards')
        shapes = ((len(inputs), 17, 64, 48),) + ((len(inputs), 17),) * 3
        if (not isinstance(output, (tuple, list)) or len(output) != 4 or
                any(not isinstance(v, torch.Tensor) or tuple(v.shape) != shape
                    for v, shape in zip(output, shapes))):
            raise ValueError('PMPose returned malformed heatmaps or probabilities.')
        # One scalar readback validates every output, including both flip views.
        with pipeline_timing('pmpose_output_validation'):
            valid = torch.stack([(torch.isfinite(v) & (v >= 0) & (v <= 1)).all() for v in output]).all()
            if not bool(valid):
                raise ValueError('PMPose returned malformed/non-finite heatmaps or probabilities.')
        return [value.float() for value in output]

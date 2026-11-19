"""Read-only scene browsing and explicit single-cue audit inference."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path

from stash_ai.runtime.cleanup import cleanup, closing
from .config import plain


@contextmanager
def open_scene(services, config, scene_id):
    """Keep one complete VTT source open, with no persistent recognition/media writes."""
    data = plain(config.data)
    data['execution'].update(max_cues=0, media_cache_bytes=0)
    with closing(services.client(url=os.getenv('STASH_URL', 'http://localhost:9999'),
                                 api_key=os.getenv('STASH_API_KEY', ''),
                                 timeout=data['execution']['timeout_seconds'])) as client:
        scene = client.scene(scene_id)
        with closing(services.source(client, scene, replace(config, data=data))) as source:
            if (not source.cues or not source.coverage['complete_sampling'] or
                    len(source.cues) != source.coverage['total_cues']):
                raise ValueError('Detection audit requires a nonempty, complete VTT cue source.')
            yield source


@dataclass(frozen=True)
class CueFrame:
    scene_id: str
    index: int
    cue: object
    image: object
    previous: tuple | None = None


def read_cue(source, scene_id, index):
    """Return an owned copy of one original crop. Reading never loads a model."""
    if type(index) is not int or not 0 <= index < len(source.cues):
        raise ValueError(f'Cue index {index} is outside this scene; valid indices: 0..{len(source.cues)-1}.')
    indices = [index-1,index] if index and temporal_enabled(getattr(source,'config',None)) else [index]
    previous = None
    with closing(source.frames_indices(indices)) as frames:
        for actual,cue,image in frames:
            with closing(image):
                if actual == index-1 and len(indices)==2:
                    with image.convert('RGB') as rgb:
                        previous = (actual,cue,rgb.size,rgb.tobytes())
                elif actual == index:
                    return CueFrame(scene_id,index,cue,image.copy(),previous)
                else:
                    raise ValueError('The VTT source returned a different cue index.')
    raise ValueError('The selected VTT cue is missing.')


def temporal_enabled(config):
    pose = config.data.get('sapiensid',{}).get('pose',{}) if config is not None else {}
    refinement = pose.get('refinement',{})
    return pose.get('backend')=='pmpose' and refinement.get('enabled') and refinement.get('recover_prompt_points')


def audit_context(image,detector,scene_id,index,cue,previous):
    from .perspective_shots import advance
    from .crowd_context import attach_context
    shot = advance(image,index,previous,evidence=True)
    # A single-cue replay knows its local shot even if the scene's first cue was skipped.
    if previous is None and shot[1] is None:
        shot = (shot[0],index,*shot[2:])
    image._recognition_scan = scene_id
    image._recognition_cue_signature = shot[:3]
    image._previous_cue_available = bool(previous and previous[0]==index-1)
    attach_context(detector,image,scene_id,index,cue.start)
    return shot


@contextmanager
def trial_image(frame, detector):
    from PIL import Image
    previous = None
    if frame.previous and temporal_enabled(getattr(detector,'config',None)):
        index,cue,size,pixels = frame.previous
        with Image.frombytes('RGB',size,pixels) as image, detector.prepared_frame(image):
            previous = audit_context(image,detector,frame.scene_id,index,cue,None)
            detector.detect(image,force_pose=True)
    with frame.image.copy() as image:
        audit_context(image,detector,frame.scene_id,frame.index,frame.cue,previous)
        yield image


def detector_factory(config, paths):
    from .sapiensid_backend import SapiensIDFaces
    return SapiensIDFaces(config, paths, load_identity=False)


def load_detector(services, config, factory, write):
    paths = services.assets(config, keys=[config.data['models']['detector']], write=write)
    return factory(config, paths)


def same_settings(left, right):
    """Compare effective settings and asset roots, independent of YAML file names/comments."""
    if left is right:
        return True
    left_root = left.source_path.parent if left.source_path else Path.cwd()
    right_root = right.source_path.parent if right.source_path else Path.cwd()
    return left_root == right_root and left.model_dir == right.model_dir and left.data == right.data


class CueDetector:
    """Owned by one inference thread. Keep at most one configured model set loaded."""
    def __init__(self, services, *, factory=detector_factory, write=print):
        self.services = services
        self.factory = factory
        self.write = write
        self.config = self.detector = None

    def render_phases(self, frame, config):
        return self.render(frame, config, capture_phases=True)

    def render(self, frame, config, *, mask_options=None, capture_phases=False):
        from .detection_audit import render_frame
        from .detection_phases import capture, DetectionResult
        if self.detector is None or not same_settings(self.config, config):
            self.close()
            self.detector = load_detector(self.services, config, self.factory, self.write)
            self.config = config
        try:
            # A fresh image also isolates single-cue inference from browsing history.
            with trial_image(frame,self.detector) as image, capture(
                    image, enabled=capture_phases) as phases, render_frame(
                    image, self.detector, frame.scene_id, frame.index, frame.cue,
                    mask_options=mask_options) as (result, record):
                if phases is not None:
                    return DetectionResult(result.copy(), record, phases)
                return result.copy(), record
        except Exception:
            cleanup([self.close])
            raise

    def close(self):
        detector, self.detector = self.detector, None
        self.config = None
        if detector is not None:
            try:
                detector.close()
            finally:
                del detector
                from .gpu_memory import reclaim, torch_memory
                if torch_memory() is not None:
                    reclaim()

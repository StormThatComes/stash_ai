"""Read-only scene browsing and explicit single-cue audit inference."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import os

from stash_ai.runtime.cleanup import closing
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


def read_cue(source, scene_id, index):
    """Return an owned copy of one original crop. Reading never loads a model."""
    if type(index) is not int or not 0 <= index < len(source.cues):
        raise ValueError(f'Cue index {index} is outside this scene; valid indices: 0..{len(source.cues)-1}.')
    with closing(source.frames_indices([index])) as frames:
        try:
            actual, cue, image = next(frames)
        except StopIteration:
            raise ValueError('The selected VTT cue is missing.') from None
        with closing(image):
            if actual != index:
                raise ValueError('The VTT source returned a different cue index.')
            return CueFrame(scene_id, index, cue, image.copy())


def detector_factory(config, paths):
    from .sapiensid_backend import SapiensIDFaces
    return SapiensIDFaces(config, paths, load_identity=False)


def load_detector(services, config, factory, write):
    paths = services.assets(config, keys=[config.data['models']['detector']], write=write)
    return factory(config, paths)


class CueDetector:
    """Owned by one inference thread. Keep at most one configured model set loaded."""
    def __init__(self, services, *, factory=detector_factory, write=print):
        self.services = services
        self.factory = factory
        self.write = write
        self.config = self.detector = None

    def render(self, frame, config, *, mask_options=None):
        from .detection_audit import render_frame
        if self.detector is None or self.config != config:
            self.close()
            self.detector = load_detector(self.services, config, self.factory, self.write)
            self.config = config
        try:
            # A fresh image also isolates single-cue inference from browsing history.
            with frame.image.copy() as image, render_frame(
                    image, self.detector, frame.scene_id, frame.index, frame.cue,
                    mask_options=mask_options) as (result, record):
                return result.copy(), record
        except Exception:
            self.close()
            raise

    def close(self):
        detector, self.detector = self.detector, None
        self.config = None
        if detector is not None:
            detector.close()

"""On-demand VTT previews, including detections rejected before crop encoding.

Owned by the inspector's detail worker. SQLite snapshots end before network
access, and reconstructed pixels never become recognition evidence.
"""
from contextlib import closing
from dataclasses import replace
import hashlib
import json
import math
import os

from .config import plain
from .inspection_runtime import open_readonly as _open_readonly, check_cancelled, Cancelled
from .identity_inspection import TABLES
from .media import SpriteInput, verify_portrait_cue
from .models import namespace


class PreviewUnavailable(ValueError):
    """An explanation safe to display without request URLs or credentials."""


def _client(timeout):
    from .stash import RecognitionStash
    return RecognitionStash(url=os.getenv('STASH_URL', 'http://localhost:9999'),
                            api_key=os.getenv('STASH_API_KEY', ''), timeout=timeout)


class VttPreviews:
    def __init__(self, config, *, client_factory=None, source_factory=SpriteInput):
        # Only bounded in-memory sprite reuse. Do not create a media SQLite cache.
        data = plain(config.data)
        data['execution']['media_cache_bytes'] = 0
        data['execution']['timeout_seconds'] = min(10, data['execution']['timeout_seconds'])
        data['execution']['sprite_cache_bytes'] = min(data['execution']['sprite_cache_bytes'], 64 * 1048576)
        self.config = replace(config, data=data)
        self.namespace = namespace(config)
        self.client_factory = client_factory or (lambda: _client(data['execution']['timeout_seconds']))
        self.source_factory = source_factory
        self.client = self.source = self.selection = None

    def fork(self):
        return type(self)(self.config, client_factory=self.client_factory, source_factory=self.source_factory)

    def image(self, kind, address, scope, retained, *, geometry=None):
        if retained[0] is not None:
            return retained
        check_cancelled()
        caption = retained[1]
        try:
            table = TABLES[kind][0]
            with closing(_open_readonly(self.config.database)) as db:
                row = db.execute(f'SELECT o.id,o.scan,o.cue,o.seconds,o.geometry,o.quality,'
                    + ('o.source,o.rect,' if kind == 'face' else '') +
                    f's.scope,s.scene,s.input_sha FROM {table} o JOIN scans s ON s.id=o.scan '
                    'WHERE o.id=? AND s.scope=? AND s.namespace=?', (address, scope, self.namespace)).fetchone()
                if row is None:
                    raise PreviewUnavailable('Observation unavailable in this scope and model.')
                row = dict(row)
            # No database connection or long-lived read transaction crosses this boundary.
            if self.client is None:
                self.client = self.client_factory()
            from .run import instance_scope
            if instance_scope(self.client.url) != row['scope']:
                raise PreviewUnavailable('STASH_URL does not match this observation\'s Stash instance.')
            selection = row['scope'], row['scan']
            if selection != self.selection:
                self._close_source()
                source = self.source_factory(self.client, self.client.scene(row['scene']), self.config)
                self.source, self.selection = source, selection
            check_cancelled()
            if self.source.fingerprint != row['input_sha']:
                raise PreviewUnavailable('VTT source changed after recognition.')
            index = row['cue']
            if type(index) is not int or not 0 <= index < len(self.source.cues) or self.source.cues[index].start != row['seconds']:
                raise PreviewUnavailable('VTT cue does not match the recorded observation.')
            if kind == 'face':
                try:
                    verify_portrait_cue(self.source, row)
                except ValueError:
                    raise PreviewUnavailable('VTT cue does not match the recorded observation.') from None
            quality, source_geometry = json.loads(row['quality']), json.loads(row['geometry'])
            with self.source.thumbnail(index) as frame:
                # PIL crops at rounded integer pixel edges; use that exact
                # viewport when mapping source geometry onto the preview.
                check_cancelled()
                bounds = tuple(round(v) for v in _bounds(kind, source_geometry, quality, frame.size))
                with frame.crop(bounds) as crop:
                    image = crop.convert('RGB')
            try:
                if not image.width or not image.height:
                    raise PreviewUnavailable('Recorded geometry has no source pixels.')
                expected = (quality.get('evidence_pixel_sha256') if kind == 'face' else
                            (quality.get('quality') or {}).get('pixel_digest'))
                if expected is not None:
                    actual = hashlib.sha256(str(image.size).encode('ascii') + image.tobytes()).hexdigest()
                    if actual != expected:
                        raise PreviewUnavailable('Reconstructed pixels differ from the recorded signature.')
                verified = 'pixels verified' if expected is not None else 'source verified; original crop signature unavailable'
                if geometry is not None:
                    geometry['bounds'] = list(bounds)
                return image, f"Scene {row['scene']} · cue {index} · {kind} · Reconstructed from VTT · {verified}"
            except BaseException:
                image.close()
                raise
        except Cancelled:
            self._close_source()
            raise
        except PreviewUnavailable as error:
            self._close_source()
            return None, caption + ' · Preview unavailable: ' + str(error)
        except Exception as error:
            self._close_source()
            # Requests/GraphQL errors can embed authenticated URLs. Keep metadata
            # and rejection reasons usable while exposing only a safe failure type.
            return None, caption + f' · VTT preview unavailable ({type(error).__name__}). Check Stash access and VTT sprites.'

    def _close_source(self):
        source, self.source = self.source, None
        self.selection = None
        if source is not None:
            source.close()

    def close(self):
        try:
            self._close_source()
        finally:
            client, self.client = self.client, None
            if client is not None:
                client.close()


def _bounds(kind, geometry, quality, size):
    width, height = size
    if kind == 'face':
        box = geometry[:4] if isinstance(geometry, list) else []
        if not _numbers(box) or box[2] <= 0 or box[3] <= 0:
            raise PreviewUnavailable('Recorded face bounds are invalid.')
        x, y, w, h = box
        if min(width, x+w) <= max(0, x) or min(height, y+h) <= max(0, y):
            raise PreviewUnavailable('Recorded face bounds are outside the VTT frame.')
        bounds = quality.get('crop_bounds')
        if bounds is None:
            # Same framing as quality.crop_bytes, including partially visible faces.
            bounds = (max(0, int(x-w*.15)), max(0, int(y-h*.15)),
                      min(width, math.ceil(x+w*1.15)), min(height, math.ceil(y+h*1.15)))
    else:
        bounds = geometry.get('bounds') if isinstance(geometry, dict) else None
    if not _numbers(bounds) or not (0 <= bounds[0] < bounds[2] <= width and 0 <= bounds[1] < bounds[3] <= height):
        raise PreviewUnavailable('Recorded crop bounds are missing, invalid or outside the VTT frame.')
    return bounds


def _numbers(values):
    return isinstance(values, (list, tuple)) and len(values)==4 and all(type(v) in (int, float) and math.isfinite(v) for v in values)

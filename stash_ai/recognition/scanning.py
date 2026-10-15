"""Checkpoint selected-scene detections without publishing anything to Stash."""
import json
import uuid

from stash_ai.runtime.cleanup import closing

from .config import digest
from .database import encoded
from .media import safe_url
from .quality import assess, rejection, crop_bytes, alignable
from .models import normalized_embedding, checked_detection
from .prefetch import frames as frame_inputs
from .visibility import CONTRACT_ID, VisibilityError, VisibilityGeometryError, cached as visibility_cached


def scan_identity(config, namespace):
    visibility_contract = CONTRACT_ID if config.data['detection']['min_visible_face_fraction'] > 0 else None
    return digest({'namespace': namespace, 'detection': config.data['detection'],
                   'max_cues': config.data['execution']['max_cues'], 'quality_format': 1,
                   'visibility_contract': visibility_contract})


def scan_scene(database, scope, scene, config, namespace, source, backend_factory, *,
               visibility=None, progress=lambda **kw: None, frame_progress=lambda **kw: None):
    db = database.connection
    policy = scan_identity(config, namespace)
    identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, encoded([scope, str(scene['id']), namespace,
                                                           source.fingerprint, policy])))
    with db:
        db.execute('INSERT OR IGNORE INTO scans(id,scope,scene,namespace,input_sha,policy_sha,manifest,coverage,total) '
                   'VALUES (?,?,?,?,?,?,?,?,?)', (identifier, scope, str(scene['id']), namespace,
                   source.fingerprint, policy, encoded(source.manifest), encoded(source.coverage), len(source.cues)))
    record = db.execute('SELECT * FROM scans WHERE id=?', (identifier,)).fetchone()
    if (record['status'] not in ('partial', 'complete') or
            not 0 <= record['cursor'] <= record['total'] or record['total'] != len(source.cues)):
        raise ValueError('Invalid recognition scan checkpoint.')
    # Recheck only legacy geometry, without loading models or losing valid scans.
    with db:
        for row in db.execute('SELECT id,geometry,quality FROM observations WHERE scan=? AND rejected IS NULL', (identifier,)):
            quality = json.loads(row['quality'])
            if 'alignment_valid' not in quality:
                quality['alignment_valid'] = alignable(checked_detection(json.loads(row['geometry'])))
                db.execute('UPDATE observations SET quality=?,rejected=?,embedding=CASE WHEN ? THEN embedding ELSE NULL END,'
                           'crop=CASE WHEN ? THEN crop ELSE NULL END WHERE id=?',
                           (encoded(quality), None if quality['alignment_valid'] else 'face_landmarks_invalid',
                            quality['alignment_valid'], quality['alignment_valid'], row['id']))
    if record['status'] == 'complete':
        if record['cursor'] != record['total']:
            raise ValueError('Incomplete recognition scan marked complete.')
        frame_progress(completed=record['total'], total=record['total'], reused=True)
        progress(reused=record['total'], scene=str(scene['id']))
        return identifier, True
    if record['cursor'] == record['total']:
        with db:
            db.execute("UPDATE scans SET status='complete' WHERE id=?", (identifier,))
        frame_progress(completed=record['total'], total=record['total'], reused=True)
        progress(reused=record['total'], scene=str(scene['id']))
        return identifier, True
    frame_progress(completed=record['cursor'], total=record['total'])
    backend = backend_factory()
    pending, cursor = [], record['cursor']

    def checkpoint():
        with db:
            db.executemany('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', pending)
            db.execute('UPDATE scans SET cursor=? WHERE id=?', (cursor, identifier))
        pending.clear()
        progress(scanned=cursor, total=record['total'], scene=str(scene['id']))

    try:
        with closing(frame_inputs(source, cursor, config.data['execution']['prefetch_frames']),
                     name='close recognition frame input') as frames:
            for index, cue, image in frames:
                with closing(image, name='close recognition cue image'):
                    faces = sorted(backend.detect(image), key=lambda f: tuple(f.row))
                    for face_index, face in enumerate(faces):
                        quality = assess(image, face)
                        reason = rejection(quality, config.data['detection'])
                        if reason is None and config.data['detection']['min_visible_face_fraction'] > 0:
                            if visibility is None:
                                raise VisibilityError('Recognition visibility gate requires a visibility model.')
                            try:
                                quality.update(visibility.measure(image, face))
                            except VisibilityGeometryError:
                                reason = 'face_visibility_unmeasurable'
                            else:
                                if not visibility_cached(quality):
                                    raise VisibilityError('Recognition visibility gate received an invalid measurement.')
                                if quality['visible_face_fraction'] < config.data['detection']['min_visible_face_fraction']:
                                    reason = 'face_occluded'
                        vector = None
                        crop = None
                        if reason is None:
                            vector = normalized_embedding(backend.embed(image, face), backend.dimensions).astype('<f4').tobytes()
                            encoding = {}
                            crop, bounds = crop_bytes(image, face, limit=config.data['storage']['max_crop_bytes'], details=encoding)
                            quality['crop_bounds'] = bounds
                            quality['crop_encoding'] = encoding
                        observation = str(uuid.uuid5(uuid.UUID(identifier), encoded([index, face_index, face.row])))
                        pending.append((observation, identifier, index, cue.start, safe_url(cue.image), encoded(list(cue.rect)),
                                        encoded(list(face.row)), encoded(quality), reason, backend.dimensions, vector, crop))
                cursor = index + 1
                frame_progress(completed=cursor, total=record['total'])
                if cursor % config.data['execution']['checkpoint_cues'] == 0:
                    checkpoint()
        checkpoint()
        with db:
            db.execute("UPDATE scans SET status='complete' WHERE id=?", (identifier,))
        return identifier, False
    except BaseException:
        # Only completed cues are durable. The current cue's tentative faces
        # must not enter a checkpoint if inference fails halfway through it.
        pending[:] = [row for row in pending if row[2] < cursor]
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('save recognition checkpoint', checkpoint)])
        raise
    finally:
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('close recognition backend', backend.close)])

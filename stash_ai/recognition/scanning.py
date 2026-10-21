"""Checkpoint selected-scene detections without publishing anything to Stash."""
import json
import uuid
from types import SimpleNamespace

from stash_ai.runtime.cleanup import closing

from .config import digest
from .database import encoded
from .media import safe_url
from .quality import assess, rejection, crop_bytes, alignable
from .models import normalized_embedding, checked_detection
from .prefetch import frames as frame_inputs
from .visibility import CONTRACT_ID, VisibilityError, VisibilityGeometryError, cached as visibility_cached
from .pose_policy import measurement_required


def scan_identity(config, namespace):
    visibility_contract = CONTRACT_ID if measurement_required(config) else None
    return digest({'namespace': namespace, 'detection': config.data['detection'],
                   'max_cues': config.data['execution']['max_cues'], 'quality_format': 2,
                   'visibility_contract': visibility_contract})


def geometry_contract(face, cue):
    return digest(['real-source-alignment-coverage-v3', list(face.row), list(cue.rect)])


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
    # Validate retained geometry even when an old alignment flag says true.
    # Reuse valid vectors; retire only evidence invalid under this same policy.
    with db:
        for row in db.execute('SELECT id,cue,geometry,quality FROM observations WHERE scan=? AND rejected IS NULL', (identifier,)):
            quality = json.loads(row['quality'])
            face = checked_detection(json.loads(row['geometry']))
            if type(row['cue']) is not int or not 0 <= row['cue'] < len(source.cues):
                raise ValueError('Retained face observation has an invalid cue.')
            cue = source.cues[row['cue']]
            stamp = geometry_contract(face, cue)
            if quality.get('geometry_contract') != stamp or face.score <= config.data['detection']['score_threshold']:
                quality['alignment_valid'] = alignable(face, SimpleNamespace(width=cue.rect[2], height=cue.rect[3]))
                quality['geometry_contract'] = stamp
                reason = ('face_detection_score' if face.score <= config.data['detection']['score_threshold']
                          else None if quality['alignment_valid'] else 'face_landmarks_invalid')
                if reason:
                    # Earlier SFace scans may already have matching memberships.
                    # Keep raw geometry/origins, retire only this invalid support,
                    # and replay matching instead of serving its old cached result.
                    db.execute('DELETE FROM members WHERE observation=?', (row['id'],))
                    db.execute('DELETE FROM candidates WHERE scan=?', (identifier,))
                    db.execute('DELETE FROM metadata WHERE key=?', ('identity-rebuild:' + digest([scope, namespace]),))
                db.execute('UPDATE observations SET quality=?,rejected=?,embedding=CASE WHEN ? THEN embedding ELSE NULL END,'
                           'crop=CASE WHEN ? THEN crop ELSE NULL END WHERE id=?',
                           (encoded(quality), reason, reason is None, reason is None, row['id']))
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
    backend = backend_factory()
    frame_progress(completed=record['cursor'], total=record['total'])
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
                        quality['geometry_contract'] = geometry_contract(face, cue)
                        reason = ('face_detection_score' if face.score <= config.data['detection']['score_threshold']
                                  else rejection(quality, config.data['detection']))
                        if reason is None and measurement_required(config):
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

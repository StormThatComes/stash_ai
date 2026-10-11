"""Checkpoint selected-scene detections without publishing anything to Stash."""
import uuid

from .config import digest
from .database import encoded
from .media import safe_url
from .quality import assess, rejection, crop_bytes
from .models import normalized_embedding


def scan_identity(config, namespace):
    return digest({'namespace': namespace, 'detection': config.data['detection'],
                   'max_cues': config.data['execution']['max_cues'], 'quality_format': 1})


def scan_scene(database, scope, scene, config, namespace, source, backend_factory, *, progress=lambda **kw: None):
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
    if record['status'] == 'complete':
        if record['cursor'] != record['total']:
            raise ValueError('Incomplete recognition scan marked complete.')
        progress(reused=record['total'], scene=str(scene['id']))
        return identifier, True
    backend = backend_factory()
    pending, cursor = [], record['cursor']

    def checkpoint():
        with db:
            db.executemany('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', pending)
            db.execute('UPDATE scans SET cursor=? WHERE id=?', (cursor, identifier))
        pending.clear()
        progress(scanned=cursor, total=record['total'], scene=str(scene['id']))

    try:
        for index, cue, image in source.frames(cursor):
            faces = sorted(backend.detect(image), key=lambda f: tuple(f.row))
            for face_index, face in enumerate(faces):
                quality = assess(image, face)
                reason = rejection(quality, config.data['detection'])
                vector = None
                crop = None
                if reason is None:
                    vector = normalized_embedding(backend.embed(image, face), backend.dimensions).astype('<f4').tobytes()
                    crop, bounds = crop_bytes(image, face, limit=config.data['storage']['max_crop_bytes'])
                    quality['crop_bounds'] = bounds
                observation = str(uuid.uuid5(uuid.UUID(identifier), encoded([index, face_index, face.row])))
                pending.append((observation, identifier, index, cue.start, safe_url(cue.image), encoded(list(cue.rect)),
                                encoded(list(face.row)), encoded(quality), reason, backend.dimensions, vector, crop))
            cursor = index + 1
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

"""Checkpoint selected-scene detections without publishing anything to Stash."""
from .profiling import pipeline_stage
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
                   'max_cues': config.data['execution']['max_cues'], 'quality_format': 3,
                   'visibility_contract': visibility_contract})


def geometry_contract(face, cue):
    return digest(['real-source-alignment-coverage-v3', list(face.row), list(cue.rect)])


@pipeline_stage('scan')
def scan_scene(database, scope, scene, config, namespace, source, backend_factory, *,
               visibility=None, progress=lambda **kw: None, frame_progress=lambda **kw: None, on_cue=None):
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
        from .scan_heads import activate
        activate(db, record)
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
    pending_bytes=0
    collector=getattr(on_cue,'__self__',None)
    grouped=callable(getattr(collector,'collect_records',None))
    columns=('id','scan','cue','seconds','source','rect','geometry','quality','rejected','dimensions','embedding','crop')

    def checkpoint():
        nonlocal pending_bytes
        supplemental=collector.take_records() if grouped else []
        with db:
            from .pixel_storage import compact
            compacted=[]
            for row in pending:
                row=list(row);row[7]=compact(db,row[7]);compacted.append(row)
            db.executemany('INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', compacted)
            if supplemental:
                collector.write_records(db,supplemental)
            db.execute('UPDATE scans SET cursor=? WHERE id=?', (cursor, identifier))
        pending.clear();pending_bytes=0
        progress(scanned=cursor, total=record['total'], scene=str(scene['id']))

    try:
        from .scan_pipeline import CpuStage
        with closing(CpuStage(config.data['execution']['scan_cpu_workers'], config.data['execution']['scan_queue_bytes']//2),
                     name='stop recognition pixel workers') as cpu, closing(frame_inputs(source, cursor, config.data['execution']['prefetch_frames']),
                     name='close recognition frame input') as frames:
            from .scan_pipeline import cue_batches, process_cues
            # Batch size one retains the previous failure/checkpoint boundary.
            maximum = config.data['execution']['recognizer_batch_size']
            with closing(cue_batches(frames, maximum, config.data['execution']['scan_queue_bytes']//2),
                         name='close cue microbatch') as batches:
                for batch in batches:
                    for index, cue, records in process_cues(batch, backend, visibility, config, stage=cpu,
                            assess=assess,rejection=rejection,crop_bytes=crop_bytes,geometry_contract=geometry_contract):
                        start=len(pending)
                        for face_index, face, quality, reason, vector, crop in records:
                            observation = str(uuid.uuid5(uuid.UUID(identifier), encoded([index, face_index, face.row])))
                            pending.append((observation, identifier, index, cue.start, safe_url(cue.image), encoded(list(cue.rect)),
                                            encoded(list(face.row)), encoded(quality), reason, backend.dimensions, vector, crop))
                        cursor = index + 1
                        frame_progress(completed=cursor, total=record['total'])
                        if grouped:
                            image=next(image for i,_,image in batch if i==index)
                            collector.collect_records(db,identifier,index,image,
                                sorted((dict(zip(columns,row)) for row in pending[start:]),key=lambda r:r['id']),seconds=cue.start)
                        pending_bytes+=sum(sum(len(v) for v in row if isinstance(v,(str,bytes)))
                            +(config.data['storage']['max_crop_bytes']+32768 if grouped else 0) for row in pending[start:])
                        supplemental_bytes=collector.pending_bytes() if grouped else 0
                        if (on_cue is not None and not grouped or cursor % config.data['execution']['checkpoint_cues'] == 0
                                or pending_bytes+supplemental_bytes>=config.data['execution']['scan_queue_bytes']//4):checkpoint()
                        if on_cue is not None and not grouped:
                            image=next(image for i,_,image in batch if i==index)
                            on_cue(db,identifier,index,image)

        checkpoint()
        with db:
            db.execute("UPDATE scans SET status='complete' WHERE id=?", (identifier,))
        return identifier, False
    except BaseException:
        # Only completed cues are durable. The current cue's tentative faces
        # must not enter a checkpoint if inference fails halfway through it.
        pending[:] = [row for row in pending if row[2] < cursor]
        from stash_ai.runtime.cleanup import cleanup
        if grouped:
            # Face and body checkpoints are atomic. Incomplete grouped work is
            # replayed from the last durable cursor, never half-published.
            pending.clear()
            cleanup([('abort person checkpoint',collector.abort)])
        else:cleanup([('save recognition checkpoint', checkpoint)])
        raise
    finally:
        from stash_ai.runtime.cleanup import cleanup
        cleanup([('close recognition backend', backend.close)])

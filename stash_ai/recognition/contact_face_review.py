"""Reconcile only disputed RetinaFace observations after both scan passes."""
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math

from stash_ai.runtime.cleanup import closing
from . import contact_faces
from .body_observations import contract, repair_projections
from .database import encoded
from .models import checked_detection
from .rotated_faces import head_crop, detect_head, store_recovered


@dataclass
class CueReview:
    rows: list
    bodies: list
    faces: list
    people: list
    measured: list
    mixed: dict
    retries: list
    search: set
    centers: dict

    def needed(self):
        return bool(self.mixed or self.retries or self.search or self.centers)


def _inputs(db, scan, cue, expected, size, strict):
    rows = list(db.execute('SELECT id,geometry,rejected FROM observations WHERE scan=? AND cue=? ORDER BY id',
                           (scan, cue)))
    bodies = list(db.execute('SELECT id,geometry,face FROM body_observations '
                             'WHERE scan=? AND contract=? AND cue=? ORDER BY id', (scan, expected, cue)))
    faces = [checked_detection(json.loads(row['geometry'])) for row in rows]
    people = [json.loads(body['geometry']) for body in bodies]
    measured = contact_faces.measured_people(people, size)
    mixed, retries, search, centers = contact_faces.cue_work(rows, faces, measured, strict)
    rejected = {rows[i]['id'] for i in mixed.keys() | centers.keys()}
    search = {i for i in search if bodies[i]['face'] is None or bodies[i]['face'] in rejected}
    return CueReview(rows, bodies, faces, people, measured, mixed, retries, search, centers)


def _reject(db, row, face, bodies, sources, reason):
    proof = encoded([bodies[i]['id'] for i in sources])
    db.execute("UPDATE observations SET rejected=?,geometry=?,embedding=NULL,crop=NULL,"
               "quality=json_set(quality,'$.conflicting_pose_heads',json(?)) WHERE id=?",
               (reason, encoded(face.row), proof, row['id']))
    db.execute('DELETE FROM members WHERE observation=?', (row['id'],))
    db.execute('DELETE FROM person_evidence WHERE observation=?', (row['id'],))
    db.execute("UPDATE body_observations SET face=NULL,quality=json_set("
               "json_remove(quality,'$.quality.contact_face_evidence'),'$.face_status','absent') WHERE face=?",
               (row['id'],))


def _center_replaced(backend, image, face, measured, owners):
    """A midpoint alone might be a third person. Vote only if a fresh crop sees exactly both heads."""
    from .rotated_faces import restore
    x, y, w, h = face.box
    box = (max(0, math.floor(x-w/2)), max(0, math.floor(y-h/2)),
           min(image.width, math.ceil(x+1.5*w)), min(image.height, math.ceil(y+1.5*h)))
    if box == (0, 0, image.width, image.height):
        return False
    with image.crop(box) as tile:
        found = backend.contact_candidates(tile)
    if len(found) != 2:
        return False
    claims = set()
    for local in found:
        candidate = restore(local, box, 0)
        heads = contact_faces.nearby_heads(candidate, measured)
        claims.add(contact_faces.ownership(candidate, measured, heads)[0])
    return claims == set(owners)


def _apply(db, scan, index, cue, image, config, model, visibility, inputs):
    rows, faces, bodies = inputs.rows, inputs.faces, inputs.bodies
    mixed = inputs.mixed
    # Crop-only mode has discarded the original points. Re-measure only the
    # disputed boxes, without admitting any result until pose ownership agrees.
    repairs = {}
    if inputs.retries:
        candidates = model().contact_candidates(image, [faces[i] for i in inputs.retries])
        for i, candidate in zip(inputs.retries, candidates, strict=True):
            heads = contact_faces.nearby_heads(candidate, inputs.measured)
            owner, sources = contact_faces.ownership(candidate, inputs.measured, heads)
            if sources:
                mixed[i] = sources
                faces[i] = candidate
            elif owner is None and (pair := contact_faces.intermediate_pair(candidate, heads)):
                inputs.centers[i] = pair
                faces[i] = candidate
            elif owner is not None:
                repairs.setdefault(owner, []).append((i, candidate))

    replaced = {i: owners for i, owners in inputs.centers.items()
                if _center_replaced(model(), image, faces[i], inputs.measured, owners)}
    for i, sources in (mixed | replaced).items():
        reason = contact_faces.BETWEEN if i in replaced else contact_faces.MIXED
        _reject(db, rows[i], faces[i], bodies, sources, reason)
    discard = mixed.keys() | replaced.keys()
    removed = {rows[i]['id'] for i in discard}
    occupied = {j for j,b in enumerate(bodies) if b['face'] and b['face'] not in removed}
    existing = [f for i,f in enumerate(faces) if i not in discard and rows[i]['rejected'] is None]
    def store(owner, face, original=None):
        body = bodies[owner]
        return store_recovered(db, scan, index, cue, image, config, model(), visibility, face, body,
                               contract_key=contact_faces.CONTRACT, provenance={'contact_head_source': body['id']},
                               original=original)

    for owner, choices in repairs.items():
        if len(choices) != 1:
            continue
        i, candidate = choices[0]
        if owner in occupied and bodies[owner]['face'] != rows[i]['id']:
            continue
        if any(b['face'] == rows[i]['id'] for j,b in enumerate(bodies) if j != owner):
            continue
        if any(contact_faces.conflicting_duplicate(candidate, other, inputs.measured, owner) for other in existing):
            continue
        if store(owner, candidate, rows[i]['id']):
            existing.append(candidate)
            occupied.add(owner)

    maximum = config.data['detection']['max_faces_per_frame']
    # Keep existing quality rejections in force. Only confirmed mixed geometry
    # and unavailable landmarks may give way to new independent measurements.
    blockers = [f for i,f in enumerate(faces) if i not in discard and rows[i]['rejected']
                and rows[i]['rejected'] not in (contact_faces.MIXED, contact_faces.BETWEEN, 'face_landmarks_invalid')]
    for owner in sorted(inputs.search - occupied):
        if len(existing) >= maximum:
            break
        crop = head_crop(inputs.people[owner], image.size)
        if crop is None:
            continue
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.rotated_face_checked',1) WHERE id=?",
                   (bodies[owner]['id'],))
        result = detect_head(model(), image, *crop, inputs.people, owner, existing + blockers, contact=True)
        if result and store(owner, result[0]):
            existing.append(result[0])
    repair_projections(db, config, scan, cue=index)


def review(db, scan, source, config, acquire, visibility, release):
    from .dependencies import supported, version, store
    expected = contract(config)
    kind = contact_faces.CONTRACT
    durable = supported(db)
    def generation():
        return encoded([version(db, 'raw-body', scan), version(db, 'raw-observations', scan)])
    if durable and db.execute('SELECT 1 FROM evidence_fingerprints WHERE kind=? AND key=? AND policy=? AND digest=?',
                              (kind, scan, expected, generation())).fetchone():
        return
    strict = config.data['detection'].get('landmarks_from_facebox', False)
    # Bound retained geometry to one cue. Avoid sprite decoding or loading
    # RetinaFace for ordinary pairs whose detections already agree with pose.
    crowded = db.execute('SELECT b.cue FROM body_observations b WHERE b.scan=? AND b.contract=? '
                         'AND EXISTS (SELECT 1 FROM observations o WHERE o.scan=b.scan AND o.cue=b.cue) '
                         """GROUP BY b.cue HAVING COUNT(*) >= 2 AND (COUNT(b.face) < COUNT(*)
                         OR EXISTS (SELECT 1 FROM observations o WHERE o.scan=b.scan AND o.cue=b.cue
                             AND (o.rejected='face_landmarks_invalid' OR NOT EXISTS (
                                 SELECT 1 FROM body_observations linked WHERE linked.face=o.id
                                 AND linked.contract=b.contract)))) ORDER BY b.cue""", (scan, expected))
    def inputs(i):
        return _inputs(db, scan, i, expected, source.cues[i].rect[2:], strict)
    planned = []
    for (i,) in crowded:
        if inputs(i).needed():
            planned.append(i)
    if planned:
        with ExitStack() as stack:
            backend = None
            def model():
                nonlocal backend
                if backend is None:
                    stack.callback(release)
                    backend = stack.enter_context(closing(acquire(), name='close contact face backend'))
                return backend
            frames = source.frames_indices(planned) if hasattr(source,'frames_indices') else source.frames(planned[0])
            pending = set(planned)
            with closing(frames, name='close contact face cues') as stream:
                for i, cue, image in stream:
                    with image:
                        if i in pending:
                            with db:
                                _apply(db, scan, i, cue, image, config, model, visibility, inputs(i))
                            pending.remove(i)
            if pending:
                raise ValueError('The VTT source omitted a requested face-contact cue.')
    if durable:
        with db:
            store(db, kind, scan, expected, version(db, 'raw-body', scan), generation())

"""Reuse historical frame evidence/JPEGs when quality preferences change."""
import json

from .config import plain
from .database import encoded
from .expression import cached as cached_expression, required as expression_required
from .visibility import cached as cached_visibility, required as visibility_required


def reuse(database, config, scope, target, observation, scene, seconds, namespace, identifier, policy_sha, selection):
    policy, db = config.data['portraits'], database.connection
    old = db.execute('''SELECT * FROM portraits WHERE scope=? AND person=? AND scene=? AND image IS NOT NULL
      AND json_extract(metadata,'$.source_type')='video' AND json_extract(metadata,'$.namespace')=?
      AND json_extract(metadata,'$.scan')=? AND json_extract(metadata,'$.observation')=?
      AND json_extract(metadata,'$.requested_seconds')=?
      AND json_extract(metadata,'$.policy.margin') IS ?
      AND json_extract(metadata,'$.policy.aspect_ratio') IS ?
      AND json_extract(metadata,'$.policy.jpeg_quality') IS ? ORDER BY rowid DESC LIMIT 1''',
      (scope, target['person'], scene, namespace, observation['scan'], observation['id'], seconds,
       policy['margin'], policy['aspect_ratio'], policy['jpeg_quality'])).fetchone()
    if old is None:
        return None
    metadata = json.loads(old['metadata'])
    previous_policy = dict(metadata.get('policy', {}))
    current_policy = plain(policy)
    for field in ('weights', 'min_face_pixels', 'min_sharpness', 'max_pose_deviation',
                  'min_exposure', 'max_exposure', 'max_clipped_fraction', 'min_visible_face_fraction'):
        previous_policy.pop(field, None)
        current_policy.pop(field, None)
    if previous_policy != current_policy:
        return None
    quality = metadata.get('quality', {})
    if ('identity_embedding' not in metadata or
            expression_required(policy) and not cached_expression(quality) or
            visibility_required(policy) and not cached_visibility(quality)):
        return None
    metadata.update(policy_sha=policy_sha, policy=plain(policy), shortlist=selection,
                    performer=target['performer'], reused_from=old['id'])
    # Current identity eligibility and soft scores are checked before reuse
    # can authorize publication. The historical record itself is never relabeled.
    with db:
        db.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
                   (identifier, scope, target['person'], scene, encoded(metadata), old['image'], old['digest'], 'candidate'))
    return db.execute('SELECT * FROM portraits WHERE id=?', (identifier,)).fetchone()

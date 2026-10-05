"""Retire obsolete replay engines while retaining publication geometry and ownership."""
from stash_ai.db.snapshot_migration import decode_legacy
from stash_ai.evidence.formats import ANALYSIS_FORMAT, current
from stash_ai.storage.snapshots import encode, validate_policy


def retire(item):
    result = dict(item)
    options = item.get('build_options')
    if options is not None:
        if options.get('algorithm') not in ('temporal', 'historical'):
            raise ValueError('Unsupported frozen algorithm.')
        # Validate historical policies structurally before retiring their engine.
        validate_policy({'build_options': {**options, 'algorithm': 'temporal'}})
    if 'scene_evidence' in result:
        result['scene_evidence'] = [retire(hit) for hit in result['scene_evidence']]
    meta = item.get('meta', {})
    version = meta.get('analysis_format')
    if version is not None and (type(version) is not int or not 1 <= version <= ANALYSIS_FORMAT):
        raise ValueError('Unsupported inference evidence format; update stash_ai.')
    obsolete = (meta.get('origin') == 'legacy'
                or (meta.get('origin') == 'inference' and not current(meta))
                or (options is not None and options['algorithm'] != 'temporal')
                or any(hit.get('requires_rebase') for hit in result.get('scene_evidence', [])))
    if obsolete:
        result['requires_rebase'] = True
        if options is not None:
            result['retired_build_options'] = result.pop('build_options')
    return result


def normalize(snapshot):
    return {**snapshot, 'contributions': [retire(item) for item in snapshot['contributions']]}


def migrate(db):
    for scope, scene, kind, published, pending in db.execute(
            'SELECT scope,scene,kind,published,pending FROM snapshots').fetchall():
        values = [encode(normalize(decode_legacy(blob))) if blob is not None else None
                  for blob in (published, pending)]
        db.execute('UPDATE snapshots SET published=?,pending=? WHERE scope=? AND scene=? AND kind=?',
                   (*values, scope, scene, kind))
    for scope, scene, kind, operation, blob in db.execute(
            'SELECT scope,scene,kind,operation,data FROM recovery_publications').fetchall():
        db.execute('UPDATE recovery_publications SET data=? WHERE scope=? AND scene=? AND kind=? AND operation=?',
                   (encode(normalize(decode_legacy(blob))), scope, scene, kind, operation))

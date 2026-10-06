"""Offline proof conversion; unreplayable proofs retain geometry, not old payloads."""
from stash_ai.storage.snapshots import encode
from .snapshot_migration import decode_legacy, normalize


def convert(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, (dict, list)) and id(value) in memo:
        return memo[id(value)]
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(convert(item, memo) for item in value)
        return result
    if not isinstance(value, dict):
        return value
    result = {}
    memo[id(value)] = result
    result.update((key, convert(item, memo)) for key, item in value.items()
                  if key != 'extra_ids' and not key.startswith('retired_'))
    if any(key.startswith('retired_') for key in value):
        result['requires_rebase'] = True
    for field in ('frame_context', 'scene_context', 'support_context'):
        if field not in result:
            continue
        context = result[field]
        if not isinstance(context, dict):
            raise ValueError('Invalid historical proof context.')
        scalar, retired = {}, {}
        for source, signatures in context.items():
            if isinstance(signatures, str) and signatures:
                scalar[source] = signatures
            elif isinstance(signatures, list) and signatures and all(isinstance(sig, str) and sig for sig in signatures):
                if len(set(signatures)) == 1:
                    scalar[source] = signatures[0]
                else:
                    retired[source] = signatures
            else:
                raise ValueError('Invalid historical proof signature.')
        result[field] = scalar
        if retired:
            result['requires_rebase'] = True
    if 'signatures' in result:
        signatures = result.pop('signatures')
        if not isinstance(signatures, list) or not signatures or any(not isinstance(sig, str) for sig in signatures):
            raise ValueError('Invalid historical source signatures.')
        if len(set(signatures)) == 1:
            result['signature'] = signatures[0]
        else:
            result['requires_rebase'] = True
    if result.get('derived') == 'shared_scene_frames' and 'frame_context' in result:
        frames = result.pop('frame_context')
        if frames != result.get('scene_context'):
            result['requires_rebase'] = True
    def retired(item):
        if isinstance(item, dict):
            return item.get('requires_rebase', False) or any(retired(v) for v in item.values())
        return isinstance(item, list) and any(retired(v) for v in item)
    if any(retired(result.get(field)) for field in ('scene_evidence', 'extra_support', 'source_evidence')):
        result['requires_rebase'] = True
    if result.get('requires_rebase'):
        result.pop('build_options', None)
    return result


def migrate(db, retired):
    from stash_ai.storage.archive import references
    def migrated(blob, scope, scene):
        snapshot = convert(normalize(decode_legacy(blob)))
        removed = retired.get((scope, scene), set())
        for item in snapshot['contributions']:
            if references(item) & removed:
                item['requires_rebase'] = True
                item.pop('build_options', None)
        return encode(snapshot)
    for scope, scene, kind in db.execute('SELECT scope,scene,kind FROM snapshots').fetchall():
        published, pending = db.execute(
            'SELECT published,pending FROM snapshots WHERE scope=? AND scene=? AND kind=?',
            (scope, scene, kind)).fetchone()
        values = [migrated(blob, scope, scene) if blob is not None else None
                  for blob in (published, pending)]
        db.execute('UPDATE snapshots SET published=?,pending=? WHERE scope=? AND scene=? AND kind=?',
                   (*values, scope, scene, kind))
    for scope, scene, kind, operation in db.execute(
            'SELECT scope,scene,kind,operation FROM recovery_publications').fetchall():
        blob = db.execute(
            'SELECT data FROM recovery_publications WHERE scope=? AND scene=? AND kind=? AND operation=?',
            (scope, scene, kind, operation)).fetchone()[0]
        db.execute('UPDATE recovery_publications SET data=? WHERE scope=? AND scene=? AND kind=? AND operation=?',
                   (migrated(blob, scope, scene), scope, scene, kind, operation))

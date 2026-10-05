"""Offline conversion of list-shaped proof contexts; archive pins remain exact."""
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
    result.update((key, convert(item, memo)) for key, item in value.items())
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
            result['retired_' + field] = retired
            result['requires_rebase'] = True
    if 'signatures' in result:
        signatures = result.pop('signatures')
        if not isinstance(signatures, list) or not signatures or any(not isinstance(sig, str) for sig in signatures):
            raise ValueError('Invalid historical source signatures.')
        if len(set(signatures)) == 1:
            result['signature'] = signatures[0]
        else:
            result['retired_signatures'] = signatures
            result['requires_rebase'] = True
    if result.get('derived') == 'shared_scene_frames' and 'frame_context' in result:
        frames = result.pop('frame_context')
        if frames != result.get('scene_context'):
            result['retired_frame_context'] = {source: [signature] for source, signature in frames.items()}
            result['requires_rebase'] = True
    def retired(item):
        if isinstance(item, dict):
            return item.get('requires_rebase', False) or any(retired(v) for v in item.values())
        return isinstance(item, list) and any(retired(v) for v in item)
    if any(retired(result.get(field)) for field in ('scene_evidence', 'extra_support', 'source_evidence')):
        result['requires_rebase'] = True
    return result


def migrate(db):
    for scope, scene, kind, published, pending in db.execute(
            'SELECT scope,scene,kind,published,pending FROM snapshots').fetchall():
        values = [encode(convert(normalize(decode_legacy(blob)))) if blob is not None else None
                  for blob in (published, pending)]
        db.execute('UPDATE snapshots SET published=?,pending=? WHERE scope=? AND scene=? AND kind=?',
                   (*values, scope, scene, kind))
    for scope, scene, kind, operation, blob in db.execute(
            'SELECT scope,scene,kind,operation,data FROM recovery_publications').fetchall():
        db.execute('UPDATE recovery_publications SET data=? WHERE scope=? AND scene=? AND kind=? AND operation=?',
                   (encode(convert(normalize(decode_legacy(blob)))), scope, scene, kind, operation))

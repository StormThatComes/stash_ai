"""Semantic publication validation and bounded lookup of pinned inference proofs."""
from functools import lru_cache

from ..payloads import _label, _number, _score
from ..common import KNOWN_MODELS


def validate_snapshot(snapshot, kind):
    if kind not in ('scene', 'markers'):
        raise ValueError('Unknown publication kind.')
    for item in snapshot['contributions']:
        from stash_ai.storage.snapshots import validate_policy
        validate_policy(item)
        if (not _label(item.get('tag')) or item.get('source') not in KNOWN_MODELS
                or not isinstance(item.get('meta'), dict)
                or 'score' not in item or (item['score'] is not None and not _score(item['score']))):
            raise ValueError('Invalid publication contribution.')
        if kind == 'markers':
            if (not _number(item.get('start')) or 'end' not in item
                    or (item['end'] is not None and (not _number(item['end']) or item['end'] < item['start']))
                    or not isinstance(item.get('extra_tags', []), list)
                    or any(not _label(tag) for tag in item.get('extra_tags', []))):
                raise ValueError('Invalid publication geometry or labels.')
            if item['meta'].get('origin') == 'occurrence':
                evidence = item.get('occurrence_evidence')
                if (not isinstance(evidence, list) or not evidence
                        or any(not isinstance(witness, dict) or witness.get('source') not in KNOWN_MODELS
                               or not _score(witness.get('score')) or not isinstance(witness.get('meta'), dict)
                               or witness.get('evidence_kind') not in ('frame', 'scene_peak')
                               for witness in evidence) or not item.get('scene_evidence')):
                    raise ValueError('Invalid Occurrence evidence.')
            elif not item.get('requires_rebase') and not isinstance(item.get('evidence_signature'), str):
                raise ValueError('Missing marker inference identity.')
        if 'statistics' in item:
            from stash_ai.evidence.formats import validate_statistics
            validate_statistics(item['statistics'])
        if 'scene_evidence' in item:
            validate_snapshot({'contributions': item['scene_evidence']}, 'scene')
    if ('owned_ids' in snapshot and (not isinstance(snapshot['owned_ids'], list)
            or any(not isinstance(value, str) for value in snapshot['owned_ids']))):
        raise ValueError('Invalid publication ownership.')
    if 'cleanup_generation' in snapshot and (type(snapshot['cleanup_generation']) is not int
                                            or snapshot['cleanup_generation'] < 0):
        raise ValueError('Invalid publication cleanup generation.')


def proof_checker(db):
    """Cache only signatures, never decoded inference payloads."""
    @lru_cache(maxsize=128)
    def active(scope, scene, source, kind):
        from stash_ai.storage.codec import decode
        from stash_ai.evidence.identity import marker_segment_signature
        row = db.execute('SELECT data FROM results WHERE scope=? AND scene=? AND model=? AND kind=?',
                         (scope, scene, source, kind)).fetchone()
        return marker_segment_signature(decode(row[0])) if row else None

    def exists(scope, scene, source, kind, signature):
        if active(scope, scene, source, kind) == signature:
            return True
        return db.execute('SELECT 1 FROM inference_archive WHERE scope=? AND scene=? AND model=? '
                          'AND kind=? AND signature=?', (scope, scene, source, kind, signature)).fetchone() is not None

    def walk(value, kind):
        if isinstance(value, list):
            for item in value:
                yield from walk(item, kind)
        elif isinstance(value, dict) and not value.get('requires_rebase'):
            signature = value.get('evidence_signature')
            if signature is not None:
                source = value.get('source')
                if source not in KNOWN_MODELS or not isinstance(signature, str) or not signature:
                    raise ValueError('Invalid pinned inference identity.')
                yield source, kind, signature
            for field, child in value.items():
                if field in ('frame_context', 'scene_context', 'support_context'):
                    for source, signature in child.items():
                        if source not in KNOWN_MODELS:
                            raise ValueError('Unknown pinned inference source.')
                        yield source, 'scene' if field == 'scene_context' else kind, signature
                elif field == 'sources' and isinstance(child, dict):
                    for source, proof in child.items():
                        if isinstance(proof, dict) and proof.get('signature') is not None:
                            yield source, kind, proof['signature']
                else:
                    yield from walk(child, 'scene' if field in ('scene_evidence', 'occurrence_evidence') else kind)

    def missing(snapshot, scope, scene, kind):
        return any(not exists(scope, scene, source, proof_kind, signature)
                   for source, proof_kind, signature in set(walk(snapshot, kind)))
    return missing

"""Versioned snapshot storage; callers continue to see expanded contributions."""

FORMAT_KEY = '__stash_ai_snapshot__'
FORMAT = 2


def compact(snapshot):
    """Intern repeated immutable proof objects without hashing their large JSON.

    Merge's range cache already shares these objects across model contributions.
    Equal but independently owned proofs stay independent; unique proofs remain
    inline. Neither the snapshot nor its borrowed evidence is modified.
    """
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('contributions', []), list):
        raise ValueError('Invalid compact snapshot format.')
    counts = {}
    for item in snapshot.get('contributions', []):
        if not isinstance(item, dict):
            raise ValueError('Invalid compact snapshot contribution.')
        validate_policy(item)
        support = item.get('extra_support', {})
        if (not isinstance(support, dict) or any(not isinstance(tag, str) or not isinstance(proof, dict)
                                                for tag, proof in support.items())):
            raise ValueError('Invalid compact snapshot support.')
        for proof in support.values():
            identity = id(proof)
            counts[identity] = counts.get(identity, 0) + 1
    proofs, indexes, contributions = [], {}, []
    for item in snapshot.get('contributions', []):
        support = item.get('extra_support', {})
        if not any(counts[id(proof)] > 1 for proof in support.values()):
            contributions.append(item)
            continue
        references = {}
        # Stable reference numbers do not depend on dictionary insertion order.
        for tag, proof in sorted(support.items()):
            identity = id(proof)
            if counts[identity] == 1:
                references[tag] = proof
                continue
            if identity not in indexes:
                indexes[identity] = len(proofs)
                proofs.append(proof)
            references[tag] = indexes[identity]
        contributions.append({**item, 'extra_support': references})
    result = {**snapshot, **({'contributions': contributions} if 'contributions' in snapshot else {})}
    return {FORMAT_KEY: FORMAT, 'snapshot': result, 'proofs': proofs}


def expand(value, *, version=FORMAT):
    """Validate/expand a declared envelope. Runtime accepts only the current format."""
    if (not isinstance(value, dict) or set(value) != {FORMAT_KEY, 'snapshot', 'proofs'}
            or type(value[FORMAT_KEY]) is not int
            or value[FORMAT_KEY] != version or not isinstance(value['snapshot'], dict)
            or not isinstance(value['snapshot'].get('contributions', []), list)
            or not isinstance(value['proofs'], list) or any(not isinstance(proof, dict) for proof in value['proofs'])):
        raise ValueError('Invalid compact snapshot format.')
    proofs, contributions = value['proofs'], []
    for item in value['snapshot'].get('contributions', []):
        if not isinstance(item, dict):
            raise ValueError('Invalid compact snapshot contribution.')
        if version == FORMAT:
            validate_policy(item)
        if 'extra_support' not in item:
            contributions.append(item)
            continue
        if not isinstance(item['extra_support'], dict):
            raise ValueError('Invalid compact snapshot support.')
        support = {}
        for tag, proof in item['extra_support'].items():
            if not isinstance(tag, str):
                raise ValueError('Invalid compact snapshot support label.')
            if isinstance(proof, dict):
                support[tag] = proof
            elif type(proof) is int and 0 <= proof < len(proofs):
                support[tag] = proofs[proof]
            else:
                raise ValueError('Invalid compact snapshot proof reference.')
        contributions.append({**item, 'extra_support': support})
    return {**value['snapshot'], **({'contributions': contributions} if 'contributions' in value['snapshot'] else {})}


def validate_policy(item):
    options = item.get('build_options')
    if options is not None and 'policy' in options:
        from stash_ai.evidence.temporal.policy import TemporalPolicy
        TemporalPolicy.from_snapshot(options['policy'])


def decode(value, report=None):
    from stash_ai.runtime.timing import stage
    from stash_ai.storage.codec import decode as decode_json
    packed = decode_json(value, report)
    with stage(report, 'state_snapshot_expand'):
        return expand(packed)


def encode(value, report=None):
    from stash_ai.runtime.timing import stage
    from stash_ai.storage.codec import encode as encode_json
    with stage(report, 'state_snapshot_compact'):
        packed = compact(value)
    return encode_json(packed, report)

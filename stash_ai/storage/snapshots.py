"""Versioned snapshot storage; callers continue to see expanded contributions."""

FORMAT_KEY = '__stash_ai_snapshot__'
FORMAT = 1


def compact(snapshot):
    """Intern repeated immutable proof objects without hashing their large JSON.

    Merge's range cache already shares these objects across model contributions.
    Equal but independently owned proofs stay independent; unique proofs remain
    inline. Neither the snapshot nor its borrowed evidence is modified.
    """
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('contributions'), list):
        return snapshot
    counts = {}
    for item in snapshot['contributions']:
        if not isinstance(item, dict):
            return snapshot
        support = item.get('extra_support', {})
        if (not isinstance(support, dict) or any(not isinstance(tag, str) or not isinstance(proof, dict)
                                                for tag, proof in support.items())):
            return snapshot
        for proof in support.values():
            identity = id(proof)
            counts[identity] = counts.get(identity, 0) + 1
    if not any(count > 1 for count in counts.values()):
        return snapshot

    proofs, indexes, contributions = [], {}, []
    for item in snapshot['contributions']:
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
    return {FORMAT_KEY: FORMAT, 'snapshot': {**snapshot, 'contributions': contributions}, 'proofs': proofs}


def expand(value):
    """Read old JSON unchanged, or validate/expand the compact storage envelope."""
    if not isinstance(value, dict) or FORMAT_KEY not in value:
        return value
    if (set(value) != {FORMAT_KEY, 'snapshot', 'proofs'} or type(value[FORMAT_KEY]) is not int
            or value[FORMAT_KEY] != FORMAT or not isinstance(value['snapshot'], dict)
            or not isinstance(value['snapshot'].get('contributions'), list)
            or not isinstance(value['proofs'], list) or any(not isinstance(proof, dict) for proof in value['proofs'])):
        raise ValueError('Invalid compact snapshot format.')
    proofs, contributions = value['proofs'], []
    for item in value['snapshot']['contributions']:
        if not isinstance(item, dict):
            raise ValueError('Invalid compact snapshot contribution.')
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
    return {**value['snapshot'], 'contributions': contributions}

"""Versioned snapshot storage; callers continue to see expanded contributions."""

FORMAT_KEY = '__stash_ai_snapshot__'
FORMAT = 4


def compact(snapshot):
    """Intern repeated immutable proof objects without hashing their large JSON.

    Merge's range cache already shares these objects across model contributions.
    Equal but independently owned proofs stay independent; unique proofs remain
    inline. Neither the snapshot nor its borrowed evidence is modified.
    """
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('contributions'), list):
        raise ValueError('Invalid compact snapshot format.')
    counts = {}
    for item in snapshot['contributions']:
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
    result = {**snapshot, 'contributions': contributions}
    return {FORMAT_KEY: FORMAT, 'snapshot': result, 'proofs': proofs}


def expand(value, *, version=FORMAT, validate=True):
    """Validate/expand a declared envelope. Runtime accepts only the current format."""
    if (not isinstance(value, dict) or set(value) != {FORMAT_KEY, 'snapshot', 'proofs'}
            or type(value[FORMAT_KEY]) is not int
            or value[FORMAT_KEY] != version or not isinstance(value['snapshot'], dict)
            or not isinstance(value['snapshot'].get('contributions', [] if version < FORMAT else None), list)
            or not isinstance(value['proofs'], list) or any(not isinstance(proof, dict) for proof in value['proofs'])):
        raise ValueError('Invalid compact snapshot format.')
    proofs, contributions = value['proofs'], []
    for item in value['snapshot'].get('contributions', []):
        if not isinstance(item, dict):
            raise ValueError('Invalid compact snapshot contribution.')
        if version == FORMAT and validate:
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
    return {**value['snapshot'], 'contributions': contributions}


def validate_policy(item):
    options = item.get('build_options')
    if options is not None:
        from stash_ai.evidence.temporal.policy import frozen_policy
        from stash_ai.models import SOURCES
        marker = {'policy', 'settings', 'marker_thresholds', 'core_exclusions', 'tag_exclusions'}
        scene = {'policy', 'max_tags', 'scene_thresholds', 'scene_exclusions'}
        keys = set(options) - {"algorithm"} if isinstance(options, dict) else set()
        if not isinstance(options, dict) or keys not in (marker, scene):
            raise ValueError('Incomplete frozen build options; run --db-update.')
        if 'algorithm' not in options:
            raise ValueError('Missing frozen algorithm; run --db-update.')
        frozen_policy(options)
        fields = ('core_exclusions', 'tag_exclusions') if keys == marker else ('scene_exclusions',)
        for field in fields:
            values = options[field]
            if (not isinstance(values, dict) or set(values) != set(SOURCES)
                    or any(not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)
                           for tags in values.values())):
                raise ValueError('Incomplete frozen exclusions; run --db-update.')
        thresholds = options['marker_thresholds' if keys == marker else 'scene_thresholds']
        if not isinstance(thresholds, dict):
            raise ValueError('Invalid frozen thresholds.')
        if keys == marker:
            settings = options['settings']
            if not isinstance(settings, dict) or set(settings) != {
                    'MARKER_MIN_HITS', 'MARKER_MERGE_GAP_SECONDS',
                    'MARKER_MAX_TAGS_PER_FRAME', 'MAX_MARKERS_PER_SCENE'}:
                raise ValueError('Incomplete frozen marker settings; run --db-update.')
    if 'scene_evidence' in item:
        if not isinstance(item['scene_evidence'], list) or any(
                not isinstance(hit, dict) for hit in item['scene_evidence']):
            raise ValueError('Invalid frozen scene evidence.')
        for hit in item['scene_evidence']:
            validate_policy(hit)


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


def require_rebased(contributions):
    if any(item.get('requires_rebase') for item in contributions):
        raise RuntimeError('Retired publication proof; run fresh inference for its models and --merge-only before cleanup.')

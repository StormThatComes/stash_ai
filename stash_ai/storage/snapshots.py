"""Versioned snapshot storage; callers continue to see expanded contributions."""

FORMAT_KEY = '__stash_ai_snapshot__'
FORMAT = 7


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


def expand(value):
    """Validate and expand the current snapshot envelope."""
    if (not isinstance(value, dict) or set(value) != {FORMAT_KEY, 'snapshot', 'proofs'}
            or type(value[FORMAT_KEY]) is not int or value[FORMAT_KEY] != FORMAT
            or not isinstance(value['snapshot'], dict)
            or not isinstance(value['snapshot'].get('contributions'), list)
            or not isinstance(value['proofs'], list)
            or any(not isinstance(proof, dict) for proof in value['proofs'])):
        raise ValueError('Invalid compact snapshot format.')
    for item in value['snapshot']['contributions']:
        if not isinstance(item, dict):
            raise ValueError('Invalid compact snapshot contribution.')
        validate_policy(item)
    return _expand_proofs(value['snapshot'], value['proofs'])


def _expand_proofs(snapshot, proofs):
    """Mechanical proof-reference expansion shared with offline upgrades."""
    contributions = []
    for item in snapshot['contributions']:
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
    return {**snapshot, 'contributions': contributions}


def validate_policy(item):
    if 'extra_ids' in item or any(key.startswith('retired_') for key in item):
        raise ValueError('Retired publication fields; run --db-update.')
    for field in ('frame_context', 'scene_context', 'support_context'):
        if field in item and (not isinstance(item[field], dict) or any(
                not isinstance(source, str) or not isinstance(signature, str) or not signature
                for source, signature in item[field].items())):
            raise ValueError('Current publication contexts require one signature per source; run --db-update.')
    options = item.get('build_options')
    if options is not None:
        from stash_ai.evidence.temporal.policy import frozen_policy
        from stash_ai.models import SOURCES
        marker = {'policy', 'settings', 'marker_thresholds', 'core_exclusions', 'tag_exclusions'}
        scene = {'policy', 'scene_thresholds', 'scene_exclusions'}
        keys = set(options) if isinstance(options, dict) else set()
        if not isinstance(options, dict) or keys not in (marker, scene):
            raise ValueError('Incomplete frozen build options; run --db-update.')
        frozen_policy(options)
        fields = ('core_exclusions', 'tag_exclusions') if keys == marker else ('scene_exclusions',)
        for field in fields:
            values = options[field]
            if (not isinstance(values, dict) or set(values) != set(SOURCES)
                    or any(not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)
                           for tags in values.values())):
                raise ValueError('Incomplete frozen exclusions; run --db-update.')
        thresholds = options['marker_thresholds' if keys == marker else 'scene_thresholds']
        if (not isinstance(thresholds, dict) or any(
                not isinstance(signature, str) or not signature or type(value) not in (int, float)
                or not 0 <= value <= 1 for signature, value in thresholds.items())):
            raise ValueError('Invalid frozen thresholds.')
        if keys == marker:
            settings = options['settings']
            if not isinstance(settings, dict) or set(settings) != {
                    'MARKER_MIN_HITS', 'MARKER_MERGE_GAP_SECONDS',
                    'MARKER_MAX_TAGS_PER_FRAME'}:
                raise ValueError('Incomplete frozen marker settings; run --db-update.')
            for name in ('MARKER_MIN_HITS', 'MARKER_MAX_TAGS_PER_FRAME'):
                if type(settings[name]) is not int or settings[name] < (1 if name == 'MARKER_MIN_HITS' else 0):
                    raise ValueError(f'Invalid frozen marker setting {name}.')
            gap = settings['MARKER_MERGE_GAP_SECONDS']
            if type(gap) not in (int, float) or not 0 <= gap < float('inf'):
                raise ValueError('Invalid frozen marker setting MARKER_MERGE_GAP_SECONDS.')
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

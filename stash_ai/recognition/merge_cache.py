"""Bounded exact constituent proofs and owned support prefixes in replay RAM."""
import weakref

from .evidence_cache import current, MISSING, same_owners
from .profiling import count


def versions(groups):
    owners, keys = [], []
    for group in groups:
        if not all(not sample.flags.writeable and isinstance(sample.base, bytes) for sample in group.samples):
            return None
        samples = tuple(group.samples)
        owners.append(samples)
        keys.append(tuple(id(sample) for sample in samples))
    return tuple(keys), tuple(owners)


def weak_owners(owners):
    return tuple(tuple(weakref.ref(sample) for sample in group) for group in owners)


def valid_owners(refs, owners):
    return len(refs) == len(owners) and all(same_owners(a, b) for a, b in zip(refs, owners))


def cross_identity(first, second, config):
    version = versions((first, second))
    if version is None:
        return None
    keys, owners = version
    key = ('cross', keys, config.data['recognition']['match_threshold'],
           config.data['clustering']['cohesion_threshold'])
    return key, owners


def find_cross(cache, identity):
    if cache is None or identity is None:
        return MISSING
    key, owners = identity
    found = cache.cache.get(key)
    if found is not MISSING and valid_owners(found[1], owners):
        count('reused_constituent_pairs')
        return found[0]
    return MISSING


def save_cross(cache, identity, result):
    if cache is None or identity is None:
        return
    key, owners = identity
    cache.cache.put(key, (result, weak_owners(owners)), sum(sample.nbytes+128 for group in owners for sample in group))


def cached_cross(first, second, config, compute):
    cache = current()
    identity = cross_identity(first, second, config) if cache is not None else None
    found = find_cross(cache, identity)
    if found is not MISSING:
        return found
    result = compute(first, second, config)
    save_cross(cache, identity, result)
    return result


def support_identity(cache, selected, policy, source, target, components, config, minimum):
    if cache is None:
        return None
    version = versions(components[source] + components[target])
    if version is None:
        return None
    keys, owners = version
    key = ('votes', source, target, policy, selected['scope'], selected['namespace'], selected['policy_sha'],
           config.data['recognition']['match_threshold'], config.data['clustering']['cohesion_threshold'],
           minimum, keys)
    return key, owners


def public_votes(votes):
    from .independent_evidence import representatives
    return [{key: vote[key] for key in ('scene', 'seconds', 'observation')}
            for vote in representatives(votes)]


def save_votes(cache, identity, votes):
    if identity is not None and all(type(vote['order'][0]) is int and type(vote['order'][1]) is int
                                    and isinstance(vote['order'][2], str) for vote in votes):
        key, owners = identity
        cache.cache.put(key, (votes, weak_owners(owners)), len(votes)*512 +
                        sum(sample.nbytes+128 for group in owners for sample in group))


def incremental_votes(cache, identity, selected, nomination, components, config, minimum, matches):
    if identity is None or nomination is None:
        return MISSING
    found = cache.cache.get(identity[0])
    if found is MISSING or not valid_owners(found[1], identity[1]):
        return MISSING
    votes = list(found[0])
    inserted, row = nomination
    if not inserted:
        count('reused_support_prefixes')
        return public_votes(votes)
    if type(row['cue']) is not int or not isinstance(row['id'], str):
        return MISSING
    if (votes and len(public_votes(votes)) >= minimum and
            (selected['scan_order'], row['cue'], row['id']) >= votes[-1]['order']):
        count('reused_support_prefixes')
        return public_votes(votes)
    # A newly inserted edge may change the maximum matching, including which
    # earlier duplicate supplies a timestamp. Revalidate the ordered prefix;
    # unchanged nominations still reuse it without issuing any SQL.
    return MISSING

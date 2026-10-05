"""Retire obsolete replay engines while retaining publication geometry and ownership."""
from stash_ai.evidence.formats import ANALYSIS_FORMAT, current
from stash_ai.storage.snapshots import validate_policy


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
            result.pop('build_options')
    return result


def normalize(snapshot):
    return {**snapshot, 'contributions': [retire(item) for item in snapshot['contributions']]}

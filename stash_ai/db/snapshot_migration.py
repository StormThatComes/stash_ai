"""Historical publication formats are converted only by offline maintenance."""
from dataclasses import asdict

from stash_ai.storage.codec import decode
from stash_ai.storage.snapshots import FORMAT, FORMAT_KEY, expand, encode
from stash_ai.models import SOURCES


def complete_policy(values):
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    if not isinstance(values, dict):
        raise ValueError('Invalid historical temporal policy.')
    if 'enabled' in values and type(values['enabled']) is not bool:
        raise ValueError('Invalid historical temporal discriminator.')
    saved = {key: value for key, value in values.items() if not key.startswith('refine_') and key != 'enabled'}
    for name in ('shared_marker_frames', 'shared_marker_extras', 'short_marker_extras',
                 'shared_scene_tags', 'occurrence_markers'):
        saved.setdefault(name, False)
    policy = TemporalPolicy(**saved)
    policy.validate()
    return asdict(policy)


def decode_legacy(blob):
    value = decode(blob)
    if isinstance(value, dict) and FORMAT_KEY in value:
        version = value[FORMAT_KEY]
        if type(version) is not int or version not in range(1, FORMAT + 1):
            raise ValueError('Unsupported historical snapshot format.')
        value = expand(value, version=version, validate=False)
    if not isinstance(value, dict):
        raise ValueError('Invalid historical snapshot format.')
    return value


def complete_options(options):
    if not isinstance(options, dict) or 'policy' not in options:
        raise ValueError('Invalid historical build options.')
    result = {**options, 'algorithm': options.get('algorithm', 'temporal' if options['policy'].get('enabled', True) else 'historical'),
              'policy': complete_policy(options['policy'])}
    if 'max_tags' in options:
        result.setdefault('scene_thresholds', {})
        fields = ('scene_exclusions',)
    else:
        result.setdefault('marker_thresholds', {})
        fields = ('core_exclusions', 'tag_exclusions')
    for field in fields:
        # Only tag exclusions had historical empty-map defaults. Missing core
        # or scene bans cannot be reconstructed from today's configuration.
        required = field != 'tag_exclusions'
        saved = result.get(field, {})
        if (not isinstance(saved, dict) or set(saved) - set(SOURCES)
                or (required and set(saved) != set(SOURCES))
                or any(not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)
                       for tags in saved.values())):
            raise ValueError('Invalid historical frozen exclusions.')
        result[field] = {source: list(saved.get(source, [])) for source in SOURCES}
    retired = result.pop('blocklist', [])
    if not isinstance(retired, list) or any(not isinstance(tag, str) for tag in retired):
        raise ValueError('Invalid historical marker blocklist.')
    if retired:
        if 'tag_exclusions' not in result:
            raise ValueError('Unexpected historical scene blocklist.')
        result['tag_exclusions'] = {source: sorted(set(tags) | set(retired))
                                    for source, tags in result['tag_exclusions'].items()}
    return result


def normalize_item(item):
    if not isinstance(item, dict):
        raise ValueError('Invalid historical snapshot contribution.')
    result = dict(item)
    if item.get('build_options') is not None:
        result['build_options'] = complete_options(item['build_options'])
    if 'scene_evidence' in item:
        if not isinstance(item['scene_evidence'], list):
            raise ValueError('Invalid historical scene evidence.')
        result['scene_evidence'] = [normalize_item(hit) for hit in item['scene_evidence']]
    return result


def normalize(snapshot):
    # Preserve shared proof objects. Only contribution build options are copied.
    if 'contributions' not in snapshot:
        return {**snapshot, 'contributions': []}
    if not isinstance(snapshot['contributions'], list) or any(
            not isinstance(item, dict) for item in snapshot['contributions']):
        raise ValueError('Invalid historical snapshot contributions.')
    contributions = [normalize_item(item) for item in snapshot['contributions']]
    from .publication_rebase import normalize
    return normalize({**snapshot, 'contributions': contributions})


def migrate(db):
    for scope, scene, kind, published, pending in db.execute(
            'SELECT scope,scene,kind,published,pending FROM snapshots'):
        values = [encode(normalize(decode_legacy(blob))) if blob is not None else None
                  for blob in (published, pending)]
        db.execute('UPDATE snapshots SET published=?,pending=? WHERE scope=? AND scene=? AND kind=?',
                   (*values, scope, scene, kind))
    for scope, scene, kind, operation, blob in db.execute(
            'SELECT scope,scene,kind,operation,data FROM recovery_publications'):
        db.execute('UPDATE recovery_publications SET data=? WHERE scope=? AND scene=? AND kind=? AND operation=?',
                   (encode(normalize(decode_legacy(blob))), scope, scene, kind, operation))

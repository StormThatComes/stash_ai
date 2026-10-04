"""Historical publication formats are converted only by offline maintenance."""
from dataclasses import asdict

from stash_ai.storage.codec import decode
from stash_ai.storage.snapshots import FORMAT_KEY, expand, encode


def complete_policy(values):
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    if not isinstance(values, dict):
        raise ValueError('Invalid historical temporal policy.')
    saved = {key: value for key, value in values.items() if not key.startswith('refine_')}
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
        if type(version) is not int or version not in (1, 2):
            raise ValueError('Unsupported historical snapshot format.')
        value = expand(value, version=version)
    if not isinstance(value, dict):
        raise ValueError('Invalid historical snapshot format.')
    return value


def normalize(snapshot):
    # Preserve shared proof objects. Only contribution build options are copied.
    if 'contributions' not in snapshot:
        return snapshot
    if not isinstance(snapshot['contributions'], list) or any(
            not isinstance(item, dict) for item in snapshot['contributions']):
        raise ValueError('Invalid historical snapshot contributions.')
    contributions = []
    for item in snapshot['contributions']:
        options = item.get('build_options')
        if options is not None and 'policy' in options:
            item = {**item, 'build_options': {**options, 'policy': complete_policy(options['policy'])}}
        contributions.append(item)
    return {**snapshot, 'contributions': contributions}


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

"""Strict auto portrait tiers, based on current retained face/person evidence."""
import json

from .database import encoded


AUTO_PRECEDENCE = ('face+pose', 'face', 'pose')


def pose_contract(config):
    from .person_signal import contract
    return contract(config) if config.data['sapiensid']['enabled'] else None


def face_clause(db, config, category):
    """SQL filter for face observations aliased as o; no embedding/crop reads."""
    if category is None:
        return '', ()
    if category not in ('face+pose', 'face'):
        raise ValueError('Unknown face portrait category.')
    from .person_signal import available
    expected = pose_contract(config)
    if expected is None or not available(db):
        return (' AND 0' if category == 'face+pose' else ''), ()
    ready = ("EXISTS (SELECT 1 FROM person_evidence pe WHERE pe.observation=o.id "
             "AND pe.contract=? AND pe.status='ready' AND json_valid(pe.metadata) "
             "AND json_type(pe.metadata,'$.person')='object' "
             "AND json_extract(pe.metadata,'$.identity_rejection') IS NULL)")
    return ' AND ' + (ready if category == 'face+pose' else 'NOT ' + ready), (expected,)


def category(db, config, metadata):
    if metadata.get('portrait_kind') == 'person':
        return 'pose'
    # A decoded video face has no pose evidence for those actual pixels.
    if metadata.get('source_type') == 'video':
        return 'face'
    clause, parameters = face_clause(db, config, 'face+pose')
    paired = db.execute('SELECT 1 FROM observations o WHERE o.id=?' + clause,
                        (metadata.get('observation'), *parameters)).fetchone()
    return 'face+pose' if paired else 'face'


def refresh_category(database, portrait, config):
    """Historical portrait priority must follow today's evidence, not old metadata."""
    metadata = json.loads(portrait['metadata'])
    current = category(database.connection, config, metadata)
    if config.data['portraits']['kind'] == 'face':
        current = 'face'
    from .paired_portraits import framing, matches, FACE_FRAMING
    try:
        expected = (framing(database.connection, config, metadata.get('observation'))
                    if current == 'face+pose' else FACE_FRAMING)
    except (ValueError, TypeError):
        return None
    if not matches(metadata, expected):
        return None
    if metadata.get('selection_category') != current:
        metadata['selection_category'] = current
        with database.connection:
            database.connection.execute('UPDATE portraits SET metadata=? WHERE id=?',
                                        (encoded(metadata), portrait['id']))
        portrait = database.connection.execute('SELECT * FROM portraits WHERE id=?', (portrait['id'],)).fetchone()
    return portrait

"""Retire obsolete inference offline; publication geometry survives in snapshots."""
from stash_ai.storage.codec import decode
from stash_ai.evidence.formats import ANALYSIS_FORMAT, UnorderedFrames, current, validate_current
from stash_ai.evidence.identity import marker_segment_signature
from .payloads import valid_result


def obsolete(record, kind):
    if (not isinstance(record, dict) or not isinstance(record.get('meta'), dict)
            or record['meta'].get('origin') != 'inference'):
        raise ValueError('State DB requires a single inference record; inspect the state DB.')
    version = record['meta'].get('analysis_format')
    if version is not None and (type(version) is not int or not 1 <= version <= ANALYSIS_FORMAT):
        raise ValueError('Unsupported inference evidence format; update stash_ai.')
    if not current(record['meta']):
        return True
    try:
        validate_current(record)
    except UnorderedFrames:
        return True
    if not valid_result(record, kind):
        raise ValueError('Invalid current inference record; inspect the state DB.')
    return False


def migrate(db):
    retired = {}
    def remember(params, record):
        retired.setdefault(params[:2], set()).add((params[2], marker_segment_signature(record)))
    # Fetch keys only; one scene's evidence may contain many thousands of frames.
    for scope, scene, model, kind in db.execute('SELECT scope,scene,model,kind FROM results').fetchall():
        params = scope, scene, model, kind
        blob = db.execute('SELECT data FROM results WHERE scope=? AND scene=? AND model=? AND kind=?',
                          params).fetchone()[0]
        record = decode(blob)
        if not obsolete(record, kind):
            continue
        remember(params, record)
        db.execute('DELETE FROM results WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        db.execute('DELETE FROM result_index WHERE scope=? AND scene=? AND model=? AND kind=?', params)
        db.execute('INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)', params)
        for table in ('checkpoints', 'checkpoint_frames', 'checkpoint_analysis'):
            db.execute(f'DELETE FROM {table} WHERE scope=? AND scene=? AND model=?', params[:3])
    for params in db.execute('SELECT scope,scene,model,kind,signature FROM inference_archive').fetchall():
        blob = db.execute('SELECT data FROM inference_archive WHERE scope=? AND scene=? AND model=? '
                          'AND kind=? AND signature=?', params).fetchone()[0]
        record = decode(blob)
        old = obsolete(record, params[3])
        if marker_segment_signature(record) != params[4]:
            raise ValueError('Archived proof signature mismatch.')
        if old:
            remember(params, record)
            db.execute('DELETE FROM inference_archive WHERE scope=? AND scene=? AND model=? '
                       'AND kind=? AND signature=?', params)
    return retired

"""Repair identical aggregate duplicates and their pinned identities without models."""
from collections import defaultdict, Counter

from stash_ai.storage.codec import decode, encode
from stash_ai.evidence.identity import marker_segment_signature


def canonical(record):
    if not isinstance(record, dict) or record.get('mode') != 'scene':
        return None
    items = record.get('items')
    if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
        return None
    unique = {}
    for item in items:
        tag = item.get('tag')
        if not isinstance(tag, str):
            return None
        if tag in unique and unique[tag] != item:
            raise ValueError(f'Conflicting scene aggregates for {tag!r}; restore this scene/model evidence.')
        unique.setdefault(tag, item)
    return {**record, 'items': list(unique.values())} if len(unique) != len(items) else None


def rewrite(value, signatures, field=None, memo=None):
    """Preserve shared proof objects while rewriting only identity-bearing fields."""
    memo = {} if memo is None else memo
    if isinstance(value, (list, dict)) and id(value) in memo:
        return memo[id(value)]
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(rewrite(item, signatures, field, memo) for item in value)
        return result
    if not isinstance(value, dict):
        if isinstance(value, str) and field in ('evidence_signature', 'signature', 'frame_context',
                                               'scene_context', 'support_context'):
            return signatures.get(value, value)
        return value
    result = {}
    memo[id(value)] = result
    for key, child in value.items():
        changed_key = signatures.get(key, key) if field in ('scene_thresholds', 'marker_thresholds') else key
        child_field = field if field in ('frame_context', 'scene_context', 'support_context') else key
        result[changed_key] = rewrite(child, signatures, child_field, memo)
    return result


def migrate(db, *, metadata=False):
    """Decode each active record once for canonicalization and metadata repair."""
    replacements = defaultdict(dict)
    changes, summaries = [], []
    counts = Counter(results_checked=0, archives_checked=0, duplicates_repaired=0, metadata_repaired=0)
    from stash_ai.storage.metadata import FORMAT, decode as decode_metadata, encode as encode_metadata, summarize
    def metadata_row(scope, scene, source, kind, record, version, data, *, force=False):
        derived = encode_metadata(summarize(record))
        if version == FORMAT and not force:
            try:
                if encode_metadata(decode_metadata(data)) == derived:
                    return None
            except (TypeError, ValueError):
                pass
        counts['metadata_repaired'] += 1
        return scope, scene, source, kind, FORMAT, derived
    for table in ('results', 'inference_archive'):
        if table == 'results' and metadata:
            query = ('SELECT r.scope,r.scene,r.model,r.data,r.kind,m.format,m.data '
                     'FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)')
        else:
            extra = ',signature' if table == 'inference_archive' else ''
            query = f"SELECT scope,scene,model,data{extra} FROM {table} WHERE kind='scene'"
        for scope, scene, source, blob, *extra in db.execute(query):
            counts['results_checked' if table == 'results' else 'archives_checked'] += 1
            kind = extra[0] if table == 'results' and metadata else 'scene'
            record = decode(blob)
            updated = canonical(record) if kind == 'scene' else None
            if table == 'results' and metadata:
                row = metadata_row(scope, scene, source, kind, updated or record, *extra[1:],
                                   force=updated is not None)
                if row is not None:
                    if updated is not None:
                        summaries.append(row)  # Raw UPDATE invalidates the old metadata.
                    else:
                        db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)', row)
            if updated is None:
                continue
            from stash_ai.evidence.formats import validate_current
            validate_current(updated)
            before = marker_segment_signature(record)
            after = marker_segment_signature(updated)
            replacements[scope, scene][before] = after
            counts['duplicates_repaired'] += 1
            old = extra if table == 'inference_archive' else []
            changes.append((table, scope, scene, source, old, after, encode(updated)))
    for table, scope, scene, source, old, after, blob in changes:
        if table == 'results':
            db.execute("UPDATE results SET data=? WHERE scope=? AND scene=? AND model=? AND kind='scene'",
                       (blob, scope, scene, source))
        else:
            db.execute('INSERT OR IGNORE INTO inference_archive VALUES (?,?,?,?,?,?)',
                       (scope, scene, source, 'scene', after, blob))
            db.execute("DELETE FROM inference_archive WHERE scope=? AND scene=? AND model=? AND kind='scene' AND signature=?",
                       (scope, scene, source, old[0]))
    for row in summaries:
        db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)', row)
    from stash_ai.storage.snapshots import decode as decode_snapshot, encode as encode_snapshot
    for (scope, scene), signatures in replacements.items():
        for kind, published, pending in db.execute(
                'SELECT kind,published,pending FROM snapshots WHERE scope=? AND scene=?', (scope, scene)).fetchall():
            values = [encode_snapshot(rewrite(decode_snapshot(blob), signatures)) if blob is not None else None
                      for blob in (published, pending)]
            db.execute('UPDATE snapshots SET published=?,pending=? WHERE scope=? AND scene=? AND kind=?',
                       (*values, scope, scene, kind))
        for kind, operation, blob in db.execute(
                'SELECT kind,operation,data FROM recovery_publications WHERE scope=? AND scene=?', (scope, scene)).fetchall():
            db.execute('UPDATE recovery_publications SET data=? WHERE scope=? AND scene=? AND kind=? AND operation=?',
                       (encode_snapshot(rewrite(decode_snapshot(blob), signatures)), scope, scene, kind, operation))

    return dict(counts)

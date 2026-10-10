"""Repair identical aggregate duplicates and their pinned identities without models."""
from collections import Counter
from contextlib import closing
import sqlite3

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


class _Signatures:
    """Indexed, disk-backed replacement lookup; no per-scene history in RAM."""
    def __init__(self, spool, scope, scene):
        self.spool, self.scope, self.scene = spool, scope, scene

    def get(self, before, default=None):
        row = self.spool.execute(
            'SELECT after FROM signatures WHERE scope=? AND scene=? AND before=?',
            (self.scope, self.scene, before)).fetchone()
        return row[0] if row else default


def migrate(db, *, metadata=False, progress=None):
    """Spool rewrites to disk; the caller owns the single atomic transaction.

    Scanning finishes before modifying archive keys, so newly inserted canonical
    proofs cannot enter the scan again. SQLite's bounded page cache holds the
    spool, while Python retains only the current evidence/snapshot record.
    """
    with closing(sqlite3.connect('')) as spool:
        spool.execute('PRAGMA cache_size=-2048')
        spool.executescript("""
            CREATE TABLE changes (target TEXT, scope TEXT, scene TEXT, model TEXT,
                                  old TEXT, signature TEXT, data BLOB, summary TEXT);
            CREATE TABLE signatures (scope TEXT, scene TEXT, before TEXT, after TEXT,
                                     PRIMARY KEY(scope,scene,before));
        """)
        return _migrate(db, spool, metadata=metadata, progress=progress)


def _migrate(db, spool, *, metadata, progress):
    from stash_ai.storage.metadata import FORMAT, decode as decode_metadata, encode as encode_metadata, summarize
    counts = Counter(results_checked=0, archives_checked=0, duplicates_repaired=0, metadata_repaired=0)
    def notify(phase, completed, total):
        if progress is not None:
            progress(phase, completed, total)
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
        all_results = table == 'results' and metadata
        if all_results:
            query = ('SELECT r.scope,r.scene,r.model,r.data,r.kind,m.format,m.data '
                     'FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)')
        else:
            extra = ',signature' if table == 'inference_archive' else ''
            query = f"SELECT scope,scene,model,data{extra} FROM {table} WHERE kind='scene'"
        total = db.execute(f'SELECT COUNT(*) FROM {table}' + ('' if all_results else " WHERE kind='scene'")).fetchone()[0]
        notify(table, 0, total)
        for scope, scene, source, blob, *extra in db.execute(query):
            counter = 'results_checked' if table == 'results' else 'archives_checked'
            counts[counter] += 1
            kind = extra[0] if all_results else 'scene'
            record = decode(blob)
            updated = canonical(record) if kind == 'scene' else None
            summary = None
            if all_results:
                row = metadata_row(scope, scene, source, kind, updated or record, *extra[1:],
                                   force=updated is not None)
                if row is not None:
                    if updated is not None:
                        summary = row[-1]  # Raw UPDATE invalidates the old metadata.
                    else:
                        db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)', row)
            if updated is not None:
                from stash_ai.evidence.formats import validate_current
                validate_current(updated)
                before = marker_segment_signature(record)
                after = marker_segment_signature(updated)
                spool.execute('INSERT OR REPLACE INTO signatures VALUES (?,?,?,?)', (scope, scene, before, after))
                counts['duplicates_repaired'] += 1
                old = extra[0] if table == 'inference_archive' else None
                spool.execute('INSERT INTO changes VALUES (?,?,?,?,?,?,?,?)',
                              (table, scope, scene, source, old, after, encode(updated), summary))
            notify(table, counts[counter], total)
    total = counts['duplicates_repaired']
    notify('canonical writes', 0, total)
    for completed, (table, scope, scene, source, old, after, blob, summary) in enumerate(
            spool.execute('SELECT * FROM changes'), 1):
        if table == 'results':
            db.execute("UPDATE results SET data=? WHERE scope=? AND scene=? AND model=? AND kind='scene'",
                       (blob, scope, scene, source))
            if summary is not None:
                db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                           (scope, scene, source, 'scene', FORMAT, summary))
        else:
            db.execute('INSERT OR IGNORE INTO inference_archive VALUES (?,?,?,?,?,?)',
                       (scope, scene, source, 'scene', after, blob))
            if old != after:
                db.execute("DELETE FROM inference_archive WHERE scope=? AND scene=? AND model=? AND kind='scene' AND signature=?",
                           (scope, scene, source, old))
        notify('canonical writes', completed, total)
    from stash_ai.storage.snapshots import decode as decode_snapshot, encode as encode_snapshot
    for table in ('snapshots', 'recovery_publications'):
        columns = 'kind,published,pending' if table == 'snapshots' else 'kind,operation,data'
        # Only rewrite publications whose scene actually had an identity change.
        # The spool has a separate connection. Keep SQL joins out of the state DB.
        scenes = spool.execute('SELECT DISTINCT scope,scene FROM signatures')
        total = sum(db.execute(f'SELECT COUNT(*) FROM {table} WHERE scope=? AND scene=?', row).fetchone()[0]
                    for row in scenes)
        notify(table, 0, total)
        completed = 0
        for scope, scene in spool.execute('SELECT DISTINCT scope,scene FROM signatures'):
            signatures = _Signatures(spool, scope, scene)
            for kind, first, second in db.execute(
                    f'SELECT {columns} FROM {table} WHERE scope=? AND scene=?', (scope, scene)):
                if table == 'snapshots':
                    values = [encode_snapshot(rewrite(decode_snapshot(blob), signatures)) if blob is not None else None
                              for blob in (first, second)]
                    db.execute('UPDATE snapshots SET published=?,pending=? WHERE scope=? AND scene=? AND kind=?',
                               (*values, scope, scene, kind))
                else:
                    db.execute('UPDATE recovery_publications SET data=? WHERE scope=? AND scene=? AND kind=? AND operation=?',
                               (encode_snapshot(rewrite(decode_snapshot(second), signatures)), scope, scene, kind, first))
                completed += 1
                notify(table, completed, total)
    return dict(counts)

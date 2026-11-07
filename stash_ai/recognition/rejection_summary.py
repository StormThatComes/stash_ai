"""Separate face rejection, independent body rejection and unavailable projections."""
import json
import sqlite3

from .body_observations import contract as body_contract, enabled as bodies_enabled
from .database import encoded
from .dependencies import cache_writable, snapshot, version
from .person_signal import contract

PREFIX = 'recognition-rejections-v1:'


def counts(db, scan, kind, policy):
    if kind == 'face':
        return dict(db.execute('SELECT rejected,COUNT(*) FROM observations WHERE scan=? '
                               'AND rejected IS NOT NULL GROUP BY rejected', (scan,)))
    if kind == 'body':
        return dict(db.execute('SELECT rejected,COUNT(*) FROM body_observations WHERE scan=? '
                               'AND contract=? AND rejected IS NOT NULL GROUP BY rejected', (scan, policy)))
    return dict(db.execute("SELECT COALESCE(json_extract(p.metadata,'$.identity_rejection'),'unavailable'),COUNT(*) "
        "FROM person_evidence p JOIN observations o ON o.id=p.observation WHERE o.scan=? "
        "AND p.contract=? AND p.status='unavailable' GROUP BY 1", (scan, policy)))


def summary(db, scan, config):
    policies = dict(face='', body=body_contract(config) if bodies_enabled(config) else None,
                    person_evidence_unavailable=contract(config) if config.data['sapiensid']['enabled'] else None)
    if db.execute('PRAGMA user_version').fetchone()[0] < 13:
        return {kind: counts(db, scan, kind, policy) if policy is not None else {}
                for kind, policy in policies.items()}
    from .dependency_schema import validate_triggers
    from .local_dependencies import scene_key
    validate_triggers(db, required=True)
    key = PREFIX + scan
    with snapshot(db):
        saved = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        try:
            cached = json.loads(saved[0]) if saved else {}
        except (ValueError, TypeError):
            cached = {}
        result, fresh = {}, {}
        for kind, policy in policies.items():
            if policy is None:
                result[kind] = {}
                continue
            dependency = {'face': 'raw-observations', 'person_evidence_unavailable': 'raw-person', 'body': 'raw-body'}[kind]
            stamp = [policy, version(db, dependency, scan)]
            if kind == 'body':
                # The scene clock includes the old address of a cross-scan SQLite REPLACE.
                stamp.append(version(db, 'scene', scene_key(db, scan)))
            entry = cached.get(kind) if isinstance(cached, dict) else None
            values = entry.get('values') if isinstance(entry, dict) and entry.get('stamp') == stamp else None
            if not isinstance(values, dict) or any(type(n) is not int or n < 0 for n in values.values()):
                values = counts(db, scan, kind, policy)
            result[kind] = values
            fresh[kind] = dict(stamp=stamp, values=values)
        if fresh != cached and cache_writable(db):
            try:
                db.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)', (key, encoded(fresh)))
            except sqlite3.OperationalError as error:
                # Concurrent writers or read-only reports can always use the measured result.
                code = (getattr(error, 'sqlite_errorcode', 0) or 0) & 255
                if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_READONLY, sqlite3.SQLITE_LOCKED):
                    raise
        return result

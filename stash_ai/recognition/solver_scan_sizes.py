"""Per-scan admission counts signed by the face, projection and body revisions."""
import json
from .dependencies import snapshot
from .database import encoded

PREFIX = 'solver-input-size-v1:'


def measure(db, scan):
    totals = [0, 0]
    # Crop-retention annotations do not participate in replay. Their updates
    # intentionally leave raw-person/raw-body revisions unchanged.
    def replay_size(column):
        return f"length(CASE WHEN json_valid({column}) THEN json_remove({column},'$.crop_status','$.crop_attempt_limit') ELSE {column} END)"
    for query in (
        'SELECT SUM(coalesce(length(embedding),0)+length(quality)+length(geometry)),COUNT(*) FROM observations WHERE scan=?',
        f"SELECT SUM(coalesce(length(embedding),0)+{replay_size('quality')}+length(geometry)),COUNT(*) FROM body_observations WHERE scan=?",
        f"SELECT SUM(coalesce(length(embedding),0)+{replay_size('metadata')}),COUNT(*) FROM person_evidence "
        'WHERE observation IN (SELECT id FROM observations WHERE scan=?)',
    ):
        size, count = db.execute(query, (scan,)).fetchone()
        totals[0] += size or 0
        totals[1] += count
    return totals


def records(db, where, args):
    return db.execute("""
        SELECT s.id,coalesce(f.version,0),coalesce(p.version,0),coalesce(b.version,0),m.value
        FROM scans s
        LEFT JOIN evidence_versions f ON f.kind='raw-observations' AND f.key=s.id
        LEFT JOIN evidence_versions p ON p.kind='raw-person' AND p.key=s.id
        LEFT JOIN evidence_versions b ON b.kind='raw-body' AND b.key=s.id
        LEFT JOIN metadata m ON m.key=?||s.id WHERE """ + where, (PREFIX, *args))


def counts(db, row):
    scan, *versions, value = row
    if value:
        try:
            stamp, amounts = json.loads(value)
            if stamp == versions and len(amounts) == 2 and all(type(n) is int and n >= 0 for n in amounts):
                return amounts, True
        except (TypeError, ValueError):
            pass
    return measure(db, scan), False


def retain(db, scan):
    if db.execute('PRAGMA user_version').fetchone()[0] < 10:
        return
    with snapshot(db):
        row = records(db, 's.id=?', (scan,)).fetchone()
        if row is None:
            return
        amounts, hit = counts(db, row)
        if not hit:
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                       (PREFIX + scan, encoded([list(row[1:4]), amounts])))


def scoped(db, scope, namespace):
    with snapshot(db):
        total = rows = 0
        for row in records(db, 's.scope=? AND s.namespace=?', (scope, namespace)):
            (size, count), _ = counts(db, row)
            total += size
            rows += count
        return total, rows

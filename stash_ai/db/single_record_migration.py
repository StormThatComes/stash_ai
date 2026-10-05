"""Convert the final active history to one record and one numeric threshold."""
from types import SimpleNamespace
from stash_ai.storage.codec import encode
from stash_ai.storage.metadata import FORMAT, summarize, encode as encode_metadata
from stash_ai.storage.progress import index_result
from stash_ai.evidence.formats import validate_current
from stash_ai.storage.thresholds import separate
from .result_history import history, overlay, threshold_column


def migrate(db):
    column = threshold_column(db)
    # Keep only keys in memory; evidence can contain many thousands of frames.
    keys = db.execute('SELECT scope,scene,model,kind FROM results').fetchall()
    # Orphan/invalid policies must not silently disappear during table replacement.
    if db.execute('SELECT 1 FROM result_thresholds AS t WHERE NOT EXISTS '
                  '(SELECT 1 FROM results AS r WHERE r.scope=t.scope AND r.scene=t.scene '
                  'AND r.model=t.model AND r.kind=t.kind) LIMIT 1').fetchone():
        raise ValueError('Orphan result thresholds; run --db-check.')
    db.execute('CREATE TABLE result_thresholds_update (scope TEXT NOT NULL, scene TEXT NOT NULL, '
               'model TEXT NOT NULL, kind TEXT NOT NULL, threshold REAL NOT NULL '
               "CHECK(typeof(threshold) IN ('real','integer') AND threshold BETWEEN 0 AND 1), "
               'PRIMARY KEY(scope,scene,model,kind))')
    state = SimpleNamespace(connection=db, scope='')
    for scope, scene, model, kind in keys:
        params = scope, scene, model, kind
        blob, policy = db.execute(
            f'SELECT r.data,t.{column} FROM results AS r LEFT JOIN result_thresholds AS t '
            'USING(scope,scene,model,kind) WHERE r.scope=? AND r.scene=? AND r.model=? AND r.kind=?',
            params).fetchone()
        previous = overlay(history(blob), kind, policy)
        active = [record for record in previous if record['meta'].get('origin') == 'inference']
        if not active:
            db.execute('DELETE FROM results WHERE scope=? AND scene=? AND model=? AND kind=?', params)
            db.execute('DELETE FROM result_index WHERE scope=? AND scene=? AND model=? AND kind=?', params)
            db.execute('INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)', params)
            continue
        if len(active) != 1:
            raise ValueError('Multiple active inference records after history migration.')
        record = active[0]
        validate_current(record)
        raw, threshold = separate(record, kind)
        db.execute('UPDATE results SET data=? WHERE scope=? AND scene=? AND model=? AND kind=?', (encode(raw), *params))
        if threshold is not None:
            db.execute('INSERT INTO result_thresholds_update VALUES (?,?,?,?,?)', (*params, threshold))
        db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                   (*params, FORMAT, encode_metadata(summarize(raw))))
        index_result(state, scene, model, kind, record, scope=scope)
    db.execute('DROP TRIGGER IF EXISTS results_delete_threshold')
    db.execute('DROP TABLE result_thresholds')
    db.execute('ALTER TABLE result_thresholds_update RENAME TO result_thresholds')
    install_triggers(db)


def prepare_history(db, version):
    """Offline readers consume old history even when repairing a partial upgrade."""
    if version < 10 or threshold_column(db) != 'threshold':
        return
    from .result_history import encode_thresholds
    rows = db.execute('SELECT scope,scene,model,kind,threshold FROM result_thresholds').fetchall()
    db.execute('CREATE TABLE result_thresholds_history (scope TEXT NOT NULL, scene TEXT NOT NULL, '
               'model TEXT NOT NULL, kind TEXT NOT NULL, thresholds TEXT NOT NULL CHECK(json_valid(thresholds)), '
               'PRIMARY KEY(scope,scene,model,kind))')
    for *params, threshold in rows:
        db.execute('INSERT INTO result_thresholds_history VALUES (?,?,?,?,?)', (*params, encode_thresholds([threshold])))
    db.execute('DROP TRIGGER IF EXISTS results_delete_threshold')
    db.execute('DROP TABLE result_thresholds')
    db.execute('ALTER TABLE result_thresholds_history RENAME TO result_thresholds')
    install_triggers(db)


def install_triggers(db):
    db.execute('CREATE TRIGGER results_delete_threshold AFTER DELETE ON results BEGIN '
               'DELETE FROM result_thresholds WHERE scope=OLD.scope AND scene=OLD.scene '
               'AND model=OLD.model AND kind=OLD.kind; END')
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='merge_checks'").fetchone():
        for event, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
            db.execute(f'CREATE TRIGGER perf_result_thresholds_{event.lower()} AFTER {event} '
                       f'ON result_thresholds BEGIN DELETE FROM merge_checks '
                       f'WHERE scope={row}.scope AND scene={row}.scene; END')

"""Offline migrations. Ordinary state readers never import this module."""
from types import SimpleNamespace

from stash_ai.storage.codec import decode
from stash_ai.storage.metadata import FORMAT, decode as decode_metadata, encode, summarize
from stash_ai.storage.progress import index_result
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION, initialize_features


def canonicalize_results(database):
    """Move old inline policies to per-segment rows without changing proof identity."""
    from stash_ai.storage.codec import encode as pack
    from stash_ai.storage.thresholds import encode as encode_thresholds, THRESHOLD_KEYS
    db = database.connection
    old_policy = 'threshold' in {row[1] for row in db.execute('PRAGMA table_info(result_thresholds)')}
    column = 'threshold' if old_policy else 'thresholds'
    if db.execute('''SELECT 1 FROM result_thresholds AS t WHERE NOT EXISTS (
            SELECT 1 FROM results AS r WHERE r.scope=t.scope AND r.scene=t.scene
                AND r.model=t.model AND r.kind=t.kind) LIMIT 1''').fetchone():
        raise ValueError('Orphan result thresholds; inspect the database with --db-check.')
    db.execute('''CREATE TABLE result_thresholds_update (
        scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
        kind TEXT NOT NULL, thresholds TEXT NOT NULL CHECK(json_valid(thresholds)),
        PRIMARY KEY(scope,scene,model,kind))''')
    for scope, scene, model, kind, blob, saved in db.execute(
            f'SELECT r.scope,r.scene,r.model,r.kind,r.data,t.{column} FROM results AS r '
            'LEFT JOIN result_thresholds AS t USING(scope,scene,model,kind)'):
        key = (scope, scene, model, kind)
        segments = decode(blob)
        override, _ = THRESHOLD_KEYS[kind]
        values = ([saved] * len(segments) if old_policy and saved is not None else
                  [segment.get(override) for segment in segments])
        if not old_policy and saved is not None:
            from stash_ai.storage.thresholds import decode as decode_thresholds
            values = decode_thresholds(saved)
            if len(values) != len(segments):
                raise ValueError('Result thresholds do not match segment history.')
        changed = False
        for segment in segments:
            for field in ('scene_threshold_override', 'marker_threshold_override', 'refinement'):
                if field in segment:
                    del segment[field]
                    changed = True
        if changed:
            db.execute('UPDATE results SET data=? WHERE scope=? AND scene=? AND model=? AND kind=?',
                       (pack(segments), *key))
        if any(value is not None for value in values):
            db.execute('INSERT INTO result_thresholds_update VALUES (?,?,?,?,?)',
                       (*key, encode_thresholds(values)))
    db.execute('DROP TABLE result_thresholds')
    db.execute('DROP TRIGGER IF EXISTS results_delete_threshold')
    db.execute('ALTER TABLE result_thresholds_update RENAME TO result_thresholds')
    db.execute('''CREATE TRIGGER results_delete_threshold AFTER DELETE ON results BEGIN
        DELETE FROM result_thresholds WHERE scope=OLD.scope AND scene=OLD.scene
            AND model=OLD.model AND kind=OLD.kind; END''')


def migrate(database):
    db = database.connection
    state = SimpleNamespace(connection=db, scope='', preview_source=False)
    # Rename/swap and payload edits are atomic. Retry never loses historical
    # overrides or pending plans, even if later feature setup is interrupted.
    if database.version < 10:
        with db:
            db.execute('BEGIN')
            canonicalize_results(database)
    initialize_features(state)
    with db:
        if database.version < 13:
            from .snapshot_migration import migrate as migrate_snapshots
            migrate_snapshots(db)
        if database.version < 10:
            from stash_ai.storage.thresholds import overlay
            for scope, scene, model, kind, payload, thresholds in db.execute(
                    'SELECT r.scope,r.scene,r.model,r.kind,r.data,t.thresholds FROM results AS r '
                    'LEFT JOIN result_thresholds AS t USING(scope,scene,model,kind)'):
                index_result(state, scene, model, kind, overlay(decode(payload), kind, thresholds), scope=scope)
        db.execute('DROP TABLE IF EXISTS refinement_progress')
        for event in ('insert', 'update', 'delete'):
            db.execute(f'DROP TRIGGER IF EXISTS perf_legacy_imports_{event}')
        # Historical metadata is derived once here. Runtime repair remains
        # necessary for corruption and invalidation by direct SQLite writers.
        for scope, scene, model, kind, payload, version, data in db.execute(
                'SELECT r.scope,r.scene,r.model,r.kind,r.data,m.format,m.data '
                'FROM results AS r LEFT JOIN result_metadata AS m USING(scope,scene,model,kind)'):
            if version == FORMAT:
                try:
                    decode_metadata(data)
                    continue
                except (TypeError, ValueError):
                    pass
            db.execute('INSERT OR REPLACE INTO result_metadata VALUES (?,?,?,?,?,?)',
                       (scope, scene, model, kind, FORMAT, encode(summarize(decode(payload)))))
        db.execute(f'PRAGMA user_version={LATEST_SCHEMA_VERSION}')


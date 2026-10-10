"""Transactionally invalidated publication checks, independent of Stash access."""
from stash_ai.storage.schema import MERGE_INVALIDATING_TABLES


def initialize(store):
    """Derived cache invalidated transactionally, including writes by older code."""
    db = store.connection
    db.executescript('''
        CREATE TABLE IF NOT EXISTS merge_checks (
            scope TEXT NOT NULL, scene TEXT NOT NULL, policy TEXT NOT NULL,
            fingerprint TEXT NOT NULL, PRIMARY KEY(scope, scene));
        CREATE INDEX IF NOT EXISTS snapshots_pending_scope
            ON snapshots(scope, scene, kind) WHERE pending IS NOT NULL;
    ''')
    if getattr(store, 'maintenance_migration', False):
        from stash_ai.storage.schema import repair_trigger_definitions
        repair_trigger_definitions(db, (
            name for name in __import__('stash_ai.storage.schema', fromlist=['_trigger_sql'])._trigger_sql()
            if name.startswith('perf_')))
    for table in MERGE_INVALIDATING_TABLES:
        for event, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
            db.execute(f'''CREATE TRIGGER IF NOT EXISTS perf_{table}_{event.lower()}
                AFTER {event} ON {table} BEGIN
                    DELETE FROM merge_checks WHERE scope={row}.scope AND scene={row}.scene;
                END''')
    for event, row in (('INSERT', 'NEW'), ('UPDATE', 'NEW'), ('DELETE', 'OLD')):
        db.execute(f'''CREATE TRIGGER IF NOT EXISTS perf_cleanup_{event.lower()}
            AFTER {event} ON cleanup_state BEGIN
                DELETE FROM merge_checks WHERE scope={row}.scope;
            END''')


def checked(store, scene, policy, fingerprint):
    return check_reason(store, scene, policy, fingerprint) == 'reused'


def check_reason(store, scene, policy, fingerprint):
    from stash_ai.storage.preview import check_source
    check_source(store)
    row = store.connection.execute('SELECT policy,fingerprint FROM merge_checks WHERE scope=? AND scene=?',
        (store.scope, str(scene))).fetchone()
    if row is None:
        return 'cache_miss'
    if row[0] != policy:
        return 'policy_changed'
    if row[1] != fingerprint:
        return 'stash_changed'
    return 'reused'


def record_check(store, scene, policy, fingerprint):
    """Caller owns the transaction, including publication promotion if needed."""
    store.connection.execute('INSERT INTO merge_checks VALUES (?,?,?,?) '
        'ON CONFLICT(scope,scene) DO UPDATE SET policy=excluded.policy,fingerprint=excluded.fingerprint',
        (store.scope, str(scene), policy, fingerprint))


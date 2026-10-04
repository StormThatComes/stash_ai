"""Transactionally invalidated publication checks, independent of Stash access."""


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
    # legacy_imports is only an anti-duplication ledger. Any newly imported
    # legacy evidence also writes results, which already invalidates merge_checks.
    # Drop triggers created by older versions so pruning dead ledger rows does not
    # force unrelated scenes through merge again.
    for event in ('insert', 'update', 'delete'):
        db.execute(f'DROP TRIGGER IF EXISTS perf_legacy_imports_{event}')
    for table in ('results', 'result_thresholds', 'snapshots'):
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
    db.execute('PRAGMA cache_size=-65536')


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

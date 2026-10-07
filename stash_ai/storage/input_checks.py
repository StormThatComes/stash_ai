"""Deferred input verification, independent from requests for fresh predictions."""

REASONS = frozenset(('missing_media', 'read_error', 'deleted'))


def initialize(store):
    store.connection.execute('''CREATE TABLE IF NOT EXISTS input_checks (
        scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
        reason TEXT NOT NULL CHECK(reason IN ('missing_media','read_error','deleted')),
        PRIMARY KEY(scope,scene,model))''')
    store.connection.commit()


def defer(store, scene, model, reason='missing_media'):
    if reason not in REASONS:
        raise ValueError('Unknown input verification reason.')
    with store.connection:
        store.connection.execute('INSERT INTO input_checks VALUES (?,?,?,?) '
            'ON CONFLICT(scope,scene,model) DO UPDATE SET reason=excluded.reason',
            (store.scope, str(scene), model, reason))


def verified(store, scene, model):
    from stash_ai.storage.preview import delete
    with store.connection:
        delete(store, 'input_checks', scene, model)
        store.connection.execute('DELETE FROM input_checks WHERE scope=? AND scene=? AND model=?',
                                 (store.scope, str(scene), model))


def pending(store, scene):
    return dict(store.connection.execute('SELECT model,reason FROM ' + store.table('input_checks') +
        ' WHERE scope=? AND scene=?', (store.scope, str(scene))))


def selected(store, scenes):
    from stash_ai.storage.selection import selected_rows
    checks = {}
    for scene, model, reason in selected_rows(store,
            'SELECT scene,model,reason FROM ' + store.table('input_checks') + ' WHERE scope=?',
            (store.scope,), scenes):
        checks.setdefault(scene, {})[model] = reason
    return checks


def removed_scenes(store):
    return {row[0] for row in store.connection.execute('SELECT DISTINCT scene FROM ' +
        store.table('input_checks') + " WHERE scope=? AND reason='deleted'", (store.scope,))}

"""Durable authorization and completeness barrier for rebuilding a fresh DB."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json

from stash_ai.storage.schema import KNOWN_MODELS, STATE_TABLES


def initialize(store):
    store.connection.executescript('''
        CREATE TABLE IF NOT EXISTS db_initializations (
            scope TEXT PRIMARY KEY, models TEXT NOT NULL, selection TEXT NOT NULL,
            scanned INTEGER NOT NULL CHECK(scanned IN (0,1)));
        CREATE TABLE IF NOT EXISTS db_init_scenes (
            scope TEXT NOT NULL, scene TEXT NOT NULL, owned_ids TEXT NOT NULL,
            completed INTEGER NOT NULL CHECK(completed IN (0,1)),
            PRIMARY KEY(scope,scene));
    ''')
    if store.preview_source:
        store.connection.execute(
            'INSERT INTO db_initializations SELECT * FROM cleanup_source.db_initializations WHERE scope=?',
            (store.scope,))
        store.connection.commit()


def string_list(value, *, nullable=False):
    values = json.loads(value)
    if nullable and values is None:
        return None
    if (not isinstance(values, list) or any(not isinstance(item, str) or not item.strip() for item in values)
            or len(set(values)) != len(values)):
        raise ValueError('Invalid DB initialization list.')
    return tuple(values)


@dataclass(frozen=True)
class InitializationRun:
    models: tuple[str, ...]
    selection: tuple[str, ...] | None
    scanned: bool

    @classmethod
    def decode(cls, models, selection, scanned):
        models = string_list(models)
        selection = string_list(selection, nullable=True)
        if (not models or not set(models) <= KNOWN_MODELS or type(scanned) is not int or scanned not in (0, 1)
                or selection == ()):
            raise ValueError('Invalid DB initialization scope/models.')
        return cls(tuple(sorted(models)), None if selection is None else tuple(sorted(selection)), bool(scanned))


def scene_owned_ids(tags):
    """Adopt leaf associations, excluding categories and reserved marker tags."""
    from stash_ai.evidence.labels import AI, raw_label
    result = set()
    for tag in tags:
        if not tag['name'].startswith(AI.prefix):
            continue
        try:
            raw_label(tag['name'][len(AI.prefix):])
        except ValueError:
            continue
        result.add(str(tag['id']))
    return result


class InitializationStore:
    """Own the init manifest; ordinary databases have no adoption permission."""

    def __init__(self, store):
        self.store = store
        row = store.connection.execute(
            'SELECT models,selection,scanned FROM db_initializations WHERE scope=?', (store.scope,)).fetchone()
        self.run = InitializationRun.decode(*row) if row else None
        self.scenes, self.owned = {}, {}
        from stash_ai.storage.input_checks import removed_scenes
        self.removed = removed_scenes(store)
        for scene, owned, completed in store.connection.execute(
                'SELECT scene,owned_ids,completed FROM ' + store.table('db_init_scenes') + ' WHERE scope=?',
                (store.scope,)):
            if (self.run is None or not isinstance(scene, str) or not scene.strip()
                    or type(completed) is not int or completed not in (0, 1)
                    or (self.run.selection is not None and scene not in self.run.selection)):
                raise ValueError('Invalid DB initialization scene.')
            self.scenes[scene] = bool(completed)
            identifiers = string_list(owned)
            if not completed:
                self.owned[scene] = set(identifiers)

    @property
    def active(self):
        return self.run is not None and (not self.run.scanned or not all(self.scenes.values()))

    def pending(self, scene):
        return str(scene) in self.scenes and not self.scenes[str(scene)]

    def configure(self, app, args, settings):
        requested = args.db_init
        if not requested and not self.active:
            return
        if not app.APPLY_SCENE_TAGS or not app.CREATE_MARKERS or args.restart or args.abandon_pending:
            raise RuntimeError('DB initialization requires both outputs; resume without partial-output, restart or abandonment modifiers.')
        if args.rethreshold_scenes or args.rethreshold_markers:
            raise RuntimeError('Finish DB initialization before rethresholding.')
        models = tuple(sorted(config.source.key for config in settings))
        selection = tuple(sorted(set(map(str, args.scene_id)))) if args.scene_id else None
        planned = InitializationRun.decode(json.dumps(models), json.dumps(selection), 0)
        if self.run is None:
            # Check the entire database, not just this Stash URL's scope.
            databases = ('main', 'cleanup_source') if self.store.preview_source else ('main',)
            if any(self.store.connection.execute(f'SELECT 1 FROM {database}.{table} LIMIT 1').fetchone()
                   for database in databases for table in STATE_TABLES):
                raise RuntimeError('--db-init requires a new or empty state DB; choose another --state-db.')
            with self.store.connection:
                self.store.connection.execute('INSERT INTO db_initializations VALUES (?,?,?,0)',
                    (self.store.scope, json.dumps(models), json.dumps(selection)))
            self.run = planned
        if models != self.run.models or selection != self.run.selection:
            raise RuntimeError('Resume DB initialization with the original --models and --scene-id selection; '
                               'a different initialization requires another --state-db.')
        # Initialization is always resumable, including a repeated --db-init.
        args.resume = True

    def enroll(self, scene):
        if self.run is None or self.run.scanned or str(scene['id']) in self.scenes:
            return
        scene_id = str(scene['id'])
        owned = scene_owned_ids(scene.get('tags', []))
        with self.store.connection:
            self.store.connection.execute('INSERT INTO db_init_scenes VALUES (?,?,?,0)',
                (self.store.scope, scene_id, json.dumps(sorted(owned))))
            for model in self.run.models:
                for kind in ('scene', 'markers'):
                    self.store.connection.execute('INSERT OR IGNORE INTO refresh_requests VALUES (?,?,?,?)',
                        (self.store.scope, scene_id, model, kind))
        self.scenes[scene_id], self.owned[scene_id] = False, owned

    def finish_scan(self, observed, *, errors):
        if not self.active:
            return
        missing = {scene for scene, done in self.scenes.items() if not done} - set(observed) - self.removed
        if self.run.selection is not None:
            missing.update(set(self.run.selection) - set(observed) - self.removed)
        if missing:
            raise RuntimeError('DB initialization scenes could not be read: ' + ', '.join(sorted(missing)[:12]))
        if not errors and not self.run.scanned:
            with self.store.connection:
                self.store.connection.execute('UPDATE db_initializations SET scanned=1 WHERE scope=?', (self.store.scope,))
            self.run = replace(self.run, scanned=True)

    def retire(self, scene):
        """Retire only a scene whose absence was confirmed by a successful API read."""
        if not self.active:
            return
        scene = str(scene)
        with self.store.connection:
            self.store.connection.execute('INSERT INTO db_init_scenes VALUES (?,?,?,1) '
                'ON CONFLICT(scope,scene) DO UPDATE SET completed=1',
                (self.store.scope, scene, '[]'))
            for model in self.run.models:
                self.store.connection.execute('INSERT INTO input_checks VALUES (?,?,?,?) '
                    'ON CONFLICT(scope,scene,model) DO UPDATE SET reason=excluded.reason',
                    (self.store.scope, scene, model, 'deleted'))
        self.scenes[scene] = True
        self.owned.pop(scene, None)
        self.removed.add(scene)

    def require_ready(self, scene):
        if not self.pending(scene):
            return
        ready = set(self.store.connection.execute(
            'SELECT i.model,i.kind FROM ' + self.store.table('result_index') + ' AS i '
            'JOIN ' + self.store.table('results') + ' AS r USING(scope,scene,model,kind) '
            'WHERE i.scope=? AND i.scene=? AND i.signature IS NOT NULL AND NOT EXISTS '
            '(SELECT 1 FROM ' + self.store.table('refresh_requests') + ' AS f '
            'WHERE f.scope=i.scope AND f.scene=i.scene AND f.model=i.model AND f.kind=i.kind)',
            (self.store.scope, str(scene))))
        missing = [(model, kind) for model in self.run.models for kind in ('scene', 'markers')
                   if (model, kind) not in ready]
        if missing:
            raise RuntimeError('DB initialization awaits complete inference for ' +
                               ', '.join(model + '/' + kind for model, kind in missing) +
                               '; existing scene outputs are retained. Resume without --merge-only.')

    def adopt(self, scene, tags):
        if not self.pending(scene):
            return set()
        self.require_ready(scene)
        scene = str(scene)
        owned = self.owned[scene] | scene_owned_ids(tags)
        # Persist ownership before any scene association can be overwritten.
        with self.store.connection:
            self.store.connection.execute('INSERT INTO db_init_scenes VALUES (?,?,?,0) '
                'ON CONFLICT(scope,scene) DO UPDATE SET owned_ids=excluded.owned_ids',
                (self.store.scope, scene, json.dumps(sorted(owned))))
        self.owned[scene] = owned
        return owned

    def finish_in_transaction(self, scene, kinds):
        if not self.pending(scene):
            return
        if set(kinds) != {'scene', 'markers'}:
            raise RuntimeError('DB initialization publishes both outputs together.')
        self.require_ready(scene)
        self.store.connection.execute('INSERT INTO db_init_scenes VALUES (?,?,?,1) '
            'ON CONFLICT(scope,scene) DO UPDATE SET completed=1',
            (self.store.scope, str(scene), json.dumps(sorted(self.owned[str(scene)]))))

    def published(self, scene):
        if self.pending(scene):
            self.scenes[str(scene)] = True
            self.owned.pop(str(scene), None)

    def report(self, report, stats):
        if self.run is None:
            return
        completed = sum(done for scene, done in self.scenes.items() if scene not in self.removed)
        pending = sum(not done for done in self.scenes.values())
        stats['db_init_completed_scenes'], stats['db_init_pending_scenes'] = completed, pending
        stats['db_init_removed_scenes'] = len(self.removed & self.scenes.keys())
        report.write(f'DB initialization: {completed:,} scene(s) rebuilt; {pending:,} pending; '
                     f'{stats["db_init_removed_scenes"]:,} removed; '
                     f'scan={"complete" if self.run.scanned else "incomplete"}.')

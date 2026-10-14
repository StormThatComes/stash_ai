"""Independent, short-transaction SQLite journal of command invocations."""
import json
from pathlib import Path
import sqlite3

APPLICATION_ID = 0x53544148
FORMAT = 3


def history_path(args):
    return (args.run_history_db or Path(str(args.state_db.expanduser().resolve()) + '.runs.sqlite3')).expanduser().resolve()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class HistoryStore:
    def __init__(self, path, *, readonly=False):
        self.path, self.connection = Path(path).expanduser().resolve(), None
        self.pending = []
        try:
            if readonly:
                self.connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=1)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.connection = sqlite3.connect(self.path, timeout=1)
            db = self.connection
            if not readonly:
                db.execute('BEGIN IMMEDIATE')
            app = db.execute('PRAGMA application_id').fetchone()[0]
            version = db.execute('PRAGMA user_version').fetchone()[0]
            populated = bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone())
            if (app not in (0, APPLICATION_ID) or (populated and app != APPLICATION_ID)
                    or version not in (0, 1, 2, FORMAT) or (readonly and (app != APPLICATION_ID or version not in (1, 2, FORMAT)))):
                raise ValueError('Not a supported run history DB; choose another --run-history-db.')
            self._validate_schema(complete=readonly and version == FORMAT, allow_empty=not readonly)
            if not readonly:
                db.execute('''CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, command TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
                    wall_seconds REAL, cpu_seconds REAL, configuration TEXT NOT NULL,
                    models TEXT NOT NULL DEFAULT '{}', counts TEXT NOT NULL DEFAULT '{}',
                    outcomes TEXT NOT NULL DEFAULT '{}', metrics TEXT NOT NULL DEFAULT '[]',
                    issues TEXT NOT NULL DEFAULT '{}')''')
                columns = {row[1] for row in db.execute('PRAGMA table_info(runs)')}
                for name, declaration in (('last_progress_at', 'TEXT'), ('progress_seq', 'INTEGER NOT NULL DEFAULT 0'),
                                          ('workload', "TEXT NOT NULL DEFAULT '{}'")):
                    if name not in columns:
                        db.execute(f'ALTER TABLE runs ADD COLUMN {name} {declaration}')
                db.execute('CREATE INDEX IF NOT EXISTS runs_scope_started ON runs(scope,started_at DESC,id)')
                db.execute('''CREATE TABLE IF NOT EXISTS reuse_decisions (
                    run_id TEXT NOT NULL, scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL,
                    action TEXT NOT NULL, status TEXT NOT NULL, reasons TEXT NOT NULL, details TEXT NOT NULL,
                    PRIMARY KEY(run_id,scene,model))''')
                db.execute('CREATE INDEX IF NOT EXISTS reuse_scope_scene ON reuse_decisions(scope,scene,model,run_id)')
                db.execute('''CREATE TABLE IF NOT EXISTS reuse_proofs (
                    scope TEXT NOT NULL, scene TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
                    signature TEXT NOT NULL, proof TEXT NOT NULL, PRIMARY KEY(scope,scene,model,kind))''')
                self._validate_schema(complete=True)
                db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                db.execute(f'PRAGMA user_version={FORMAT}')
                db.commit()
        except BaseException:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([self.connection.rollback, self.close] if self.connection is not None else [self.close])
            raise

    def _validate_schema(self, *, complete, allow_empty=False):
        """Known legacy subsets are readable; current format needs every feature."""
        db = self.connection
        expected = {
            'runs': {'id':'TEXT', 'scope':'TEXT', 'command':'TEXT', 'started_at':'TEXT',
                     'finished_at':'TEXT', 'status':'TEXT', 'wall_seconds':'REAL', 'cpu_seconds':'REAL',
                     'configuration':'TEXT', 'models':'TEXT', 'counts':'TEXT', 'outcomes':'TEXT',
                     'metrics':'TEXT', 'issues':'TEXT', 'last_progress_at':'TEXT',
                     'progress_seq':'INTEGER', 'workload':'TEXT'},
            'reuse_decisions': {'run_id':'TEXT', 'scope':'TEXT', 'scene':'TEXT', 'model':'TEXT',
                                'action':'TEXT', 'status':'TEXT', 'reasons':'TEXT', 'details':'TEXT'},
            'reuse_proofs': {'scope':'TEXT', 'scene':'TEXT', 'model':'TEXT', 'kind':'TEXT',
                             'signature':'TEXT', 'proof':'TEXT'},
        }
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables and allow_empty:
            return
        if tables - expected.keys() or 'runs' not in tables or complete and tables != expected.keys():
            raise ValueError('Incomplete/unsupported run history schema; reopen writable history to repair known missing features.')
        for table in tables:
            columns = {row[1]: row[2].upper() for row in db.execute(f'PRAGMA table_info({table})')}
            required = set(expected[table])
            if table == 'runs' and not complete:
                required -= {'last_progress_at', 'progress_seq', 'workload'}
            if not required <= columns.keys() or any(expected[table].get(name) != kind for name,kind in columns.items()):
                raise ValueError(f'Unsupported run history column shape: {table}; restore a consistent backup.')
        for table in tables:
            nullable = {'id','finished_at','wall_seconds','cpu_seconds','last_progress_at'} if table=='runs' else set()
            if any(bool(row[3]) != (row[1] not in nullable) for row in db.execute(f'PRAGMA table_info({table})')):
                raise ValueError(f'Unsupported run history nullability: {table}; restore a consistent backup.')
        indexes = {'runs_scope_started': ('runs',['scope','started_at','id']),
                   'reuse_scope_scene': ('reuse_decisions',['scope','scene','model','run_id'])}
        for name,(table,columns) in indexes.items():
            present = db.execute("SELECT tbl_name FROM sqlite_master WHERE type='index' AND name=?", (name,)).fetchone()
            if (present is None and complete or present is not None and
                    (present[0] != table or [row[2] for row in db.execute(f'PRAGMA index_info({name})')] != columns)):
                raise ValueError(f'Incomplete/unsupported run history index shape: {name}.')
        keys = {'runs':['id'], 'reuse_decisions':['run_id','scene','model'],
                'reuse_proofs':['scope','scene','model','kind']}
        for table in tables:
            primary = sorted((row[5],row[1]) for row in db.execute(f'PRAGMA table_info({table})') if row[5])
            if [name for _,name in primary] != keys[table]:
                raise ValueError(f'Unsupported run history primary key: {table}; restore a consistent backup.')

    def start(self, identifier, scope, command, started, configuration):
        with self.connection:
            self.connection.execute('INSERT INTO runs(id,scope,command,started_at,status,configuration) VALUES (?,?,?,?,?,?)',
                                    (identifier, scope, command, started, 'running', encode(configuration)))

    def finish(self, identifier, finished, status, wall, cpu, *, models, counts, outcomes, metrics, issues, workload=None):
        self.flush()
        with self.connection:
            self.connection.execute('''UPDATE runs SET finished_at=?,status=?,wall_seconds=?,cpu_seconds=?,
                models=?,counts=?,outcomes=?,metrics=?,issues=?,last_progress_at=?,progress_seq=progress_seq+1,
                workload=COALESCE(?,workload) WHERE id=?''',
                (finished, status, wall, cpu, encode(models), encode(counts), encode(outcomes),
                 encode(metrics), encode(issues), finished, encode(workload) if workload is not None else None, identifier))

    def progress(self, identifier, updated, wall, cpu, *, models, counts, outcomes, metrics, issues, workload):
        self.flush()
        with self.connection:
            self.connection.execute('''UPDATE runs SET last_progress_at=?,progress_seq=progress_seq+1,
                wall_seconds=?,cpu_seconds=?,models=?,counts=?,outcomes=?,metrics=?,issues=?,workload=?
                WHERE id=? AND status='running' AND finished_at IS NULL''',
                (updated, wall, cpu, encode(models), encode(counts), encode(outcomes), encode(metrics),
                 encode(issues), encode(workload), identifier))

    def rows(self, scope, *, limit=50, command=None, status=None, model=None):
        filters, params = ['scope=?'], [scope]
        for name, value in (('command', command), ('status', status)):
            if value:
                filters.append(name + '=?')
                params.append(value)
        if model:
            filters.append('CASE WHEN json_valid(models) THEN json_type(models,?) END IS NOT NULL')
            params.append('$.' + model)
        params.append(limit)
        cursor = self.connection.execute('SELECT * FROM runs WHERE ' + ' AND '.join(filters) +
                                         ' ORDER BY started_at DESC,id DESC LIMIT ?', params)
        names = [column[0] for column in cursor.description]
        for values in cursor:
            row = dict(zip(names, values))
            invalid = []
            for name in ('configuration', 'models', 'counts', 'outcomes', 'metrics', 'issues', 'workload'):
                if name == 'workload' and name not in row:
                    row[name] = {}
                    continue
                try:
                    row[name] = json.loads(row[name])
                except (TypeError, ValueError):
                    row[name] = None
                    invalid.append(name)
            if invalid:
                row['invalid_json_fields'] = invalid
            row.setdefault('last_progress_at', None)
            row.setdefault('progress_seq', 0)
            row['partial'] = row['finished_at'] is None
            yield row

    def close(self):
        self.pending.clear()
        if self.connection is not None:
            connection, self.connection = self.connection, None
            connection.close()

    def record_decision(self, run_id, scope, scene, model, action, status, reasons, details):
        self.queue('decision', (run_id, scope, str(scene), model, action, status, encode(reasons), encode(details)))

    def put_proof(self, scope, scene, model, kind, signature, proof):
        self.queue('proof', (scope, str(scene), model, kind, signature, encode(proof)))

    def queue(self, operation, values):
        self.pending.append((operation, values))
        if len(self.pending) >= 64:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        with self.connection:
            for operation, values in self.pending:
                if operation == 'decision':
                    self.connection.execute('INSERT INTO reuse_decisions VALUES (?,?,?,?,?,?,?,?) '
                        'ON CONFLICT(run_id,scene,model) DO UPDATE SET action=excluded.action,status=excluded.status,'
                        'reasons=excluded.reasons,details=excluded.details', values)
                else:
                    self.connection.execute('INSERT INTO reuse_proofs VALUES (?,?,?,?,?,?) '
                        'ON CONFLICT(scope,scene,model,kind) DO UPDATE SET signature=excluded.signature,proof=excluded.proof '
                        'WHERE reuse_proofs.signature<>excluded.signature', values)
        self.pending.clear()

    def get_proof(self, scope, scene, model, kind, signature):
        for operation, values in reversed(self.pending):
            if operation == 'proof' and values[:4] == (scope, str(scene), model, kind):
                return json.loads(values[5]) if values[4] == signature else None
        row = self.connection.execute('SELECT proof FROM reuse_proofs '
            'WHERE scope=? AND scene=? AND model=? AND kind=? AND signature=?',
            (scope, str(scene), model, kind, signature)).fetchone()
        return json.loads(row[0]) if row else None

    def decisions(self, scope, *, limit=1, scenes=None, models=None, command=None, status=None):
        if not self.connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='reuse_decisions'").fetchone():
            return []
        conditions, params = ['r.scope=?'], [scope]
        for name, value in (('command', command), ('status', status)):
            if value:
                conditions.append('r.' + name + '=?')
                params.append(value)
        decisions, decision_params = [], []
        for field, values in (('scene', scenes), ('model', models)):
            if values:
                decisions.append('d.' + field + ' IN (' + ','.join('?' for _ in values) + ')')
                decision_params.extend(values)
        suffix = (' AND ' + ' AND '.join(decisions)) if decisions else ''
        selected = ('SELECT r.id FROM runs AS r WHERE ' + ' AND '.join(conditions) +
            ' AND EXISTS (SELECT 1 FROM reuse_decisions AS d WHERE d.run_id=r.id' + suffix +
            ') ORDER BY r.started_at DESC,r.id DESC LIMIT ?')
        cursor = self.connection.execute('WITH selected AS (' + selected + ') '
            'SELECT d.run_id,d.scene,d.model,d.action,d.status,d.reasons,d.details,'
            'r.started_at,r.status AS run_status FROM reuse_decisions AS d JOIN runs AS r ON r.id=d.run_id '
            'WHERE r.id IN (SELECT id FROM selected)' + suffix + ' ORDER BY r.started_at DESC,r.id DESC,d.scene,d.model',
            [*params, *decision_params, limit, *decision_params])
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, (*row[:5], json.loads(row[5]), json.loads(row[6]), *row[7:]))) for row in cursor]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            if args[0] is None:
                self.flush()
        finally:
            self.close()

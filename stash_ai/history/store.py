"""Independent, short-transaction SQLite journal of command invocations."""
import json
from pathlib import Path
import sqlite3

APPLICATION_ID = 0x53544148
FORMAT = 1


def history_path(args):
    return (args.run_history_db or Path(str(args.state_db.expanduser().resolve()) + '.runs.sqlite3')).expanduser().resolve()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


class HistoryStore:
    def __init__(self, path, *, readonly=False):
        self.path, self.connection = Path(path).expanduser().resolve(), None
        try:
            if readonly:
                self.connection = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=1)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.connection = sqlite3.connect(self.path, timeout=1)
            db = self.connection
            app = db.execute('PRAGMA application_id').fetchone()[0]
            version = db.execute('PRAGMA user_version').fetchone()[0]
            populated = bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone())
            if (app not in (0, APPLICATION_ID) or (populated and app != APPLICATION_ID)
                    or version not in (0, FORMAT) or (readonly and (app != APPLICATION_ID or version != FORMAT))):
                raise ValueError('Not a supported run history DB; choose another --run-history-db.')
            if not readonly:
                with db:
                    db.execute(f'PRAGMA application_id={APPLICATION_ID}')
                    db.execute(f'PRAGMA user_version={FORMAT}')
                    db.execute('''CREATE TABLE IF NOT EXISTS runs (
                        id TEXT PRIMARY KEY, scope TEXT NOT NULL, command TEXT NOT NULL,
                        started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
                        wall_seconds REAL, cpu_seconds REAL, configuration TEXT NOT NULL,
                        models TEXT NOT NULL DEFAULT '{}', counts TEXT NOT NULL DEFAULT '{}',
                        outcomes TEXT NOT NULL DEFAULT '{}', metrics TEXT NOT NULL DEFAULT '[]',
                        issues TEXT NOT NULL DEFAULT '{}')''')
                    db.execute('CREATE INDEX IF NOT EXISTS runs_scope_started ON runs(scope,started_at DESC,id)')
        except BaseException:
            self.close()
            raise

    def start(self, identifier, scope, command, started, configuration):
        with self.connection:
            self.connection.execute('INSERT INTO runs(id,scope,command,started_at,status,configuration) VALUES (?,?,?,?,?,?)',
                                    (identifier, scope, command, started, 'running', encode(configuration)))

    def finish(self, identifier, finished, status, wall, cpu, *, models, counts, outcomes, metrics, issues):
        with self.connection:
            self.connection.execute('''UPDATE runs SET finished_at=?,status=?,wall_seconds=?,cpu_seconds=?,
                models=?,counts=?,outcomes=?,metrics=?,issues=? WHERE id=?''',
                (finished, status, wall, cpu, encode(models), encode(counts), encode(outcomes),
                 encode(metrics), encode(issues), identifier))

    def rows(self, scope, *, limit=50, command=None, status=None, model=None):
        filters, params = ['scope=?'], [scope]
        for name, value in (('command', command), ('status', status)):
            if value:
                filters.append(name + '=?')
                params.append(value)
        if model:
            filters.append('json_type(models,?) IS NOT NULL')
            params.append('$.' + model)
        params.append(limit)
        cursor = self.connection.execute('SELECT * FROM runs WHERE ' + ' AND '.join(filters) +
                                         ' ORDER BY started_at DESC,id DESC LIMIT ?', params)
        names = [column[0] for column in cursor.description]
        for values in cursor:
            row = dict(zip(names, values))
            for name in ('configuration', 'models', 'counts', 'outcomes', 'metrics', 'issues'):
                row[name] = json.loads(row[name])
            yield row

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

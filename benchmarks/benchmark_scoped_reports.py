"""Selected cache report and rethreshold workflows over a larger local scope."""
import argparse
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scenes', type=int, default=9000)
    parser.add_argument('--selected', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.scenes, args.selected, args.repeats) < 1 or args.selected > args.scenes:
        parser.error('positive counts required, with selected <= scenes')
    sys.path.insert(0, str(args.project.resolve()))
    from benchmarks.fixtures import record
    from stash_ai.models import ModelSettings, SOURCES
    from stash_ai.reports import cache
    from stash_ai.rethreshold.execution import run as rethreshold
    from stash_ai.runtime.report import RunReport
    from stash_ai.runtime.settings import RuntimeSettings
    from stash_ai.storage.store import StateStore

    def evidence_hash(store):
        h = hashlib.sha256()
        for row in store.connection.execute('SELECT data FROM ' + store.table('results') +
                                            ' ORDER BY scope,scene,model,kind'):
            h.update(row[0])
        return h.hexdigest()

    settings = [ModelSettings(source, Path('.'), scene_threshold=.4, marker_threshold=.4,
                              batch_size=1) for source in SOURCES.values()]
    targets = [ModelSettings(source, Path('.'), scene_threshold=.55, marker_threshold=.55,
                             batch_size=1) for source in SOURCES.values()]
    app = RuntimeSettings()
    chosen = [str(i) for i in range(args.selected)]
    trials = {name: [] for name in ('cache_preview', 'rethreshold_writer', 'rethreshold_preview')}
    with tempfile.TemporaryDirectory(prefix='stash-scoped-report-') as directory:
        path = Path(directory) / 'state.sqlite3'
        rows = [dict(start=0., end=1., tags=[['label', .9]])]
        with StateStore(path, 'synthetic') as store:
            store.put('seed', 'wd', {'scene': record('synthetic', rows, scene=True, threshold=.4),
                                    'markers': record('synthetic', rows, threshold=.4)})
            with store.connection:
                seeds = {table: store.connection.execute('SELECT * FROM ' + table).fetchall()
                         for table in ('results', 'result_metadata', 'result_index')}
                for table, seed in seeds.items():
                    store.connection.execute('DELETE FROM ' + table)
                    sql = 'INSERT INTO ' + table + ' VALUES (' + ','.join('?' for _ in seed[0]) + ')'
                    store.connection.executemany(sql, (
                        ('synthetic', str(scene), model, *row[3:])
                        for scene in range(args.scenes) for model in SOURCES for row in seed))
            initial = evidence_hash(store)
        for name in trials:
            for _ in range(args.repeats):
                with StateStore(path, 'synthetic', dry_run=name != 'rethreshold_writer') as store:
                    report = RunReport(write=lambda _: None)
                    store.report = report
                    instructions = [0]
                    store.connection.set_progress_handler(lambda: instructions.__setitem__(0, instructions[0] + 100), 100)
                    changes = store.connection.total_changes
                    with redirect_stderr(io.StringIO()):
                        started = perf_counter()
                        if name == 'cache_preview':
                            payload = cache.build_report(app, store, chosen, settings)
                            result = {key: payload[key] for key in ('totals', 'models', 'issues', 'plan')}
                        else:
                            result = rethreshold(app, store, chosen, targets, report, markers=True)
                        seconds = perf_counter() - started
                    store.connection.set_progress_handler(None, 0)
                    trials[name].append(dict(seconds=seconds, sql_vm_steps=instructions[0],
                        changed_rows=store.connection.total_changes - changes,
                        result_sha256=hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest(),
                        evidence_unchanged=evidence_hash(store) == initial,
                        decoded_payloads=report.timing_calls['state_json_decode']))
                    if name == 'rethreshold_writer':
                        with redirect_stderr(io.StringIO()):
                            rethreshold(app, store, chosen, settings, report, markers=True)
    for name, rows in trials.items():
        if len({row['result_sha256'] for row in rows}) != 1 or not all(row['evidence_unchanged'] for row in rows):
            raise RuntimeError('Workflow results or raw evidence changed: ' + name)
    print(json.dumps(dict(scenes=args.scenes, selected_scenes=args.selected, models=len(settings),
        scope='Synthetic end-to-end selected report and Check/Apply; no network or inference.',
        summaries={name: dict(median_seconds=statistics.median(row['seconds'] for row in rows),
                             median_sql_vm_steps=statistics.median(row['sql_vm_steps'] for row in rows))
                   for name, rows in trials.items()}, trials=trials), indent=2))


if __name__ == '__main__':
    main()

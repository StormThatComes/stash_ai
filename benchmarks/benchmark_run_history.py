"""Offline telemetry overhead for synthetic fresh, cached and resumed inference."""
import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter, process_time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def trial(root, case, enabled, frames, labels):
    import numpy as np
    from PIL import Image
    from stash_ai.commands.arguments import parse_args
    from stash_ai.history import lifecycle
    from stash_ai.history.store import HistoryStore, history_path
    from stash_ai.history.retention import prune
    from stash_ai.inference.inputs import RunInputs
    from stash_ai.models import ModelSettings, SOURCES, Vocabulary
    from stash_ai.runs.merged_model import MergedModelRun
    from stash_ai.runtime.context import RuntimeContext
    from stash_ai.runtime.report import RunReport
    from stash_ai.runtime.memory import process_rss
    from stash_ai.storage.store import StateStore, scope_for
    from stash_ai.vtt import format_time

    root = Path(root)
    assets = root / 'assets'
    assets.mkdir(exist_ok=True)
    state = root / f'{case}-{enabled}.sqlite3'
    app = RuntimeContext(PREFETCH_BATCHES=0, PREPARE_WORKERS=1, CHECKPOINT_BATCHES=1)
    config = ModelSettings(SOURCES['wd'], assets, scene_threshold=.4, marker_threshold=.4, batch_size=4, device='cpu')
    vocab = Vocabulary([f'label_{i}' for i in range(labels)], {})
    values = np.array([[.3 + ((i+j*3) % 70)/100 for j in range(labels)]
                       for i in range(frames)], dtype=np.float32)
    image = Image.new('RGB', (frames, 1))
    for i in range(frames):
        image.putpixel((i, 0), (i % 256, (i//256) % 256, (i//65536) % 256))
    png = BytesIO(); image.save(png, format='PNG'); image.close()
    vtt = 'WEBVTT\n\n' + '\n\n'.join(
        f'{format_time(i)} --> {format_time(i+1)}\nsprite.png#xywh={i},0,1,1' for i in range(frames))
    media = type('Media', (), {'get_media': lambda self, url: png.getvalue() if url.endswith('.png') else vtt.encode()})()
    scenes = {'1': {'id': '1', 'paths': {'sprite': '/sprite.png', 'vtt': '/sprite.vtt'}}}
    scope = scope_for(app.STASH_URL)
    work, counters = dict(fail=None), Counter()

    class Tagger:
        def __init__(self, settings):
            self.settings, self.batch_size, self.labels, self.device = settings, settings.batch_size, vocab.labels, 'cpu'
        def prepare(self, image):
            r, g, b = image.getpixel((0, 0))
            return r + 256*g + 65536*b
        def predict_batch(self, indexes):
            if work['fail'] is not None and min(indexes) >= work['fail']:
                raise KeyboardInterrupt('Synthetic checkpoint interruption')
            counters['predicted_frames'] += len(indexes)
            return values[indexes]
        def close(self):
            pass
    app.services.load_tagger = lambda settings, *a: Tagger(settings)

    def execute(history):
        args = parse_args(app, ['--models', 'wd', '--scene-id', '1', '--prefetch-batches', '0',
                               '--prepare-workers', '1', '--checkpoint-batches', '1', '--state-db', str(state)]
                               + ([] if history else ['--no-run-history']))
        args.report = RunReport(write=lambda _: None)
        with lifecycle.RunHistory(app, args) as run:
            if history and run.store is None:
                raise RuntimeError('Enabled history did not open')
            with StateStore(state, scope) as store:
                store.report = args.report
                stats = Counter()
                args.report.watch_counts(stats)
                MergedModelRun(app, args, store, config, vocab, stats=stats, report=args.report).run(
                    scenes, ['scene', 'markers'], RunInputs(media, store.cache, args.report))
                counters['state_sql_writes'] += store.connection.total_changes
                digest = hashlib.sha256()
                for blob, in store.connection.execute('SELECT data FROM results ORDER BY scope,scene,model,kind'):
                    digest.update(blob)
                return args, digest.hexdigest(), dict(stats)

    if case == 'cached':
        execute(False)
    elif case == 'resumed':
        work['fail'] = max(4, (frames//2//4)*4)
        try:
            execute(False)
        except KeyboardInterrupt:
            pass
        else:
            raise RuntimeError('Missing synthetic interruption')
        work['fail'] = None
    counters.clear()
    history_file = Path(str(state)+'.runs.sqlite3')
    if enabled:
        with HistoryStore(history_file) as store:
            for i in range(40):
                store.start(f'old-{i}', scope, 'inference', '2025-01-01T00:00:00+00:00', {'fingerprint': 'fixture'})
                store.finish(f'old-{i}', '2025-01-01T00:00:00+00:00', 'finished', 1, 1,
                             models={}, counts={}, outcomes={}, metrics=[], issues={})

    class CountedHistory(HistoryStore):
        def close(self):
            if self.connection is not None:
                counters['history_sql_writes'] += self.connection.total_changes
            super().close()
    lifecycle.HistoryStore = CountedHistory
    initial_rss = process_rss()
    started, cpu = perf_counter(), process_time()
    args, fingerprint, stats = execute(enabled)
    seconds, cpu_seconds = perf_counter()-started, process_time()-cpu
    peak = process_rss()
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    except ImportError:
        pass
    retention = None
    if enabled:
        with HistoryStore(history_file) as store:
            before_proofs = store.connection.execute('SELECT * FROM reuse_proofs').fetchall()
            changes, started = store.connection.total_changes, perf_counter()
            retained = prune(store, scope, keep_last=1, dry_run=False)
            retention = dict(seconds=perf_counter()-started, changed_rows=store.connection.total_changes-changes,
                             removed_runs=retained['candidate_runs'],
                             proofs_unchanged=before_proofs == store.connection.execute('SELECT * FROM reuse_proofs').fetchall())
    return dict(case=case, history=enabled, wall_seconds=seconds, cpu_seconds=cpu_seconds,
                initial_rss_bytes=initial_rss, peak_process_rss_bytes=peak,
                state_bytes=state.stat().st_size, history_bytes=history_file.stat().st_size if enabled else 0,
                evidence_sha256=fingerprint, counts=dict(counters), stats=stats, retention=retention)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=128)
    parser.add_argument('--labels', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--case', choices=('fresh', 'cached', 'resumed'))
    parser.add_argument('--history', choices=('on', 'off'))
    parser.add_argument('--workdir', type=Path)
    args = parser.parse_args()
    if args.frames < 8 or args.labels < 1 or args.repeats < 1:
        parser.error('frames >= 8; labels and repeats must be positive')
    if args.case:
        if args.workdir is None or args.history is None:
            parser.error('internal trial requires workdir and history')
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            result = trial(args.workdir, args.case, args.history == 'on', args.frames, args.labels)
        print(json.dumps(result))
        return
    samples = []
    with tempfile.TemporaryDirectory(prefix='stash-history-benchmark-') as directory:
        root = Path(directory)
        (root / 'assets').mkdir()
        for i in range(args.repeats):
            for case in ('fresh', 'cached', 'resumed'):
                for history in ('off', 'on'):
                    output = subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_run_history',
                        '--case', case, '--history', history, '--workdir', str(root),
                        '--frames', str(args.frames), '--labels', str(args.labels)],
                        capture_output=True, text=True, check=True, timeout=60,
                        cwd=Path(__file__).resolve().parents[1])
                    samples.append(json.loads(output.stdout))
                    for path in root.glob(case+'-*sqlite3*'):
                        path.unlink()
    if len({row['evidence_sha256'] for row in samples}) != 1:
        raise RuntimeError('Telemetry changed completed raw evidence')
    for row in samples:
        if row['case'] == 'cached' and row['counts'].get('predicted_frames', 0) != 0:
            raise RuntimeError('Cached benchmark forced inference')
        if row['retention'] and not row['retention']['proofs_unchanged']:
            raise RuntimeError('Retention changed reuse proofs')
    groups = {}
    for case in ('fresh', 'cached', 'resumed'):
        groups[case] = {}
        for enabled in (False, True):
            rows = [row for row in samples if row['case'] == case and row['history'] == enabled]
            groups[case]['on' if enabled else 'off'] = dict(
                median_seconds=statistics.median(row['wall_seconds'] for row in rows),
                median_history_writes=statistics.median(row['counts'].get('history_sql_writes', 0) for row in rows))
    print(json.dumps(dict(scope='Synthetic CPU fixture; actual orchestration/input verification/checkpoints, no model backend or network. Each sample is a fresh process; peak RSS includes setup.',
                          frames=args.frames, labels=args.labels, identical_evidence=True,
                          groups=groups, trials=samples), indent=2))


if __name__ == '__main__':
    main()

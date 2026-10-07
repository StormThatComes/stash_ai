"""End-to-end result replacement with large published/pending/recovery proofs.

Both variants run current persistence; baseline disables only the per-call lazy
pin reader to perform the previous independent reads. No model or Stash API runs.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]


def peak_rss():
    try:
        import resource
        scale = 1 if sys.platform == 'darwin' else 1024
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    except ImportError:
        from stash_ai.runtime.memory import process_rss
        return process_rss()


def trial(args):
    sys.path.insert(0, str(ROOT))
    from benchmarks.fixtures import record
    from stash_ai.evidence.identity import marker_segment_signature
    from stash_ai.models import SOURCES
    from stash_ai.storage import archive
    from stash_ai.storage.recovery import abandon
    from stash_ai.storage.store import StateStore

    rows = [dict(start=float(i), end=float(i+1),
                 tags=[[f'tag_{label}', .9] for label in range(args.labels)])
            for i in range(args.frames)]
    def outputs(source, identity):
        return {'scene': record(source, rows, scene=True, identity=identity),
                'markers': record(source, rows, identity=identity)}
    old = {source: outputs(source, 'old') for source in SOURCES}
    new = {source: outputs(source, 'replacement') for source in SOURCES}
    snapshots = {}
    for kind in ('scene', 'markers'):
        contributions = []
        for source in SOURCES:
            for label in range(args.labels):
                # Each range proof is independent, as for different merged
                # markers; compact encoding must not erase this workload.
                support = dict(frame_context={source: marker_segment_signature(old[source]['scene'])},
                               sample_times=[float(i)+label*.001 for i in range(args.frames)])
                contributions.append(dict(source=source, tag=f'tag_{label}', score=.9,
                    start=0., end=float(args.frames), extra_tags=['support'], meta={'origin':'benchmark'},
                    evidence_signature=marker_segment_signature(old[source][kind]),
                    extra_support={'support': support}))
        snapshots[kind] = {'contributions': contributions}

    tables = ('results', 'inference_archive', 'snapshots', 'recovery_publications',
              'result_index', 'result_thresholds', 'result_metadata', 'merge_checks')
    def state_hash(store, selected):
        value = hashlib.sha256()
        for table in selected:
            records = store.read_rows('SELECT * FROM ' + store.table(table))
            normalized = sorted([[(cell.hex() if isinstance(cell, bytes) else cell)
                                  for cell in row] for row in records], key=repr)
            value.update(json.dumps([table, normalized], sort_keys=True,
                                    separators=(',', ':')).encode())
        return value.hexdigest()

    with tempfile.TemporaryDirectory(prefix='stash-pinned-proofs-') as directory:
        path = Path(directory)/'state.sqlite3'
        with StateStore(path, 'synthetic') as store:
            for source in SOURCES:
                store.put('1', source, old[source])
            store.publish('1', snapshots)
            store.prepare('1', snapshots)
            with redirect_stdout(io.StringIO()):
                abandon(store, lambda _: None)
            # Abandonment uses an operation timestamp/UUID; fix only this fixture
            # identifier so the two variants can compare every persisted byte.
            with store.connection:
                store.connection.execute("UPDATE recovery_publications SET operation='fixture-recovery'")
            store.prepare('1', snapshots)
        source_bytes = path.read_bytes()
        if args.variant == 'baseline':
            archive.lazy_pins = lambda *a: None
        with StateStore(path, 'synthetic', dry_run=args.preview) as store:
            proof_before = state_hash(store, ('snapshots', 'recovery_publications'))
            original, decoded = store.unpack_snapshot, []
            def decode(blob):
                decoded.append(1)
                return original(blob)
            store.unpack_snapshot = decode
            writes = store.connection.total_changes
            cpu, wall = process_time(), perf_counter()
            for source in SOURCES:
                store.put('1', source, new[source])
            wall = perf_counter()-wall
            cpu = process_time()-cpu
            write_count = store.connection.total_changes-writes
            store.unpack_snapshot = original
            archived = store.read_rows('SELECT model,kind,signature FROM ' + store.table('inference_archive'))
            if len(archived) != len(SOURCES)*2:
                raise RuntimeError('Pinned evidence was lost during replacement')
            proofs_unchanged = proof_before == state_hash(store, ('snapshots', 'recovery_publications'))
            result = dict(variant=args.variant, preview=args.preview, wall_seconds=wall,
                          cpu_seconds=cpu, peak_process_rss_bytes=peak_rss(),
                          decoded_snapshots=len(decoded), state_sql_writes=write_count,
                          archived_records=len(archived), state_sha256=state_hash(store, tables),
                          proofs_unchanged=proofs_unchanged)
        result['source_unchanged'] = path.read_bytes() == source_bytes if args.preview else None
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--labels', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variant', choices=('baseline', 'candidate'))
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if min(args.frames, args.labels, args.repeats) < 1:
        parser.error('positive frames, labels and repeats required')
    if args.variant:
        print(json.dumps(trial(args), sort_keys=True))
        return
    trials = []
    for preview in (False, True):
        for repeat in range(args.repeats):
            for variant in ('baseline', 'candidate'):
                command = [sys.executable, str(Path(__file__).resolve()), '--frames', str(args.frames),
                           '--labels', str(args.labels), '--variant', variant]
                if preview:
                    command.append('--preview')
                done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
                if done.returncode:
                    raise RuntimeError(done.stderr)
                trials.append(dict(repeat=repeat, **json.loads(done.stdout)))
    groups = {}
    for preview in (False, True):
        scope = [row for row in trials if row['preview'] == preview]
        if (len({row['state_sha256'] for row in scope}) != 1 or
                any(not row['proofs_unchanged'] for row in scope) or
                any(not row['source_unchanged'] for row in scope if preview)):
            raise RuntimeError('State, proofs or preview source changed between variants')
        for variant in ('baseline', 'candidate'):
            group = [row for row in scope if row['variant'] == variant]
            groups[('preview' if preview else 'writer') + '_' + variant] = dict(
                median_seconds=statistics.median(row['wall_seconds'] for row in group),
                median_decoded_snapshots=statistics.median(row['decoded_snapshots'] for row in group),
                median_state_sql_writes=statistics.median(row['state_sql_writes'] for row in group))
    print(json.dumps(dict(scope='synthetic CPU result writes; no inference or Stash network',
                          frames=args.frames, labels=args.labels, identical_state=True,
                          groups=groups, trials=trials), sort_keys=True))


if __name__ == '__main__':
    main()

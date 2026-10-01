"""Offline combined rethreshold over temporary SQLite evidence, without inference."""
import argparse
from contextlib import redirect_stderr
import gc
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
import tempfile
from time import perf_counter
from types import SimpleNamespace

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evidence_hash(connection):
    digest = hashlib.sha256()
    for blob, in connection.execute('SELECT data FROM results ORDER BY scope,scene,model,kind'):
        digest.update(blob)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--scenes', type=int, default=6)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--labels', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.scenes, args.frames, args.labels, args.repeats) < 1:
        parser.error('scenes, frames, labels and repeats must be positive')
    sys.path.insert(0, str(args.project.resolve()))
    from stash_ai import scene_cache
    from stash_ai.runtime import RunReport
    from stash_ai.state import StateStore
    from stash_ai.taggers import SOURCES
    from stash_ai.temporal import TemporalPolicy
    app = SimpleNamespace(TEMPORAL_POLICY=TemporalPolicy())
    settings = [SimpleNamespace(source=source, scene_threshold=.55, marker_threshold=.55)
                for source in SOURCES.values()]
    frames = [dict(start=float(i), end=float(i + 1), tags=[
        ['label_' + str(j), .3 + ((i * 7 + j * 19) % 600) / 1000]
        for j in range(args.labels)]) for i in range(args.frames)]
    meta = dict(origin='inference', inference_signature='synthetic',
                scene_threshold=.4, marker_threshold=.4, scene_frames_format=1,
                scene_score_floor=.1, scene_support_floor=.34, marker_score_floor=.1)
    outputs = {
        'scene': dict(mode='scene', meta=meta, frames=frames,
                      items=[dict(tag='label_' + str(j), score=.9) for j in range(args.labels)]),
        'markers': dict(mode='frames', meta=meta, items=frames),
    }
    scenes = [str(i) for i in range(args.scenes)]
    trials = []
    for _ in range(args.repeats):
        with tempfile.TemporaryDirectory(prefix='stash-rethreshold-benchmark-') as directory:
            with StateStore(Path(directory) / 'state.sqlite3', 'synthetic') as store:
                for scene in scenes:
                    for source in SOURCES:
                        store.put(scene, source, outputs)
                before = evidence_hash(store.connection)
                report = RunReport(write=lambda _: None)
                store.report = report
                writes, packed_bytes = 0, 0
                replace_records, pack = store.replace_records, store.pack

                def count_write(*values):
                    nonlocal writes
                    writes += 1
                    return replace_records(*values)

                def count_pack(value):
                    nonlocal packed_bytes
                    blob = pack(value)
                    packed_bytes += len(blob)
                    return blob

                store.replace_records, store.pack = count_write, count_pack
                gc.collect()
                with redirect_stderr(io.StringIO()):
                    started = perf_counter()
                    result = scene_cache.rethreshold(app, store, scenes, settings, report, markers=True)
                    seconds = perf_counter() - started
                if result != {'scene_thresholds_updated': args.scenes * len(settings)}:
                    raise RuntimeError('Unexpected threshold update result: ' + repr(result))
                codec_calls = {name: report.timing_calls[name] for name in (
                    'state_json_decode', 'state_decompress', 'state_json_encode', 'state_compress')}
                # Compare the effective view as well as immutable source bytes.
                effective = hashlib.sha256()
                for scene in scenes:
                    for kind in ('scene', 'markers'):
                        effective.update(json.dumps(store.records(scene, kind), sort_keys=True).encode())
                trials.append(dict(seconds=seconds, evidence_writes=writes, packed_bytes=packed_bytes,
                                   evidence_unchanged=before == evidence_hash(store.connection),
                                   effective_sha256=effective.hexdigest(),
                                   codec_calls=codec_calls))
    if len({trial['effective_sha256'] for trial in trials}) != 1:
        raise RuntimeError('Non-deterministic effective evidence')
    print(json.dumps(dict(scenes=args.scenes, models=len(settings), frames=args.frames,
                         labels=args.labels, median_seconds=statistics.median(t['seconds'] for t in trials),
                         scope='Synthetic local rethreshold only; no inference, merge or Stash API.',
                         trials=trials), indent=2))


if __name__ == '__main__':
    main()

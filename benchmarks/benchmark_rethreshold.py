"""Offline rethreshold over temporary SQLite evidence, without inference."""
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
    parser.add_argument('--mode', choices=('combined', 'scene', 'markers'), default='combined')
    parser.add_argument('--legacy-metadata', action='store_true',
                        help='Remove derived metadata before Check to measure first-time backfill.')
    args = parser.parse_args()
    if min(args.scenes, args.frames, args.labels, args.repeats) < 1:
        parser.error('scenes, frames, labels and repeats must be positive')
    sys.path.insert(0, str(args.project.resolve()))
    from stash_ai.rethreshold import (
        execution as rethreshold_execution, markers as marker_updates, scenes as scene_updates)
    from stash_ai.runtime.report import RunReport
    from stash_ai.storage.store import StateStore
    from stash_ai.models import ModelSettings, SOURCES
    from stash_ai.evidence.temporal.policy import TemporalPolicy
    app = SimpleNamespace(TEMPORAL_POLICY=TemporalPolicy())
    settings = [ModelSettings(source, Path('.'), scene_threshold=.4 if args.mode == 'markers' else .55,
                              marker_threshold=.55)
                for source in SOURCES.values()]
    frames = [dict(start=float(i), end=float(i + 1), tags=[
        ['label_' + str(j), .3 + ((i * 7 + j * 19) % 600) / 1000]
        for j in range(args.labels)]) for i in range(args.frames)]
    meta = dict(origin='inference', inference_signature='synthetic',
                scene_threshold=.4, marker_threshold=.4, scene_frames_format=1,
                scene_score_floor=.1, scene_support_floor=.34, marker_score_floor=.1,
                marker_tag_blocklist=[])
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
                if args.legacy_metadata:
                    with store.connection:
                        store.connection.execute('DELETE FROM result_metadata')
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
                planner = marker_updates if args.mode == 'markers' else scene_updates
                plan_name = 'validate' if args.mode == 'markers' else 'plan'
                original_plan = getattr(planner, plan_name)
                phases = {phase: dict(seconds=0., decode_calls=0) for phase in ('check', 'apply')}
                plan_calls = 0

                def timed_plan(*values, **options):
                    nonlocal plan_calls
                    phase = 'check' if plan_calls < args.scenes * len(settings) else 'apply'
                    plan_calls += 1
                    decodes = report.timing_calls['state_json_decode']
                    started = perf_counter()
                    try:
                        return original_plan(*values, **options)
                    finally:
                        phases[phase]['seconds'] += perf_counter() - started
                        phases[phase]['decode_calls'] += report.timing_calls['state_json_decode'] - decodes

                setattr(planner, plan_name, timed_plan)
                gc.collect()
                try:
                    with redirect_stderr(io.StringIO()):
                        started = perf_counter()
                        result = rethreshold_execution.run(
                            app, store, scenes, settings, report,
                            scenes=args.mode != 'markers', markers=args.mode != 'scene')
                        seconds = perf_counter() - started
                finally:
                    setattr(planner, plan_name, original_plan)
                updated_key = 'marker_thresholds_updated' if args.mode == 'markers' else 'scene_thresholds_updated'
                if result != {updated_key: args.scenes * len(settings)}:
                    raise RuntimeError('Unexpected threshold update result: ' + repr(result))
                codec_calls = {name: report.timing_calls[name] for name in (
                    'state_json_decode', 'state_decompress', 'state_json_encode', 'state_compress')}
                # Compare the effective view as well as immutable source bytes.
                effective = hashlib.sha256()
                for scene in scenes:
                    for kind in ('scene', 'markers'):
                        effective.update(json.dumps(store.records(scene, kind), sort_keys=True).encode())
                index = store.connection.execute(
                    'SELECT * FROM result_index ORDER BY scope,scene,model,kind').fetchall()
                trials.append(dict(seconds=seconds, evidence_writes=writes, packed_bytes=packed_bytes,
                                   evidence_sha256=before,
                                   evidence_unchanged=before == evidence_hash(store.connection),
                                   effective_sha256=effective.hexdigest(),
                                   index_sha256=hashlib.sha256(json.dumps(index).encode()).hexdigest(),
                                   codec_calls=codec_calls, phases=phases,
                                   metadata_rows=store.connection.execute(
                                       'SELECT COUNT(*) FROM result_metadata').fetchone()[0]))
    if len({trial['effective_sha256'] for trial in trials}) != 1:
        raise RuntimeError('Non-deterministic effective evidence')
    if len({trial['index_sha256'] for trial in trials}) != 1:
        raise RuntimeError('Non-deterministic completion index')
    print(json.dumps(dict(scenes=args.scenes, models=len(settings), frames=args.frames,
                         labels=args.labels, mode=args.mode, legacy_metadata=args.legacy_metadata,
                         median_seconds=statistics.median(t['seconds'] for t in trials),
                         median_check_seconds=statistics.median(t['phases']['check']['seconds'] for t in trials),
                         median_apply_plan_seconds=statistics.median(t['phases']['apply']['seconds'] for t in trials),
                         scope='Synthetic local rethreshold only; no inference, merge or Stash API.',
                         trials=trials), indent=2))


if __name__ == '__main__':
    main()

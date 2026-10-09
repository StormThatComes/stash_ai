"""Compare zlib levels on identical synthetic retained scene/marker evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.fixtures import record
from stash_ai.storage.codec import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=729)
    parser.add_argument('--labels', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if min(args.frames, args.labels, args.repeats) < 1:
        parser.error('frames, labels and repeats must be positive')
    rng = random.Random(0)
    rows = [dict(start=float(i), end=float(i + 1), tags=sorted(
        [[f'tag_{label}', .1 + .9 * rng.random()] for label in range(args.labels)],
        key=lambda item: -item[1])) for i in range(args.frames)]
    payloads = {'scene': record('oppai', rows, scene=True), 'markers': record('oppai', rows)}
    trials = []
    for kind, payload in payloads.items():
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(',', ':')).encode('utf-8')
        expected = json.loads(raw)
        for repeat in range(args.repeats):
            # Alternate order to avoid always favoring the first/last level.
            for level in ([6, 1] if repeat % 2 == 0 else [1, 6]):
                started = perf_counter()
                blob = zlib.compress(raw, level)
                elapsed = perf_counter() - started
                if decode(blob) != expected or zlib.decompress(blob) != raw:
                    raise RuntimeError('Compression changed retained evidence')
                trials.append(dict(kind=kind, level=level, seconds=elapsed,
                                   json_bytes=len(raw), blob_bytes=len(blob),
                                   logical_sha256=hashlib.sha256(raw).hexdigest()))
    groups = []
    for kind in payloads:
        for level in (6, 1):
            values = [row for row in trials if row['kind'] == kind and row['level'] == level]
            groups.append(dict(kind=kind, level=level,
                median_seconds=statistics.median(row['seconds'] for row in values),
                json_bytes=values[0]['json_bytes'], blob_bytes=values[0]['blob_bytes']))
    print(json.dumps(dict(scope='Synthetic compression only; excludes JSON encoding, SQLite and GPU',
        frames=args.frames, labels=args.labels, identical_evidence=True, groups=groups, trials=trials)))


if __name__ == '__main__':
    main()

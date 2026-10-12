"""Isolated CPU inference/asset/RSS smoke; synthetic input, no accuracy claims."""
from contextlib import closing
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import time


def measure(config):
    import numpy as np
    from PIL import Image
    from stash_ai.recognition.models import checked_detection, ensure_assets, load_backend, namespace
    from stash_ai.runtime.memory import process_rss
    samples, stop = [], threading.Event()
    def sample():
        while not stop.wait(.01):
            value = process_rss()
            if value is not None:
                samples.append(value)
    watcher = threading.Thread(target=sample)
    paths = ensure_assets(config, write=lambda message: None)
    watcher.start()
    try:
        started = time.perf_counter()
        with closing(load_backend(config, paths)) as adapter:
            load_seconds = time.perf_counter() - started
            grid = np.indices((128,128)).sum(axis=0) % 2
            pixels = np.repeat((grid[...,None] * 210 + 20).astype('uint8'), 3, axis=2)
            face = checked_detection([0,0,128,128,38,44,88,44,64,66,42,90,86,90,.99])
            durations = []
            with Image.fromarray(pixels) as image:
                detected = len(adapter.detect(image))
                adapter.embed(image, face)  # warmup on the same explicit fixture
                for _ in range(5):
                    start = time.perf_counter()
                    result = adapter.embed(image, face)
                    durations.append(time.perf_counter() - start)
            return {'namespace': namespace(config), 'device': adapter.device, 'dimensions': len(result),
                    'load_seconds': load_seconds, 'median_embedding_seconds': statistics.median(durations),
                    'sampled_peak_rss_bytes': max(samples) if samples else None, 'rss_interval_seconds': .01,
                    'artifact_bytes': sum(path.stat().st_size for path in paths.values()),
                    'fixture': 'synthetic texture with supplied landmarks', 'detected_faces': detected,
                    'accuracy': None, 'accuracy_reason': 'no labeled human-face pilot in this smoke'}
    finally:
        stop.set()
        watcher.join()


def main():
    from stash_ai.recognition.config import load
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        print(json.dumps(measure(load(Path(sys.argv[2]))), sort_keys=True))
        return
    with tempfile.TemporaryDirectory(prefix='recognition-benchmark-') as directory:
        root, results = Path(directory), {}
        for name in ('sface', 'auraface'):
            path = root / f'{name}.yml'
            path.write_text(f'version: 1\nmodels: {{recognizer: {name}, device: cpu, directory: {name}}}\n', encoding='utf-8')
            worker = subprocess.run([sys.executable, '-m', 'benchmarks.benchmark_recognition_adapters', '--worker', str(path)],
                                     capture_output=True, text=True, check=True, timeout=180)
            results[name] = json.loads(worker.stdout)
        print(json.dumps({'scope': 'isolated-synthetic-runtime-only', 'adapters': results}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

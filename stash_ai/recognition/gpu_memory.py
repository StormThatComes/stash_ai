"""Phase-owned GPU cleanup and bounded, best-effort memory diagnostics."""
from .profiling import pipeline_stage
from contextlib import contextmanager
import gc
import os
import shutil
import subprocess
import sys
from time import monotonic
from threading import Lock, Thread

from stash_ai.runtime.cleanup import cleanup
from stash_ai.runtime import shutdown


def torch_memory():
    # Cached/CPU runs must not import PyTorch or initialize a CUDA context.
    torch = sys.modules.get('torch')
    cuda = getattr(torch, 'cuda', None)
    if cuda is None or not cuda.is_initialized():
        return None
    device = cuda.current_device()
    return dict(device=device, allocated_bytes=cuda.memory_allocated(device),
                reserved_bytes=cuda.memory_reserved(device))


def process_memory(executable):
    """All NVIDIA devices for this PID, including ONNX; unknown is never zero."""
    if executable is None:
        return None
    result = subprocess.run([executable, '--query-compute-apps=pid,used_gpu_memory',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, timeout=2, check=True)
    values = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) != 2 or fields[0] != str(os.getpid()):
            continue
        try:
            value = int(fields[1])
        except ValueError:
            return None  # e.g. Windows WDDM reports N/A
        if value < 0:
            return None
        values.append(value * 1024 * 1024)
    return sum(values) if values else None


def reclaim():
    """After models/leases are gone, return unused PyTorch blocks to CUDA."""
    shutdown.require_idle()
    gc.collect()
    torch = sys.modules.get('torch')
    cuda = getattr(torch, 'cuda', None)
    if cuda is not None and cuda.is_initialized():
        cuda.empty_cache()


class GpuMemory:
    def __init__(self, config, report, database, run):
        self.enabled = config.data['execution']['log_vram']
        self.report, self.database, self.run = report, database, run
        self.executable = None
        if self.enabled and config.data['models']['device'] != 'cpu':
            try:
                self.executable = shutil.which('nvidia-smi')
            except OSError:
                pass
        self.samples = 0
        self.latest = {}
        self.peaks = {}
        self.display = ''
        self.reported_unavailable = False
        self.batch_sizes = {}
        self.batch_lock = Lock()
        self.process_thread = None
        self.process_latest = (None, None)
        self.process_requested = float('-inf')
        self.sampled_batches = {}
        self.sampled_scene = None
        self.scene_sample_at = float('-inf')

    def batch_change(self, kind, previous, maximum, retry):
        with self.batch_lock:
            row = self.batch_sizes.setdefault(kind, dict(retries=0))
            row['maximum'] = maximum
            if retry:
                row.update(previous=previous, reduced_to=maximum, retries=row['retries']+1)

    def batch_line(self):
        with self.batch_lock:
            return ', '.join(f"{kind} {v['previous']}→{v['reduced_to']} · effective {v['maximum']} · {v['retries']} retries"
                             for kind, v in sorted(self.batch_sizes.items()) if v['retries'])

    def poll_process(self):
        if self.executable is None:
            return None
        now = monotonic()
        if now-self.process_requested >= 10 and not (self.process_thread and self.process_thread.is_alive()):
            self.process_requested = now
            def probe():
                try:
                    value = process_memory(self.executable)
                except (OSError, subprocess.SubprocessError, ValueError):
                    value = None
                self.process_latest = (monotonic(), value)
            self.process_thread = Thread(target=probe, name='recognition-vram', daemon=True)
            self.process_thread.start()
        return self.process_latest[1]

    def live_display(self):
        sampled, _ = self.process_latest
        return self.display + (f' · process sample {monotonic()-sampled:.0f}s ago' if sampled is not None else '')

    def snapshot(self):
        values = {}
        for name, query in (('torch', torch_memory),
                            ('process_bytes', self.poll_process)):
            try:
                values[name] = query()
            except Exception:
                values[name] = None
        return values

    def record(self, stage, point, *, scene=None):
        if self.enabled:
            self.report.optional('recognition_vram', self._record, stage, point, scene)

    def _record(self, stage, point, scene):
        if stage in ('faces', 'scene') and scene is None:
            if point == 'before_release':
                last, _ = self.sampled_batches.get(stage, (float('-inf'), False))
                now = monotonic()
                keep = now-last >= 10
                self.sampled_batches[stage] = (now if keep else last, keep)
            if not self.sampled_batches.get(stage, (0, False))[1]:
                return
        if scene is not None and scene != self.sampled_scene:
            now = monotonic()
            if now - self.scene_sample_at < 10:
                return
            self.sampled_scene, self.scene_sample_at = scene, now
        if stage == 'final' and point == 'after_release' and self.process_thread:
            self.process_thread.join(timeout=2.25)
        values = self.snapshot()
        if self.process_latest[0] is not None:
            values['process_sample_age_seconds'] = max(0., monotonic()-self.process_latest[0])
        self.samples += 1
        key = stage + ':' + point
        # Stage names are fixed by orchestration. Retain only the last scene
        # per boundary, never one event per scene in a 10,000-scene run.
        self.latest[key] = dict(values, scene=scene)
        measured = {'process_bytes': values['process_bytes']}
        if values['torch'] is not None:
            measured.update({name: values['torch'][name] for name in ('allocated_bytes', 'reserved_bytes')})
        for name, value in measured.items():
            if value is not None:
                self.peaks[name] = max(self.peaks.get(name, 0), value)
        # Observability must never commit a caller's tentative evidence writes.
        # The next boundary/final summary will persist the latest measurements.
        connection = getattr(self.database, 'connection', None)
        if not getattr(connection, 'in_transaction', False):
            self.database.progress(self.run, 'gpu_memory', self.summary())
        def size(value):
            return 'unavailable' if value is None else f'{value / 1048576:.1f} MiB'
        tensor = values['torch'] or {}
        line = (f'VRAM {stage} {point}' + (f' scene={scene}' if scene is not None else '') +
                f": process={size(values['process_bytes'])}"
                f" torch_allocated={size(tensor.get('allocated_bytes'))}"
                f" torch_reserved={size(tensor.get('reserved_bytes'))}")
        from .live_resources import gib
        current = values['process_bytes']
        self.display = (f'{stage} {point} · process {gib(current)} · sampled max {gib(self.peaks.get("process_bytes"))}'
                        f' · torch reserved {gib(tensor.get("reserved_bytes"))}')
        unavailable = current is None and not any(v is not None for v in tensor.values())
        if self.report.verbose:
            self.report.detail(line)
        elif unavailable and not self.reported_unavailable and not (self.process_thread and self.process_thread.is_alive()):
            self.reported_unavailable = True
            self.report.write('VRAM measurement unavailable for this process.')


    def summary(self):
        return dict(samples=self.samples, sampled_peak_bytes=dict(self.peaks),
                    latest=dict(self.latest))

    @contextmanager
    def phase(self, stage):
        self.record(stage, 'before')
        try:
            yield
        finally:
            self.record(stage, 'after')

    @pipeline_stage('model_release')
    def release(self, stage, callbacks, *, scene=None, stop=()):
        self.record(stage, 'before_release', scene=scene)
        try:
            # Never free resources retained by quarantined workers. The final
            # cleanup uses the same deferred-release mechanism as LazyBackend.
            actions = [(name, lambda action=action: shutdown.release(
                getattr(action, '__self__', action), action)) for name, action in callbacks]
            cleanup([*stop, *actions, ('reclaim unused GPU memory', self.reclaim)], report=self.report)
        finally:
            self.record(stage, 'release_deferred' if shutdown.busy() else 'after_release', scene=scene)

    def reclaim(self):
        shutdown.release(self, reclaim)

"""One scene of read-only input lookahead; the caller owns all durable state."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from stash_ai.runtime.cancellation import Cancellation, use


class SceneInputPrefetch:
    """Retain one verified input, without decoding images or invoking a model."""

    def __init__(self, app, inputs, report, model):
        self.app, self.inputs, self.report, self.model = app, inputs, report, model
        self.executor = self.pending = self.reader = self.token = None
        self.scene = None

    def submit(self, scene, paths, identity):
        if self.pending is not None:
            return
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scene-inputs')
        self.scene, self.token = scene, Cancellation()
        self.pending = self.executor.submit(self._prepare, dict(paths), identity, self.token)

    def _prepare(self, paths, identity, token):
        with use(token), self.report.model(self.model) if self.report else nullcontext():
            if self.reader is None:
                self.reader = self.inputs.open_background_reader()
            with self.inputs.read_with(self.reader):
                timer = (self.report.measure('scene_input_prefetch', model=self.model,
                                             units=1, unit='scenes')
                         if self.report else nullcontext())
                with timer:
                    result = self.inputs.inspect(self.app, paths, identity)
                    token.check()
                    return result

    def take(self, scene):
        if self.pending is None or self.scene != scene:
            return None
        future = self.pending
        try:
            timer = (self.report.measure('scene_input_wait', model=self.model,
                                         units=1, unit='scenes')
                     if self.report else nullcontext())
            with timer:
                return future.result()
        finally:
            # A caller's Ctrl+C can interrupt result() while the read is still
            # active. Keep its token/future until close() can cancel that read.
            if future.done():
                self.pending = self.scene = self.token = None

    def close(self):
        if self.executor is None:
            return
        if self.token is not None:
            self.token.cancel()
        if self.pending is not None:
            self.pending.cancel()
        try:
            # Cleanup is queued after the cancelled read, on its owner thread.
            self.executor.submit(self._close_reader).result()
        finally:
            self.executor.shutdown(wait=True)
            self.executor = self.pending = self.token = self.scene = None

    def _close_reader(self):
        if self.reader is not None:
            try:
                self.reader.close()
            finally:
                self.reader = None

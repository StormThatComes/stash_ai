"""One scene of read-only input lookahead; the caller owns all durable state."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Lock

from stash_ai.runtime.cancellation import Cancellation, Cancelled, use


class SceneInputPrefetch:
    """Verify one next input; optionally retain two prepared CPU batches."""

    def __init__(self, app, inputs, report, model):
        self.app, self.inputs, self.report, self.model = app, inputs, report, model
        self.executor = self.pending = self.reader = self.token = None
        self.scene = None
        self.prepared = {}
        self.streams = []
        self.batch_executor = None
        self.batch_close_errors = []
        self.lock, self.closed = Lock(), False

    def submit(self, scene, paths, identity, *, tagger=None, prefetch=0, workers=1, transforms=None):
        if self.pending is not None:
            return
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='scene-inputs')
        self.scene, self.token = scene, Cancellation()
        with self.lock:
            self.streams = [stream for stream in self.streams if not stream.stopped]
        self.pending = self.executor.submit(self._prepare, scene, dict(paths), identity, self.token,
                                            tagger, prefetch, workers, transforms)

    def _prepare(self, scene, paths, identity, token, tagger, prefetch, workers, transforms):
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
                if tagger is not None:
                    from .batch_stream import PreparedBatchStream
                    stream = PreparedBatchStream(min(prefetch, 2), self.report, self.model)
                    with self.lock:
                        if self.closed:
                            raise Cancelled('Input prefetch was closed.')
                        self.prepared[scene] = stream
                        self.streams.append(stream)
                        if self.batch_executor is None:
                            self.batch_executor = ThreadPoolExecutor(
                                max_workers=2, thread_name_prefix='scene-batches')
                        stream.task = self.batch_executor.submit(self._prepare_batches, stream, tagger,
                                                                  paths, result, workers, transforms)
                return result

    def _prepare_batches(self, stream, tagger, paths, result, workers, transforms):
        cues, media, _ = result
        reader = None
        try:
            # Verification of C must not wait for a full CPU queue of current
            # scene B. Every CPU producer also owns an isolated HTTP session.
            reader = self.inputs.open_background_reader()
            with self.inputs.read_with(reader), self.report.model(self.model) if self.report else nullcontext():
                if self.report is not None:
                    self.report.work('lookahead')
                timer = (self.report.measure('scene_batch_prefetch', model=self.model, units=1, unit='scenes')
                         if self.report else nullcontext())
                with timer:
                    stream.produce(self.app, tagger, media, paths, cues, workers, transforms)
        except BaseException as error:
            stream._send(error)
            stream._send(stream.end)
        finally:
            if reader is not None:
                try:
                    reader.close()
                except BaseException as error:
                    # HTTP cleanup is reported after durable scene processing,
                    # like cleanup of the input verifier's own reader.
                    with self.lock:
                        if not self.batch_close_errors:
                            self.batch_close_errors.append(error)

    def take_batches(self, scene, processed):
        with self.lock:
            stream = self.prepared.pop(scene, None)
        if stream is None:
            return None
        if processed:
            # Resume is authorized on the SQLite owner after verifying inputs.
            # Discard speculative CPU work; prepare only the restored suffix.
            stream.close()
            return None
        return stream

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
        with self.lock:
            self.closed = True
            streams = tuple(self.streams)
        if self.token is not None:
            self.token.cancel()
        if self.pending is not None:
            self.pending.cancel()
        try:
            for stream in streams:
                stream.close()
            # Cleanup is queued after the cancelled read, on its owner thread.
        finally:
            try:
                self.executor.submit(self._close_reader).result()
            finally:
                try:
                    if self.batch_executor is not None:
                        self.batch_executor.shutdown(wait=True, cancel_futures=True)
                finally:
                    self.executor.shutdown(wait=True)
                    self.executor = self.pending = self.token = self.scene = self.batch_executor = None
                    self.prepared.clear()
                    self.streams.clear()
                    errors = tuple(self.batch_close_errors)
                    self.batch_close_errors.clear()
        if errors:
            raise errors[0]

    def _close_reader(self):
        if self.reader is not None:
            try:
                self.reader.close()
            finally:
                self.reader = None

"""Bounded CPU inputs prepared ahead of the next scene's GPU task."""
from contextlib import nullcontext
from queue import Empty, Full, Queue

from stash_ai.runtime.cancellation import Cancellation, use
from .preparation import prepared_batches


class PreparedBatchStream:
    def __init__(self, limit, report, model):
        self.queue = Queue(maxsize=limit)
        self.report, self.model = report, model
        self.cancellation = Cancellation()
        self.stopped = False
        self.task = None
        self.end = object()

    def _send(self, value):
        timer = (self.report.measure('scene_batch_queue_wait', model=self.model)
                 if self.report else nullcontext())
        with timer:
            while not self.stopped:
                try:
                    self.queue.put(value, timeout=.05)
                    return
                except Full:
                    pass

    def produce(self, app, tagger, media, paths, cues, workers, transforms):
        batches = None
        try:
            with use(self.cancellation):
                batches = prepared_batches(app, tagger, media, paths, cues, workers=workers,
                                           report=self.report, collate=True,
                                           cancellation=self.cancellation,
                                           transform_executor=transforms, frame_lookahead=True)
                for batch in batches:
                    self._send(batch)
                    self.cancellation.check()
        except BaseException as error:
            self._send(error)
        finally:
            if batches is not None:
                batches.close()
            self._send(self.end)

    def __iter__(self):
        return self

    def __next__(self):
        timer = (self.report.measure('inference_input_wait', model=self.model)
                 if self.report else nullcontext())
        with timer:
            value = self._receive()
            if isinstance(value, BaseException):
                raise value
        if value is self.end:
            raise StopIteration
        return value

    def _receive(self):
        while not self.stopped:
            try:
                value = self.queue.get(timeout=.05)
            except Empty:
                # A task cancelled before it started has no queue sentinel.
                if self.task is not None and self.task.done():
                    self.task.result()
                    return self.end
                continue
            return value
        return self.end

    def close(self):
        self.stopped = True
        self.cancellation.cancel()
        try:
            if self.task is not None:
                self.task.cancel()
                if not self.task.cancelled():
                    self.task.result()
        finally:
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break

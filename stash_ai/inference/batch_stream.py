"""Bounded CPU inputs prepared ahead of the next scene's GPU task."""
from contextlib import nullcontext

from stash_ai.runtime.cancellation import use
from .preparation import prepared_batches
from .buffer_usage import queued_batches
from .managed_stream import ManagedStream


class PreparedBatchStream:
    def __init__(self, limit, report, model):
        self.channel = ManagedStream(limit)
        self.queue = self.channel.queue
        self.report, self.model = report, model
        self.cancellation, self.end = self.channel.cancellation, self.channel.end

    @property
    def stopped(self):
        return self.channel.stopped.is_set()

    @property
    def task(self):
        return self.channel.worker

    @task.setter
    def task(self, worker):
        self.channel.worker = worker

    def buffer_usage(self):
        return queued_batches(self.queue)

    def _send(self, value):
        timer = (self.report.measure('scene_batch_queue_wait', model=self.model)
                 if self.report else nullcontext())
        with timer:
            self.channel.send(value)

    def produce(self, app, tagger, media, paths, cues, workers, transforms):
        batches = None
        try:
            with use(self.cancellation):
                batches = prepared_batches(app, tagger, media, paths, cues, workers=workers,
                                           report=self.report, collate=True,
                                           cancellation=self.cancellation,
                                           transform_executor=transforms, frame_lookahead=True)
                while True:
                    self.queue.producing(True)
                    try:
                        batch = next(batches)
                    except StopIteration:
                        break
                    self._send(batch)
                    self.cancellation.check()
        except BaseException as error:
            self._send(error)
        finally:
            self.queue.producing(False)
            from stash_ai.runtime.cleanup import cleanup
            cleanup([lambda: batches.close() if batches is not None else None,
                     lambda: self._send(self.end)], report=self.report)

    def __iter__(self):
        return self

    def __next__(self):
        self.queue.consuming(False)
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
        return self.channel.receive()

    def stop(self):
        self.channel.stop()

    def close(self):
        self.channel.close()

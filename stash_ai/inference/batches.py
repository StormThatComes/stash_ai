"""Ordered inference lookahead with a bounded result queue and one model owner."""
from contextlib import closing, nullcontext
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread

from stash_ai.inference.preparation import PreparedBatch, prepared_batches
from stash_ai.models import validate_scores
from stash_ai.runtime.cancellation import Cancellation


@dataclass(frozen=True)
class _Failure:
    error: BaseException


class BatchPredictions:
    """Start GPU production now; consume/checkpoint CPU results later.

    A bounded number of CPU results and one prediction can precede the consumer.
    Model calls always use the borrowed backend owner, including across scenes.
    """
    def __init__(self, app, tagger, stash, paths, cues, *, prefetch,
                 prepare_workers=1, report=None, inference_executor=None,
                 prepared_inputs=None, transform_executor=None, result_prefetch=32):
        if type(result_prefetch) is not int or not 1 <= result_prefetch <= 128:
            raise ValueError('result_prefetch must be an integer in 1..128.')
        self.report, self.model = report, tagger.settings.source.key
        self.work = report.work() if report is not None else ''
        self.tagger, self.cancellation = tagger, Cancellation()
        self.results, self.stopped = Queue(maxsize=result_prefetch), Event()
        self.finished, self.external_inputs = object(), prepared_inputs
        self.inputs = (prepared_inputs if prepared_inputs is not None else
                       prepared_batches(app, tagger, stash, paths, cues, prefetch,
                                        prepare_workers, report=report, collate=True,
                                        cancellation=self.cancellation,
                                        transform_executor=transform_executor))
        self.borrowed = inference_executor is not None
        if self.borrowed:
            self.worker = inference_executor.submit(self._produce)
        else:
            self.worker = Thread(target=self._produce, name='inference-batches')
            self.worker.start()

    def measure(self, name, count=0):
        return (self.report.measure(name, model=self.model, work=self.work, units=count,
                                    unit='frames') if self.report is not None else nullcontext())

    def _send(self, value):
        with self.measure('inference_result_queue_wait'):
            while not self.stopped.is_set():
                try:
                    self.results.put(value, timeout=.05)
                    return
                except Full:
                    pass

    def _produce(self):
        try:
            with self.report.model(self.model) if self.report is not None else nullcontext():
                if self.report is not None:
                    self.report.work(self.work)
                with closing(self.inputs) as batches:
                    for prepared, batch_cues in batches:
                        self.cancellation.check()
                        with self.measure('inference_batch', len(prepared)):
                            scores = (self.tagger.predict_collated_batch(prepared.inputs, prepared.count)
                                      if isinstance(prepared, PreparedBatch)
                                      else self.tagger.predict_batch(prepared))
                            result = validate_scores(scores, len(prepared), self.tagger.labels,
                                                     self.tagger.settings.source.title)
                        self.cancellation.check()
                        self._send((result, batch_cues))
        except BaseException as error:
            self._send(_Failure(error))
        finally:
            self._send(self.finished)

    def __iter__(self):
        return self

    def __next__(self):
        with self.measure('inference_result_wait'):
            item = self.results.get()
        if item is self.finished:
            raise StopIteration
        if isinstance(item, _Failure):
            raise item.error
        return item

    def close(self):
        self.stopped.set()
        self.cancellation.cancel()
        try:
            if self.external_inputs is not None:
                self.external_inputs.close()
        finally:
            if self.borrowed:
                self.worker.cancel()
                if not self.worker.cancelled():
                    self.worker.result()
            else:
                self.worker.join()
        # Release prefetched scores immediately after cancellation/completion.
        while not self.results.empty():
            self.results.get_nowait()


def predicted_batches(app, tagger, stash, paths, cues, *, prefetch=0,
                      prepare_workers=1, report=None, inference_executor=None,
                      prepared_inputs=None, transform_executor=None, result_prefetch=32):
    """Lazy compatibility iterator; prefetch=0 keeps caller-owned inference."""
    if prefetch:
        with closing(BatchPredictions(app, tagger, stash, paths, cues, prefetch=prefetch,
                                     prepare_workers=prepare_workers, report=report,
                                     inference_executor=inference_executor,
                                     prepared_inputs=prepared_inputs,
                                     transform_executor=transform_executor,
                                     result_prefetch=result_prefetch)) as batches:
            yield from batches
        return
    with closing(prepared_batches(app, tagger, stash, paths, cues, workers=prepare_workers,
                                  report=report, collate=True)) as batches:
        for prepared, batch_cues in batches:
            timer = (report.measure('inference_batch', model=tagger.settings.source.key,
                                    units=len(prepared), unit='frames')
                     if report is not None else nullcontext())
            with timer:
                scores = (tagger.predict_collated_batch(prepared.inputs, prepared.count)
                          if isinstance(prepared, PreparedBatch) else tagger.predict_batch(prepared))
                result = validate_scores(scores, len(prepared), tagger.labels, tagger.settings.source.title)
            yield result, batch_cues

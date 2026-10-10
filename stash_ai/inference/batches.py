"""Ordered inference lookahead with a bounded result queue and one model owner."""
from contextlib import nullcontext
from threading import Thread

from stash_ai.inference.preparation import PreparedBatch, prepared_batches
from stash_ai.models import validate_scores
from .buffer_usage import PreparedBatchUsage, queued_batches
from .managed_stream import ManagedStream, StreamFailure as _Failure
from stash_ai.runtime.cleanup import cleanup, closing


def prediction_scores(tagger, prepared):
    """Validate once at the pipeline boundary, including external adapters.

    Only the exact native types expose their internal raw calculation here;
    subclasses and injected backends retain the checked public-adapter path.
    Direct native backend calls also remain validated.
    """
    from stash_ai.backends.loader import JoyTagger
    from stash_ai.backends.onnx import OnnxTagger
    collated = isinstance(prepared, PreparedBatch)
    if type(tagger) in (JoyTagger, OnnxTagger):
        inputs = prepared.inputs if collated else tagger.collate_batch(prepared)
        scores = tagger._predict_collated_scores(inputs, len(prepared))
    else:
        scores = (tagger.predict_collated_batch(prepared.inputs, prepared.count) if collated
                  else tagger.predict_batch(prepared))
    return validate_scores(scores, len(prepared), tagger.labels, tagger.settings.source.title)


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
        self.channel = ManagedStream(result_prefetch)
        self.tagger, self.cancellation = tagger, self.channel.cancellation
        self.results, self.stopped = self.channel.queue, self.channel.stopped
        self.finished, self.external_inputs = self.channel.end, prepared_inputs
        self.input_usage = PreparedBatchUsage(prefetch)
        self.inputs = (prepared_inputs if prepared_inputs is not None else
                       prepared_batches(app, tagger, stash, paths, cues, prefetch,
                                        prepare_workers, report=report, collate=True,
                                        cancellation=self.cancellation,
                                        transform_executor=transform_executor,
                                        buffer_usage=self.input_usage))
        self.borrowed = inference_executor is not None
        if self.borrowed:
            self.worker = inference_executor.submit(self._produce)
        else:
            self.worker = Thread(target=self._produce, name='inference-batches')
            self.worker.start()
        self.channel.worker = self.worker

    def measure(self, name, count=0):
        return (self.report.measure(name, model=self.model, work=self.work, units=count,
                                    unit='frames') if self.report is not None else nullcontext())

    def buffer_usage(self):
        inputs = (self.external_inputs.buffer_usage() if self.external_inputs is not None
                  and hasattr(self.external_inputs, 'buffer_usage') else self.input_usage.snapshot())
        return dict(input=inputs, result=queued_batches(self.results))

    def _send(self, value):
        with self.measure('inference_result_queue_wait'):
            self.channel.send(value)

    def _produce(self):
        try:
            with self.report.model(self.model) if self.report is not None else nullcontext():
                if self.report is not None:
                    self.report.work(self.work)
                with closing(self.inputs, report=self.report) as batches:
                    for prepared, batch_cues in batches:
                        self.cancellation.check()
                        with self.measure('inference_batch', len(prepared)):
                            result = prediction_scores(self.tagger, prepared)
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
            item = self.channel.receive()
        if item is self.finished:
            raise StopIteration
        if isinstance(item, _Failure):
            raise item.error
        return item

    def close(self):
        callbacks = [self.channel.stop]
        if self.external_inputs is not None:
            stop = getattr(self.external_inputs, 'stop', None)
            if stop is not None:
                callbacks.append(stop)
            callbacks.append(self.external_inputs.close)
        cleanup(callbacks + [self.channel.join, self.channel.drain], report=self.report)


def predicted_batches(app, tagger, stash, paths, cues, *, prefetch=0,
                      prepare_workers=1, report=None, inference_executor=None,
                      prepared_inputs=None, transform_executor=None, result_prefetch=32,
                      on_start=None):
    """Lazy compatibility iterator; prefetch=0 keeps caller-owned inference."""
    if type(prefetch) is not int or not 0 <= prefetch <= 128:
        raise ValueError('prefetch must be an integer in 0..128.')
    if prefetch:
        with closing(BatchPredictions(app, tagger, stash, paths, cues, prefetch=prefetch,
                                     prepare_workers=prepare_workers, report=report,
                                     inference_executor=inference_executor,
                                     prepared_inputs=prepared_inputs,
                                     transform_executor=transform_executor,
                                     result_prefetch=result_prefetch), report=report) as batches:
            if on_start is not None:
                on_start(batches)
            yield from batches
        return
    with closing(prepared_batches(app, tagger, stash, paths, cues, workers=prepare_workers,
                                  report=report, collate=True), report=report) as batches:
        for prepared, batch_cues in batches:
            timer = (report.measure('inference_batch', model=tagger.settings.source.key,
                                    units=len(prepared), unit='frames')
                     if report is not None else nullcontext())
            with timer:
                result = prediction_scores(tagger, prepared)
            yield result, batch_cues

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


def predicted_batches(app, tagger, stash, paths, cues, *, prefetch=0,
                      prepare_workers=1, report=None):
    """Overlap the next prediction with ordered CPU result/checkpoint handling.

    Two queued CPU results plus one running prediction bound inference lookahead.
    The consumer owns all scene state and SQLite writes. With prefetch=0,
    prediction and result handling run in the caller; the CPU transform pool
    remains available, preserving the previous troubleshooting path.
    """
    cancellation = Cancellation()
    work = report.work() if report is not None else ''
    model = tagger.settings.source.key

    def measure(name, count=0):
        return (report.measure(name, model=model, work=work, units=count,
                               unit='frames') if report is not None else nullcontext())

    def predict(prepared):
        with measure('inference_batch', len(prepared)):
            scores = (tagger.predict_collated_batch(prepared.inputs, prepared.count)
                      if isinstance(prepared, PreparedBatch) else tagger.predict_batch(prepared))
            return validate_scores(scores, len(prepared), tagger.labels, tagger.settings.source.title)

    def inputs():
        return prepared_batches(app, tagger, stash, paths, cues, prefetch,
                                prepare_workers, report=report, collate=True,
                                cancellation=cancellation)

    if not prefetch:
        with closing(inputs()) as batches:
            for prepared, batch_cues in batches:
                yield predict(prepared), batch_cues
        return

    results = Queue(maxsize=2)
    stopped = Event()
    finished = object()

    def send(value):
        with measure('inference_result_queue_wait'):
            while not stopped.is_set():
                try:
                    results.put(value, timeout=.05)
                    return
                except Full:
                    pass

    def produce():
        try:
            with report.model(model) if report is not None else nullcontext():
                if report is not None:
                    report.work(work)
                with closing(inputs()) as batches:
                    for prepared, batch_cues in batches:
                        cancellation.check()
                        result = predict(prepared)
                        cancellation.check()
                        send((result, batch_cues))
        except BaseException as error:
            send(_Failure(error))
        finally:
            send(finished)

    worker = Thread(target=produce, name='inference-batches')
    worker.start()
    try:
        while True:
            with measure('inference_result_wait'):
                item = results.get()
            if item is finished:
                break
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        stopped.set()
        cancellation.cancel()
        worker.join()

"""Owner-local CUDA OOM recovery with deterministic, finite batch splitting."""
import logging
from contextlib import contextmanager
from contextvars import ContextVar

_sizes = ContextVar('recognition_batch_sizes', default=None)
_observer = ContextVar('recognition_gpu_batches', default=None)


@contextmanager
def observe_batches(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


@contextmanager
def batch_sizes():
    from collections import Counter
    histogram = Counter()
    token = _sizes.set(histogram)
    try:
        yield histogram
    finally:
        _sizes.reset(token)


class Batches:
    def __init__(self, maximum, kind):
        self.maximum, self.kind, self.retries = maximum, kind, 0
        self.report_batch = _observer.get()
        if self.report_batch:
            self.report_batch(kind, maximum, maximum, False)
    def run(self,values,calculate,device):
        import torch
        from stash_ai.runtime.cancellation import current
        results=[];position=0
        while position<len(values):
            token=current()
            if token is not None:token.check()
            batch=values[position:position+self.maximum]
            try:
                result=calculate(batch)
            except torch.cuda.OutOfMemoryError as error:
                if str(device).split(':')[0]!='cuda' or len(batch)==1:
                    error.add_note(f'Recognition {self.kind}: batch_size={len(batch)}; no smaller CUDA batch is available.')
                    raise
                self.maximum=max(1,(len(batch)+1)//2);self.retries+=1
                if self.report_batch:
                    self.report_batch(self.kind, len(batch), self.maximum, True)
                logging.getLogger(__name__).warning('Recognition %s CUDA OOM: retrying with batch size %s.',self.kind,self.maximum)
                from .profiling import pipeline_count
                pipeline_count('cuda_oom_retries')
            else:
                from .profiling import pipeline_count
                key=self.kind.lower().replace(' ','_').replace('-','_')
                pipeline_count(key+'_batch_'+str(len(batch)))
                histogram = _sizes.get()
                if histogram is not None:
                    histogram[self.kind, len(batch)] += 1
                results.extend(result);position+=len(batch);continue
            # Leave the except block first so the failed forward traceback no
            # longer retains device tensors. No CPU/device migration or skips.
            if torch.cuda.is_initialized():torch.cuda.empty_cache()
        return results

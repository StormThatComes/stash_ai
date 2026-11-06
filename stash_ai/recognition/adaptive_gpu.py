"""Owner-local CUDA OOM recovery with deterministic, finite batch splitting."""
import logging


class Batches:
    def __init__(self,maximum,kind):self.maximum=maximum;self.kind=kind;self.retries=0
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
                logging.getLogger(__name__).warning('Recognition %s CUDA OOM: retrying with batch size %s.',self.kind,self.maximum)
                from .profiling import pipeline_count
                pipeline_count('cuda_oom_retries')
            else:
                results.extend(result);position+=len(batch);continue
            # Leave the except block first so the failed forward traceback no
            # longer retains device tensors. No CPU/device migration or skips.
            if torch.cuda.is_initialized():torch.cuda.empty_cache()
        return results

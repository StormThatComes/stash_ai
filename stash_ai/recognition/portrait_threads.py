"""Optional native CPU limits, installed before workers and restored after join."""
from contextlib import ExitStack


def encoding_workers(config):
    budget=config.data['execution'].get('portrait_cpu_threads',0)
    return 2 if not budget else 0 if budget==1 else min(2,budget//2)


def preparation_workers(config):
    budget=config.data['execution'].get('portrait_cpu_threads',0)
    return 4 if not budget else min(8,max(0,budget-encoding_workers(config)-2))


def native_threads(config):
    budget=config.data['execution'].get('portrait_cpu_threads',0)
    return 1 if not budget else max(1,budget-encoding_workers(config)-preparation_workers(config)-1)


class NativeBudget:
    def __init__(self,config):self.config,self.stack=config,ExitStack()
    def __enter__(self):
        if not self.config.data['portraits']['enabled']:return self
        if not self.config.data['execution'].get('portrait_cpu_threads',0):
            from threadpoolctl import threadpool_limits
            self.stack.enter_context(threadpool_limits(limits=1,user_api='blas'))
            return self
        import torch
        from threadpoolctl import threadpool_limits
        previous=torch.get_num_threads()
        self.stack.callback(torch.set_num_threads,previous)
        try:
            torch.set_num_threads(native_threads(self.config))
            self.stack.enter_context(threadpool_limits(limits=1,user_api='blas'))
        except BaseException:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([('restore portrait native CPU limits',self.stack.close)])
            raise
        return self
    def __exit__(self,*error):
        from stash_ai.runtime.cleanup import cleanup
        from stash_ai.runtime.shutdown import release
        cleanup([('restore portrait native CPU limits',lambda:release(self,self.stack.close))],primary=error[1])

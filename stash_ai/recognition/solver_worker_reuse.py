"""One spawned worker owns immutable indices and a bounded numerical cache."""
from contextlib import ExitStack
from .config import digest, plain


class Reuse:
    def __init__(self):
        self.stack, self.contract, self.cache = None, None, None

    def prepare(self, request):
        contract=(request.seed_path,id(request.goals),request.execution,
                  digest(plain(request.config.data['execution'])),request.shared_directory,request.shared_bytes)
        if self.contract==contract:return True
        self.close()
        stack=ExitStack()
        try:
            from .solver_execution import scope as execution
            from .run_cache import scope as cache
            from .merge_goal_index import prepared
            from .solver_shared import scope as shared
            stack.enter_context(execution(request.execution))
            self.cache=stack.enter_context(cache(request.config))
            stack.enter_context(prepared(request.goals))
            stack.enter_context(shared(request.shared_directory,request.shared_bytes))
        except BaseException:
            stack.close()
            raise
        self.stack,self.contract=stack,contract
        return False

    def completed(self):
        if self.cache is not None:self.cache.retain_numeric()

    def close(self):
        stack,self.stack=self.stack,None
        self.contract=self.cache=None
        if stack is not None:stack.close()

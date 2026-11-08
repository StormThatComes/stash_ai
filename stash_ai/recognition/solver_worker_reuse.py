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
            from .disk_budget import scope as disk_scope
            stack.enter_context(disk_scope(request.config))
            stack.enter_context(private_session())
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


from contextlib import contextmanager
from contextvars import ContextVar

_private = ContextVar('solver_private_trial', default=None)


class PrivateTrial:
    def __init__(self):
        self.db = None

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    @contextmanager
    def borrow(self, settings, seed, observations):
        from .offline_evaluation import _ram
        from .trial_inputs import prepare
        from .solver_slot_index import Slots
        from .profiling import count
        try:
            if self.db is None:
                self.db = _ram(settings)
                prepare(self.db, seed)
                if isinstance(observations, Slots):
                    observations.attach(self.db)
                self.db.commit()
            else:
                count('trial_private_image_reuses')
            # Production rebuild borrows the enclosing transaction. Roll back
            # its changed pages before returning the connection to this worker.
            self.db.execute('SAVEPOINT isolated_trial')
            yield self.db
            self.db.execute('ROLLBACK TO isolated_trial')
            self.db.execute('RELEASE isolated_trial')
        except BaseException:
            self.close()
            raise


@contextmanager
def private_session():
    owner = PrivateTrial()
    token = _private.set(owner)
    try:
        yield
    finally:
        _private.reset(token)
        owner.close()


def current_private():
    return _private.get()

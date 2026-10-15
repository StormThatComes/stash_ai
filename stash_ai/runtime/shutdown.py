"""Bound coordinator waits while retaining resources still owned by live work.

Opaque native work cannot be killed safely in a Python thread. Quarantine keeps
its owners and dependent releases reachable and blocks subsequent inference.
Non-daemon native/executor workers may still require process termination; a
timeout reports stopping, never that the foreign operation has stopped.
"""
from concurrent.futures import TimeoutError as FutureTimeout, wait
from threading import Event, Lock, Thread

TIMEOUT = 1.
_lock = Lock()
_pending = {}
_releases = {}
_calls = {}


class ShutdownIncomplete(RuntimeError):
    """Live work retains its resources; inference cannot safely continue."""


def quarantine(done, owner, *, key=None):
    key = owner if key is None else key
    with _lock:
        _pending[id(key)] = (done, owner, key)
    raise ShutdownIncomplete('Worker shutdown is incomplete; live resources are quarantined. '
                             'No subsequent model may run until the worker exits; '
                             'an unresponsive native worker may require process termination.')


def busy():
    with _lock:
        for key, (done, _, _) in tuple(_pending.items()):
            if done():
                del _pending[key]
        for key, (_, done, _, worker) in tuple(_calls.items()):
            if done.is_set() and not worker.is_alive():
                del _calls[key]
        return bool(_pending)


def _release_one(resource, callback):
    occupied = busy()
    with _lock:
        if occupied or _pending:
            _releases[id(resource), callback] = (resource, callback)
            return False
    callback()
    return True


def release(resource, callback):
    """Run only after all timed-out work has relinquished borrowed resources."""
    if busy():
        return _release_one(resource, callback)
    with _lock:
        callbacks = list(_releases.values())
        _releases.clear()
    from .cleanup import cleanup
    cleanup([lambda owner=owner, action=action: _release_one(owner, action)
             for owner, action in [*callbacks, (resource, callback)]])
    return not busy()


def require_idle():
    if busy():
        raise ShutdownIncomplete('Earlier worker shutdown is incomplete; new inference is blocked.')
    with _lock:
        callbacks = list(_releases.values())
        _releases.clear()
    from .cleanup import cleanup
    cleanup([lambda owner=owner, action=action: _release_one(owner, action) for owner, action in callbacks])
    if busy():
        raise ShutdownIncomplete('Deferred resource cleanup is still stopping; new inference is blocked.')


def join(worker, owner, timeout=None):
    timeout = TIMEOUT if timeout is None else timeout
    if worker is None:
        return
    if hasattr(worker, 'result'):
        if worker.cancelled():
            return
        try:
            worker.result(timeout=timeout)
        except FutureTimeout:
            if not worker.done():
                quarantine(worker.done, owner)
            raise
    elif worker.ident is not None:
        worker.join(timeout=timeout)
        if worker.is_alive():
            quarantine(lambda: not worker.is_alive(), owner)


def call(callback, owner, timeout=None):
    """Bound thread-safe lifecycle waits such as executor.shutdown()."""
    with _lock:
        current = _calls.get(id(owner))
        if current is not None and current[1].is_set() and not current[3].is_alive():
            current = None
        if current is None:
            done, errors = Event(), []
            def run():
                try:
                    callback()
                except BaseException as error:
                    errors.append(error)
                finally:
                    done.set()
            worker = Thread(target=run, name='owned-shutdown', daemon=True)
            _calls[id(owner)] = owner, done, errors, worker
            try:
                worker.start()
            except BaseException:
                if worker.ident is None:
                    _calls.pop(id(owner), None)
                else:
                    _pending[id(worker)] = (lambda: not worker.is_alive(), owner, worker)
                raise
        else:
            _, done, errors, worker = current
    if not done.wait(TIMEOUT if timeout is None else timeout):
        quarantine(done.is_set, (owner, worker, errors), key=worker)
    worker.join(timeout=TIMEOUT if timeout is None else timeout)
    if worker.is_alive():
        quarantine(lambda: not worker.is_alive(), (owner, worker, errors), key=worker)
    with _lock:
        if _calls.get(id(owner), (None,))[0] is owner:
            _calls.pop(id(owner), None)
    if errors:
        raise errors[0]


def futures(items, owner):
    items = tuple(items)
    if not items:
        return
    _, pending = wait(items, timeout=TIMEOUT)
    if pending:
        quarantine(lambda: all(item.done() for item in pending), (owner, items))

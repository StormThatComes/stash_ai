"""Owned elapsed HTTP boundaries; retry policy belongs to the caller."""
from queue import Empty, Full, Queue
from threading import Event, Thread, Lock, BoundedSemaphore
from time import monotonic
from contextvars import ContextVar

from requests import Response, Timeout


class ReadDeadlineExceeded(Timeout):
    pass


MAX_ACTIVE_READS = 4
_slots = BoundedSemaphore(MAX_ACTIVE_READS)
_closing = Queue(maxsize=8)
_close_lock, _closer = Lock(), None
_owned_read = ContextVar('owned_stash_read_deadline', default=None)


def close_later(response):
    """One bounded daemon closes readers without blocking the command thread."""
    global _closer
    def close_worker():
        while True:
            value = _closing.get()
            try:
                value.close()
            except Exception:
                pass
            finally:
                _closing.task_done()
    with _close_lock:
        if _closer is None:
            worker = Thread(target=close_worker, name='stash-close-read', daemon=True)
            try:
                worker.start()
            except BaseException:
                # A start interrupted after launch still owns the queue.
                if worker.is_alive():
                    _closer = worker
                raise
            _closer = worker
    try:
        _closing.put_nowait(response)
    except Full:
        control = getattr(response, '_stash_read_control', None)
        if control is not None:
            control.cancel()


def remaining(deadline):
    value = deadline - monotonic()
    if value <= 0:
        raise ReadDeadlineExceeded('Stash read exceeded its total deadline.')
    return value


def call_with_deadline(operation, deadline, *, cancel=None, label='read'):
    """Run one owned network operation; discard late results, never retry it.

    requests' socket read timeout measures inactivity, so trickling bytes alone
    cannot enforce a wall-clock budget. A daemon performs this one blocking read;
    the caller remains bounded. Proven reads and single mutation attempts share
    the bounded worker capacity. Mutation callers own a pre-dispatch expiry gate
    and durable uncertainty handling; this helper never repeats an operation.
    """
    from stash_ai.runtime.cancellation import current, use
    from stash_ai.stash_api.read_connections import ReadControl
    owned = _owned_read.get()
    if owned is not None and owned[0] == deadline:
        _, control, callbacks, callback_lock = owned
        remaining(deadline)
        if cancel is not None:
            with callback_lock:
                callbacks.add(cancel)
        try:
            if cancel is not None and control.cancelled.is_set():
                cancel()
            token = current()
            if token is not None:
                token.check()
            value = operation()
            if isinstance(value, Response):
                value._stash_read_control = control
            remaining(deadline)
            return value
        finally:
            if cancel is not None:
                with callback_lock:
                    callbacks.discard(cancel)
    budget, completed, cancelled = remaining(deadline), Queue(maxsize=1), Event()
    started, token, control = monotonic(), current(), ReadControl()
    gate = Lock()
    slot_lock, slot_returned = Lock(), False
    callbacks, callback_lock = set(), Lock()

    def return_slot():
        nonlocal slot_returned
        with slot_lock:
            if not slot_returned:
                slot_returned = True
                _slots.release()

    def abort(primary=None):
        with callback_lock:
            active = tuple(callbacks)
        from stash_ai.runtime.cleanup import cleanup
        cleanup([control.cancel, *active, *([cancel] if cancel is not None else [])], primary=primary)

    if token is not None:
        token.subscribe(abort)

    def discard(value):
        if isinstance(value, Response):
            close_later(value)

    def run():
        try:
            from contextlib import nullcontext
            with use(token) if token is not None else nullcontext():
                with control.use():
                    owned_token = _owned_read.set((deadline, control, callbacks, callback_lock))
                    try:
                        value, failed = operation(), False
                    finally:
                        _owned_read.reset(owned_token)
            if isinstance(value, Response):
                value._stash_read_control = control
        except BaseException as exc:
            value, failed = exc, True
        finally:
            return_slot()
        with gate:
            abandoned = cancelled.is_set()
            if not abandoned:
                completed.put((failed, value))
        if abandoned:
            discard(value)

    value, succeeded = None, False
    try:
        # Even an uninterruptible resolver or foreign read cannot spawn an
        # unbounded number of workers. Waiting for capacity shares the deadline.
        while True:
            if token is not None:
                token.check()
            left = budget - (monotonic() - started)
            if left <= 0:
                raise Empty
            if _slots.acquire(timeout=min(left, .05)):
                break
        worker = None
        try:
            worker = Thread(target=run, name='stash-deadline-' + label, daemon=True)
            worker.start()
        except BaseException:
            # start() can be interrupted after the worker has actually started.
            # In that case only its finally block may return the capacity.
            if worker is None or not getattr(worker, 'is_alive', lambda: False)():
                return_slot()
            raise
        while True:
            if token is not None:
                token.check()
            left = budget - (monotonic() - started)
            if left <= 0:
                raise Empty
            try:
                failed, value = completed.get(timeout=min(left, .05) if token is not None else left)
                break
            except Empty:
                if token is None:
                    raise
        if token is not None:
            token.check()
        if failed:
            raise value
        remaining(deadline)
        succeeded = True
        return value
    except Empty:
        error = ReadDeadlineExceeded('Stash read exceeded its total deadline.')
        abort(error)
        raise error from None
    except (ReadDeadlineExceeded, KeyboardInterrupt):
        abort()
        raise
    finally:
        if token is not None:
            token.unsubscribe(abort)
        with gate:
            cancelled.set()
            late = completed.get_nowait()[1] if not completed.empty() else None
        discard(late)
        if not succeeded:
            discard(value)


def abort_response(response):
    """Interrupt a streaming body before closing its buffered reader."""
    from stash_ai.stash_api.read_connections import shutdown
    control = getattr(response, '_stash_read_control', None)
    if control is not None:
        control.cancel()
    connection = getattr(getattr(response, 'raw', None), 'connection', None)
    if connection is not None:
        shutdown(connection.sock)
    # Closing a buffered reader can wait for a read lock; never block the caller.
    close_later(response)

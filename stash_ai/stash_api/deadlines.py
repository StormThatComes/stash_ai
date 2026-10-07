"""Hard caller deadlines for proven reads, including a trickling response body."""
from queue import Empty, Full, Queue
from threading import Event, Thread, Lock, BoundedSemaphore
from time import monotonic

from requests import Response, Timeout


class ReadDeadlineExceeded(Timeout):
    pass


MAX_ACTIVE_READS = 4
_slots = BoundedSemaphore(MAX_ACTIVE_READS)
_closing = Queue(maxsize=8)
_close_lock, _closer = Lock(), None


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
            _closer = Thread(target=close_worker, name='stash-close-read', daemon=True)
            _closer.start()
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


def call_with_deadline(operation, deadline, *, cancel=None):
    """Only a read runs here. A late result is discarded and closed, never applied.

    requests' socket read timeout measures inactivity, so trickling bytes alone
    cannot enforce a wall-clock budget. A daemon performs this one blocking read;
    the caller remains bounded. It never schedules retries or any writes.
    """
    from stash_ai.runtime.cancellation import current, use
    from stash_ai.stash_api.read_connections import ReadControl
    budget, completed, cancelled = remaining(deadline), Queue(maxsize=1), Event()
    started, token, control = monotonic(), current(), ReadControl()
    gate = Lock()

    def abort():
        control.cancel()
        if cancel is not None:
            cancel()

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
                    value, failed = operation(), False
            if isinstance(value, Response):
                value._stash_read_control = control
        except BaseException as exc:
            value, failed = exc, True
        finally:
            _slots.release()
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
        Thread(target=run, name='stash-deadline-read', daemon=True).start()
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
        abort()
        raise ReadDeadlineExceeded('Stash read exceeded its total deadline.') from None
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
    connection = getattr(response.raw, 'connection', None)
    if connection is not None:
        shutdown(connection.sock)
    # Closing a buffered reader can wait for a read lock; never block the caller.
    close_later(response)

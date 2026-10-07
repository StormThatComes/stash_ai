"""Hard caller deadlines for proven reads, including a trickling response body."""
from queue import Empty, Queue
from threading import Event, Thread, Lock
from time import monotonic

from requests import Response, Timeout


class ReadDeadlineExceeded(Timeout):
    pass


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
    budget, completed, cancelled = remaining(deadline), Queue(maxsize=1), Event()
    gate = Lock()

    def discard(value):
        if isinstance(value, Response):
            Thread(target=value.close, name='stash-close-read', daemon=True).start()

    def run():
        try:
            value, failed = operation(), False
        except BaseException as exc:
            value, failed = exc, True
        with gate:
            abandoned = cancelled.is_set()
            if not abandoned:
                completed.put((failed, value))
        if abandoned:
            discard(value)

    Thread(target=run, name='stash-deadline-read', daemon=True).start()
    value, succeeded = None, False
    try:
        failed, value = completed.get(timeout=budget)
        if failed:
            raise value
        remaining(deadline)
        succeeded = True
        return value
    except Empty:
        if cancel is not None:
            cancel()
        raise ReadDeadlineExceeded('Stash read exceeded its total deadline.') from None
    except (ReadDeadlineExceeded, KeyboardInterrupt):
        if cancel is not None:
            cancel()
        raise
    finally:
        with gate:
            cancelled.set()
            late = completed.get_nowait()[1] if not completed.empty() else None
        discard(late)
        if not succeeded:
            discard(value)


def abort_response(response):
    """Interrupt a streaming body before closing its buffered reader."""
    import socket
    connection = response.raw.connection
    if connection is not None and connection.sock is not None:
        try:
            connection.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    # Closing a buffered reader can wait for a read lock; never block the caller.
    Thread(target=response.close, name='stash-close-read', daemon=True).start()

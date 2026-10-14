"""Register submitted-task ownership before its operation can acquire resources."""
from threading import Event
from .cleanup import cleanup


def submit_owned(executor, operation, register):
    ready, aborted = Event(), Event()
    future = None
    def run():
        ready.wait()
        if not aborted.is_set():
            return operation()
    try:
        future = executor.submit(run)
        register(future)
        return future
    except BaseException:
        aborted.set()
        cleanup([future.cancel] if future is not None else [])
        raise
    finally:
        ready.set()

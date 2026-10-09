"""A failed read-only job must not wait indefinitely for other workers."""
from contextlib import contextmanager

from .cleanup import cleanup


@contextmanager
def read_process_pool(factory, **options):
    executor = factory(**options)
    try:
        yield executor
        executor.shutdown(wait=True)
    except BaseException:
        # Supported Python 3.14 provides a public, immediate termination API.
        # Read-only workers hold no publication or state-write ownership.
        cleanup([executor.kill_workers])
        raise

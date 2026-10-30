"""Bound cooperative shutdown before stopping unresponsive trial processes."""
from concurrent.futures import wait

GRACE_SECONDS = 2.


def shutdown(executor, futures):
    futures = tuple(future for future in futures if future is not None)
    _, pending = wait(futures, timeout=GRACE_SECONDS) if futures else ((), ())
    if pending:
        # Trials have private databases; completed results were harvested before
        # cancellation. Python 3.14 owns process termination and queue cleanup.
        executor.kill_workers()
        _, pending = wait(futures, timeout=GRACE_SECONDS)
        if pending:
            raise RuntimeError('Solver workers did not exit after forced shutdown.')
    else:
        executor.shutdown(wait=True, cancel_futures=True)

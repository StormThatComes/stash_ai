"""A real spawn worker deliberately never completes its read-only operation."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Event, active_children, get_context
import os
from pathlib import Path
import sys

from stash_ai.runtime.process_pool import read_process_pool


def blocked(path):
    Path(path).write_text(str(os.getpid()))
    Event().wait()


if __name__ == '__main__':
    try:
        with read_process_pool(ProcessPoolExecutor, max_workers=1,
                               mp_context=get_context('spawn')) as pool:
            pool.submit(blocked, sys.argv[1]).result()
    except KeyboardInterrupt:
        # Multiprocessing's active_children also reaps finished children.
        for child in active_children():
            child.join(3)
        assert not active_children()

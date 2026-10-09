"""One advisory writer-lock protocol for state DB writers and maintenance."""
from contextlib import contextmanager
import os


def acquire_writer_lock(path):
    lock = open(str(path) + '.lock', 'a+b')
    try:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another merged-output process is using this state DB.') from exc
        return lock
    except BaseException:
        lock.close()
        raise


@contextmanager
def writer_lock(path):
    lock = acquire_writer_lock(path)
    try:
        yield
    finally:
        lock.close()

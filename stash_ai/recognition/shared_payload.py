"""Immutable process input in a parent-owned shared segment."""
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
import errno
import os
import sys


@dataclass(frozen=True)
class Rows:
    name: str
    metadata: tuple


def pack(rows):
    size=sum(len(row[-1]) for row in rows if row[-1] is not None)
    if not size:
        return rows, None
    try:
        segment = SharedMemory(create=True, size=size)
    except OSError:
        return rows, None
    # Linux tmpfs can accept ftruncate even when writing the mapped pages
    # would raise SIGBUS. Commit capacity before touching the mapping.
    if sys.platform.startswith('linux') and hasattr(os, 'posix_fallocate'):
        try:
            os.posix_fallocate(segment._fd, 0, size)
        except OSError as error:
            release(segment)
            if error.errno in (errno.ENOSPC, errno.ENOMEM, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOSYS):
                return rows, None
            raise
    try:
        offset=0
        records=[]
        for row in rows:
            data=row[-1]
            if data is None:
                records.append(tuple(row[:-1]) + (offset, -1))
                continue
            segment.buf[offset:offset+len(data)]=data
            records.append(tuple(row[:-1])+(offset,len(data)))
            offset+=len(data)
        return Rows(segment.name,tuple(records)),segment
    except BaseException:
        segment.close();segment.unlink()
        raise


@contextmanager
def unpack(rows):
    if not isinstance(rows,Rows):
        yield rows
        return
    segment=SharedMemory(name=rows.name)
    try:
        # SQLite owns each inserted BLOB; do not retain borrowed buffer views.
        yield (row[:-2]+(None if row[-1] < 0 else bytes(segment.buf[row[-2]:row[-2]+row[-1]]),) for row in rows.metadata)
    finally:
        segment.close()


def release(segment):
    try:
        segment.close()
    finally:
        try:
            segment.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def borrowed(rows):
    """Read-only zero-copy embedding buffers, owned for the entire worker call."""
    if not isinstance(rows, Rows):
        yield rows
        return
    segment = SharedMemory(name=rows.name)
    buffers = []
    try:
        values = []
        for row in rows.metadata:
            if row[-1] < 0:
                values.append(row[:-2] + (None,))
                continue
            view = segment.buf[row[-2]:row[-2]+row[-1]].toreadonly()
            buffers.append(view)
            values.append(row[:-2]+(view,))
        yield values
    finally:
        for view in buffers:
            view.release()
        segment.close()

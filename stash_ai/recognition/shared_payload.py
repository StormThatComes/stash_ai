"""Immutable process input in a parent-owned shared segment."""
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory


@dataclass(frozen=True)
class Rows:
    name: str
    metadata: tuple


def pack(rows):
    size=sum(len(row[-1]) for row in rows)
    if not size:
        return rows, None
    segment=SharedMemory(create=True,size=size)
    try:
        offset=0
        records=[]
        for row in rows:
            data=row[-1]
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
        yield (row[:-2]+(bytes(segment.buf[row[-2]:row[-2]+row[-1]]),) for row in rows.metadata)
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

"""Explicit read-only cursor streams, with bounded SQLite fetch overhead."""


def rows(db,sql,parameters=(),*,batch_size=64):
    # Opt-in at read-only loops; never prefetch cursors whose table is mutated
    # between rows or change fetchone()/cursor iteration semantics globally.
    cursor=db.execute(sql,parameters)
    try:
        while batch:=cursor.fetchmany(batch_size):
            yield from batch
    finally:
        cursor.close()

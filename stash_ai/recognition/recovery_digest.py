"""Transactional dirty ranges for convergence and exact cycle detection."""
from contextlib import contextmanager
from itertools import groupby
import hashlib

from .database import encoded
from .profiling import count

TABLES=(("candidates","id",("id","person","status")),
        ("identity_merges","source",("source","target")))


def _digest(rows):
    digest=hashlib.sha256()
    for row in rows:
        digest.update(encoded(tuple(row)).encode())
    return digest.digest()


@contextmanager
def snapshots(db):
    # Ordered key ranges work with the existing primary-key indexes. Replacing
    # or rolling back a row dirties exactly the ranges SQLite actually changed.
    parts={}
    triggers=[]
    db.execute("CREATE TEMP TABLE recovery_ranges(t TEXT,k TEXT,PRIMARY KEY(t,k))")
    try:
        for table,key,fields in TABLES:
            rows=db.execute(f"SELECT {','.join(fields)} FROM {table} ORDER BY {key}")
            for prefix,group in groupby(rows,key=lambda row:row[0][:2]):
                parts[table,prefix]=_digest(group)
            for event,sides in (("INSERT",("new",)),("DELETE",("old",)),("UPDATE",("old","new"))):
                trigger=f"recovery_range_{table}_{event}"
                when=(" WHEN "+" OR ".join(f"new.{f} IS NOT old.{f}" for f in fields)) if event=="UPDATE" else ""
                mark="".join(f"INSERT OR IGNORE INTO recovery_ranges VALUES ('{table}',substr({side}.{key},1,2));" for side in sides)
                db.execute(f"CREATE TEMP TRIGGER {trigger} AFTER {event} ON {table}{when} BEGIN {mark} END")
                triggers.append(trigger)
        def snapshot():
            for table,key,fields in TABLES:
                for prefix, in db.execute("SELECT k FROM recovery_ranges WHERE t=?",(table,)):
                    sql=f"SELECT {','.join(fields)} FROM {table} WHERE {key}>=? AND substr({key},1,2)=?"
                    args=[prefix,prefix]
                    if prefix and ord(prefix[-1])<0x10ffff:
                        sql+=f" AND {key}<?"
                        args.append(prefix[:-1]+chr(ord(prefix[-1])+1))
                    value=_digest(db.execute(sql+f" ORDER BY {key}",args))
                    if value==hashlib.sha256().digest():parts.pop((table,prefix),None)
                    else:parts[table,prefix]=value
                    count('recovery_ranges_refreshed')
            db.execute("DELETE FROM recovery_ranges")
            digest=hashlib.sha256()
            for key,value in sorted(parts.items()):
                digest.update(encoded(key).encode()+value)
            return digest.hexdigest()
        yield snapshot
    finally:
        for trigger in triggers:db.execute(f"DROP TRIGGER {trigger}")
        db.execute("DROP TABLE recovery_ranges")


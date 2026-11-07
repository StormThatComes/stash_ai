"""Shared immutable policy/admission documents for durable candidate rows."""
import hashlib
import json

FIELDS=('policy','representative_limit','resource_limits')
SCHEMA='''
CREATE TABLE decision_contexts(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE INDEX candidates_context ON candidates(CASE WHEN json_valid(payload) THEN json_extract(payload,'$.decision_context') END);
'''


class Packer:
    """Scoped to one assignment iterator/transaction, so rollback cannot leave cached IDs."""
    def __init__(self,db):
        self.db=db
        self.enabled=db.execute("SELECT 1 FROM sqlite_master WHERE name='decision_contexts'").fetchone()
        self.last=None
        self.key=None

    def __call__(self,payload):
        from .database import encoded
        if not self.enabled:return payload
        value=json.loads(payload)
        shared={k:value.pop(k) for k in FIELDS if k in value}
        if not shared:return payload
        if shared!=self.last:
            raw=encoded(shared);self.key=hashlib.sha256(raw.encode()).hexdigest()
            self.db.execute('INSERT OR IGNORE INTO decision_contexts VALUES (?,?)',(self.key,raw))
            self.last=shared
        value['decision_context']=self.key
        return encoded(value)


def pack(db,payload):
    return Packer(db)(payload)


def unpack(db,payload):
    value=json.loads(payload)
    key=value.pop('decision_context',None)
    if key is not None:
        row=db.execute('SELECT payload FROM decision_contexts WHERE id=?',(key,)).fetchone()
        if row is None or hashlib.sha256(row[0].encode()).hexdigest()!=key:
            raise ValueError('Missing or corrupt candidate decision context.')
        value=dict(json.loads(row[0]),**value)
    return value


def limits_sql(db,alias='c'):
    inline=f"json_extract({alias}.payload,'$.resource_limits')"
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='decision_contexts'").fetchone():return inline
    return (f"coalesce({inline},(SELECT json_extract(payload,'$.resource_limits') FROM decision_contexts "
            f"WHERE id=json_extract({alias}.payload,'$.decision_context')))")


def collect(db):
    db.execute("DELETE FROM decision_contexts WHERE NOT EXISTS (SELECT 1 FROM candidates "
               "WHERE CASE WHEN json_valid(payload) THEN json_extract(payload,'$.decision_context') END=decision_contexts.id)")

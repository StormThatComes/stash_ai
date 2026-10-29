"""Warm body rebuilds require unchanged global evidence, owners and aliases."""
import json
from .config import digest,plain
from .database import encoded
from .dependencies import version, supported


def key(selected,config):
    from .body_identity import policy
    return 'body-rebuild:'+digest([sorted((i['scan'],i['scene']) for i in selected),
                                  plain(config.data),policy(config)])


def cached(db,selected,config):
    if not supported(db):return False
    row=db.execute('SELECT value FROM metadata WHERE key=?',(key(selected,config),)).fetchone()
    return bool(row and json.loads(row[0])==dict(format=1,generation=version(db)))


def stamp(db,selected,config):
    if supported(db):
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   (key(selected,config),encoded(dict(format=1,generation=version(db)))))
        db.execute("DELETE FROM metadata WHERE key LIKE 'body-rebuild:%' AND rowid NOT IN "
                   "(SELECT rowid FROM metadata WHERE key LIKE 'body-rebuild:%' ORDER BY rowid DESC LIMIT 8)")

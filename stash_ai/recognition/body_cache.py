"""Warm body rebuilds require unchanged global evidence, owners and aliases."""
import json
from .config import digest,plain
from .database import encoded
from .dependencies import version, supported


def key(selected,config):
    from .body_identity import policy
    return 'body-rebuild:'+digest([sorted((i['scan'],i['scene']) for i in selected),
                                  matching_settings(config),policy(config)])


def matching_settings(config):
    """Recognition dependencies only; portrait/storage/execution tuning is unrelated."""
    from .scanning import scan_identity
    from .models import namespace
    person = config.data['sapiensid']
    return ['profile-recovery-v3-global-fragments',scan_identity(config, namespace(config)), plain(config.data['recognition']),
            plain(config.data['clustering']), person['enabled'], plain(person['identities']),
            plain(person['matching']), plain(person['global_matching']), plain(person['execution'])]


def cached(db,selected,config):
    if not supported(db):return False
    if db.execute('PRAGMA user_version').fetchone()[0]>=17 and selected:
        scope,namespace = db.execute('SELECT scope,namespace FROM scans WHERE id=?',(selected[0]['scan'],)).fetchone()
        if db.execute('SELECT 1 FROM pending_body_merges WHERE scope=? AND namespace=? AND NOT EXISTS '
                '(SELECT 1 FROM json_each(scenes) WHERE value NOT IN (SELECT value FROM json_each(?))) LIMIT 1',
                (scope,namespace,encoded([s['scene'] for s in selected]))).fetchone():
            return False
    row=db.execute('SELECT value FROM metadata WHERE key=?',(key(selected,config),)).fetchone()
    return bool(row and json.loads(row[0])==dict(format=2,generation=version(db)))


def stamp(db,selected,config):
    if supported(db):
        db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                   (key(selected,config),encoded(dict(format=2,generation=version(db)))))
        db.execute("DELETE FROM metadata WHERE key LIKE 'body-rebuild:%' AND rowid NOT IN "
                   "(SELECT rowid FROM metadata WHERE key LIKE 'body-rebuild:%' ORDER BY rowid DESC LIMIT 8)")

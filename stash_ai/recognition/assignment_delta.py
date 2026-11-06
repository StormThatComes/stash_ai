"""Apply selected replay differences without rewriting unchanged membership rows."""
import json
from .database import encoded


def prune(db,replay,selected,scope,namespace,policy):
    changed=set()
    db.execute('CREATE TEMP TABLE desired_members(observation TEXT PRIMARY KEY)')
    db.execute('CREATE TEMP TABLE desired_candidates(id TEXT PRIMARY KEY)')
    try:
        db.executemany('INSERT INTO desired_members VALUES (?)',replay.execute('SELECT observation FROM members'))
        db.executemany('INSERT INTO desired_candidates VALUES (?)',replay.execute('SELECT id FROM candidates WHERE policy=?',(policy,)))
        for item in selected:
            deleted=db.execute('DELETE FROM members WHERE observation IN (SELECT o.id FROM observations o JOIN scans s '
                'ON s.id=o.scan WHERE s.scope=? AND s.namespace=? AND s.scene=?) '
                'AND observation NOT IN (SELECT observation FROM desired_members)',(scope,namespace,item['scene'])).rowcount
            deleted+=db.execute('DELETE FROM candidates WHERE scan=? AND policy=? AND id NOT IN '
                '(SELECT id FROM desired_candidates)',(item['scan'],policy)).rowcount
            if deleted:changed.add(item['scene'])
    finally:
        db.execute('DROP TABLE desired_members');db.execute('DROP TABLE desired_candidates')
    return changed


def candidates(db,rows,changed):
    from .reference_roles import observation_sql
    for row in rows:
        old=db.execute('SELECT * FROM candidates WHERE id=?',(row[0],)).fetchone()
        fresh=json.loads(row[5]);observation=fresh['observation']
        latest=db.execute('SELECT id FROM candidates c WHERE '+observation_sql('c')+'=? ORDER BY rowid DESC LIMIT 1',(observation,)).fetchone()
        same=False
        if old is not None and tuple(old)[:5]==row[:5]:
            try:
                prior=json.loads(old['payload'])
                for key in ('reference_context','reference_selected'):
                    prior.pop(key,None);fresh.pop(key,None)
                same=prior==fresh
            except (ValueError,TypeError,AttributeError):pass
        # Refresh precedence across policies even when the payload is identical.
        if not same or not latest or latest[0]!=row[0]:
            db.execute('DELETE FROM candidates WHERE id=?',(row[0],))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',row)
            changed.add(db.execute('SELECT scene FROM scans WHERE id=?',(row[1],)).fetchone()[0])
        yield row


def enqueue(db,scope,namespace,item,config,changed,reason):
    """An empty first result or new policy is still a publication decision."""
    from .config import digest
    from .publication_queue import enqueue as publish,matching_policy,body_policy
    key='recognition-publication-input:'+digest([scope,namespace,item['scene']])
    value=encoded([item['scan'],matching_policy(config),body_policy(config)])
    old=db.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
    altered=old is None or old[0]!=value
    if changed or altered:publish(db,scope,namespace,item['scene'],config,reason)
    if altered:db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',(key,value))

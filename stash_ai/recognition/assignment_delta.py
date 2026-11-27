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
    from .decision_contexts import unpack,Packer
    pack=Packer(db)
    from itertools import islice
    rows = iter(rows)
    while batch := list(islice(rows, 256)):
        payloads = [json.loads(row[5]) for row in batch]
        old = {r[0]:r for r in db.execute("SELECT * FROM candidates WHERE id IN (SELECT value FROM json_each(?))",
                                          (encoded([row[0] for row in batch]),))}
        obs = observation_sql('c')
        latest = dict(db.execute("SELECT value,(SELECT id FROM candidates c WHERE "+obs+
                                 "=j.value ORDER BY rowid DESC LIMIT 1) FROM json_each(?) j",
                                 (encoded([v['observation'] for v in payloads]),)))
        scenes = dict(db.execute("SELECT id,scene FROM scans WHERE id IN (SELECT value FROM json_each(?))",
                                 (encoded(sorted({row[1] for row in batch})),)))
        distinct = len({row[0] for row in batch}) == len(batch)
        writes = []
        for row,fresh in zip(batch,payloads):
            prior_row = old.get(row[0])
            observation = fresh['observation']
            same = False
            if prior_row is not None and tuple(prior_row)[:5] == row[:5]:
                try:
                    prior = unpack(db,prior_row[5])
                    for key in ('reference_context','reference_selected'):
                        prior.pop(key,None)
                        fresh.pop(key,None)
                    same = prior == fresh
                except (ValueError, TypeError, AttributeError):
                    pass
            if not same or latest.get(observation) != row[0]:
                packed = (*row[:5],pack(row[5]))
                if distinct:
                    writes.append(packed)
                else:
                    db.execute('DELETE FROM candidates WHERE id=?',(row[0],))
                    db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',packed)
                old[row[0]] = packed
                latest[observation] = row[0]
                changed.add(scenes[row[1]])
        if writes:
            db.execute('DELETE FROM candidates WHERE id IN (SELECT value FROM json_each(?))',
                       (encoded([r[0] for r in writes]),))
            db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)',writes)
        yield from batch


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

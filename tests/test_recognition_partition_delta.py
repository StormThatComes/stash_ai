import random
import pytest

from stash_ai.recognition.replay_simulation import impact
from stash_ai.recognition.partition_delta import PartitionImpact, Predictions


@pytest.mark.parametrize('fraction', [0, .01, .1, 1])
def test_delta_metrics_match_full_partition(fraction):
    rng = random.Random(14871)
    people = [None, 'a', 'b', 'c']
    retained = {i: rng.choice(people) for i in range(401)}
    baseline = {i: rng.choice(people) for i in range(400)}
    changes = {i: rng.choice(people + ['new']) for i in baseline if rng.random() < fraction}
    after = Predictions(baseline, changes)
    for excluded in (set(), {0, 1, 50}):
        expected = impact({k:v for k,v in retained.items() if k not in excluded},
                          {k:v for k,v in after.items() if k not in excluded})
        assert PartitionImpact(retained, baseline, excluded).measure(changes) == expected


def test_delta_records_deletion_new_owner_and_seed_difference(tmp_path):
    import sqlite3
    from types import SimpleNamespace
    from stash_ai.recognition.solver_slot_index import Slots, track
    cfg = SimpleNamespace(data={'sapiensid': {'enabled': False}})
    with sqlite3.connect(tmp_path/'slots.db') as index, sqlite3.connect(':memory:', uri=True) as db:
        index.execute('CREATE TABLE slots(ordinal INTEGER PRIMARY KEY,id TEXT UNIQUE,scene TEXT,retained TEXT,baseline TEXT)')
        index.execute('CREATE TABLE input_counts(body_enabled INTEGER)')
        index.execute('INSERT INTO input_counts VALUES (0)')
        index.executemany('INSERT INTO slots VALUES (?,?,?,?,?)',
                          [(1,'a','1','old','base'), (2,'b','1','left','left'), (3,'c','2',None,None),
                           (4,'d','2',None,None), (5,'e','2',None,None)])
        index.commit()
        db.execute('CREATE TABLE members(observation TEXT PRIMARY KEY,person TEXT)')
        db.execute('CREATE TABLE body_members(observation TEXT PRIMARY KEY,person TEXT,status TEXT)')
        db.executemany('INSERT INTO members VALUES (?,?)',[('a','old'),('b','left')])
        slots = Slots(index,'baseline',5)
        slots.attach(db)
        track(db)
        db.execute("DELETE FROM members WHERE observation='b'")
        db.execute("INSERT INTO members VALUES ('c','new')")
        assert slots.prediction_delta(db,cfg) == {'a':'old','b':None,'c':'new'}


def test_body_report_delta_counts_reason_only_and_removed_memberships():
    from collections import Counter
    import sqlite3
    from stash_ai.recognition.solver_body_counts import groups, incremental, report
    with sqlite3.connect(':memory:') as db, sqlite3.connect(':memory:') as index:
        db.execute("ATTACH DATABASE ':memory:' AS retained")
        for schema in ('main','retained'):
            db.execute(f'CREATE TABLE {schema}.body_members(observation TEXT PRIMARY KEY,person TEXT,status TEXT,role TEXT,proof TEXT)')
        db.execute('CREATE TABLE body_observations(id TEXT PRIMARY KEY,scan TEXT,contract TEXT)')
        db.executemany('INSERT INTO body_observations VALUES (?,\'s\',\'policy\')', [('a',),('b',),('c',)])
        db.executemany('INSERT INTO retained.body_members VALUES (?,?,?,?,?)', [
            ('a','p','accepted','anchor','{"reason":"old"}'), ('b','p','accepted','auxiliary','{}')])
        db.execute('INSERT INTO main.body_members SELECT * FROM retained.body_members')
        db.execute("UPDATE main.body_members SET proof='{}' WHERE observation='a'")
        db.execute("DELETE FROM main.body_members WHERE observation='b'")
        db.execute("INSERT INTO main.body_members VALUES ('c',NULL,'withheld','auxiliary','{}')")
        index.execute('CREATE TABLE retained_body_groups(owner,status,role,reason,amount)')
        index.executemany('INSERT INTO retained_body_groups VALUES (?,?,?,?,?)',groups(db,['s'],'policy',source='retained'))
        db.execute('CREATE TEMP TABLE trial_changed(id PRIMARY KEY)')
        db.executemany('INSERT INTO trial_changed VALUES (?)',[('a',),('b',),('c',)])
        coverage = dict(contract='policy')
        expected = report(Counter({tuple(r[:4]):r[4] for r in groups(db,['s'],'policy')}),coverage)
        assert incremental(index,db,coverage,['s'],100) == expected
        assert expected['reasons'] == {'unknown':2}


def test_indexed_outsiders_apply_moves_to_and_from_goal_destinations():
    import sqlite3
    from stash_ai.recognition.partition_delta import Predictions
    from stash_ai.recognition.solver_slot_index import Slots
    with sqlite3.connect(':memory:') as index:
        index.execute('CREATE TABLE slots(id PRIMARY KEY,baseline,retained,goal)')
        index.execute('CREATE TABLE outsiders(owner,retained,amount)')
        index.executemany('INSERT INTO slots VALUES (?,?,?,?)',[
            ('q','A','R',1),('old','A','R',0),('new','B',None,0),('lost','A',None,0)])
        index.execute('INSERT INTO outsiders SELECT baseline,retained,COUNT(*) FROM slots WHERE goal=0 GROUP BY baseline,retained')
        delta = Predictions(Slots(index,'baseline',4),{'q':None,'old':'B','new':'A','lost':None})
        assert +delta.outsider_counts({'A'}) == {None:1}
        assert +delta.outsider_counts({'B'}) == {'R':1}

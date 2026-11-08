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
                          [(1,'a','1','old','base'), (2,'b','1','left','left'), (3,'c','2',None,None)])
        index.commit()
        db.execute('CREATE TABLE members(observation TEXT PRIMARY KEY,person TEXT)')
        db.execute('CREATE TABLE body_members(observation TEXT PRIMARY KEY,person TEXT,status TEXT)')
        db.executemany('INSERT INTO members VALUES (?,?)',[('a','old'),('b','left')])
        slots = Slots(index,'baseline',3)
        slots.attach(db)
        track(db)
        db.execute("DELETE FROM members WHERE observation='b'")
        db.execute("INSERT INTO members VALUES ('c','new')")
        assert slots.prediction_delta(db,cfg) == {'a':'old','b':None,'c':'new'}

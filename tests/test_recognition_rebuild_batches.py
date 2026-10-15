import sqlite3

import numpy as np
import pytest

from stash_ai.recognition.clustering import cluster_scene, matching_policy
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.rebuild import _inserted, _results
from tests.test_recognition_clustering import setup, scan


def test_support_uses_independent_times_and_avoids_per_observation_member_reads(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as database:
        identifier, _ = scan(database, config, '1', [np.array([1., 0, 0, 0])] * 3
                              + [np.array([0., 1, 0, 0])] * 3, count=6)
        statements = []
        database.connection.set_trace_callback(statements.append)
        result = cluster_scene(database, identifier, config)
        assert sorted(row['observations'] for row in result if row['status'] == 'accepted') == [3, 3]
        assert not any(sql.startswith('SELECT person FROM members WHERE observation=') for sql in statements)
        assert sum('COUNT(DISTINCT o.seconds)' in sql for sql in statements) == 1


def test_bounded_result_collection_keeps_empty_scans_and_null_person_groups(tmp_path):
    config = setup(tmp_path)
    with FaceDatabase(config.database) as database:
        db = database.connection
        selected = [{'scene': str(i), 'scan': 's' + str(i)} for i in range(260)]
        for item in selected:
            db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (item['scan'], 'scope', item['scene'], 'ns', 'input', 'policy', '{}', '{}', 0, 0, 'complete'))
        for index, person in enumerate((None, 'b', 'a')):
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                       (str(index), 's0', matching_policy(config), person, 'withheld', encoded({})))
        statements = []
        db.set_trace_callback(statements.append)
        result = _results(db, selected, config)
        assert list(result) == [item['scan'] for item in selected]
        assert result['s0'] == [{'person': p, 'status': 'withheld', 'observations': 1} for p in (None, 'a', 'b')]
        assert all(not result[item['scan']] for item in selected[1:])
        assert sum(sql.startswith('SELECT scan,person,status,COUNT(*)') for sql in statements) == 2


def test_later_sql_batch_failure_rolls_back_earlier_batches():
    db = sqlite3.connect(':memory:')
    try:
        db.execute('CREATE TABLE items(id INTEGER CHECK(id<>300))')
        with pytest.raises(sqlite3.IntegrityError):
            with db:
                list(_inserted(db, 'INSERT INTO items VALUES (?)', ((i,) for i in range(700))))
        assert db.execute('SELECT COUNT(*) FROM items').fetchone()[0] == 0
    finally:
        db.close()

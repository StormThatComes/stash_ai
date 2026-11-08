from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sqlite3
import numpy as np

from stash_ai.recognition.solver_shared import Arrays, scope, numerical, key


def _read_shared(directory):
    array = np.arange(64, dtype='float64').reshape(8, 8)
    with scope(directory, 16384):
        result = numerical('test', (array,), lambda: array @ array.T)
        return result.tolist(), result.flags.writeable, isinstance(result, np.memmap)


def test_shared_arrays_are_exact_readonly_and_reusable_across_processes(tmp_path):
    Arrays.prepare(tmp_path)
    first = _read_shared(str(tmp_path))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_read_shared, [str(tmp_path)]*4))
    assert all(r == first for r in results)
    assert first[1:] == (False, True)
    assert len(list(tmp_path.glob('*.npy'))) == 1


def test_shared_storage_budget_and_content_generation(tmp_path):
    Arrays.prepare(tmp_path)
    array = np.arange(8, dtype='float64')
    with scope(tmp_path, 600) as shared:
        original = numerical('test', (array,), lambda: array*2)
        mutated = array.copy(); mutated[0] = 100
        changed = numerical('test', (mutated,), lambda: mutated*2)
        assert original[0] == 0 and changed[0] == 200
        assert key('test', (array,)) != key('test', (mutated,))
        assert key('test', (array,)) != key('test', (array.reshape(2, 4),))
        assert shared.db.execute('SELECT SUM(bytes) FROM entries').fetchone()[0] <= 600
    assert len(list(tmp_path.glob('*.npy'))) == 1


def test_numeric_cache_never_reuses_decisions_or_partial_files(tmp_path):
    Arrays.prepare(tmp_path)
    array = np.ones(8, dtype='float64')
    with scope(tmp_path, 10000):
        first = numerical('scores', (array,), lambda: array*2, extra=('float64',))
        def unexpected():
            raise AssertionError('numerical evidence was recomputed')
        assert np.array_equal(first, numerical('scores', (array,), unexpected, extra=('float64',)))
        other = numerical('scores', (array,), lambda: array*3, extra=('different-rule',))
        assert other[0] == 3


def test_empty_banks_have_a_stable_content_key():
    import numpy as np
    from stash_ai.recognition.solver_shared import key
    assert key('empty', (np.empty((0, 128), dtype='float64'),)) == key('empty', (np.empty((0, 128), dtype='float64'),))

def test_busy_reservation_can_retry_without_disabling_worker(tmp_path):
    Arrays.prepare(tmp_path)
    with scope(tmp_path, 65536) as shared, sqlite3.connect(tmp_path/'ledger.sqlite') as other:
        shared.db.execute('PRAGMA busy_timeout=1')
        other.execute('BEGIN IMMEDIATE')
        value = np.arange(8)
        signature = key('lock', (value,))
        assert shared.put(signature, value) is value
        assert not shared.disabled
        other.rollback()
        shared.put(signature, value)
        assert shared.get(signature) is not None
        assert shared.db.execute('SELECT bytes FROM usage').fetchone()[0] == value.nbytes + 512


def test_trial_template_restores_private_state_and_shared_views(tmp_path):
    from contextlib import closing
    from stash_ai.recognition.database import SCHEMA
    from stash_ai.recognition.trial_inputs import prepare
    from stash_ai.recognition.solver_images import session, current
    with closing(sqlite3.connect(tmp_path/'seed.db')) as seed, session(seed, 32 << 20):
        seed.executescript(SCHEMA)
        seed.execute("INSERT INTO metadata VALUES ('marker','seed')")
        seed.commit()
        for attempt in range(2):
            with closing(sqlite3.connect(':memory:', uri=True)) as private:
                private.executescript(SCHEMA)
                prepare(private, seed)
                assert private.execute("SELECT value FROM metadata WHERE key='marker'").fetchone() == ('seed',)
                private.execute("UPDATE metadata SET value='changed' WHERE key='marker'")
                assert private.execute('SELECT count(*) FROM observations').fetchone() == (0,)
                assert (current().directory/'trial.sqlite').is_file()

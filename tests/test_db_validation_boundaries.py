"""Derived metadata and serialized plans cannot bypass complete validation."""
import json
import sqlite3

import pytest

from stash_ai.db.update import update_database
from stash_ai.storage.metadata import decode
from stash_ai.storage.schema import LATEST_SCHEMA_VERSION
from stash_ai.storage.store import StateStore
from stash_ai.runtime.report import RunReport
from tests.test_db_update import seed_version, rows
from tests.test_result_thresholds import outputs, config
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_temporal')


@pytest.mark.parametrize('field,value', [
    ('meta', {}), ('meta.origin', None), ('meta.analysis_format', None), ('meta.analysis_format', 4),
    ('meta.analysis_format', True), ('meta.inference_signature', ''), ('meta.frame_input_signature', 123),
    ('meta.scene_threshold', None), ('meta.marker_threshold', True), ('meta.marker_score_floor', -1),
    ('meta.scene_score_floor', 1.1), ('meta.scene_support_floor', '0.1'), ('has_frames', False),
])
def test_invalid_nested_metadata_stays_dirty_and_repairs_without_evidence_changes(tmp_path, field, value):
    path = tmp_path/'state.sqlite'
    seed_version(path, LATEST_SCHEMA_VERSION)
    evidence = rows(path, 'results')
    with sqlite3.connect(path) as db:
        record = json.loads(db.execute("SELECT data FROM result_metadata WHERE scope='scope-a'").fetchone()[0])
        target, key = record, field
        if '.' in field:
            parent, key = field.split('.')
            target = record[parent]
        target[key] = value
        serialized = json.dumps(record)
        with pytest.raises(ValueError): decode(serialized)
        db.execute("UPDATE result_metadata SET data=? WHERE scope='scope-a'", (serialized,))
        assert db.execute('SELECT scope FROM metadata_dirty').fetchall() == [('scope-a',)]
    details = {}
    update_database(path, details=details)
    assert not details['noop'] and details['metadata_repaired'] == 1
    assert rows(path, 'results') == evidence
    with sqlite3.connect(path) as db:
        assert not db.execute('SELECT * FROM metadata_dirty').fetchall()
        decode(db.execute("SELECT data FROM result_metadata WHERE scope='scope-a'").fetchone()[0])
    update_database(path, details=details)
    assert details['noop']


def test_untrusted_old_ledger_reseeds_semantically_invalid_rows(tmp_path):
    from stash_ai.storage.metadata_ledger import definitions, trusted
    path = tmp_path/'state.sqlite'
    seed_version(path, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE result_metadata SET data=? WHERE scope='scope-a'",
                   (json.dumps({'mode': 'scene', 'meta': {}, 'has_frames': True, 'populated': False}),))
        db.execute('DELETE FROM metadata_dirty')
        db.execute('DROP TRIGGER ' + next(iter(definitions())))
        assert not trusted(db)
    details = {}
    update_database(path, details=details)
    assert details['metadata_repaired'] == 1
    with sqlite3.connect(path) as db:
        assert trusted(db)


def test_db_check_counts_and_examples_share_the_same_wal_snapshot(tmp_path, monkeypatch):
    import stash_ai.db.checks as checks
    path = tmp_path/'state.sqlite'
    seed_version(path, LATEST_SCHEMA_VERSION)
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA journal_mode=WAL')
        original_row = db.execute("SELECT * FROM result_metadata WHERE scope='scope-a'").fetchone()
        db.execute("DELETE FROM result_metadata WHERE scope='scope-a'")
    original = checks._bounded_examples
    def repair_between_counts_and_examples(db, errors):
        with sqlite3.connect(path) as writer:
            writer.execute('INSERT INTO result_metadata VALUES (?,?,?,?,?,?)', original_row)
        return original(db, errors)
    monkeypatch.setattr(checks, '_bounded_examples', repair_between_counts_and_examples)
    result = checks.check_database(path, 'scope-a')
    issue = next(row for row in result['errors'] if row['code'] == 'result_metadata_missing')
    assert issue['count'] == 1
    assert issue['examples'] == [{'scope': 'scope-a', 'scene': '1', 'model': 'wd', 'kind': 'scene'}]
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT COUNT(*) FROM result_metadata').fetchone()[0] == 2


@pytest.mark.parametrize('damage', ['json', 'mode', 'model', 'delete'])
def test_entire_spool_is_verified_before_any_threshold_writes(tmp_path, monkeypatch, damage):
    from stash_ai.rethreshold.execution import run
    from stash_ai.rethreshold.spool import SpoolUsage
    original = SpoolUsage.flushed
    changed = []
    def corrupt(self, *args):
        original(self, *args)
        if self.plans == 129 and not changed:
            changed.append(True)
            if damage == 'json':
                self.spool.execute("UPDATE prepared SET outputs='broken' WHERE id=129")
            elif damage == 'mode':
                self.spool.execute("UPDATE prepared SET outputs=json_set(outputs,'$.scene.mode','frames') WHERE id=129")
            elif damage == 'model':
                self.spool.execute("UPDATE prepared SET model='joytag' WHERE id=129")
            else:
                self.spool.execute('DELETE FROM prepared WHERE id=129')
            self.spool.commit()
    monkeypatch.setattr(SpoolUsage, 'flushed', corrupt)
    with StateStore(tmp_path/'state.sqlite', 'scope') as store:
        scenes = [str(i) for i in range(129)]
        for scene in scenes: store.put(scene, 'wd', outputs())
        before = store.connection.execute('SELECT * FROM results').fetchall()
        report = RunReport(write=lambda *args: None)
        with pytest.raises(ValueError):
            run(app, store, scenes, [config(.8)], report, scenes=True, markers=True)
        assert changed and not report.counts['threshold_pairs_committed']
        assert not store.connection.execute('SELECT * FROM result_thresholds').fetchall()
        assert store.connection.execute('SELECT * FROM results').fetchall() == before
        assert report.threshold_preflight['applied'] == 0
        assert report.workload['rethreshold_spool']['state'] == 'invalid'

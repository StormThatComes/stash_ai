"""Empty retention avoids JSON work; compact indexes preserve legacy witnesses."""
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition import db_retention, retention_index
from stash_ai.recognition.profiling import pipeline_scope, pipeline_snapshot
from tests.test_recognition_retention import settings, scans


def test_fresh_database_does_not_build_reference_index(tmp_path):
    config = settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state, config, count=1)
        db = state.connection
        with db:
            db.executemany("INSERT INTO candidates VALUES (?,'s0','p',NULL,'accepted',?)",
                           [(str(i), encoded({'description': 'large unused document ' * 200})) for i in range(100)])
        statements = []
        db.set_trace_callback(statements.append)
        with pipeline_scope(True), db:
            db.execute('BEGIN IMMEDIATE')
            result = db_retention.on_connection(db, config, apply=True)
            profile = pipeline_snapshot()
        db.set_trace_callback(None)
        assert result['actions'] == [] and result['scans'] == {'keep': 1}
        assert profile['counts']['retention_empty_targets'] == 1
        assert not any('json_tree' in sql for sql in statements)
        assert not db.execute('SELECT 1 FROM metadata WHERE key=?', (retention_index.MARKER,)).fetchone()


def test_compact_index_omits_prose_and_protects_late_legacy_address(tmp_path):
    config = settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state, config)
        db = state.connection
        with db:
            db.execute("INSERT INTO candidates VALUES ('external','s3','p',NULL,'accepted',?)",
                       (encoded({'witnesses': ['late-legacy', 'face-s0'], 'description': 'not an address'}),))
            assert retention_index.refresh(db, True)
        values = {row[0] for row in db.execute("SELECT value FROM retention_links WHERE doc='external'")}
        assert values == {b'face-s0'}
        # A historical JSON document can precede insertion of an arbitrary legacy
        # address. Its unchanged index must not authorize removing that evidence.
        with db:
            db.execute("UPDATE observations SET id='late-legacy' WHERE id='face-s1'")
        with db:
            db.execute('BEGIN IMMEDIATE')
            result = db_retention.on_connection(db, config, apply=True)
        assert result['scans'] == {'archive': 2, 'keep': 2}
        assert db.execute("SELECT embedding FROM observations WHERE id='late-legacy'").fetchone()[0]


def test_empty_obsolete_scans_still_removed(tmp_path):
    config = settings(tmp_path)
    with FaceDatabase(config.database) as state:
        scans(state, config)
        with state.connection as db:
            db.execute('DELETE FROM observations')
        with state.connection as db:
            db.execute('BEGIN IMMEDIATE')
            result = db_retention.on_connection(db, config, apply=True)
        assert result['scans'] == {'delete': 2, 'keep': 2}
        assert {row[0] for row in db.execute('SELECT id FROM scans')} == {'s2', 's3'}

import json
import sqlite3

import pytest

from stash_ai.recognition.database import FaceDatabase,SCHEMA_V13,APPLICATION_ID,encoded
from stash_ai.recognition.cold_payloads import put,get,pack_scene,collect
from stash_ai.recognition.decision_contexts import pack,unpack
from tests.test_recognition_retention import settings,scans


def test_v13_migration_preserves_inline_evidence_and_deduplicates_on_pack(tmp_path):
    config=settings(tmp_path)
    with sqlite3.connect(config.database) as db:
        db.executescript(SCHEMA_V13)
        db.execute('INSERT INTO metadata VALUES (?,?)',('installation','00000000-0000-0000-0000-000000000001'))
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=13')
    with FaceDatabase(config.database) as state:
        db=state.connection
        selected=scans(state,config,2)
        with db:
            for scan in selected:pack_scene(db,scan)
        assert db.execute('SELECT COUNT(*) FROM crop_payloads').fetchone()[0]==1
        values=[r[0] for r in db.execute('SELECT crop FROM observations')]
        assert all(len(v)==69 and get(db,v)==b'pixels'*1000 for v in values)
        with db:
            db.execute('UPDATE observations SET crop=NULL WHERE scan=?',(selected[0],))
            collect(db)
        assert get(db,values[1])==b'pixels'*1000
        with db:
            db.execute('UPDATE observations SET crop=NULL')
            collect(db)
        assert db.execute('SELECT COUNT(*) FROM crop_payloads').fetchone()[0]==0


def test_payload_checksum_and_rollback(tmp_path):
    with FaceDatabase(settings(tmp_path).database) as state:
        db=state.connection
        key=put(db,b'x'*1000)
        db.rollback()
        with pytest.raises(ValueError,match='Missing'):get(db,key)
        with db:
            key=put(db,b'x'*1000)
            db.execute('UPDATE crop_payloads SET data=?',(b'corrupt',))
        with pytest.raises(ValueError,match='checksum'):get(db,key)


def test_decision_context_is_shared_and_round_trips_original_explanation(tmp_path):
    with FaceDatabase(settings(tmp_path).database) as state:
        db=state.connection
        for i in range(3):
            value=dict(observation=str(i),reason='known_identity',policy='test',
                       resource_limits={'max_people_per_lookup':100000},representative_limit=128)
            compact=pack(db,encoded(value))
            assert 'resource_limits' not in json.loads(compact)
            assert unpack(db,compact)==value
        assert db.execute('SELECT COUNT(*) FROM decision_contexts').fetchone()[0]==1

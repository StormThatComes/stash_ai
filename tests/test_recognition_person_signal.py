import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import defaults, load, validate
from stash_ai.recognition.database import FaceDatabase, SCHEMA_V5, APPLICATION_ID
from stash_ai.recognition.person_signal import contract, read, save


def test_contract_ignores_policy_paths_and_budgets():
    data=defaults(); config=SimpleNamespace(data=data)
    original=contract(config)
    data['sapiensid']['matching']['person_threshold']=.9
    data['sapiensid']['model']['source_directory']='/different'
    data['sapiensid']['execution']['batch_size']=8
    assert contract(config)==original
    data['sapiensid']['pose']['keypoint_score_threshold']=.7
    assert contract(config)!=original


def test_enabled_requires_assets_but_does_not_load_them(tmp_path):
    data=defaults();data['sapiensid']['enabled']=True
    with pytest.raises(ValueError,match='local path'):validate(data)
    data['sapiensid']['model'].update(source_directory='missing',checkpoint_directory='missing')
    data['sapiensid']['pose'].update(path='missing',sha256='a'*64)
    validate(data)


def test_v5_readonly_then_writer_migration(tmp_path):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    path=tmp_path/'old.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V5)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=5')
        db.execute("INSERT INTO metadata VALUES ('installation','00000000-0000-0000-0000-000000000001')")
    before=path.read_bytes()
    with _open_readonly(path) as db:
        assert read(db,'missing','contract') is None
    assert path.read_bytes()==before
    with FaceDatabase(path) as state:
        db=state.connection
        assert db.execute('PRAGMA user_version').fetchone()[0]==6
        db.execute("INSERT INTO scans(id,scope,scene,namespace,input_sha,policy_sha,manifest,coverage,total) VALUES ('s','x','1','n','i','p','{}','{}',1)")
        db.execute("INSERT INTO observations VALUES ('o','s',0,0,'url','[]','[]','{}',NULL,512,NULL,NULL)")
        version=db.execute("SELECT version FROM evidence_versions WHERE kind='scan' AND key='s'").fetchone()[0]
        save(db,'o','v1',{'association':'unique'},np.ones(4096))
        assert len(read(db,'o','v1')['embedding'])==16384
        assert read(db,'o','v2') is None
        assert db.execute("SELECT version FROM evidence_versions WHERE kind='scan' AND key='s'").fetchone()[0]>version
        save(db,'o','v1',{'reason':'no_person'})
        assert read(db,'o','v1')['status']=='unavailable'
        with pytest.raises(ValueError):save(db,'o','v1',{},np.zeros(4096))
        with pytest.raises(ValueError):save(db,'o','v1',{},np.ones(512))

from types import SimpleNamespace

import numpy as np
from PIL import Image

from stash_ai.recognition.config import defaults
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition import body_observations as body, person_signal


def setup(db):
    config=SimpleNamespace(data=defaults())
    config.data['sapiensid']['enabled']=True
    config.data['sapiensid']['collection']['enabled']=True
    db.execute("INSERT INTO scans VALUES ('s','scope','1','ns','i','p','{}','{}',1,1,'complete')")
    db.execute("INSERT INTO observations VALUES ('f','s',0,0,'','[]','{}','{}',NULL,2,?,NULL)",
               (np.array([1,0],dtype='<f4').tobytes(),))
    vector=np.zeros(4096,dtype='<f4');vector[0]=1
    metadata=dict(person=dict(bounds=[0,0,16,16]),identity_rejection=None,
        _body=dict(scan='s',cue=0,seconds=0,contract=body.contract(config),face='f',face_accepted=True,rejected=None))
    return config,vector,metadata


def test_fresh_projection_never_inserts_a_duplicate_feature_and_survives_replacement(tmp_path):
    with FaceDatabase(tmp_path/'db') as state:
        db=state.connection;c,vector,metadata=setup(db)
        # Observe actual inserted values, including transient payloads.
        db.execute('CREATE TEMP TABLE inserted_sizes(bytes INTEGER)')
        db.execute('CREATE TEMP TRIGGER observe_projection AFTER INSERT ON main.person_evidence '
                   'BEGIN INSERT INTO inserted_sizes VALUES(length(new.embedding)); END')
        body.save_records(db,c,[('b',metadata,vector,b'crop')])
        assert db.execute('SELECT bytes FROM inserted_sizes').fetchone()[0]==6
        assert person_signal.read(db,'f',person_signal.contract(c))['embedding']==vector.tobytes()
        db.commit()
        db.execute("UPDATE body_observations SET embedding=zeroblob(16384),crop=NULL WHERE id='b'")
        assert person_signal.read(db,'f',person_signal.contract(c))['embedding']==vector.tobytes()
        db.rollback()
        assert db.execute('SELECT length(embedding) FROM person_evidence').fetchone()[0]==6


def test_repair_and_crop_backfill_keep_shared_payloads(tmp_path):
    with FaceDatabase(tmp_path/'db') as state:
        db=state.connection;c,vector,metadata=setup(db)
        body.save_records(db,c,[('b',metadata,vector,None),
            (None,{'_body_cue':('s',body.contract(c),0)},None,None)])
        db.execute('DELETE FROM person_evidence')
        assert body.repair_projections(db,c,'s')==1
        assert db.execute('SELECT length(embedding) FROM person_evidence').fetchone()[0]==6
        from stash_ai.recognition.person_crop_backfill import backfill
        source=SimpleNamespace(frames_indices=lambda indices: ((0,None,Image.new('RGB',(16,16),'red')) for _ in indices))
        backfill(db,'s',source,c)
        assert tuple(db.execute('SELECT length(embedding),length(crop) FROM person_evidence').fetchone())==(6,6)
        restored=person_signal.read(db,'f',person_signal.contract(c))
        assert restored['embedding']==vector.tobytes() and restored['crop'].startswith(b'\x89PNG')

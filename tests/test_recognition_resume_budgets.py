import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.body_guards import FaceGuard
from stash_ai.recognition.body_identity import Scores
from tests.test_recognition_body_identity import configuration
from tests.test_recognition_merge_report import seeded
from tests.test_recognition_hybrid_recovery import enable
from tests.test_recognition_global_identity_lifecycle import add_bodies_to_faces


def test_identical_face_vetoes_share_one_budgeted_comparison(tmp_path):
    c=configuration(tmp_path);c.data['sapiensid']['execution']['max_witness_comparisons']=1
    face=np.array([1.,0.],dtype='<f4');guard=FaceGuard(None,c)
    with patch('numpy.dot',wraps=np.dot) as dot:
        assert guard.allows({'_face_vector':face},[{'_face_vector':face.copy()} for _ in range(20)])
    assert dot.call_count==guard.scores.count==1


def test_distinct_face_vetoes_use_the_body_comparison_budget(tmp_path):
    c=configuration(tmp_path);c.data['sapiensid']['execution']['max_witness_comparisons']=1
    scores=Scores(c);guard=FaceGuard(None,c,scores=scores)
    face=np.array([1.,0.],dtype='<f4')
    with pytest.raises(ValueError,match='max_witness_comparisons'):
        guard.allows({'_face_vector':face},[{'_face_vector':face},{'_face_vector':np.array([.8,.6],dtype='<f4')}])
    assert scores.count==1


def test_face_veto_stream_observes_cancellation(tmp_path):
    from stash_ai.runtime.cancellation import Cancellation,Cancelled,use
    c=configuration(tmp_path);guard=FaceGuard(None,c);token=Cancellation()
    face=np.array([1.,0.],dtype='<f4')
    with use(token):
        token.cancel()
        with pytest.raises(Cancelled):guard.allows({'_face_vector':face},[{'_face_vector':face}])
    assert guard.scores.count==0


@pytest.mark.parametrize('direct',[False,True])
def test_completed_cue_repairs_missing_projections_without_inference_or_decode(tmp_path,direct):
    from stash_ai.recognition.person_collection import PersonCollector
    from stash_ai.recognition.person_signal import contract
    c,people=seeded(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db);c.data['sapiensid']['collection']['enabled']=True
        c.data['sapiensid']['storage']['retain_person_crops']=False
        add_bodies_to_faces(db,c,people)
        scan=db.execute('SELECT id FROM scans LIMIT 1').fetchone()[0]
        bodies=db.execute('SELECT * FROM body_observations WHERE scan=? ORDER BY cue',(scan,)).fetchall()
        db.execute("UPDATE body_observations SET rejected='person_overlap' WHERE id=?",(bodies[0]['id'],))
        db.execute('UPDATE body_observations SET face=NULL WHERE id=?',(bodies[1]['id'],))
        db.execute('DELETE FROM person_evidence WHERE observation IN (SELECT id FROM observations WHERE scan=?)',(scan,));db.commit()
        collector=PersonCollector(c,{})
        def unexpected(*args,**kwargs):raise AssertionError('no decoding or inference is needed')
        source=SimpleNamespace(cues=[0,1,2],frames_indices=unexpected)
        with patch.object(collector,'_load',side_effect=unexpected):
            for _ in range(2):
                if direct:
                    for cue in range(3):collector.collect_frame(db,scan,cue,None)
                else:collector.enrich(db,scan,source)
        rows=db.execute('SELECT e.* FROM person_evidence e JOIN observations o ON o.id=e.observation '
            'WHERE o.scan=? AND e.contract=? ORDER BY o.cue',(scan,contract(c))).fetchall()
        assert [r['status'] for r in rows]==['unavailable','unavailable','ready']
        assert json.loads(rows[0]['metadata'])['identity_rejection']=='person_overlap'
        assert json.loads(rows[1]['metadata'])['identity_rejection']=='person_face_association_unavailable'
        assert rows[2]['embedding']==bodies[2]['embedding']
        collector.close()


def test_incomplete_cue_does_not_receive_a_synthetic_projection(tmp_path):
    from stash_ai.recognition.body_observations import repair_projections
    c,people=seeded(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection;c=enable(c,db);c.data['sapiensid']['collection']['enabled']=True
        add_bodies_to_faces(db,c,people)
        scan=db.execute('SELECT id FROM scans LIMIT 1').fetchone()[0]
        db.execute('DELETE FROM body_cues WHERE scan=? AND cue=0',(scan,))
        db.execute('DELETE FROM person_evidence WHERE observation IN (SELECT id FROM observations WHERE scan=? AND cue=0)',(scan,))
        with db:assert repair_projections(db,c,scan)==0
        assert db.execute('SELECT COUNT(*) FROM person_evidence e JOIN observations o ON o.id=e.observation '
            'WHERE o.scan=? AND o.cue=0',(scan,)).fetchone()[0]==0

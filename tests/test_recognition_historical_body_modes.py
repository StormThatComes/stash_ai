import json
from pathlib import Path
import numpy as np
from stash_ai.recognition.hybrid_modes import body_modes,covered_score
from stash_ai.recognition.hybrid_references import body_quality
from stash_ai.recognition.config import defaults
from types import SimpleNamespace


def test_modes_cover_all_supported_looks_without_discarding_singletons():
    config=SimpleNamespace(data=defaults());rows=[]
    for i in range(6):
        v=np.zeros(4096,dtype='<f4');v[i//3]=1
        rows.append(dict(id=str(i),scan='s',seconds=i,quality='{}',embedding=v.tobytes()))
    modes=body_modes(rows,config)
    assert len(modes)==2
    score=covered_score(modes,modes,config.data['sapiensid']['matching'])
    assert score['minimum']==1 and score['mode_coverage']==[2,2]
    assert body_modes(rows[:4],config)==()
    assert covered_score(modes,modes[:1],config.data['sapiensid']['matching']) is None


def test_body_quality_uses_body_input_not_face_score():
    def row(score,points):return dict(person_metadata=json.dumps(dict(person=dict(
        bounds=[0,0,100,300],score=score,keypoints=points))))
    assert body_quality(row(.9,[[20,20]]*17))>body_quality(row(.6,[[-1,-1]]*12+[[20,20]]*5))


def test_query_can_match_original_core_after_merge(tmp_path):
    from tests.test_recognition_merge_report import seeded
    from tests.test_recognition_hybrid_recovery import enable
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.hybrid_merge import apply
    from stash_ai.recognition.hybrid_queries import recover
    vectors=[[float(np.cos(a)),float(np.sin(a)),0,0] for a in np.radians([0,60,100])]
    cfg,people=seeded(tmp_path,vectors=vectors)
    with FaceDatabase(cfg.database) as state:
        db=state.connection;cfg=enable(cfg,db)
        db.execute('DELETE FROM members WHERE person=?',(people[2],))
        db.execute("UPDATE candidates SET person=NULL,status='withheld' WHERE person=?",(people[2],));db.commit()
        selected=[dict(scan=r['id'],scene=r['scene']) for r in db.execute('SELECT * FROM scans')]
        apply(state,selected,cfg)
        assert recover(state,selected,cfg)==3
        assert recover(state,selected,cfg)==0
        assert db.execute("SELECT COUNT(*) FROM candidates WHERE json_extract(payload,'$.hybrid_query.core_observations') IS NOT NULL").fetchone()[0]==3

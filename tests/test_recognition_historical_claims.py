from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.historical_references import competing_claims
from tests.test_recognition_pose_policy import FRONT, OTHER, configured
from tests.test_recognition_clustering import scan


def test_same_cue_historical_owner_requires_supported_separated_winner(tmp_path):
    c=configured(tmp_path)
    with FaceDatabase(c.database) as state:
        db=state.connection
        sid,_ = scan(state,c,'1',[FRONT,FRONT,FRONT,OTHER],count=4)
        for who in 'AB':
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',(who,'scope','ns',1,None,who))
        for oid,cue in db.execute('SELECT id,cue FROM observations WHERE scan=?',(sid,)).fetchall():
            db.execute('INSERT INTO members VALUES (?,?)',(oid,'A' if cue<3 else 'B'))
        db.execute('UPDATE observations SET cue=2 WHERE scan=? AND cue=3',(sid,))
        def claims(a,b):
            return competing_claims(db,{p:('T',dict(mode='strict',mean_score=score))
                                        for p,score in [('B',b),('A',a)]},c)
        assert claims(.95,.7)=={'B'}
        assert claims(.95,.93)=={'A','B'}
        assert claims(.7,.95)=={'A','B'}

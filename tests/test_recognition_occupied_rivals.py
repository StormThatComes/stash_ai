import json

from stash_ai.recognition.auxiliary import distinct_anchored_peers
from stash_ai.recognition.database import FaceDatabase
from tests.test_recognition_pose_policy import FRONT, OTHER, configured
from tests.test_recognition_clustering import scan


def test_same_cue_anchor_excludes_only_a_disjoint_distinct_person(tmp_path):
    config = configured(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        sid, _ = scan(state, config, '1', [FRONT, OTHER], count=2)
        query, peer = list(db.execute('SELECT * FROM observations WHERE scan=? ORDER BY cue',(sid,)))
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('rival','scope','ns',1,None,peer['id']))
        db.execute('INSERT INTO members VALUES (?,?)',(peer['id'],'rival'))
        geometry = json.loads(query['geometry'])
        geometry[0] += geometry[2]+10
        db.execute('UPDATE observations SET geometry=?, cue=0 WHERE id=?',(json.dumps(geometry),peer['id']))
        check = lambda: distinct_anchored_peers(db,query,config.data['clustering'],.95)
        assert check()=={'rival':[peer['id']]}
        db.execute('UPDATE observations SET embedding=? WHERE id=?',(query['embedding'],peer['id']))
        assert check()=={}
        db.execute('UPDATE observations SET geometry=?, embedding=? WHERE id=?',
                   (query['geometry'],peer['embedding'],peer['id']))
        assert check()=={}

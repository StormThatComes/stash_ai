import sqlite3
import numpy as np
from types import SimpleNamespace
from stash_ai.recognition.hybrid_queries import choose
from stash_ai.recognition.clustering import Group,co_presence_allows
from stash_ai.recognition.config import defaults


def test_valid_own_core_survives_invalid_higher_mean():
    assert choose([(.99,'a',False,None),(.88,'a',True,None)],.1)[:2]==(.88,'a')
    assert choose([(.88,'a',True,None),(None,'b',False,None)],.1) is None
    assert choose([(.88,'a',True,None),(.85,'b',False,None)],.1) is None


def test_same_owner_core_cannot_explain_away_copresence():
    def v(deg):return np.array([np.cos(np.deg2rad(deg)),np.sin(np.deg2rad(deg))],dtype='float32')
    db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
    try:
        db.execute('CREATE TABLE observations(id,scan,cue,geometry,dimensions,embedding,rejected)')
        for name,deg in [('query',0),('peer',60)]:
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,NULL)',(name,'s',0,'[]',2,v(deg).tobytes()))
        row=db.execute("SELECT * FROM observations WHERE id='query'").fetchone()
        a=Group('same',[v(15)],True,('a',));b=Group('same',[v(60)],True,('b',))
        assert not co_presence_allows(db,row,a,[a,b],v(0),SimpleNamespace(data=defaults()))
    finally:db.close()


def test_historical_body_modes_preserve_complete_coverage():
    from dataclasses import replace
    from stash_ai.recognition.hybrid_evidence import Identity,pair
    from stash_ai.recognition.hybrid_modes import query_score
    a=np.array([[1.,0.]]*3,dtype='float32');b=np.array([[0.,1.]]*3,dtype='float32')
    i=Identity('a',('face',),a[:1],np.vstack([a,b]),tuple(range(6)),body_modes=(a,b))
    settings=defaults()['sapiensid']['matching']
    assert not pair(replace(i,guard_persons=a),replace(i,person='b'),settings)['blockers']
    assert query_score(a[:1],i.persons,settings,i.body_modes)['minimum']==1

from stash_ai.recognition.inspection_search import SearchReader
from stash_ai.recognition.scene_inspection import SceneReader
from stash_ai.recognition.inspection_navigation import History
from tests.test_recognition_identity_inspector import scene_fixture


def test_search_filters_complete_scope_before_limit(tmp_path,monkeypatch):
    from stash_ai.recognition import inspection_search
    config,person=scene_fixture(tmp_path)
    source=SceneReader(config,'2')
    monkeypatch.setattr(inspection_search,'PAGE_SIZE',2)
    search=SearchReader(source,{'scene':'10','kind':'face','state':'anchor'})
    rows,more=search.page()
    assert rows and not more
    assert all(r['scene']=='10' and r['state']=='anchor' for r in rows)
    assert search.owner(rows[0]).selector[1]==person
    rows,_=SearchReader(source,{'state':'blurred'}).page()
    assert [r['id'] for r in rows]==['rejected-face']
    assert SearchReader(source,{'text':"' OR 1=1 --"}).page()==([],False)
    rows,_=SearchReader(source,{'performer':'900','kind':'face','policy':'current','reason':'identity','max_margin':'.3','sort':'Lowest lead'}).page()
    assert len(rows)==2 and all(r['margin']<.3 and r['identity']==person for r in rows)


def test_history_is_bounded_and_discards_forward_branch():
    h=History(3)
    for i in range(4):h.record(dict(scene=str(i),selected=None))
    assert len(h.entries)==3
    assert h.move(-1)['scene']=='2'
    h.record(dict(scene='new',selected=('face','x')))
    assert h.move(1) is None
    h.record(dict(scene='new',selected=('face','x'),scroll=.5))
    assert len(h.entries)==3 and h.entries[-1]['scroll']==.5

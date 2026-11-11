import gc
import time

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.identity_inspection import IdentityReader
from tests.test_recognition_identity_report import seed


def test_paged_inspection_reads_only_selected_crop(tmp_path, monkeypatch):
    from stash_ai.recognition import identity_inspection
    config, person, _ = seed(tmp_path, 6)
    before = config.database.read_bytes()
    monkeypatch.setattr(identity_inspection, 'PAGE_SIZE', 2)
    reader = IdentityReader(config, performer_id='900')
    with monkeypatch.context() as context:
        context.setattr(identity_inspection, 'cold_crop', lambda *_: pytest.fail('List loaded pixels'))
        rows, more = reader.page(scene='2', cue=0)
        assert len(rows) == 2 and not more
        first, more = reader.page()
        assert more
        second, _ = reader.page(offset=2)
        assert {row['id'] for row in first}.isdisjoint(row['id'] for row in second)
    address = rows[0]['id']
    filtered, _ = reader.page(observation=address)
    assert filtered == rows[:1]
    detail = reader.detail('face', address)
    assert detail['image'][0].size == (37,43)
    assert detail['trace']['decisions'][0]['metrics']['score'] == .87
    assert reader.reference('face', 'missing')[0] is None
    assert config.database.read_bytes() == before
    args = parser().parse_args(['inspect-identity','--identity-id',person,'--observation-id',address])
    assert args.observation_id == address


def test_native_window_selects_observation_and_displays_recorded_reason(tmp_path):
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError as error:
        import os
        if os.environ.get('STASH_AI_DESKTOP_TEST'):
            raise
        pytest.skip(f'No graphical desktop: {error}')
    from stash_ai.recognition.identity_window import Inspector
    config, person, _ = seed(tmp_path, 1)
    reader = IdentityReader(config, identity_id=person)
    address = reader.page()[0][0]['id']
    ui = Inspector(root, reader, 'Test identity', address)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            root.update()
            if address in ui.status.get() and 'known_identity' in ui.notes.get('1.0','end'):
                break
            time.sleep(.005)
        assert 'known_identity' in ui.notes.get('1.0','end')
        assert ui.images.panels['A'].image.size == (37,43)
        assert 'No recorded reference' in ui.images.panels['B'].caption.get()
    finally:
        ui.close()
        root.destroy()
        del ui, root
        gc.collect()


def scene_fixture(tmp_path):
    import json
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.body_identity import policy
    from stash_ai.recognition.models import namespace
    config, person, blob = seed(tmp_path, 6)
    with FaceDatabase(config.database) as state:
        db = state.connection
        scan = db.execute("SELECT id FROM scans WHERE scene='2'").fetchone()[0]
        template = dict(db.execute('SELECT * FROM observations LIMIT 1').fetchone())
        for address, scene, rejected in (('rejected-face',scan,'blurred'), ('unmatched-face',scan,None),
                                        ('internal-face',scan,None), ('outside-rejected',
                                        db.execute("SELECT id FROM scans WHERE scene='10'").fetchone()[0], 'too-small')):
            row = dict(template, id=address, scan=scene, rejected=rejected)
            db.execute('INSERT INTO observations VALUES ('+','.join('?' for _ in row)+')', tuple(row.values()))
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('internal','scope',namespace(config),2,None,'internal-token'))
        db.execute('INSERT INTO members VALUES (?,?)', ('internal-face','internal'))
        proof = dict(observation='unmatched-face', reason='candidate_group', scores=[.67,.66],
                     face_comparison=dict(threshold=.8, minimum=.61), compared_references=[template['id']])
        from stash_ai.recognition.decision_contexts import Packer
        payload = Packer(db)(json.dumps(proof))
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                   ('unmatched',scan,'historical-policy',None,'withheld',payload))
        for address, owner, status, role, rejected in (
                ('accepted-body',person,'accepted','auxiliary',None),
                ('provisional-body',person,'provisional','anchor',None),
                ('withheld-body',None,'withheld','auxiliary',None),
                ('rejected-body',None,None,None,'person_blurred')):
            db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (address,scan,0,0.,contract(config),None,'{}','{}',rejected,None,blob))
            if status:
                db.execute('INSERT INTO body_members VALUES (?,?,?,?,?,?)',
                    (address,owner,role,status,policy(config),json.dumps(dict(reason='body_scene_core',threshold=.8,score=.65))))
        db.commit()
    return config, person


def test_scene_tree_partitions_memberships_and_pages_all_identity_scenes(tmp_path, monkeypatch):
    from stash_ai.recognition import scene_inspection, identity_inspection
    from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
    config, person = scene_fixture(tmp_path)
    before = config.database.read_bytes()
    reader = SceneReader(config, '2')
    # Browsing the catalogue and every list page must not decode pixels.
    with monkeypatch.context() as context:
        context.setattr(identity_inspection, 'cold_crop', lambda *_: pytest.fail('List loaded pixels'))
        result = reader.catalogue()
        assert result['unassigned'] == 5
        assert {g['identity']:g['count'] for g in result['groups']} == {person:4,'internal':1}
        assert next(g for g in result['groups'] if g['identity']=='internal')['performer'] is None
        assigned, more = reader.identity_reader(person, 'scope').page(accepted_only=True)
        assert not more and len(assigned) == 7
        assert {r['scene'] for r in assigned} == {'2','10'}
        assert 'accepted-body' in {r['id'] for r in assigned}
        unassigned = UnassignedReader(config, result['scan'])
        context.setattr(scene_inspection, 'PAGE_SIZE', 2)
        pages = [unassigned.page(offset=i) for i in (0,2,4)]
        assert [more for _,more in pages] == [True,True,False]
        ids = [r['id'] for page,_ in pages for r in page]
        assert len(ids) == len(set(ids)) == 5
        assert set(ids) == {'rejected-face','unmatched-face','provisional-body','withheld-body','rejected-body'}
        assert set(ids).isdisjoint(r['id'] for r in assigned)
    detail = unassigned.detail('face','rejected-face')
    assert detail['rejection'] == 'blurred'
    assert detail['image'][0].size == (37,43)
    assert detail['trace']['status'] == 'unassigned'
    assert any(c['parameter']=='detection.min_sharpness' and c['status']=='fail'
               for g in detail['admission']['groups'] for c in g['checks'])
    assert detail['admission']['groups'][0]['configuration_source']=='current_configuration_only_historical_unavailable'
    detail = unassigned.detail('face','unmatched-face')
    assert detail['recorded_decisions'][0]['recorded']['compared_references']
    assert detail['recorded_decisions'][0]['current'] is False
    assert detail['trace']['decisions'][0]['metrics']['face_comparison.threshold'] == .8
    for address in ('provisional-body','withheld-body','rejected-body'):
        detail = unassigned.detail('body',address)
        assert detail['trace']['status']=='unassigned'
        assert detail['admission']['evidence_kind']=='person'
    with pytest.raises(ValueError,match='no longer unassigned'):
        unassigned.detail('body','accepted-body')
    assert config.database.read_bytes()==before


def test_scene_uses_current_scan_head_and_requires_unambiguous_scope(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.scene_inspection import SceneReader
    from stash_ai.recognition.scan_heads import activate
    config, _ = scene_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        original = dict(db.execute("SELECT * FROM scans WHERE scene='2'").fetchone())
        for row in (dict(original,id='new-partial',input_sha='new',status='partial'),
                    dict(original,id='foreign-scope',scope='foreign')):
            db.execute('INSERT INTO scans VALUES ('+','.join('?' for _ in row)+')',tuple(row.values()))
        db.commit()
    with pytest.raises(ValueError,match='--scope'):
        SceneReader(config, '2').catalogue()
    current = SceneReader(config,'2',scope='scope').catalogue()
    assert current['scan']['status']=='partial' and current['groups']==[] and current['unassigned']==0
    with FaceDatabase(config.database) as state:
        activate(state.connection,original)
        state.connection.commit()
    current = SceneReader(config,'2',scope='scope').catalogue()
    assert current['scan']['id']==original['id'] and current['unassigned']==5
    with pytest.raises(ValueError,match='No retained scan'):
        SceneReader(config,'999').catalogue()


def test_scene_cli_requires_positive_id_and_unassigned_refresh_detects_assignment(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
    for value in ('0','-1','anything'):
        with pytest.raises(SystemExit):
            parser().parse_args(['inspect-scene','--scene-id',value])
    args=parser().parse_args(['inspect-scene','--scene-id','002','--scope','scope'])
    assert args.scene_id=='2' and args.scope=='scope'
    config, person = scene_fixture(tmp_path)
    reader=UnassignedReader(config,SceneReader(config,'2').catalogue()['scan'])
    with FaceDatabase(config.database) as state:
        state.connection.execute('INSERT INTO members VALUES (?,?)',('unmatched-face',person))
        state.connection.commit()
    assert 'unmatched-face' not in {r['id'] for r in reader.page()[0]}
    with pytest.raises(ValueError,match='Refresh'):
        reader.detail('face','unmatched-face')
    assert reader.reference('face','outside-scope')[0] is None


def test_scene_includes_body_only_performer_and_keeps_missing_crop_visible(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
    config, _ = scene_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)', ('body-only','scope',namespace(config),3,'901','body-token'))
        db.execute("UPDATE body_members SET person='body-only' WHERE observation='accepted-body'")
        db.execute("UPDATE body_observations SET crop=NULL WHERE id='rejected-body'")
        db.commit()
    reader = SceneReader(config,'2')
    catalogue = reader.catalogue()
    assert any(g['identity']=='body-only' and g['performer']=='901' for g in catalogue['groups'])
    rows, more = reader.identity_reader('body-only','scope').page(accepted_only=True)
    assert not more and [r['id'] for r in rows]==['accepted-body']
    detail = UnassignedReader(config,catalogue['scan']).detail('body','rejected-body')
    assert detail['image'][0] is None and 'Crop not retained' in detail['image'][1]
    assert detail['rejection']=='person_blurred'


def test_native_scene_tree_expands_all_scenes_and_unassigned_rejections(tmp_path,monkeypatch):
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError as error:
        import os
        if os.environ.get('STASH_AI_DESKTOP_TEST'):
            raise
        pytest.skip(f'No graphical desktop: {error}')
    from stash_ai.recognition import identity_inspection, scene_inspection
    from stash_ai.recognition.scene_window import SceneInspector
    monkeypatch.setattr(identity_inspection,'PAGE_SIZE',2)
    monkeypatch.setattr(scene_inspection,'PAGE_SIZE',2)
    config, person = scene_fixture(tmp_path)
    ui = SceneInspector(root, scene_inspection.SceneReader(config,'2'), 'Test scene')

    def wait_for(condition):
        deadline = time.monotonic()+8
        while not condition() and time.monotonic()<deadline:
            root.update()
            time.sleep(.005)
        assert condition(), ui.status.get()

    def expand(item):
        ui.listing.focus(item)
        ui.listing.event_generate('<<TreeviewOpen>>')
        ui.listing.item(item,open=True)

    try:
        wait_for(lambda: len(ui.branches)==3)
        owner = next(k for k,v in ui.branches.items() if not v['unassigned'] and v['reader'].selector[1]==person)
        unknown = next(k for k,v in ui.branches.items() if v['unassigned'])
        expand(owner)
        expand(unknown)  # Queued expansions must both complete.
        wait_for(lambda: ui.branches[owner]['loaded'] and ui.branches[unknown]['loaded'])
        while ui.branches[owner]['offset']<7:
            marker=ui.branches[owner]['marker']
            old=ui.branches[owner]['offset']
            ui.listing.selection_set(marker)
            wait_for(lambda: ui.branches[owner]['offset']>old)
        assert {row['scene'] for row in ui.rows.values() if row['reader']==ui.branches[owner]['reader']}=={'2','10'}
        while 'rejected-face' not in {r['id'] for r in ui.rows.values()}:
            old=ui.branches[unknown]['offset']
            ui.listing.selection_set(ui.branches[unknown]['marker'])
            wait_for(lambda: ui.branches[unknown]['offset']>old)
        leaf=next(k for k,r in ui.rows.items() if r['id']=='rejected-face')
        ui.listing.selection_set(leaf)
        wait_for(lambda: 'Retained rejection: blurred' in ui.notes.get('1.0','end'))
        assert 'detection.min_sharpness' in ui.notes.get('1.0','end')
        assert ui.images.panels['A'].image.size==(37,43)
        ui.scene_entry.delete(0,'end')
        ui.scene_entry.insert(0,'10')
        ui.search()
        wait_for(lambda: bool(ui.branches))
        assert not ui.rows
        assert not ui.notes.get('1.0','end').strip()
        assert all('rejected-face'!=row['id'] for row in ui.rows.values())
    finally:
        ui.close()
        root.destroy()
        del ui,root
        gc.collect()

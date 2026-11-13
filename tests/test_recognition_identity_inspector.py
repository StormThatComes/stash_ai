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
    assert detail['suggestions']['title'] == 'If this assignment is incorrect'
    assert detail['suggestions']['paths'][0]['changes'][0]['proposed'] == .870001
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
            if address in ui.status.get() and 'known_identity' in ui.notes.get('1.0','end') and ui.images.panels['A'].image is not None:
                break
            time.sleep(.005)
        assert 'known_identity' in ui.notes.get('1.0','end')
        assert 'If this assignment is incorrect' in ui.notes.get('1.0','end')
        assert ui.images.panels['A'].image.size == (37,43)
        assert 'Selected face: yellow' in ui.images.panels['A'].caption.get()
        clean, annotated = ui.preview_images['A']
        assert clean[0].tobytes() != annotated[0].tobytes()
        ui.show_geometry.set(False)
        ui.toggle_geometry()
        assert ui.images.panels['A'].image is clean[0]
        ui.show_geometry.set(True)
        ui.toggle_geometry()
        assert ui.images.panels['A'].image is annotated[0]
        ui.images.set_zoom('A',2.)
        assert ui.images.panels['A'].zoom==2. and ui.images.panels['B'].zoom==1.
        ui.sync_zoom.set(True);ui.sync_viewports()
        assert ui.images.panels['B'].zoom==2.
        ui.images.fit('A')
        assert .05<=ui.images.panels['A'].zoom<=8.
        assert 'No recorded reference' in ui.images.panels['B'].caption.get()
        ui.pin_selected()
        pinned=ui.images.panels['B'].image
        ui.clear_detail()
        assert ui.pin is not None and ui.images.panels['B'].image is pinned
        ui.unpin()
        assert ui.pin is None and ui.images.panels['B'].image is None
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


def test_identity_kind_filter_pages_before_limit_without_loading_crops(tmp_path, monkeypatch):
    from stash_ai.recognition import identity_inspection
    config, person = scene_fixture(tmp_path)
    before = config.database.read_bytes()
    monkeypatch.setattr(identity_inspection, 'PAGE_SIZE', 2)
    monkeypatch.setattr(identity_inspection, 'cold_crop', lambda *_: pytest.fail('List loaded pixels'))
    reader = IdentityReader(config, identity_id=person)
    pages = [reader.page(kind='face', accepted_only=True, offset=i) for i in (0,2,4)]
    assert [more for _, more in pages] == [True, True, False]
    faces = [row for rows, _ in pages for row in rows]
    assert len({row['id'] for row in faces}) == 6
    assert {row['kind'] for row in faces} == {'face'}
    assert {row['scene'] for row in faces} == {'2', '10'}
    bodies, more = reader.page(kind='body', accepted_only=True)
    assert not more and [row['id'] for row in bodies] == ['accepted-body']
    assert {row['kind'] for row in bodies} == {'body'}
    assert reader.page(kind='body', accepted_only=True, offset=1) == ([], False)
    sample = faces[-1]
    assert reader.page(kind='face', scene=sample['scene'], cue=sample['cue'],
                       observation=sample['id']) == ([sample], False)
    assert reader.page(kind='body', observation=sample['id']) == ([], False)
    with pytest.raises(ValueError, match='kind'):
        reader.page(kind='invalid')
    monkeypatch.setattr(identity_inspection, 'bodies_available', lambda _: False)
    assert reader.page(kind='body') == ([], False)
    assert config.database.read_bytes() == before


def add_grouped_unassigned(config):
    from stash_ai.recognition.database import FaceDatabase
    with FaceDatabase(config.database) as state:
        db = state.connection
        for table, source, addresses in (
                ('observations', 'rejected-face', ('blurred-1', 'blurred-2')),
                ('body_observations', 'withheld-body', ('withheld-1', 'withheld-2', 'rejected-withheld'))):
            template = dict(db.execute(f'SELECT * FROM {table} WHERE id=?', (source,)).fetchone())
            for cue, address in enumerate(addresses, 1):
                row = dict(template, id=address, cue=cue)
                db.execute(f'INSERT INTO {table} VALUES ('+','.join('?' for _ in row)+')', tuple(row.values()))
                if table == 'body_observations':
                    member = dict(db.execute('SELECT * FROM body_members WHERE observation=?', (source,)).fetchone())
                    member['observation'] = address
                    db.execute('INSERT INTO body_members VALUES ('+','.join('?' for _ in member)+')', tuple(member.values()))
        db.execute("UPDATE body_observations SET rejected='person_blurred' WHERE id='rejected-withheld'")
        template = dict(db.execute("SELECT * FROM body_observations WHERE id='rejected-body'").fetchone())
        row = dict(template, id='unmatched-body', rejected=None)
        db.execute('INSERT INTO body_observations VALUES ('+','.join('?' for _ in row)+')', tuple(row.values()))
        db.commit()


def test_unassigned_kind_state_counts_and_pages_match_displayed_state(tmp_path, monkeypatch):
    from stash_ai.recognition import scene_inspection, identity_inspection
    from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
    config, _ = scene_fixture(tmp_path)
    add_grouped_unassigned(config)
    before = config.database.read_bytes()
    monkeypatch.setattr(scene_inspection, 'PAGE_SIZE', 2)
    monkeypatch.setattr(identity_inspection, 'cold_crop', lambda *_: pytest.fail('List loaded pixels'))
    result = SceneReader(config, '2').catalogue()
    counts = {(g['kind'], g['state']):g['count'] for g in result['unassigned_groups']}
    assert counts == {('face','blurred'):3, ('face','unassigned'):1,
                      ('body','person_blurred'):2, ('body','provisional'):1,
                      ('body','unassigned'):1, ('body','withheld'):3}
    assert result['unassigned'] == sum(counts.values()) == 11
    reader = UnassignedReader(config, result['scan'])
    ids = []
    for (kind, state), count in counts.items():
        rows = []
        for offset in range(0, count, 2):
            page, more = reader.page(kind=kind, state=state, offset=offset)
            assert more == (offset+2 < count)
            rows.extend(page)
        assert len(rows) == count
        assert {(r['kind'],r['state']) for r in rows} == {(kind,state)}
        assert all(r['state'] == (r['rejected'] or (r['role'] if r['status']=='accepted' else r['status'])) for r in rows)
        ids.extend(r['id'] for r in rows)
    all_ids = [r['id'] for offset in range(0,11,2) for r in reader.page(offset=offset)[0]]
    assert len(set(ids)) == len(ids) == 11 and set(ids) == set(all_ids)
    assert {'accepted-body','internal-face','outside-rejected'}.isdisjoint(ids)
    assert reader.page(kind='face', state='withheld') == ([], False)
    assert reader.page(kind='body', state="' OR 1=1 --") == ([], False)
    assert reader.page(kind='body', state='withheld', offset=3) == ([], False)
    assert {r['kind'] for r in reader.page(state='unassigned')[0]} == {'face','body'}
    with pytest.raises(ValueError, match='kind'):
        reader.page(kind='invalid')
    monkeypatch.setattr(scene_inspection, 'bodies_available', lambda _: False)
    assert reader.page(kind='body', state='unassigned') == ([], False)
    assert config.database.read_bytes() == before


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
    assert current['unassigned_groups']==[]
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
    assert reader.page(kind='face', state='unassigned') == ([], False)
    refreshed = SceneReader(config,'2').catalogue()
    assert refreshed['unassigned']==4
    assert ('face','unassigned') not in {(g['kind'],g['state']) for g in refreshed['unassigned_groups']}
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
    add_grouped_unassigned(config)
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
        wait_for(lambda: len(ui.branches)==10)
        owner = {v['kind']:k for k,v in ui.branches.items()
                 if not v['unassigned'] and v['reader'].selector[1]==person}
        face, body = owner['face'], owner['body']
        parent = ui.listing.parent(face)
        assert ui.listing.parent(body)==parent
        assert len(ui.groups)==5
        for group in ui.listing.get_children():
            assert [ui.listing.item(child,'text') for child in ui.listing.get_children(group)]==['Face','Body']
        expand(parent)
        ui.listing.selection_set(parent)
        wait_for(lambda: person in ui.notes.get('1.0','end'))
        assert not ui.rows  # Opening the performer itself does not load observations.
        assert not any(b['loading'] or b['loaded'] for b in ui.branches.values())
        unassigned = {(v['kind'],v['state']):k for k,v in ui.branches.items() if v['unassigned']}
        unknown = unassigned['face','blurred']
        withheld = unassigned['body','withheld']
        unknown_kind = ui.listing.parent(unknown)
        unknown_root = ui.listing.parent(unknown_kind)
        assert ui.listing.item(unknown_root,'text')=='Unassigned in this scene · 11'
        assert ui.listing.item(unknown_kind,'text')=='Face'
        assert ui.listing.item(unknown,'text')=='blurred · 3'
        assert {ui.listing.item(k,'text') for k in ui.listing.get_children(unknown_kind)}=={'blurred · 3','unassigned · 1'}
        expand(unknown_root)
        expand(unknown_kind)
        ui.listing.selection_set(unknown_kind)
        wait_for(lambda: 'Face observations' in ui.notes.get('1.0','end'))
        assert not ui.rows
        expand(face)
        expand(body)
        expand(withheld)
        expand(unknown)  # Queued expansions must all complete.
        wait_for(lambda: all(ui.branches[k]['loaded'] for k in (face,body,unknown,withheld)))
        assert ui.branches[face]['offset']==0 and ui.branches[body]['offset']==0
        face_states = list(ui.listing.get_children(face))
        body_states = list(ui.listing.get_children(body))
        assert {ui.listing.item(k,'text') for k in face_states}=={'anchor · 3','auxiliary · 3'}
        assert {ui.listing.item(k,'text') for k in body_states}=={'auxiliary · 1'}
        for state in face_states+body_states:
            expand(state)
        wait_for(lambda: all(ui.branches[k]['loaded'] for k in face_states+body_states))
        for state in face_states:
            assert ui.branches[state]['offset']==2
            ui.listing.selection_set(ui.branches[state]['marker'])
            wait_for(lambda: ui.branches[state]['offset']==3)
            assert all(ui.listing.item(k,'values')[-1]==ui.branches[state]['state'] for k in ui.listing.get_children(state))
        faces = [ui.rows[k] for state in face_states for k in ui.listing.get_children(state)]
        bodies = [ui.rows[k] for state in body_states for k in ui.listing.get_children(state)]
        assert len({row['id'] for row in faces})==6
        assert {row['kind'] for row in faces}=={'face'}
        assert {row['scene'] for row in faces}=={'2','10'}
        assert [row['id'] for row in bodies]==['accepted-body']
        assert ui.branches[body_states[0]]['offset']==1
        empty = next(k for k,v in ui.branches.items() if not v['unassigned'] and v['kind']=='body' and v['reader'].selector[1]=='internal')
        expand(empty)
        wait_for(lambda: ui.branches[empty]['loaded'])
        assert [ui.listing.item(k,'text') for k in ui.listing.get_children(empty)]==['No observations in this group.']
        while ui.branches[unknown]['offset']<3:
            old=ui.branches[unknown]['offset']
            ui.listing.selection_set(ui.branches[unknown]['marker'])
            wait_for(lambda: ui.branches[unknown]['offset']>old)
        assert ui.branches[unknown]['offset']==3 and ui.branches[withheld]['offset']==2
        ui.listing.selection_set(ui.branches[withheld]['marker'])
        wait_for(lambda: ui.branches[withheld]['offset']==3)
        assert {ui.rows[k]['id'] for k in ui.listing.get_children(unknown)}=={'rejected-face','blurred-1','blurred-2'}
        assert {ui.rows[k]['id'] for k in ui.listing.get_children(withheld)}=={'withheld-body','withheld-1','withheld-2'}
        for item, state in ((unknown,'blurred'),(withheld,'withheld')):
            assert all(ui.listing.item(k,'values')[-1]==state for k in ui.listing.get_children(item))
        leaf=next(k for k,r in ui.rows.items() if r['id']=='rejected-face')
        ui.listing.selection_set(leaf)
        wait_for(lambda: 'Retained rejection: blurred' in ui.notes.get('1.0','end') and ui.images.panels['A'].image is not None)
        assert 'detection.min_sharpness' in ui.notes.get('1.0','end')
        assert 'Suggested next steps' in ui.notes.get('1.0','end')
        assert 'YAML patch to test:' in ui.notes.get('1.0','end')
        assert ui.images.panels['A'].image.size==(37,43)
        from stash_ai.recognition.identity_window import describe
        attempt = ui.branches[unknown]['reader'].detail('face','unmatched-face')['trace']
        assert 'No competitor result retained' not in describe(attempt)
        attempt['decisions'][0]['metrics'] = {'probe.rival_score': .66}
        assert 'No competitor result retained' not in describe(attempt)
        ui.scene_entry.delete(0,'end')
        ui.scene_entry.insert(0,'10')
        ui.search()
        wait_for(lambda: bool(ui.branches))
        assert len(ui.groups)==4 and len(ui.branches)==3
        unknown = next(k for k,v in ui.branches.items() if v['unassigned'])
        assert ui.branches[unknown]['state']=='too-small'
        unknown_root = ui.listing.parent(ui.listing.parent(unknown))
        empty_body = ui.listing.get_children(unknown_root)[1]
        assert [ui.listing.item(k,'text') for k in ui.listing.get_children(empty_body)]==['No observations in this group.']
        assert not ui.rows
        assert not ui.notes.get('1.0','end').strip()
        assert all('rejected-face'!=row['id'] for row in ui.rows.values())
    finally:
        ui.close()
        root.destroy()
        del ui,root
        gc.collect()


def vtt_fixture(tmp_path):
    from dataclasses import replace
    import json
    from types import SimpleNamespace
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.models import namespace
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.scanning import scan_scene
    from stash_ai.recognition.run import instance_scope
    from stash_ai.recognition.body_observations import contract
    from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
    from tests.helpers.recognition import MediaClient, Backend
    from tests.test_recognition_scanning import config as configuration
    config = configuration(tmp_path)
    data = plain(config.data)
    data['detection']['min_face_pixels'] = 256  # Rejected before embedding/crop encoding.
    data['execution']['media_cache_bytes'] = 0
    config = replace(config, data=data)
    media, backend = MediaClient(count=1), Backend()
    scope = instance_scope(media.url)
    with FaceDatabase(config.database) as state, SpriteInput(media, media.scene, config) as source:
        scan, _ = scan_scene(state, scope, media.scene, config, namespace(config), source, lambda: backend)
        db = state.connection
        face = db.execute('SELECT id,crop,rejected,quality FROM observations').fetchone()
        assert face['crop'] is None and face['rejected'] and 'evidence_pixel_sha256' not in json.loads(face['quality'])
        bounds = [10,20,80,130]
        db.execute('INSERT INTO body_observations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ('rejected-body',scan,0,0.,contract(config),None,json.dumps(dict(bounds=bounds)),
                    json.dumps(dict(person=dict(bounds=bounds),quality={})), 'person_pixels',None,None))
        db.commit()
    closed = []
    client = SimpleNamespace(url=media.url, api_key='', media=media.media,
                             scene=lambda _: media.scene, close=lambda: closed.append(True))
    scan = SceneReader(config,'1').catalogue()['scan']
    reader = UnassignedReader(config,scan)
    media.calls.clear()
    return config, reader, client, media, face['id'], backend, closed


def test_vtt_previews_restore_rejected_face_and_body_after_snapshot_closes(tmp_path,monkeypatch):
    import sqlite3
    from stash_ai.recognition import inspection_crops
    from stash_ai.recognition.media import SpriteInput
    config, reader, client, media, face, backend, closed = vtt_fixture(tmp_path)
    before = config.database.read_bytes()
    connections, sources = [], []
    original = inspection_crops._open_readonly
    def opened(path):
        db = original(path)
        connections.append(db)
        return db
    def source(client, scene, config):
        with pytest.raises(sqlite3.ProgrammingError,match='closed'):
            connections[-1].execute('SELECT 1')
        assert config.data['execution']['media_cache_bytes']==0
        assert config.data['execution']['sprite_cache_bytes']<=64*1048576
        result = SpriteInput(client,scene,config)
        sources.append(result)
        return result
    monkeypatch.setattr(inspection_crops,'_open_readonly',opened)
    previews = inspection_crops.VttPreviews(config,client_factory=lambda:client,source_factory=source)
    try:
        assert len(reader.page()[0])==2 and not media.calls
        for kind,address,size in (('face',face,(160,160)),('body','rejected-body',(70,110))):
            result=reader.detail(kind,address)
            image, caption=previews.image(kind,address,result['scope'],result['image'])
            assert image.size==size
            assert 'Reconstructed from VTT' in caption and 'original crop signature unavailable' in caption
            image.close()
        assert len(sources)==1 and len(media.calls)==2  # One VTT and one sprite, shared by both crops.
        assert backend.calls==0
        assert config.database.read_bytes()==before
        assert not config.database.with_name(config.database.name+'.media.sqlite3').exists()
    finally:
        previews.close()
    assert closed==[True] and not sources[0].sprites.entries


@pytest.mark.parametrize('damage,kind,message', [
    ('vtt','face','source changed'), ('sprite','body','source changed'),
    ('scope','face','STASH_URL'), ('cue','face','cue does not match'),
    ('geometry','body','bounds'), ('signature','face','recorded signature'),
    ('signature','body','recorded signature'), ('missing','face','RuntimeError'),
])
def test_vtt_preview_refuses_changed_or_unverifiable_evidence(tmp_path,damage,kind,message):
    import json
    from io import BytesIO
    from PIL import Image
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.inspection_crops import VttPreviews
    config, reader, client, media, face, _, _ = vtt_fixture(tmp_path)
    if damage=='vtt':
        key=client.url+'vtt/1.vtt'
        media.data[key]=media.data[key].replace(b'00:00:00.000',b'00:00:00.100')
    elif damage=='sprite':
        with Image.new('RGB',(160,160),'red') as image, BytesIO() as output:
            image.save(output,format='PNG')
            media.data[client.url+'sprite/1.jpg']=output.getvalue()
    elif damage=='scope':
        client.url='http://other-stash/'
    elif damage=='missing':
        def missing(_):
            raise RuntimeError('https://stash/?token=do-not-show-this-secret')
        client.scene=missing
    else:
        with FaceDatabase(config.database) as state:
            db=state.connection
            if damage=='cue':
                db.execute('UPDATE observations SET seconds=42')
            elif damage=='geometry':
                db.execute('UPDATE body_observations SET geometry=?',(json.dumps(dict(bounds=[0,0,9999,9999])),))
            elif kind=='face':
                db.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_pixel_sha256','wrong')")
            else:
                db.execute("UPDATE body_observations SET quality=json_set(quality,'$.quality.pixel_digest','wrong')")
            db.commit()
    address=face if kind=='face' else 'rejected-body'
    result=reader.detail(kind,address)
    before=config.database.read_bytes()
    previews=VttPreviews(config,client_factory=lambda:client)
    try:
        image,caption=previews.image(kind,address,result['scope'],result['image'])
        assert image is None and message in caption
        assert 'do-not-show-this-secret' not in caption
        assert result['rejection'] and result['trace']['status']=='unassigned'
        assert config.database.read_bytes()==before
    finally:
        previews.close()


def test_vtt_preview_checks_available_signature_and_leaves_retained_pixels_offline(tmp_path):
    import hashlib
    from PIL import Image
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.inspection_crops import VttPreviews
    from tests.helpers.recognition import textured_image
    config, reader, client, media, face, _, _ = vtt_fixture(tmp_path)
    with textured_image() as image:
        digest=hashlib.sha256(str(image.size).encode('ascii')+image.tobytes()).hexdigest()
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE observations SET quality=json_set(quality,'$.evidence_pixel_sha256',?)",(digest,))
        state.connection.commit()
    previews=VttPreviews(config,client_factory=lambda:client)
    try:
        with Image.new('RGB',(2,2)) as stored:
            retained=(stored,'Retained crop')
            assert previews.image('face',face,reader.scan['scope'],retained) is retained
            assert not media.calls and previews.client is None
        image,caption=previews.image('face',face,reader.scan['scope'],reader.detail('face',face)['image'])
        assert image.size==(160,160) and 'pixels verified' in caption
        image.close()
        assert previews.image('face',face,'different-scope',(None,'Missing'))[0] is None
    finally:
        previews.close()
    assert parser().parse_args(['inspect-scene','--scene-id','1','--offline']).offline
    assert not parser().parse_args(['inspect-scene','--scene-id','1']).offline


def test_native_scene_inspector_reconstructs_selected_unassigned_crop_on_worker(tmp_path):
    tk = pytest.importorskip('tkinter')
    try:
        root=tk.Tk()
    except tk.TclError as error:
        import os
        if os.environ.get('STASH_AI_DESKTOP_TEST'):
            raise
        pytest.skip(f'No graphical desktop: {error}')
    from threading import Event, get_ident
    from stash_ai.recognition.inspection_crops import VttPreviews
    from stash_ai.recognition.scene_inspection import SceneReader
    from stash_ai.recognition.scene_window import SceneInspector
    config, reader, client, media, face, backend, closed=vtt_fixture(tmp_path)
    main_thread=get_ident()
    started, release = Event(), Event()
    def factory():
        assert get_ident()!=main_thread
        started.set()
        assert release.wait(5)
        return client
    previews=VttPreviews(config,client_factory=factory)
    ui=SceneInspector(root,SceneReader(config,'1'),'VTT previews',previews=previews)
    def wait_for(condition):
        deadline=time.monotonic()+8
        while not condition() and time.monotonic()<deadline:
            root.update()
            time.sleep(.005)
        assert condition(), ui.status.get()
    try:
        wait_for(lambda: bool(ui.branches))
        for branch in ui.branches:
            ui.load_branch(branch)
        wait_for(lambda: all(v['loaded'] for v in ui.branches.values()))
        assert not media.calls
        row=next(k for k,v in ui.rows.items() if v['id']==face)
        ui.listing.selection_set(row)
        wait_for(started.is_set)
        # An old VTT read must not overwrite a newer observation selection.
        body=next(k for k,v in ui.rows.items() if v['id']=='rejected-body')
        ui.listing.selection_set(body)
        root.update()
        release.set()
        wait_for(lambda: 'Reconstructed from VTT' in ui.images.panels['A'].caption.get())
        assert ui.images.panels['A'].image.size==(70,110)
        assert 'rejected-body' in ui.status.get()
        ui.listing.selection_set(row)
        wait_for(lambda: face in ui.status.get() and ui.images.panels['A'].image is not None)
        assert ui.images.panels['A'].image.size==(160,160)
        assert 'Unassigned' in ui.notes.get('1.0','end')
        assert backend.calls==0
    finally:
        release.set()
        ui.close()
        root.destroy()
        del ui,root
        gc.collect()
    deadline=time.monotonic()+2
    while not closed and time.monotonic()<deadline:
        time.sleep(.005)
    assert closed and all(closed)


def advice_fixture(tmp_path):
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    config, _, _ = seed(tmp_path, 1)
    data = plain(config.data)
    data['detection'].update(min_visible_face_fraction=0, max_pose_deviation=None)
    data['sapiensid']['enabled'] = True
    data['sapiensid']['collection']['enabled'] = True
    data['sapiensid']['identities']['enabled'] = True
    return replace(config, data=data, document_version=2)


def advice_admission(config, kind='face', **measurements):
    from types import SimpleNamespace
    from stash_ai.recognition import admission_report
    policies = SimpleNamespace(resolve=lambda *_: (config,'current_configuration_hash_verified'))
    if kind == 'face':
        quality = dict(pixels=128, sharpness=30., complete=True, alignment_valid=True, pose_deviation=.1)
        quality.update(measurements)
        return admission_report.face(quality, [0]*14+[.99], config, policies, 'test')
    metadata = dict(person=dict(score=.99, bounds=[0,0,300,500]),
                    quality=dict(width=300, height=500, sharpness=100., keypoints=17, overlap=0.))
    metadata['quality'].update(measurements)
    return admission_report.person(metadata, config, policies, 'test', independent=True)


@pytest.mark.parametrize('version', [1,2])
def test_advice_emits_valid_yaml_with_strict_and_inclusive_boundaries(tmp_path, version):
    from dataclasses import replace
    import yaml
    from stash_ai.recognition.inspection_advice import suggest, describe
    from stash_ai.recognition.merge_report_policy import patched
    config = replace(advice_fixture(tmp_path), document_version=version)
    admission = advice_admission(config, sharpness=7.25)
    admission['groups'][0]['checks'][0]['measured'] = config.data['detection']['score_threshold']
    result = suggest(config,'face',admission,[], 'blurred')
    path = result['paths'][0]
    changes = {c['parameter']:c['proposed'] for c in path['changes']}
    assert changes == {'detection.score_threshold':.899999, 'detection.min_sharpness':7.25}
    patched(config,changes)  # Valid together under production config constraints.
    patch = yaml.safe_load(path['yaml'])
    assert (patch['face']['quality'] if version==2 else patch['detection'])['min_sharpness']==7.25
    assert 'fresh scan evidence' in path['next_action']
    assert 'not a verified assignment replay' in describe(result)
    assert len(result['paths'])==2


def test_advice_offers_auxiliary_before_relaxing_face_anchor_quality(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest
    config = advice_fixture(tmp_path)
    config.data['clustering'].update(min_anchor_sharpness=40., auxiliary_matching=False)
    advice = suggest(config,'face',advice_admission(config),[],None)
    auxiliary, anchor = advice['paths'][:2]
    assert {c['yaml_parameter']:c['proposed'] for c in auxiliary['changes']}=={'matching.face.auxiliary.enabled':True}
    assert {c['yaml_parameter']:c['proposed'] for c in anchor['changes']}=={'matching.face.anchors.quality.min_sharpness':30.}
    assert config.data['clustering']['min_anchor_sharpness']==40.
    assert 'rebuild assignments' in auxiliary['next_action']


@pytest.mark.parametrize('measured', [None, float('nan'), float('inf')])
def test_advice_does_not_guess_missing_measurements_or_fix_geometry_with_yaml(tmp_path, measured):
    from stash_ai.recognition.inspection_advice import suggest, describe
    config = advice_fixture(tmp_path)
    result = suggest(config,'face',advice_admission(config, sharpness=measured, alignment_valid=False),[], 'blurred')
    assert not any(p['changes'] for p in result['paths'])
    assert 'measurement unavailable' in describe(result)
    assert 'alignment_valid: fail' in describe(result)


def test_advice_rechecks_current_thresholds_instead_of_recommending_historical_values(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest, describe
    config = advice_fixture(tmp_path)
    admission = advice_admission(config, sharpness=7.25)
    admission['groups'][0]['configuration_source']='retained_configuration_hash_verified'
    config.data['detection']['min_sharpness']=5.
    result = suggest(config,'face',admission,[], 'blurred')
    assert not result['paths'][0]['changes']
    assert 'Reprocess this scene' in describe(result)
    assert 'Scan settings differ' in describe(result)


def test_advice_uses_current_independent_body_support_and_keeps_an_evidence_alternative(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest, describe
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['identities']['min_support']=3
    decision = dict(current=True, status='provisional', recorded=dict(reason='body_scene_core',independent_support=2,required_support=3))
    result = suggest(config,'body',advice_admission(config,'body'),[decision],None)
    assert {c['yaml_parameter']:c['proposed'] for p in result['paths'] for c in p['changes']}=={'matching.body.identities.min_support':2}
    assert any('collect stronger evidence' in p['title'] for p in result['paths'])
    decision['current']=False
    result = suggest(config,'body',advice_admission(config,'body'),[decision],None)
    assert not any(p['changes'] for p in result['paths'])
    assert 'No current-policy matching attempt' in describe(result)


def test_advice_keeps_body_rival_margin_and_regional_keypoint_rules(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest, describe
    from stash_ai.recognition.admission_report import check, group
    config = advice_fixture(tmp_path)
    settings = config.data['sapiensid']['identities']
    settings.update(match_threshold=.85, cohesion_threshold=.8, ambiguity_margin=.1)
    admission = advice_admission(config,'body')
    admission['groups'].append(dict(group('person_anchor_keypoints_any', [
        check('sapiensid.identities.anchor_keypoint_rules.any.0.min',1,3),
        check('sapiensid.identities.anchor_keypoint_rules.any.1.min',3,3)]), status='pass'))
    decision = dict(current=True, status='withheld', recorded=dict(
        reason='ambiguous_or_insufficient_face_body_bank',score=.8,minimum=.7,runner=.79,rival='other'))
    result = suggest(config,'body',admission,[decision],None)
    changes = {c['parameter']:c['proposed'] for p in result['paths'] for c in p['changes']}
    assert changes=={'sapiensid.identities.match_threshold':.8,'sapiensid.identities.cohesion_threshold':.7}
    assert 'rival margin also fails' in describe(result)
    assert not any('Qualify as a body anchor'==p['title'] for p in result['paths'])
    admission['groups'][-1]['status']='fail'
    result = suggest(config,'body',admission,[],None)
    assert 'Keep the configured all/any rules' in describe(result)
    assert not any('keypoint_rules' in c['parameter'] for p in result['paths'] for c in p['changes'])


def test_advice_reports_disabled_body_route_before_tuning_support(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['identities']['enabled']=False
    result = suggest(config,'body',advice_admission(config,'body'),[],None)
    assert {c['yaml_parameter']:c['proposed'] for c in result['paths'][0]['changes']}=={'matching.body.identities.enabled':True}


def test_advice_face_matching_requires_measured_candidate_without_failed_rival_margin(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest, describe
    config = advice_fixture(tmp_path)
    decision = dict(current=True, status='withheld', recorded=dict(reason='insufficient_match_or_cohesion',scores=[.62,.3]))
    admission = advice_admission(config)
    result = suggest(config,'face',admission,[decision],None)
    assert {c['yaml_parameter']:c['proposed'] for p in result['paths'] for c in p['changes']}=={'matching.face.match_threshold':.62}
    decision['recorded']['scores']=[.62,.61]
    result = suggest(config,'face',admission,[decision],None)
    assert not any(p['changes'] for p in result['paths'])
    assert 'rival margin' in describe(result)
    decision['recorded'].update(reason='co_present_identity_conflict', scores=[.9,.3])
    result = suggest(config,'face',admission,[decision],None)
    assert not any(p['changes'] for p in result['paths'])
    assert 'do not bypass' in describe(result)


def detachment_trace(reason='known_identity', **metrics):
    return dict(status='accepted', role='auxiliary', evidence_status='recorded_current_policy',
                decisions=[dict(reason=reason, stage='match', metrics=metrics, merges=[])])


def matching_patches(advice):
    return [{c['parameter']:c['proposed'] for c in p['changes']}
            for p in advice['paths'] if not p['title'].startswith(('If quality', 'Reduce the'))]


@pytest.mark.parametrize('version', [1,2])
@pytest.mark.parametrize('reason', ['known_identity','candidate_group','auxiliary_identity','anchor_view_identity'])
def test_detachment_offers_strict_matching_boundaries_in_loaded_yaml_layout(tmp_path, version, reason):
    from dataclasses import replace
    import yaml
    from stash_ai.recognition.config import plain
    from stash_ai.recognition.inspection_detachment import suggest
    from stash_ai.recognition.merge_report_policy import patched
    config = replace(advice_fixture(tmp_path), document_version=version)
    original = plain(config.data)
    trace = detachment_trace(reason, score=.91, match_score=.91, minimum=.85, runner=.7)
    advice = suggest(config, 'face', trace, advice_admission(config))
    patches = matching_patches(advice)
    if reason == 'anchor_view_identity':
        assert patches == [{'clustering.view_match_threshold':.910001},
                           {'clustering.view_cohesion_threshold':.850001},
                           {'clustering.view_ambiguity_margin':.210001}]
    else:
        assert patches == [{'recognition.match_threshold':.910001},
                           {'clustering.cohesion_threshold':.850001, 'recognition.match_threshold':.850001},
                           {'recognition.ambiguity_margin':.210001}]
    for path in advice['paths']:
        patched(config, {c['parameter']:c['proposed'] for c in path['changes']})
        assert yaml.safe_load(path['yaml']) == path['yaml_patch']
        assert all(('yaml_parameter' in c) == (version == 2) for c in path['changes'])
    assert config.data == original


def test_detachment_accounts_for_same_scene_bonus_in_effective_gates(tmp_path):
    from stash_ai.recognition.inspection_detachment import suggest
    config = advice_fixture(tmp_path)
    config.data['clustering']['same_scene_match_bonus'] = .08
    advice = suggest(config, 'face', detachment_trace('same_scene_identity', score=.68, minimum=.63), advice_admission(config))
    similarity, cohesion = matching_patches(advice)
    assert similarity['recognition.match_threshold'] == .760001
    assert similarity['recognition.match_threshold']-.08 > .68
    assert cohesion['clustering.cohesion_threshold'] == .710001
    assert cohesion['clustering.cohesion_threshold']-.08 > .63
    assert 'include this bonus' in ' '.join(advice['notes'])


@pytest.mark.parametrize('reason', ['frozen_face_body_bank','body_core_attachment','body_confirmed_core_attachment'])
def test_detachment_uses_actual_body_comparisons_and_not_certified_lower_bounds(tmp_path, reason):
    from stash_ai.recognition.inspection_detachment import suggest
    from stash_ai.recognition.inspection_advice import describe
    config = advice_fixture(tmp_path)
    admission = advice_admission(config, 'body')
    advice = suggest(config, 'body', detachment_trace(reason, score=.94, minimum=.91, runner=.7), admission)
    assert matching_patches(advice) == [
        {'sapiensid.identities.match_threshold':.940001},
        {'sapiensid.identities.cohesion_threshold':.910001},
        {'sapiensid.identities.ambiguity_margin':.240001}]
    advice = suggest(config, 'body', detachment_trace('body_core_attachment', certified_floor=.8), admission)
    assert not matching_patches(advice)
    assert 'only a lower bound' in describe(advice)


@pytest.mark.parametrize('status', ['recorded_other_policy','membership_only'])
def test_detachment_never_derives_matching_patch_from_historical_or_missing_proof(tmp_path, status):
    from stash_ai.recognition.inspection_detachment import suggest
    from stash_ai.recognition.inspection_advice import describe
    config = advice_fixture(tmp_path)
    trace = detachment_trace(score=.85, minimum=.75)
    trace['evidence_status'] = status
    advice = suggest(config, 'face', trace, advice_admission(config))
    assert not matching_patches(advice)
    assert 'No current-policy assignment proof' in describe(advice)


@pytest.mark.parametrize('route', ['merge_chain','inherited','seed','multiple','linked_face'])
def test_detachment_does_not_mistake_membership_route_for_direct_matching(tmp_path, route):
    from stash_ai.recognition.inspection_detachment import suggest
    from stash_ai.recognition.inspection_advice import describe
    config = advice_fixture(tmp_path)
    kind = 'body' if route == 'linked_face' else 'face'
    trace = detachment_trace('measured_face_association' if kind=='body' else 'candidate_group', score=.87, minimum=.75)
    step = trace['decisions'][0]
    if route == 'merge_chain':
        step['merges'] = [dict(source='old', target='new', route='global_view')]
    elif route == 'inherited':
        step['final_reason'] = 'body_global_identity'
    elif route == 'seed':
        step['stage'] = 'seed'
    elif route == 'multiple':
        trace['decisions'].append(dict(step))
    advice = suggest(config, kind, trace, advice_admission(config, kind))
    assert not matching_patches(advice)
    expected = {'merge_chain':'inherited through', 'inherited':'inherited through',
                'seed':'started a candidate group', 'multiple':'multiple or missing', 'linked_face':'linked assigned face'}
    assert expected[route] in describe(advice)


@pytest.mark.parametrize('support,expected', [(2,3),(7,8),(8,None),(None,None)])
def test_detachment_caps_support_and_warns_that_it_affects_the_whole_core(tmp_path, support, expected):
    from stash_ai.recognition.inspection_detachment import suggest
    from stash_ai.recognition.inspection_advice import describe
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['identities']['min_support'] = 2
    advice = suggest(config, 'body', detachment_trace('body_scene_core', independent_support=support), advice_admission(config,'body'))
    assert matching_patches(advice) == ([{'sapiensid.identities.min_support':expected}] if expected else [])
    if expected:
        assert 'whole core provisional' in describe(advice)


def test_detachment_quality_distinguishes_global_filter_anchor_demotion_and_scan(tmp_path):
    from stash_ai.recognition.inspection_detachment import suggest
    from stash_ai.recognition.inspection_advice import describe
    from stash_ai.recognition.merge_report_policy import patched
    from stash_ai.recognition.pose_policy import is_anchor, query_allows
    config = advice_fixture(tmp_path)
    config.data['clustering'].update(min_anchor_sharpness=20., min_auxiliary_sharpness=15.)
    trace = detachment_trace(score=.91)
    trace['role'] = 'anchor'
    advice = suggest(config,'face',trace,advice_admission(config))
    query, anchor, scan = advice['paths'][-3:]
    def modified(path):
        return patched(config, {c['parameter']:c['proposed'] for c in path['changes']})
    quality = dict(sharpness=30., pixels=128, pose_deviation=.1)
    assert not query_allows(quality, modified(query).data['clustering'])
    assert not is_anchor(quality, modified(anchor).data['clustering'])
    assert query_allows(quality, modified(anchor).data['clustering'])
    assert 'may remain assigned as auxiliary' in describe(advice)
    assert modified(scan).data['detection']['min_sharpness'] > 30.
    assert 'fresh scan evidence' in scan['next_action']
    assert not any('keypoint_rules' in c['parameter'] for p in advice['paths'] for c in p['changes'])


@pytest.mark.parametrize('score', [1., None, float('nan'), float('inf')])
def test_detachment_does_not_guess_impossible_or_missing_similarity(tmp_path, score):
    from stash_ai.recognition.inspection_detachment import suggest
    config = advice_fixture(tmp_path)
    advice = suggest(config,'face',detachment_trace(score=score),advice_admission(config))
    assert not matching_patches(advice)


def test_detachment_scan_uses_strict_detector_and_inclusive_pose_boundaries(tmp_path):
    from stash_ai.recognition.inspection_detachment import suggest
    config = advice_fixture(tmp_path)
    admission = advice_admission(config, pixels=None, sharpness=None, pose_deviation=None)
    advice = suggest(config,'face',detachment_trace(),admission)
    assert {c['parameter']:c['proposed'] for c in advice['paths'][-1]['changes']} == {'detection.score_threshold':.99}
    # Equality rejects a strict detector score gate; a pose maximum must be below the measurement.
    admission = advice_admission(config, pixels=None, sharpness=None, pose_deviation=.15)
    advice = suggest(config,'face',detachment_trace(),admission)
    assert {c['parameter']:c['proposed'] for c in advice['paths'][-1]['changes']} == {
        'detection.max_pose_deviation':.149999, 'clustering.max_anchor_pose_deviation':.149999}


def test_assigned_body_advice_reads_retained_metadata_without_mutation(tmp_path):
    import json
    from stash_ai.recognition.database import FaceDatabase
    config, person = scene_fixture(tmp_path)
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE body_members SET proof=? WHERE observation='accepted-body'",
            (json.dumps(dict(reason='body_core_attachment',score=.94,minimum=.91)),))
        state.connection.commit()
    before = config.database.read_bytes()
    reader = IdentityReader(config, identity_id=person)
    detail = reader.detail('body','accepted-body')
    assert matching_patches(detail['suggestions'])[0] == {'sapiensid.identities.match_threshold':.940001}
    assert detail['contract_current']
    assert detail['admission']['route'] == 'independent'
    assert reader.detail('body','provisional-body')['suggestions'] is None
    assert config.database.read_bytes() == before


@pytest.mark.parametrize('signal', ['face','face_and_body'])
def test_detachment_hybrid_query_never_uses_body_measurement_as_a_face_floor(tmp_path, signal):
    from stash_ai.recognition.inspection_detachment import suggest
    config = advice_fixture(tmp_path)
    advice = suggest(config, 'face', detachment_trace('hybrid_query',signal=signal,score=.94,minimum=.91,rival_score=.7),
                     advice_admission(config))
    patches = matching_patches(advice)
    if signal == 'face':
        assert patches == [{'recognition.match_threshold':.940001},
                           {'clustering.cohesion_threshold':.910001,'recognition.match_threshold':.910001},
                           {'recognition.ambiguity_margin':.240001}]
    else:
        assert patches == [{'sapiensid.matching.person_threshold':.910001},
                           {'sapiensid.matching.ambiguity_margin':.240001}]


def test_detachment_track_uses_measured_floor_and_can_pin_inherited_threshold(tmp_path):
    from stash_ai.recognition.inspection_detachment import suggest
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['identities']['track_similarity_threshold'] = None
    advice = suggest(config, 'body', detachment_trace('local_body_track',track_floor=.91,runner=.7),advice_admission(config,'body'))
    assert matching_patches(advice) == [{'sapiensid.identities.track_similarity_threshold':.910001},
                                      {'sapiensid.identities.ambiguity_margin':.210001}]


def test_detachment_candidate_seed_stage_survives_reader_projection(tmp_path):
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.inspection_advice import describe
    config, person, _ = seed(tmp_path,1)
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE candidates SET payload=json_set(payload, '$.reason','candidate_group',"
                                 "'$.face_comparison.stage','seed','$.face_comparison.required_support',2)")
        state.connection.commit()
    reader = IdentityReader(config, identity_id=person)
    detail = reader.detail('face',reader.page()[0][0]['id'])
    assert detail['trace']['decisions'][0]['stage'] == 'seed'
    assert not matching_patches(detail['suggestions'])
    assert 'started a candidate group' in describe(detail['suggestions'])


def test_member_state_groups_filter_before_pagination_and_match_detail_roles(tmp_path, monkeypatch):
    import json
    from stash_ai.recognition import identity_inspection
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.models import namespace
    config, person = scene_fixture(tmp_path)
    reader = IdentityReader(config, identity_id=person)
    faces = reader.page(kind='face')[0]
    with FaceDatabase(config.database) as state:
        db = state.connection
        candidates = [dict(db.execute('SELECT * FROM candidates WHERE json_extract(payload,\'$.observation\')=?', (r['id'],)).fetchone()) for r in faces]
        # IDs are random and multiple observations share a cue; set roles
        # explicitly instead of assuming the page preserves insertion order.
        for index,candidate in enumerate(candidates):
            payload=json.loads(candidate['payload'])
            payload['reason']='known_identity' if index in (0,4,5) else 'auxiliary_identity'
            candidate['payload']=json.dumps(payload)
            db.execute('UPDATE candidates SET payload=? WHERE id=?',(candidate['payload'],candidate['id']))
        # Current decisions win over historical roles, even when inserted earlier.
        old = dict(candidates[0], id='older-role', policy='old-policy')
        payload = json.loads(old['payload']); payload['reason']='auxiliary_identity'
        old['payload']=json.dumps(payload)
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',tuple(old.values()))
        db.execute('UPDATE candidates SET policy=? WHERE id=?',('old-policy',candidates[1]['id']))
        db.execute('DELETE FROM candidates WHERE id=?',(candidates[2]['id'],))
        db.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','hybrid_query') WHERE id=?",(candidates[3]['id'],))
        mixed = dict(candidates[4],id='mixed-current')
        payload=json.loads(mixed['payload']); payload['reason']='auxiliary_identity'
        mixed['payload']=json.dumps(payload)
        db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',tuple(mixed.values()))
        # A predecessor's accepted proof still explains membership in the survivor.
        db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',('predecessor','scope',namespace(config),0,None,'predecessor-token'))
        db.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',('predecessor',person,1,'{}','local'))
        db.execute("UPDATE candidates SET person='predecessor',payload=json_set(payload,'$.reason','known_identity') WHERE id=?",(candidates[5]['id'],))
        db.commit()
    before=config.database.read_bytes()
    expected = {r['id']:reader.detail('face',r['id'])['trace']['role'] for r in faces}
    monkeypatch.setattr(identity_inspection,'PAGE_SIZE',1)
    with monkeypatch.context() as context:
        context.setattr(identity_inspection,'cold_crop',lambda *_: pytest.fail('Grouping loaded pixels'))
        context.setattr('stash_ai.recognition.decision_contexts.unpack',lambda *_: pytest.fail('Grouping loaded full decision proofs'))
        groups=reader.states('face')
        assert {g['state']:g['count'] for g in groups}=={'anchor':2,'auxiliary':1,'attachment':1,'unknown':2}
        seen=[]
        for group in groups:
            for offset in range(group['count']):
                rows,more=reader.page(kind='face',state=group['state'],offset=offset,accepted_only=True)
                assert len(rows)==1 and more==(offset+1<group['count'])
                row=rows[0]
                assert row['state']==row['role']==expected[row['id']]==group['state']
                seen.append(row['id'])
        assert len(set(seen))==len(seen)==6
        assert reader.states('body')==[dict(state='auxiliary',count=1)]
        assert reader.page(kind='body',state='anchor',accepted_only=True)==([],False)
        assert reader.page(kind='face',state="' OR 1=1 --")==([],False)
        assert reader.page(kind='body',state='provisional')[0][0]['id']=='provisional-body'
        assert IdentityReader(config,identity_id='internal').states('face')==[dict(state='unknown',count=1)]
        with pytest.raises(ValueError,match='kind'):
            reader.states('invalid')
    assert config.database.read_bytes()==before

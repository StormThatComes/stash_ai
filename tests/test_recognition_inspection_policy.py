"""Current inspector results never mix scans or collection configurations."""
from dataclasses import replace
import json

import pytest

from stash_ai.recognition.body_observations import contract
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.identity_inspection import IdentityReader
from stash_ai.recognition.inspection_context import locate
from stash_ai.recognition.inspection_diagnostics import stages, unassigned_summary, review_queue
from stash_ai.recognition.inspection_search import SearchReader
from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
from tests.test_recognition_identity_inspector import scene_fixture


def insert(db, table, row):
    db.execute(f'INSERT INTO {table} VALUES ('+','.join('?' for _ in row)+')', tuple(row.values()))


def test_rerun_lower_keypoint_gate_hides_old_accepted_and_rejected_bodies(tmp_path):
    config, owner = scene_fixture(tmp_path)
    config = replace(config, document_version=2)
    old = replace(config, data=plain(config.data))
    old.data['sapiensid']['collection']['min_keypoints'] = 7
    assert config.data['sapiensid']['collection']['min_keypoints'] == 4
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('UPDATE body_observations SET contract=?', (contract(old),))
        run = state.start_run('scope', ['2'], old.snapshot())
        state.finish_run(run, 'success', {})
        db.commit()
    scene = SceneReader(config, '2')
    catalogue = scene.catalogue()
    # Editing YAML without collecting new evidence must not fall back to old bodies.
    assert not UnassignedReader(config, catalogue['scan']).page(kind='body')[0]
    assert not scene.identity_reader(owner, 'scope').page(kind='body')[0]
    with FaceDatabase(config.database) as state:
        db = state.connection
        template = dict(db.execute("SELECT * FROM body_observations WHERE id='rejected-body'").fetchone())
        metadata = dict(person=dict(bounds=[0, 0, 160, 160], score=.99),
                        quality=dict(keypoints=6, sharpness=100., overlap=0.))
        insert(db, 'body_observations', dict(template, id='current-body', contract=contract(config),
               rejected=None, quality=json.dumps(metadata)))
        db.commit()
    before = config.database.read_bytes()
    catalogue = scene.catalogue()
    reader = UnassignedReader(config, catalogue['scan'])
    assert [r['id'] for r in reader.page(kind='body')[0]] == ['current-body']
    assert {g['identity']: g['count'] for g in catalogue['groups']} == {owner: 3, 'internal': 1}
    assert catalogue['unassigned'] == 3
    detail = reader.detail('body', 'current-body', pixels=False)
    gate = next(c for g in stages(detail) for c in g['children'] if c['label']=='body.collection.min_keypoints')
    assert gate['value'] == '6 >= 4' and gate['status'] == 'pass'
    collection = next(g for g in detail['admission']['groups'] if g['stage']=='independent_person_collection')
    assert collection['configuration_source'] == 'current_configuration_hash_verified'
    assert 'sapiensid.collection.min_keypoints' in {c['parameter'] for c in collection['checks']}
    summary = unassigned_summary(scene)
    assert summary['total'] == 3
    assert next(g for g in summary['gates'] if g['parameter']=='body.collection.min_keypoints')['threshold'] == 4
    assert [r['id'] for r in SearchReader(scene, {'kind': 'body'}).iter_rows()] == ['current-body']
    assert review_queue(scene)['seen'] == 7
    with pytest.raises(ValueError, match='current'):
        reader.detail('body', 'rejected-body', pixels=False)
    with pytest.raises(ValueError, match='current'):
        IdentityReader(config, identity_id=owner).detail('body', 'accepted-body', pixels=False)
    with pytest.raises(ValueError, match='refresh'):
        locate(config, 'body', 'accepted-body', 'scope')
    assert config.database.read_bytes() == before


def test_all_scene_members_search_and_counts_exclude_superseded_scans(tmp_path):
    from stash_ai.recognition.scan_heads import activate
    config, owner = scene_fixture(tmp_path)
    scene = SceneReader(config, '2')
    original = scene.catalogue()
    expected = {r['id'] for r in SearchReader(scene).iter_rows()}
    with FaceDatabase(config.database) as state:
        db = state.connection
        for scan in list(db.execute('SELECT * FROM scans')):
            old = dict(scan, id='old-'+scan['scene'], input_sha='old-input')
            insert(db, 'scans', old)
            sample = dict(db.execute('SELECT * FROM observations WHERE scan=? LIMIT 1', (scan['id'],)).fetchone())
            insert(db, 'observations', dict(sample, id='old-face-'+scan['scene'], scan=old['id']))
            db.execute('INSERT INTO members VALUES (?,?)', ('old-face-'+scan['scene'], owner))
            insert(db, 'observations', dict(sample, id='old-unassigned-'+scan['scene'], scan=old['id']))
            body = dict(db.execute("SELECT * FROM body_observations WHERE id='accepted-body'").fetchone())
            insert(db, 'body_observations', dict(body, id='old-body-'+scan['scene'], scan=old['id']))
            db.execute('INSERT INTO body_members SELECT ?,person,role,status,policy,proof FROM body_members '
                       "WHERE observation='accepted-body'", ('old-body-'+scan['scene'],))
            # Reusing an earlier-created scan must override creation order.
            activate(db, scan)
        db.commit()
    assert scene.catalogue() == original
    assert {r['id'] for r in SearchReader(scene).iter_rows()} == expected
    member = scene.identity_reader(owner, 'scope')
    assert sum(g['count'] for g in member.states('face')) == 6
    assert sum(g['count'] for g in member.states('body')) == 1
    assert not member.page(observation='old-face-10')[0]
    with pytest.raises(ValueError, match='current'):
        member.detail('face', 'old-face-10', pixels=False)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute("UPDATE scans SET policy_sha='different-yaml' WHERE id=?", (original['scan']['id'],))
        db.commit()
    with pytest.raises(ValueError, match='Run recognition'):
        scene.catalogue()
    assert not member.page(scene='2')[0]


def test_reloading_yaml_preserves_cli_overrides_and_validates_new_settings(tmp_path):
    path = tmp_path/'faces.yml'
    path.write_text('version: 2\nbody: {collection: {min_keypoints: 7}}\nstorage: {database: first.sqlite3}\n')
    original = load(path, database=tmp_path/'cli.sqlite3', device='cpu')
    from_yaml = load(path)
    path.write_text('version: 2\nbody: {collection: {min_keypoints: 4}}\nstorage: {database: second.sqlite3}\n')
    reloaded = original.reload()
    assert reloaded.data['sapiensid']['collection']['min_keypoints'] == 4
    assert original.data['sapiensid']['collection']['min_keypoints'] == 7
    assert reloaded.database == original.database == tmp_path/'cli.sqlite3'
    assert reloaded.data['models']['device'] == 'cpu'
    assert from_yaml.reload().database == tmp_path/'second.sqlite3'
    path.write_text('version: 2\npose: {body: {collection: {min_keypoints: 4}}}\n')
    with pytest.raises(ValueError, match='pose.body.collection.min_keypoints'):
        original.reload()
    assert load(root=tmp_path/'no-config').reload().source_path is None


@pytest.mark.parametrize('version', [1, 2])
def test_inspector_parameter_names_follow_the_loaded_yaml_version(tmp_path, version):
    from stash_ai.recognition.identity_window import describe_diagnostics
    from stash_ai.recognition.config_layout import public_path
    config, owner = scene_fixture(tmp_path)
    config = replace(config, document_version=version)
    reader = IdentityReader(config, identity_id=owner)
    for kind in ('face', 'body'):
        detail = reader.detail(kind, reader.page(kind=kind)[0][0]['id'], pixels=False)
        expected = []
        for group in detail['admission']['groups']:
            for check in group['checks']:
                path = check['parameter']
                try:
                    public = public_path(path) if version==2 else path
                except KeyError:
                    public = path
                assert check['yaml_parameter'] == public
                expected.append(public)
        labels = [c['label'] for g in stages(detail) for c in g['children']]
        assert all(path in labels and path in describe_diagnostics(detail) for path in expected)


def test_scene_refresh_reloads_yaml_and_does_not_restore_an_obsolete_selection(tmp_path):
    import gc
    import os
    import time
    tk = pytest.importorskip('tkinter')
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if os.environ.get('STASH_AI_DESKTOP_TEST'):
            raise
        pytest.skip(f'No graphical desktop: {error}')
    from stash_ai.recognition.scene_window import SceneInspector
    config, _ = scene_fixture(tmp_path)
    original_yaml = config.source_path.read_text()
    config.source_path.write_text(original_yaml+'sapiensid: {collection: {min_keypoints: 7}}\n')
    config = config.reload()
    with FaceDatabase(config.database) as state:
        state.connection.execute('UPDATE body_observations SET contract=?', (contract(config),))
        state.connection.commit()
    ui = SceneInspector(root, SceneReader(config, '2'), 'Refresh test')

    def wait_for(condition):
        deadline = time.monotonic()+8
        while not condition() and time.monotonic()<deadline:
            root.update()
            time.sleep(.005)
        assert condition(), ui.status.get()

    try:
        wait_for(lambda: bool(ui.branches))
        ui.view_filters = {'kind': 'body', 'assignment': 'unassigned'}
        ui.search()
        wait_for(lambda: len(ui.rows)==3)
        selected = next(k for k, v in ui.rows.items() if v['id']=='rejected-body')
        ui.listing.selection_set(selected)
        wait_for(lambda: ui.detail_data is not None)
        config.source_path.write_text(original_yaml+'sapiensid: {collection: {min_keypoints: 4}}\n')
        current = config.reload()
        with FaceDatabase(config.database) as state:
            db = state.connection
            sample = dict(db.execute("SELECT * FROM body_observations WHERE id='rejected-body'").fetchone())
            insert(db, 'body_observations', dict(sample, id='new-body', contract=contract(current)))
            db.commit()
        ui.search()
        wait_for(lambda: len(ui.rows)==1 and not ui.restoring and ui.tree_worker.future is None and ui.worker.future is None)
        assert {r['id'] for r in ui.rows.values()} == {'new-body'}
        assert ui.reader.config.data['sapiensid']['collection']['min_keypoints'] == 4
        assert ui.detail_data is None and ui.current_row is None
        assert ui.reader.config.database == config.database
        config.source_path.write_text('version: 2\nunknown: invalid\n')
        ui.search()
        assert 'unknown' in ui.status.get()
        assert ui.reader.config.data['sapiensid']['collection']['min_keypoints'] == 4
        assert {r['id'] for r in ui.rows.values()} == {'new-body'}
    finally:
        ui.close()
        root.destroy()
        del ui, root
        gc.collect()

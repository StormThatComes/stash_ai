"""Recovery skips only proven committed strict prefixes, never source evidence."""
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import uuid

import numpy as np
import pytest

from stash_ai.recognition import rebuild, replay_checkpoint as checkpoint
from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase, encoded
from stash_ai.recognition.replay_memory import memory_replay
from tests.test_recognition_clustering import setup, scan


def configured(config, **execution):
    data = plain(config.data)
    data['execution'].update(rebuild_checkpoint_scenes=1, profile_rebuild=True, **execution)
    return replace(config, data=data)


def selection(state, config, count=5):
    result = []
    for i in range(count):
        vector = np.eye(4, dtype='float32')[(i//2)%4]
        identifier, _ = scan(state, config, str(i+1), [vector]*3)
        result.append(dict(scene=str(i+1),scan=identifier))
    return result


def grouping(db):
    return {table:[tuple(r) for r in db.execute(f'SELECT * FROM {table} ORDER BY 1,2')]
            for table in (*checkpoint.TABLES,'identity_origins')}


def interrupt(monkeypatch, state, selected, config, after=2):
    original = rebuild.cluster_scene
    calls = []
    def cluster(*args, **kwargs):
        if len(calls) == after:
            raise KeyboardInterrupt('process lost before next scene')
        calls.append(args[1])
        return original(*args, **kwargs)
    before = grouping(state.connection)
    with monkeypatch.context() as patch:
        patch.setattr(rebuild,'cluster_scene',cluster)
        with pytest.raises(KeyboardInterrupt):
            rebuild.rebuild_selected(state, selected, config)
    assert grouping(state.connection) == before
    return calls


def count_replay(monkeypatch):
    original = rebuild.cluster_scene
    calls = []
    def cluster(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(rebuild,'cluster_scene',cluster)
    return calls


def files(config):
    directory = Path(config.database).with_name(Path(config.database).name+'.rebuild')
    manifest = json.loads((directory/'current.json').read_text())
    return directory, manifest, directory/manifest['generation']


def invalidate_warm(state):
    state.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-rebuild%'")
    state.connection.commit()


def test_resume_exact_prefix_and_clock_matches_full_replay(tmp_path, monkeypatch):
    config = configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected = selection(state, config)
        interrupt(monkeypatch,state,selected,config,after=3)
        directory, _, generation = files(config)
        with closing(sqlite3.connect(generation)) as saved, saved:
            header = json.loads(saved.execute('SELECT value FROM metadata').fetchone()[0])
            # A known identity visit consumes a clock tick without inserting a
            # new person. max(created)+1 would silently change later evidence.
            assert header['next_clock'] == 4
            assert saved.execute('SELECT MAX(created) FROM people').fetchone()[0] == 3
            assert set(r[0] for r in saved.execute("SELECT name FROM sqlite_master WHERE type='table'")) == set(checkpoint.TABLES)|{'metadata'}
        calls = count_replay(monkeypatch)
        actual = rebuild.rebuild_selected(state,selected,config)
        assert calls == [i['scan'] for i in selected[3:]]
        assert state.last_rebuild_profile['counts']['checkpoint_resumed_scenes'] == 3
        expected = grouping(state.connection)
        invalidate_warm(state)
        calls.clear()
        disabled = replace(config,data={**config.data,'execution':{**config.data['execution'],'rebuild_checkpoint_scenes':0}})
        assert rebuild.rebuild_selected(state,selected,disabled) == actual
        assert grouping(state.connection) == expected
        assert len(calls) == 5
        assert len(list(directory.glob('replay-*.sqlite3'))) == 1


def test_known_visits_keep_next_clock_beyond_max_created(tmp_path,monkeypatch):
    config = configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected = selection(state,config)
        interrupt(monkeypatch,state,selected,config,after=2)
        _,_,generation=files(config)
        with closing(sqlite3.connect(generation)) as saved, saved:
            assert json.loads(saved.execute('SELECT value FROM metadata').fetchone()[0])['next_clock']==3
            assert saved.execute('SELECT MAX(created) FROM people').fetchone()[0]==1
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert calls==[i['scan'] for i in selected[2:]]


@pytest.mark.parametrize('mutation',[
    "UPDATE observations SET embedding=CAST(embedding||x'00' AS BLOB) WHERE scan=(SELECT id FROM scans WHERE scene='5')",
    "UPDATE scans SET input_sha='repaired' WHERE scene='5'",
])
def test_future_selected_raw_dependency_change_rejects_prefix(tmp_path,monkeypatch,mutation):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config)
        state.connection.execute(mutation);state.connection.commit()
        calls=count_replay(monkeypatch)
        try:
            rebuild.rebuild_selected(state,selected,config)
        except ValueError as error:
            assert 'embedding' in str(error)
        assert calls[:2]==[i['scan'] for i in selected[:2]]


@pytest.mark.parametrize('change',['policy','capacity','engine'])
def test_semantic_version_or_reduced_admission_replays(tmp_path,monkeypatch,change):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config)
        data=plain(config.data)
        if change=='policy':data['recognition']['match_threshold']=.9
        elif change=='capacity':data['clustering']['max_reference_bytes']//=2
        else:monkeypatch.setattr(checkpoint,'ENGINE','future-engine')
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,replace(config,data=data))
        assert len(calls)==5


@pytest.mark.parametrize('mutation',[
    "CREATE TRIGGER untrusted AFTER INSERT ON people BEGIN DELETE FROM people; END",
    "CREATE TABLE surprise(value TEXT)",
    "UPDATE candidates SET payload='{}'",
    "UPDATE metadata SET value=json_set(value,'$.cursor',500)",
])
def test_corrupt_or_executable_checkpoint_rejected_without_source_loss(tmp_path,monkeypatch,mutation):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config)
        _,_,generation=files(config)
        with closing(sqlite3.connect(generation)) as damaged, damaged:damaged.execute(mutation)
        raw=[tuple(r) for r in state.connection.execute('SELECT * FROM observations ORDER BY id')]
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==5
        assert [tuple(r) for r in state.connection.execute('SELECT * FROM observations ORDER BY id')]==raw


@pytest.mark.parametrize('foreign',['directory','manifest','symlink'])
def test_foreign_recovery_paths_untouched(tmp_path,monkeypatch,foreign):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        directory=state.path.with_name(state.path.name+'.rebuild')
        directory.mkdir()
        if foreign=='directory':
            marker=directory/'personal.txt';marker.write_bytes(b'keep me')
        else:
            (directory/'owner.json').write_text(encoded(dict(format=checkpoint.FORMAT,
                installation=state.installation,path=checkpoint.hashlib.sha256(str(state.path).encode()).hexdigest())))
            marker=directory/'current.json'
            if foreign=='manifest':marker.write_bytes(b'{"foreign":true}')
            else:
                target=tmp_path/'personal.json';target.write_bytes(b'keep me')
                try:marker.symlink_to(target)
                except OSError:pytest.skip('OS does not permit creating symlinks')
        before=marker.read_bytes()
        rebuild.rebuild_selected(state,selected,config)
        assert marker.read_bytes()==before
        assert state.last_rebuild_profile['counts']['checkpoint_failures']==1


def test_failed_staged_save_preserves_last_committed_generation(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config)
        directory,manifest,generation=files(config)
        original=checkpoint._exclusive_json
        def fail(path,value):
            if path.name.startswith('.manifest-'):
                raise OSError('full device')
            return original(path,value)
        with monkeypatch.context() as patch:
            patch.setattr(checkpoint,'_exclusive_json',fail)
            rebuild.rebuild_selected(state,selected,config)
        assert json.loads((directory/'current.json').read_text())==manifest
        assert generation.exists() and len(list(directory.glob('replay-*.sqlite3')))==1
        invalidate_warm(state)
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==3


def test_safe_append_reuses_prefix_and_unsafe_future_copresence_replays(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        rebuild.rebuild_selected(state,selected[:3],config)
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert calls==[i['scan'] for i in selected[3:]]
        assert state.last_rebuild_profile['counts']['checkpoint_append_reused_scenes']==3
        expected=grouping(state.connection)
        invalidate_warm(state)
        disabled=replace(config,data={**config.data,'execution':{**config.data['execution'],'rebuild_checkpoint_scenes':0}})
        rebuild.rebuild_selected(state,selected,disabled)
        assert grouping(state.connection)==expected
        # A second distinct face in the same future cue invalidates the proof,
        # even though it has no accepted membership in the prefix.
        identifier,_=scan(state,config,'6',[np.eye(4)[0]]*3)
        row=state.connection.execute('SELECT * FROM observations WHERE scan=? LIMIT 1',(identifier,)).fetchone()
        values=list(row);values[0]=str(uuid.uuid4());values[6]=encoded({'box':[9,9,20,20]})
        values[10]=np.eye(4,dtype='<f4')[3].tobytes()
        state.connection.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',values)
        state.connection.commit()
        calls.clear()
        rebuild.rebuild_selected(state,selected+[dict(scene='6',scan=identifier)],config)
        assert len(calls)==6


def test_same_scene_pass_only_after_entire_strict_resume(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        calls=[];original=rebuild.apply_scene_context
        def context(*args,**kwargs):
            calls.append([r[0] for r in args[0].connection.execute('SELECT DISTINCT scan FROM candidates')])
            return original(*args,**kwargs)
        monkeypatch.setattr(rebuild,'apply_scene_context',context)
        interrupt(monkeypatch,state,selected,config)
        assert not calls
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==1 and set(calls[0])=={i['scan'] for i in selected}


def test_resume_reloads_current_historical_labels(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        # Establish historical profiles before a failed private replay.
        rebuild.rebuild_selected(state,selected,replace(config,data={**config.data,'execution':
            {**config.data['execution'],'rebuild_checkpoint_scenes':0}}))
        invalidate_warm(state)
        interrupt(monkeypatch,state,selected,config)
        person=state.connection.execute('SELECT id FROM people ORDER BY created LIMIT 1').fetchone()[0]
        state.connection.execute("UPDATE people SET performer='current-label' WHERE id=?",(person,));state.connection.commit()
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==3
        assert state.connection.execute('SELECT performer FROM people WHERE id=?',(person,)).fetchone()[0]=='current-label'


def test_checkpoint_cap_failure_is_optional_and_preserves_rebuild(tmp_path,monkeypatch):
    config=configured(setup(tmp_path),rebuild_checkpoint_bytes=1048576)
    original=checkpoint._rows
    def huge(db,table,order):
        for batch in original(db,table,order):
            if table=='candidates':
                batch=[(*row[:-1],encoded({'large':'x'*1048576})) for row in batch]
            yield batch
    monkeypatch.setattr(checkpoint,'_rows',huge)
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        assert rebuild.rebuild_selected(state,selected,config)
        assert state.last_rebuild_profile['counts']['checkpoint_failures']==1
        directory=state.path.with_name(state.path.name+'.rebuild')
        assert not list(directory.glob('replay-*.sqlite3'))


def test_process_exit_during_next_generation_keeps_previous_prefix(tmp_path,monkeypatch):
    import subprocess
    import sys
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        initial=grouping(state.connection)
    script='''
import os,sys
from dataclasses import replace
from stash_ai.recognition.config import load,plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition import rebuild,replay_checkpoint as checkpoint
config=load(sys.argv[1]);data=plain(config.data)
data['execution']['rebuild_checkpoint_scenes']=1
config=replace(config,data=data)
original=checkpoint._exclusive_json
calls=0
def crash(path,value):
    global calls
    if path.name.startswith('.manifest-'):
        calls+=1
        if calls==3:os._exit(23)
    return original(path,value)
checkpoint._exclusive_json=crash
with FaceDatabase(config.database) as state:
    selected=[dict(scene=r['scene'],scan=r['id']) for r in state.connection.execute('SELECT id,scene FROM scans ORDER BY CAST(scene AS INTEGER)')]
    rebuild.rebuild_selected(state,selected,config)
'''
    child=subprocess.run([sys.executable,'-c',script,str(tmp_path/'face.yml')],capture_output=True,text=True,timeout=30)
    assert child.returncode==23,child.stderr
    directory,_,_=files(config)
    assert len(list(directory.glob('replay-*.sqlite3')))==2
    with FaceDatabase(config.database) as state:
        assert grouping(state.connection)==initial
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert calls==[i['scan'] for i in selected[2:]]
        assert len(list(directory.glob('replay-*.sqlite3')))==1


def test_resumed_merge_scores_support_aliases_and_clock_match_full_strict(tmp_path,monkeypatch):
    from tests.test_recognition_rebuild import merge_evidence
    config=configured(setup(tmp_path,{'cohesion_threshold':.35}))
    data=plain(config.data);data['recognition']['min_merge_support']=1
    config=replace(config,data=data)
    with FaceDatabase(config.database) as state:
        selected=[]
        for scene,vectors in merge_evidence().items():
            identifier,_=scan(state,config,scene,[np.asarray(v,dtype='float32') for v in vectors],count=len(vectors))
            selected.append(dict(scene=scene,scan=identifier))
        # Repeat a known profile before introducing another identity, making
        # next_clock observably different from max(person.created)+1.
        identifier,_=scan(state,config,'4',[np.array([1.,0,0,0])]*3)
        selected.append(dict(scene='4',scan=identifier))
        interrupt(monkeypatch,state,selected,config,after=2)
        snapshots=[];original=rebuild._bind
        def bind(database,replay,*args,**kwargs):
            snapshot={table:[tuple(r) for r in replay.execute(f'SELECT * FROM {table} ORDER BY 1,2')]
                      for table in checkpoint.TABLES}
            snapshot['identity_merges']=[r[:2]+(0,)+r[3:] for r in snapshot['identity_merges']]
            snapshots.append(snapshot)
            return original(database,replay,*args,**kwargs)
        monkeypatch.setattr(rebuild,'_bind',bind)
        rebuild.rebuild_selected(state,selected,config)
        assert snapshots[0]['identity_merges'] and snapshots[0]['merge_support']
        invalidate_warm(state)
        disabled=replace(config,data={**config.data,'execution':{**config.data['execution'],'rebuild_checkpoint_scenes':0}})
        rebuild.rebuild_selected(state,selected,disabled)
        assert snapshots[0]==snapshots[1]


def test_valid_checksum_cannot_hide_incomplete_or_future_grouping(tmp_path,monkeypatch):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config)
        directory,manifest,generation=files(config)
        with closing(sqlite3.connect(generation)) as damaged, damaged:
            damaged.execute('DELETE FROM candidates WHERE id=(SELECT id FROM candidates LIMIT 1)')
            damaged.commit()
            header=checkpoint._header(damaged)
            manifest['checksum']=checkpoint._checksum(damaged,header)
        (directory/'current.json').write_text(encoded(manifest))
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==5
        assert state.last_rebuild_profile['counts']['checkpoint_rejections']==1


def test_checkpoint_disabled_has_no_sidecar_and_no_dependency_hash(tmp_path,monkeypatch):
    config=setup(tmp_path)
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        # Final descriptor has one input hash; no extra checkpoint scan occurs.
        original=rebuild.rebuild_cache.input_context;calls=[]
        def inputs(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
        monkeypatch.setattr(rebuild.rebuild_cache,'input_context',inputs)
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==1
        assert not state.path.with_name(state.path.name+'.rebuild').exists()


def test_resumed_scene_progress_starts_at_committed_cursor(tmp_path,monkeypatch):
    from tests.test_recognition_phase_progress import Terminal
    from stash_ai.recognition import progress
    from stash_ai.runtime.console import present
    from stash_ai.runtime.report import RunReport
    from tqdm import tqdm
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config)
        bars=[]
        class Bar(tqdm):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);bars.append(self)
        monkeypatch.setattr(progress,'tqdm',Bar)
        report=RunReport(write=lambda _:None)
        monkeypatch.setattr('sys.stderr',Terminal())
        with present(report,'auto'):
            rebuild.rebuild_selected(state,selected,config)
        rows=[b for b in bars if b.desc=='Rebuild selected identities']
        assert rows[0].initial==2 and rows[-1].n==rows[-1].total==5
        assert all(row.disable for row in rows)


@pytest.mark.parametrize('mutation',[
    "UPDATE metadata SET value=json_set(value,'$.next_clock',1)",
    "INSERT INTO identity_merges SELECT s.id,t.id,1,'{}','pending' FROM people s JOIN people t ON s.created<t.created LIMIT 1",
    "UPDATE candidates SET id='invalid' WHERE id=(SELECT id FROM candidates LIMIT 1)",
    "UPDATE candidates SET status='pending' WHERE id=(SELECT id FROM candidates LIMIT 1)",
])
def test_valid_checksum_still_requires_grouping_invariants(tmp_path,monkeypatch,mutation):
    config=configured(setup(tmp_path))
    with FaceDatabase(config.database) as state:
        selected=selection(state,config)
        interrupt(monkeypatch,state,selected,config,after=3)
        directory,manifest,generation=files(config)
        with closing(sqlite3.connect(generation)) as damaged, damaged:
            damaged.execute(mutation);damaged.commit()
            manifest['checksum']=checkpoint._checksum(damaged,checkpoint._header(damaged))
        (directory/'current.json').write_text(encoded(manifest))
        calls=count_replay(monkeypatch)
        rebuild.rebuild_selected(state,selected,config)
        assert len(calls)==5


def test_partial_manifest_fsync_failure_removes_only_its_own_stage(tmp_path,monkeypatch):
    stage=tmp_path/'stage.json'
    def failed_sync(*args):raise OSError('failed fsync')
    monkeypatch.setattr(checkpoint.os,'fsync',failed_sync)
    with pytest.raises(OSError,match='failed fsync'):
        checkpoint._exclusive_json(stage,{'owner':'test'})
    assert not stage.exists()
    stage.write_bytes(b'foreign existing bytes')
    with pytest.raises(FileExistsError):
        checkpoint._exclusive_json(stage,{'owner':'test'})
    assert stage.read_bytes()==b'foreign existing bytes'

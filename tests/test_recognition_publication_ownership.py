import pytest
import json
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.publication import publish_person
from stash_ai.recognition.scene_sync import remove_stale_links
from stash_ai.recognition.run import instance_scope
from tests.test_recognition_run import setup as run_setup, run
from tests.test_recognition_rebuild import changed
from tests.test_recognition_run import run


def test_failed_add_does_not_adopt_a_later_manual_scene_link(tmp_path, monkeypatch):
    from stash_ai.runtime.report import RunReport
    config, client, services, _ = run_setup(tmp_path, monkeypatch)
    add=client.add_scene_person
    class NotDispatched(OSError):write_outcome='failed'
    def failed(*a):raise NotDispatched('ADD never dispatched')
    client.add_scene_person=failed
    assert run(config, services, '1') == 1
    performer=next(identifier for identifier in client.profiles if identifier!='77')
    client.links['1'].append(performer)  # A human adds the link after the failure.
    client.add_scene_person=add
    scope = instance_scope(client.url)
    with FaceDatabase(config.database) as state:
        person = state.connection.execute('SELECT * FROM people').fetchone()
        assert state.connection.execute("SELECT status FROM publications WHERE kind='link'").fetchone()[0]=='failed'
        publish_person(state, client, RunReport(False), scope, '1', person['id'], reconcile_merges=False)
        record = state.connection.execute("SELECT * FROM publications WHERE kind='link'").fetchone()
        assert json.loads(record['payload'])['managed'] is False
        assert remove_stale_links(state, client, RunReport(False), scope, '1', set(), person['namespace']) == 0
        assert person['performer'] in client.links['1']


def test_undispatched_legacy_alias_reports_withheld_metadata(tmp_path, monkeypatch):
    from tests.test_recognition_rebuild import library, two_people
    config,client,services,_=library(tmp_path,monkeypatch,two_people())
    strict=changed(config,recognition={'match_threshold':.95})
    assert run(strict,services,'1','2') == 0
    with FaceDatabase(config.database) as db:
        root,source=db.connection.execute('SELECT * FROM people ORDER BY created,id').fetchall()
        # State left by the old local merge transaction before remote dispatch.
        db.connection.execute('INSERT INTO identity_merges VALUES (?,?,?,?,?)',
            (source['id'],root['id'],1.,'{}','pending'))
        db.connection.execute('UPDATE members SET person=? WHERE person=?',(root['id'],source['id']))
        db.connection.commit()
        assert not db.connection.execute("SELECT 1 FROM publications WHERE kind='merge'").fetchone()
    calls=[]
    services.metadata=lambda *a,**kw:calls.append(a[6]) or {}
    assert run(config,services,'1','2') == 1
    assert not calls


def test_stored_token_cannot_adopt_a_manual_profile_without_installation_binding(tmp_path,monkeypatch):
    from stash_ai.recognition.publication import ensure_person
    from stash_ai.runtime.report import RunReport
    config,client,services,_=run_setup(tmp_path,monkeypatch)
    assert run(config,services,'1') == 0
    client.profiles['77']['details']='ordinary manual note'
    with FaceDatabase(config.database) as state:
        person=state.connection.execute('SELECT id FROM people').fetchone()[0]
        state.connection.execute('UPDATE people SET token=?,performer=NULL WHERE id=?',('ordinary manual note',person))
        state.connection.commit()
        with pytest.raises(ValueError, match='ownership token'):
            ensure_person(state,client,RunReport(False),instance_scope(client.url),'1',person)
        assert client.profiles['77']['details']=='ordinary manual note'


def test_noncanonical_ids_are_rejected_before_merge_dispatch():
    from stash_ai.recognition.stash import RecognitionStash,checked_id
    client=RecognitionStash.__new__(RecognitionStash)
    calls=[]
    def gql(query,variables):calls.append(variables);return {'performerMerge':{'id':'1'}}
    client.gql=gql
    with pytest.raises(RuntimeError):checked_id({'id':'01'})
    with pytest.raises(RuntimeError):client.merge_person('01','1','details')
    assert not calls

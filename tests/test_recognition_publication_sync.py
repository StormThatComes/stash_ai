"""Global DB repairs converge on Stash only through an explicit guarded action."""
import json
from types import SimpleNamespace
import numpy as np
import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import namespace
from stash_ai.recognition.publication_queue import current,enqueue,plan,inventory
from stash_ai.recognition.sync_publications import execute
from stash_ai.recognition.run import instance_scope
from tests.test_recognition_rebuild import library,changed
from tests.test_recognition_run import run
from tests.test_recognition_reference_registry import angle,pending


def args(*,apply=False,scenes=None):
    return SimpleNamespace(apply=apply,scene_id=scenes,progress_style='off')


def fragmented(tmp_path,monkeypatch):
    config,client,services,_=library(tmp_path,monkeypatch,{'1':[angle(0)]*3,'2':[angle(12)]*3,'3':[angle(6)]*3})
    config=changed(config,recognition={'min_identity_support':2,'cross_scene_provisional':True},
        clustering={'max_anchor_pose_deviation':.2,'anchor_view_matching':True,'view_prototypes_enabled':True})
    # Seed profiles from an earlier strict-only policy, before global view/birth
    # safeguards existed. Current policy must repair their publication debt.
    strict=changed(config,recognition={'match_threshold':.99,'ambiguity_margin':.001},
                   clustering={'anchor_view_matching':False,'view_prototypes_enabled':False})
    assert run(strict,services,'1')==run(strict,services,'2')==0
    original={s:client.links[s][-1] for s in ('1','2')}
    assert original['1']!=original['2']
    assert run(config,services,'3')==0
    assert client.links['2']==[original['2']]
    def no_models(*a,**k):pytest.fail('Synchronization must not load models, images or inference assets')
    services._test_source=services.source
    services.backend=services.assets=services.source=no_models
    return config,client,services,original


def queue(db,config,client):
    return plan(db.connection,instance_scope(client.url),namespace(config),config)


def test_dry_run_is_offline_readonly_and_distinguishes_profiles_from_active_ids(tmp_path,monkeypatch,capsys):
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    with FaceDatabase(config.database) as db:before=list(db.connection.iterdump())
    services.client=lambda **k:pytest.fail('Dry-run is offline')
    capsys.readouterr();assert execute(args(),config,services=services)==0
    report=json.loads(capsys.readouterr().out)
    assert [(e['scene'],e['status']) for e in report['scenes']]==[('2','ready')]
    assert report['counts']['identities_with_latest_memberships']==1
    assert [p['performer'] for p in report['counts']['retained_profiles_without_latest_memberships']]==[original['2']]
    with FaceDatabase(config.database) as db:assert list(db.connection.iterdump())==before


def test_explicit_sync_relinks_old_scene_preserves_profiles_and_manual_links(tmp_path,monkeypatch):
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    before=list(client.writes);client.profiles[original['2']]['name']='Preserved manual name'
    assert execute(args(apply=True),config,services=services)==0
    assert client.links['1']==['77',original['1']]
    assert client.links['2']==[original['1']] and client.links['3']==[original['1']]
    assert client.profiles[original['2']]['name']=='Preserved manual name'
    assert not any(w[0] in ('create','merge','delete','metadata') for w in client.writes[len(before):])
    with FaceDatabase(config.database) as db:assert queue(db,config,client)==[]
    writes=list(client.writes);assert execute(args(apply=True),config,services=services)==0
    assert client.writes==writes


def test_normal_selected_publication_leaves_and_then_acknowledges_foreign_debt(tmp_path,monkeypatch):
    config,client,services,_=fragmented(tmp_path,monkeypatch)
    with FaceDatabase(config.database) as db:assert [e['scene'] for e in queue(db,config,client)]==['2']
    # Reuse retained scans through the normal run; no extra source/model load.
    services.source=services._test_source
    assert run(config,services,'2')==0
    with FaceDatabase(config.database) as db:assert queue(db,config,client)==[]


def test_scene_filter_does_not_publish_other_queued_scenes(tmp_path,monkeypatch,capsys):
    config,client,services,_=fragmented(tmp_path,monkeypatch)
    writes=list(client.writes)
    assert execute(args(apply=True,scenes=['1']),config,services=services)==0
    assert client.writes==writes
    with FaceDatabase(config.database) as db:assert [e['scene'] for e in queue(db,config,client)]==['2']


@pytest.mark.parametrize('change',['new_scan','policy','ownership'])
def test_stale_or_unowned_plan_preserves_old_links_and_debt(tmp_path,monkeypatch,change):
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    if change=='new_scan':
        with FaceDatabase(config.database) as db:
            scan=db.connection.execute("SELECT id FROM scans WHERE scene='2'").fetchone()[0]
            pending(db,{'scan':scan})
    elif change=='policy':config=changed(config,recognition={'ambiguity_margin':.11})
    else:client.profiles[original['1']]['details']='Manual ownership edit'
    writes=list(client.writes)
    assert execute(args(apply=True),config,services=services)==1
    assert client.writes==writes and client.links['2']==[original['2']]
    with FaceDatabase(config.database) as db:assert len(queue(db,config,client))==1


def test_lost_add_response_is_observed_without_another_add(tmp_path,monkeypatch):
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    before=len(client.writes);client.fail_link=True
    assert execute(args(apply=True),config,services=services)==1
    assert original['2'] in client.links['2']
    assert execute(args(apply=True),config,services=services)==0
    assert sum(w[0]=='link' for w in client.writes[before:])==1
    assert client.links['2']==[original['1']]
    with FaceDatabase(config.database) as db:assert queue(db,config,client)==[]


def test_new_queue_revision_during_add_blocks_stale_unlink_and_ack(tmp_path,monkeypatch):
    from stash_ai.recognition import sync_publications
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    add=client.add_scene_person
    connections=[];validate=sync_publications.validate
    def captured(db,*a,**k):
        connections[:]=[db]
        return validate(db,*a,**k)
    monkeypatch.setattr(sync_publications,'validate',captured)
    def changed_during_add(scene,performer):
        result=add(scene,performer)
        with connections[0]:enqueue(connections[0],instance_scope(client.url),namespace(config),scene,config,'concurrent_rebuild')
        return result
    client.add_scene_person=changed_during_add
    assert execute(args(apply=True),config,services=services)==1
    assert set(client.links['2'])==set(original.values())
    with FaceDatabase(config.database) as db:
        assert 'concurrent_rebuild' in queue(db,config,client)[0]['reasons']
    client.add_scene_person=add
    assert execute(args(apply=True),config,services=services)==0
    assert client.links['2']==[original['1']]


@pytest.mark.parametrize('removed',[False,True])
def test_uncertain_unlink_is_observed_without_blind_remove_retry(tmp_path,monkeypatch,removed):
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    remove=client.remove_scene_person;attempts=[]
    def lost(scene,performer):
        attempts.append((scene,performer))
        if removed:remove(scene,performer)
        raise TimeoutError('lost unlink response')
    client.remove_scene_person=lost
    assert execute(args(apply=True),config,services=services)==1
    client.remove_scene_person=remove
    assert execute(args(apply=True),config,services=services)==(0 if removed else 1)
    assert len(attempts)==1
    with FaceDatabase(config.database) as db:assert bool(queue(db,config,client))==(not removed)
    if removed:assert client.links['2']==[original['1']]
    else:assert set(client.links['2'])==set(original.values())


def test_pool_backfill_enqueues_old_scene_without_widening_normal_run(tmp_path,monkeypatch):
    a=np.array([1.,0.,0.,0.])
    config,client,services,_=library(tmp_path,monkeypatch,{'1':[a],'2':[a]})
    config=changed(config,recognition={'min_identity_support':2,'cross_scene_provisional':True},
        clustering={'max_anchor_pose_deviation':.2,'anchor_view_matching':True})
    assert run(config,services,'1')==run(config,services,'2')==0
    assert client.links['1']==['77']
    with FaceDatabase(config.database) as db:assert [e['scene'] for e in queue(db,config,client)]==['1']
    services.backend=lambda *a:pytest.fail('No inference during synchronization')
    assert execute(args(apply=True),config,services=services)==0
    assert client.links['1']==['77',client.links['2'][0]]
    assert sum(w[0]=='create' for w in client.writes)==1


def test_rebuild_failure_rolls_back_publication_debt_and_memberships(tmp_path,monkeypatch):
    from stash_ai.recognition import rebuild_cache
    from stash_ai.recognition.rebuild import rebuild_selected
    from tests.test_recognition_frozen_partition import fixture
    from tests.test_recognition_anchor_views import config_for,A
    config=config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        item=fixture(db,config,'1',[A]*3)
        before=list(db.connection.iterdump())
        def fail(*a,**k):raise RuntimeError('after queued assignment')
        monkeypatch.setattr(rebuild_cache,'save',fail)
        with pytest.raises(RuntimeError,match='queued assignment'):rebuild_selected(db,[item],config)
        assert list(db.connection.iterdump())==before


def test_cli_sync_is_an_independent_explicit_action():
    value=parser().parse_args(['sync-identities','--scene-id','2','--apply','--progress-style','off'])
    assert value.command=='sync-identities' and value.scene_id==['2'] and value.apply


def test_legacy_debt_without_queue_is_discovered_readonly_and_explicitly_adopted(tmp_path,monkeypatch,capsys):
    config,client,services,original=fragmented(tmp_path,monkeypatch)
    with FaceDatabase(config.database) as db:
        db.connection.execute("DELETE FROM metadata WHERE key LIKE 'identity-publication:%'");db.connection.commit()
        before=list(db.connection.iterdump())
        entries=queue(db,config,client)
        assert [(e['scene'],e['origin']) for e in entries]==[('2','journal_inventory')]
    assert execute(args(),config,services=services)==0
    with FaceDatabase(config.database) as db:assert list(db.connection.iterdump())==before
    assert execute(args(apply=True),config,services=services)==0
    assert client.links['2']==[original['1']]
    with FaceDatabase(config.database) as db:assert queue(db,config,client)==[]


def test_sync_requires_an_existing_database(tmp_path):
    from stash_ai.recognition.config import load
    config=load(root=tmp_path)
    with pytest.raises(FileNotFoundError,match='existing face database'):execute(args(),config)

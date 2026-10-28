from io import BytesIO
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from stash_ai.recognition.database import FaceDatabase,encoded
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.body_identity import rebuild
from tests.test_recognition_body_identity import configuration,scene


def seeded(tmp_path,vectors=None):
    c=configuration(tmp_path);c.data['sapiensid']['global_matching']['enabled']=True
    vectors=vectors or [[1,0],[.88,math.sqrt(1-.88**2)],[0,-1]]
    selected=[];ids=[];people=[]
    with FaceDatabase(c.database) as state:
        db=state.connection
        for i,v in enumerate(vectors):
            item,observations=scene(db,c,str(i+1),vectors=[v]*3);selected.append(item);ids.append(observations)
        db.execute('UPDATE scans SET namespace=?,policy_sha=?',(namespace(c),scan_identity(c,namespace(c))));db.commit()
        result=rebuild(state,selected,c,{s['scan']:[] for s in selected})
        for i,s in enumerate(selected):
            p=result[s['scan']][0]['person'];people.append(p)
            db.execute('UPDATE people SET performer=? WHERE id=?',(str(900+i),p))
        db.commit()
    return c,selected,ids,people


def crop(db,ids):
    with Image.new('RGB',(120,240),(20,90,160)) as image:
        out=BytesIO();image.save(out,'PNG')
    for oid in ids:db.execute('UPDATE body_observations SET crop=?,geometry=? WHERE id=?',
        (out.getvalue(),encoded(dict(bounds=[0,0,120,240],keypoints=[[-1,-1]]*17)),oid))
    db.commit()


def test_body_replay_tune_solver_and_export_are_read_only(tmp_path):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    from stash_ai.recognition.merge_goals import resolve,evaluate
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.threshold_tuning import tune
    from stash_ai.recognition.offline_evaluation import evaluate as evaluate_truth
    from stash_ai.recognition.identity_report import export
    c,selected,ids,people=seeded(tmp_path)
    with FaceDatabase(c.database) as state:crop(state.connection,ids[0])
    before=c.database.read_bytes()
    with _open_readonly(c.database) as db:
        goals=resolve(db,c,[['900','901'],['902']])
        assert len(goals['groups'][0]['members'][0]['observations'])==3
        assert goals['groups'][0]['members'][0]['observations'][0]['evidence_kind']=='person'
        baseline=evaluate(c,goals,db)['trials']['baseline']
        assert baseline['versus_retained']['assigned']==9
        assert baseline['goal_result']['groups'][0]['sample_assignment']['face']['anchor']['total']==0
        result=solve(c,goals,db,maximum=6)
        assert result['status']=='fully_solved'
        assert any(r['parameter'].startswith('sapiensid.global_matching.') for r in result['recommendation']['changes'])
    tuned=tune(c,{'sapiensid.global_matching.match_threshold':[.87]},scope='scope')
    assert tuned['trials']['trial-1']['versus_baseline']['newly_joined_pairs']==9
    truth={oid:'same' if i<2 else 'other' for i,rows in enumerate(ids) for oid in rows}
    evaluated=evaluate_truth(c,truth,batch_sizes=(0,1))
    assert evaluated['trials']
    report=export(c,tmp_path/'report',identity_id=people[0])
    assert report['summary']['members']==report['summary']['pngs_exported']==3
    assert all(s['evidence_kind']=='person' and s['linked_face'] is None for s in report['samples'])
    assert c.database.read_bytes()==before


def test_body_portrait_modes_manual_guard_and_anchor_policy(tmp_path,monkeypatch):
    from stash_ai.recognition.portraits import fill_portrait,policy_identity
    from stash_ai.recognition.body_portraits import choose_kind
    from stash_ai.recognition.portrait_refresh import retained_targets
    from tests.helpers.recognition import Stash
    c,selected,ids,people=seeded(tmp_path,vectors=[[1,0]])
    client=Stash();person=people[0];c.data['portraits']['kind']='person'
    forbidden=lambda *a,**k:pytest.fail('Body portraits must not run face/media/model inference')
    with FaceDatabase(c.database) as state:
        db=state.connection;crop(db,ids[0])
        token=db.execute('SELECT token FROM people WHERE id=?',(person,)).fetchone()[0]
        client.profiles['900']=dict(id='900',name='Body',details=token,image_path='')
        target=retained_targets(state,c,'scope')[''+person]
        assert target['scans']==selected
        result=fill_portrait(state,client,None,c,'scope',target,{},forbidden,forbidden)
        assert result['portraits_filled']==result['portraits_person']==1
        writes=len(client.writes)
        assert fill_portrait(state,client,None,c,'scope',target,{},forbidden,forbidden)['portraits_preserved']==1
        assert len(client.writes)==writes
        client.images['900']=b'manual replacement'
        assert fill_portrait(state,client,None,c,'scope',target,{},forbidden,forbidden)['portraits_preserved']==1
        assert len(client.writes)==writes
        c.data['portraits']['anchor_only']=True
        db.execute("UPDATE body_members SET role='auxiliary'");db.commit()
        assert choose_kind(state,client,c,'scope',target,{},forbidden,forbidden,None) is None
        c.data['portraits']['anchor_only']=False;c.data['portraits']['kind']='face'
        monkeypatch.setattr('stash_ai.recognition.portraits.choose',lambda *a,**k:None)
        assert choose_kind(state,client,c,'scope',target,{},forbidden,forbidden,None) is None


def test_auto_upgrades_managed_body_portrait_to_good_face(tmp_path,monkeypatch):
    from stash_ai.recognition.portraits import fill_portrait,policy_identity
    from stash_ai.recognition.portrait_refresh import retained_targets
    from tests.helpers.recognition import Stash
    import hashlib
    c,selected,ids,people=seeded(tmp_path,vectors=[[1,0]])
    client=Stash();person=people[0]
    with FaceDatabase(c.database) as state:
        db=state.connection;crop(db,ids[0])
        token=db.execute('SELECT token FROM people WHERE id=?',(person,)).fetchone()[0]
        client.profiles['900']=dict(id='900',name='Body',details=token,image_path='')
        target=retained_targets(state,c,'scope')[person]
        monkeypatch.setattr('stash_ai.recognition.portraits.choose',lambda *a,**k:None)
        assert fill_portrait(state,client,None,c,'scope',target,{},None,None)['portraits_person']==1
        with Image.new('RGB',(120,160),(190,50,70)) as img:
            output=BytesIO();img.save(output,'JPEG');data=output.getvalue()
        metadata=dict(policy_sha=policy_identity(c),observation='new-face',quality={},source_type='vtt')
        db.execute('INSERT INTO portraits VALUES (?,?,?,?,?,?,?,?)',
            ('face','scope',person,'1',encoded(metadata),data,hashlib.sha256(data).hexdigest(),'selected'));db.commit()
        monkeypatch.setattr('stash_ai.recognition.portraits.choose',lambda *a,**k:db.execute("SELECT * FROM portraits WHERE id='face'").fetchone())
        monkeypatch.setattr('stash_ai.recognition.quality.portrait_shortfalls',lambda *a:[])
        assert fill_portrait(state,client,None,c,'scope',target,{},None,None)['portraits_updated']==1
        assert client.images['900']==data


@pytest.mark.parametrize('kind',['auto','person'])
def test_faceless_run_publishes_once_and_reuses_model_inputs(tmp_path,monkeypatch,kind):
    from tests.test_recognition_run import setup,run
    from tests.test_recognition_body_collection import Persons
    from tests.helpers.recognition import Backend,Response
    c,client,services,_=setup(tmp_path,monkeypatch)
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    c=replace(c,data=plain(c.data))
    c.data['sapiensid']['enabled']=True
    c.data['sapiensid']['collection'].update(enabled=True,min_sharpness=0)
    c.data['sapiensid']['identities'].update(enabled=True,min_appearance_distance=0)
    c.data['portraits'].update(enabled=True,kind=kind)
    # Three genuinely distinct crops, not three copies of one input image.
    pixels=np.random.default_rng(2).integers(0,255,(160,480,3),dtype='uint8')
    with Image.fromarray(pixels) as img:out=BytesIO();img.save(out,'PNG');sprite=out.getvalue()
    vtt=('WEBVTT\n\n'+'\n\n'.join(f'00:00:0{i}.000 --> 00:00:0{i+1}.000\nsprite.jpg#xywh={i*160},0,160,160' for i in range(3))).encode()
    get=client.get
    client.media.get=lambda url,**kw:Response(vtt if '/vtt/' in url else sprite) if '/vtt/' in url or '/sprite/' in url else get(url,**kw)
    backend=Backend();backend.detect=lambda image:[]
    services.backend=lambda *a:backend
    model=Persons(1)
    monkeypatch.setattr('stash_ai.recognition.sapiensid_backend.SapiensIDFaces',lambda *a,**k:model)
    assert run(c,services,'1')==0
    assert len(client.links['1'])==2
    assert client.links['1'][-1] in client.images
    assert model.calls==model.embedded==3
    writes=list(client.writes)
    assert run(c,services,'1')==0
    assert client.writes==writes and model.calls==model.embedded==3
    with FaceDatabase(c.database) as state:
        assert state.connection.execute('SELECT COUNT(*) FROM observations').fetchone()[0]==0
        assert state.connection.execute("SELECT COUNT(*) FROM body_members WHERE status='accepted'").fetchone()[0]==3


def test_faceless_alignment_audit_exports_missing_face_landmarks(tmp_path):
    from tests.test_recognition_body_collection import Persons
    from tests.helpers.recognition import MediaClient
    from stash_ai.recognition.media import SpriteInput
    from stash_ai.recognition.alignment_audit import audit_scene
    c=configuration(tmp_path);client=MediaClient();model=Persons(1)
    detect=model.detect
    model.detect=lambda image:detect(image,[])
    with SpriteInput(client,client.scene,c) as source:
        result=audit_scene(source,model,c,'1',tmp_path,maximum=1)
    assert model.calls==3 and model.embedded==0
    assert result['samples']==[] and len(result['person_samples'])==1
    data=result['person_samples'][0]['person_input']
    assert data['normalized_keypoints'][17:]==[[-1,-1]]*5
    assert (tmp_path/'scene-1'/data['input_file']).exists()


def test_body_fragmentation_and_pair_reports_show_replay_verified_proposals(tmp_path):
    from stash_ai.recognition.fragmentation_report import build as fragmentation
    from stash_ai.recognition.merge_report import build as pair_report
    c,selected,ids,people=seeded(tmp_path)
    before=c.database.read_bytes()
    report=fragmentation(c,'1')
    assert report['independent_person']['identities'][0]['person']==people[0]
    assert report['independent_person']['assignments']['anchor']==3
    report=pair_report(c,people[0],people[1],simulate=True)
    recommendation=next(r for r in report['recommendations'] if r['route']=='independent_person')
    assert recommendation['replay_verified']
    assert all(i['person_members']==3 and i['members']==0 for i in report['identities'])
    assert c.database.read_bytes()==before

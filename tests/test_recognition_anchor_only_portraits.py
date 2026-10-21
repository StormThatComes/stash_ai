"""Anchor-only is a hard source-face gate, independent of soft portrait scores."""
from dataclasses import replace
import json
import pytest
import yaml

from stash_ai.recognition.config import load,plain
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.portraits import policy_identity
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.visibility import CONTRACT_ID
from tests.test_recognition_best_available import published
from tests.test_recognition_portraits import refresh,forbidden,forbid_recognition,identity_state
from tests.test_recognition_portrait_removal import change
from tests.test_recognition_vtt_portraits import setup,image_writes
from tests.test_recognition_metadata import setup as video_setup,Decoder
from tests.test_recognition_run import run


WEIGHTS=dict(frontal=0,sharpness=1,resolution=0,exposure=0,visibility=0,
             neutral_expression=0,closed_mouth=0,open_mouth=0,open_eyes=0)


def configured(config,**clustering):
    data=yaml.safe_load(config.source_path.read_text())
    data['clustering']={'max_anchor_pose_deviation':.2,**clustering}
    data['portraits'].update(enabled=False,weights=WEIGHTS)
    config.source_path.write_text(yaml.safe_dump(data))
    return load(config.source_path)


def qualities(config,poses,*,scene='1',clear_visibility=False):
    with FaceDatabase(config.database) as db:
        rows=db.connection.execute('SELECT o.* FROM observations o JOIN scans s ON s.id=o.scan '
            'WHERE s.scene=? ORDER BY o.cue',(scene,)).fetchall()
        for row,pose in zip(rows,poses):
            quality=json.loads(row['quality']);quality.update(pose_deviation=pose,sharpness=1 if row['cue']==0 else 1000)
            if clear_visibility:
                quality.pop('visibility',None);quality.pop('visible_face_fraction',None)
            db.connection.execute('UPDATE observations SET quality=? WHERE id=?',(json.dumps(quality),row['id']))
        db.connection.commit()


@pytest.mark.parametrize('anchor_only',[False,True])
@pytest.mark.parametrize('prefetch',[False,True])
def test_hard_anchor_filter_beats_soft_quality_and_matches_prefetch(tmp_path,monkeypatch,anchor_only,prefetch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    qualities(config,[.05,.3,.3])
    config=change(config,enabled=True,anchor_only=anchor_only)
    data=plain(config.data);data['execution']['prefetch_portraits']=prefetch;config=replace(config,data=data)
    before=identity_state(config)
    forbid_recognition(monkeypatch,services,stash);services.backend=services.visibility=forbidden
    assert refresh(config,services,'--scene-id','1')==0
    metadata=published(config)[1]
    assert metadata['cue']==(0 if anchor_only else 1)
    assert identity_state(config)==before


def test_anchor_attachment_can_illustrate_its_identity(tmp_path,monkeypatch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    qualities(config,[.05,.05,.3])
    with FaceDatabase(config.database) as db:
        db.connection.execute("UPDATE observations SET quality=json_set(quality,'$.sharpness',CASE cue WHEN 0 THEN 2000 ELSE 1 END)")
        db.connection.execute("UPDATE candidates SET payload=json_set(payload,'$.reason','historical_view_identity',"
            "'$.reference_role','attachment') WHERE json_extract(payload,'$.observation') IN "
            '(SELECT id FROM observations WHERE cue=0)');db.connection.commit()
    config=change(config,enabled=True,anchor_only=True)
    assert refresh(config,services)==0 and published(config)[1]['cue']==0


def test_missing_anchor_skips_new_portrait_without_auxiliary_fallback(tmp_path,monkeypatch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    qualities(config,[.3,.3,.3]);config=change(config,enabled=True,anchor_only=True)
    forbid_recognition(monkeypatch,services,stash);services.backend=services.visibility=forbidden
    assert refresh(config,services)==0 and not stash.images
    with FaceDatabase(config.database) as db:
        reasons=[json.loads(r[0]).get('reason') for r in db.connection.execute('SELECT metadata FROM portraits')]
        assert reasons and set(reasons)=={'portrait_anchor_required'}


def test_source_failure_can_only_fall_back_to_an_anchor_crop(tmp_path,monkeypatch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    qualities(config,[.05,.3,.3]);config=change(config,enabled=True,anchor_only=True)
    services.source=lambda *a:(_ for _ in ()).throw(OSError('thumbnail unavailable'))
    assert refresh(config,services)==0
    metadata=published(config)[1]
    assert metadata['source_type']=='recognition_crop' and metadata['cue']==0


def test_enabling_anchor_only_replaces_a_higher_scoring_auxiliary_cache(tmp_path,monkeypatch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    qualities(config,[.05,.3,.3]);config=change(config,enabled=True)
    assert refresh(config,services)==0 and published(config)[1]['cue']==1
    config=change(config,anchor_only=True)
    assert refresh(config,services)==0 and published(config)[1]['cue']==0
    assert refresh(config,services)==0 and published(config)[1]['cue']==0


def test_no_anchor_preserves_an_already_published_image(tmp_path,monkeypatch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    config=change(config,enabled=True);assert refresh(config,services)==0
    images=dict(stash.images);writes=len(image_writes(stash))
    qualities(config,[.3,.3,.3]);config=change(config,anchor_only=True)
    assert refresh(config,services)==0
    assert stash.images==images and len(image_writes(stash))==writes


@pytest.mark.parametrize('fraction',[.95,.5,None])
@pytest.mark.parametrize('prefetch',[False,True])
def test_anchor_visibility_is_required_with_zero_portrait_weight(tmp_path,monkeypatch,fraction,prefetch):
    config,stash,services,_=setup(tmp_path,monkeypatch)
    config=configured(config,min_anchor_visible_face_fraction=.8,min_auxiliary_visible_face_fraction=.5)
    assert run(config,services,'1','2')==0
    qualities(config,[.05,.05,.05],scene='2',clear_visibility=True)
    config=change(config,enabled=True,anchor_only=True)
    data=plain(config.data);data['execution']['prefetch_portraits']=prefetch;config=replace(config,data=data)
    calls=[]
    class Visibility:
        def __init__(self,*a):pass
        def measure(self,*a):
            calls.append(1)
            if fraction is None:raise RuntimeError('FaceOcc unavailable')
            return dict(visible_face_fraction=fraction,visibility=dict(contract=CONTRACT_ID))
        def close(self):pass
    services.visibility=Visibility;services.backend=forbidden
    assert refresh(config,services,'--scene-id','2')==0 and calls
    assert bool(stash.images)==(fraction==.95)
    if fraction==.95:assert published(config)[1]['quality']['visible_face_fraction']==.95


def test_video_face_itself_must_pass_anchor_gate(tmp_path,monkeypatch):
    import stash_ai.recognition.portraits as portraits
    config,stash,services=video_setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    original=portraits.assess
    def profile(*a):return dict(original(*a),pose_deviation=.3)
    monkeypatch.setattr(portraits,'assess',profile)
    config=change(config,enabled=True,anchor_only=True)
    assert refresh(config,services)==0
    assert published(config)[1]['source_type'] in ('vtt','recognition_crop')
    with FaceDatabase(config.database) as db:
        video=[json.loads(r[0]) for r in db.connection.execute("SELECT metadata FROM portraits WHERE json_extract(metadata,'$.source_type')='video'")]
        assert video and all(v['exclusions'][0]['reason']=='portrait_anchor_required' for v in video)


def test_video_anchor_can_be_found_from_an_auxiliary_hint(tmp_path,monkeypatch):
    config,stash,services=video_setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    qualities(config,[.3,.3,.05]);config=change(config,enabled=True,anchor_only=True)
    assert refresh(config,services)==0
    metadata=published(config)[1];assert metadata['source_type']=='video'
    assert metadata['quality']['pose_deviation']<=.2
    with FaceDatabase(config.database) as db:
        original=json.loads(db.connection.execute('SELECT quality FROM observations WHERE id=?',(metadata['observation'],)).fetchone()[0])
        assert original['pose_deviation']==.3


def test_cached_video_anchor_can_be_requalified_without_seeking_or_embedding(tmp_path,monkeypatch):
    config,stash,services=video_setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    config=change(config,enabled=True);assert refresh(config,services)==0
    count=len(Decoder.instances);images=dict(stash.images)
    services.decoder=services.backend=forbidden
    config=change(config,anchor_only=True)
    assert refresh(config,services)==0
    assert len(Decoder.instances)==count and stash.images==images


def test_video_visibility_checks_decoded_face_even_when_hint_is_visible(tmp_path,monkeypatch):
    config,stash,services=video_setup(tmp_path,monkeypatch)
    config=configured(config,min_anchor_visible_face_fraction=.8,min_auxiliary_visible_face_fraction=.5)
    assert run(config,services,'1')==0
    calls=[]
    class Occluded:
        def __init__(self,*a):pass
        def measure(self,*a):
            calls.append(1)
            return dict(visible_face_fraction=.5,visibility=dict(contract=CONTRACT_ID))
        def close(self):pass
    services.visibility=Occluded;config=change(config,enabled=True,anchor_only=True)
    assert refresh(config,services)==0 and calls
    assert published(config)[1]['source_type'] in ('vtt','recognition_crop')


def test_tighter_anchor_policy_rejects_cached_video_eligibility(tmp_path,monkeypatch):
    import stash_ai.recognition.portraits as portraits
    config,stash,services=video_setup(tmp_path,monkeypatch)
    config=configured(config);assert run(config,services,'1')==0
    original=portraits.assess
    monkeypatch.setattr(portraits,'assess',lambda *a:dict(original(*a),pose_deviation=.1))
    config=change(config,enabled=True,anchor_only=True);assert refresh(config,services)==0
    assert published(config)[1]['quality']['pose_deviation']==.1
    previous=len(Decoder.instances)
    data=yaml.safe_load(config.source_path.read_text());data['clustering']['max_anchor_pose_deviation']=.05
    config.source_path.write_text(yaml.safe_dump(data));config=load(config.source_path)
    monkeypatch.setattr(portraits,'assess',original)
    assert refresh(config,services)==0
    assert published(config)[1]['quality']['pose_deviation']<=.05 and len(Decoder.instances)>previous


def test_anchor_only_changes_portrait_policy_without_invalidating_identity_or_scan(tmp_path):
    config=load(root=tmp_path);data=plain(config.data);data['portraits']['anchor_only']=True
    enabled=replace(config,data=data)
    assert policy_identity(config)!=policy_identity(enabled)
    assert matching_policy(config)==matching_policy(enabled)
    assert scan_identity(config,'ns')==scan_identity(enabled,'ns')


@pytest.mark.parametrize('value',['null','1','maybe'])
def test_anchor_only_option_requires_a_boolean(tmp_path,value):
    path=tmp_path/'bad.yml';path.write_text('version: 1\nportraits: {anchor_only: '+value+'}\n')
    with pytest.raises(ValueError,match='portraits.anchor_only must be boolean'):load(path)

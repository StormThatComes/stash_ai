import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.config import defaults, load, validate
from stash_ai.recognition.database import FaceDatabase, SCHEMA_V5, APPLICATION_ID
from stash_ai.recognition.person_signal import contract, read, save


def test_contract_ignores_policy_paths_and_budgets():
    data=defaults(); config=SimpleNamespace(data=data)
    original=contract(config)
    data['sapiensid']['matching']['person_threshold']=.9
    data['sapiensid']['model']['source_directory']='/different'
    data['sapiensid']['execution']['batch_size']=8
    assert contract(config)==original
    data['sapiensid']['pose']['keypoint_score_threshold']=.7
    assert contract(config)!=original


def test_enabled_accepts_download_destinations_without_loading_assets(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    data=defaults();data['sapiensid']['enabled']=True
    validate(data)
    assert not (tmp_path/'recognition_models').exists()
    data['sapiensid']['model'].update(source_directory='missing',checkpoint_directory='missing')
    data['sapiensid']['pose'].update(path='missing',sha256='a'*64)
    validate(data)


def test_empty_pose_hash_and_explicit_pinned_hash_share_evidence_contract():
    from stash_ai.recognition.sapiensid_backend import identity_contract
    data=defaults(); config=SimpleNamespace(data=data)
    original=contract(config), identity_contract(config)
    data['sapiensid']['pose']['sha256']=''
    assert (contract(config), identity_contract(config))==original


@pytest.mark.parametrize('variant', ['', 'wb8m', 'WB12M', None, [], 12])
def test_variant_validation_rejects_unsupported_values(variant):
    data=defaults();data['sapiensid']['model']['variant']=variant
    with pytest.raises(ValueError,match='model.variant'):
        validate(data)


def test_wb12m_default_and_wb4m_boundary_revision_invalidates_old_evidence():
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.sapiensid_backend import identity_contract
    from stash_ai.recognition.sapiensid_registry import MODELS, LEGACY_POSE_SHA256
    data=defaults();config=SimpleNamespace(data=data)
    assert data['sapiensid']['model']['variant']=='wb12m'
    wb12m=contract(config);s=data['sapiensid']
    assert identity_contract(config)['model_sha256']==MODELS['wb12m']['sha256']
    s['model']['variant']='wb4m'
    s['pose']['sha256']=LEGACY_POSE_SHA256
    legacy=digest(dict(format='sapiensid-wb4m-person-v2',model_yaml=s['model']['model_yaml_sha256'],
                       pose={k:s['pose'][k] for k in ('sha256','score_threshold','keypoint_score_threshold',
                                                      'nms_threshold','max_people_per_frame')}))
    assert contract(config,legacy_boundaries=True)==legacy
    assert contract(config)!=legacy and contract(config)!=wb12m
    assert identity_contract(config)['model_sha256']==MODELS['wb4m']['sha256']


@pytest.mark.parametrize('variant', ['wb12m', 'wb4m'])
def test_yolo26_default_separates_evidence_from_explicit_legacy_pose(variant):
    from stash_ai.recognition.config import digest
    from stash_ai.recognition.sapiensid_backend import identity_contract
    from stash_ai.recognition.sapiensid_registry import LEGACY_POSE_SHA256, POSE_SHA256, MODELS, UPSTREAM_COMMIT
    data=defaults();settings=data['sapiensid'];settings['model']['variant']=variant
    config=SimpleNamespace(data=data)
    assert settings['pose']['path'].endswith('/yolo26m-pose.pt')
    assert settings['pose']['sha256']==POSE_SHA256
    current=contract(config)
    settings['pose'].update(path='old/yolov8n-pose.pt',sha256=LEGACY_POSE_SHA256)
    previous=dict(format='sapiensid-person-v3',model_yaml=settings['model']['model_yaml_sha256'],
                  pose={k:settings['pose'][k] for k in ('sha256','score_threshold','keypoint_score_threshold',
                                                       'nms_threshold','max_people_per_frame')})
    if variant=='wb4m':previous['format']='sapiensid-wb4m-person-v2'
    else:previous.update(variant=variant,model_sha256=MODELS[variant]['sha256'],source_revision=UPSTREAM_COMMIT)
    assert contract(config,legacy_boundaries=True)==digest(previous)
    assert contract(config)!=digest(previous) and contract(config)!=current
    assert identity_contract(config)['contract']=='sapiensid-v1-yolov8-pose-face-anchor-square384-v3'
    settings['pose'].update(path='',sha256='')
    assert contract(config)==current
    assert identity_contract(config)['pose_sha256']==POSE_SHA256
    assert 'coco17' in identity_contract(config)['contract']


@pytest.mark.parametrize('variant', ['wb12m', 'wb4m'])
def test_changed_weights_invalidate_evidence_even_with_identical_yaml(monkeypatch,variant):
    from stash_ai.recognition.sapiensid_registry import MODELS
    data=defaults();data['sapiensid']['model']['variant']=variant
    config=SimpleNamespace(data=data);original=contract(config)
    monkeypatch.setitem(MODELS,variant,{**MODELS[variant],'sha256':'f'*64})
    assert contract(config)!=original


def test_v5_readonly_then_writer_migration(tmp_path):
    from stash_ai.recognition.fragmentation_report import _open_readonly
    path=tmp_path/'old.sqlite'
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA_V5)
        db.execute(f'PRAGMA application_id={APPLICATION_ID}')
        db.execute('PRAGMA user_version=5')
        db.execute("INSERT INTO metadata VALUES ('installation','00000000-0000-0000-0000-000000000001')")
    before=path.read_bytes()
    with _open_readonly(path) as db:
        assert read(db,'missing','contract') is None
    assert path.read_bytes()==before
    with FaceDatabase(path) as state:
        db=state.connection
        from stash_ai.recognition.database import SCHEMA_VERSION
        assert db.execute('PRAGMA user_version').fetchone()[0]==SCHEMA_VERSION
        db.execute("INSERT INTO scans(id,scope,scene,namespace,input_sha,policy_sha,manifest,coverage,total) VALUES ('s','x','1','n','i','p','{}','{}',1)")
        db.execute("INSERT INTO observations VALUES ('o','s',0,0,'url','[]','[]','{}',NULL,512,NULL,NULL)")
        version=db.execute("SELECT version FROM evidence_versions WHERE kind='scan' AND key='s'").fetchone()[0]
        save(db,'o','v1',{'association':'unique'},np.ones(4096))
        assert len(read(db,'o','v1')['embedding'])==16384
        assert read(db,'o','v2') is None
        assert db.execute("SELECT version FROM evidence_versions WHERE kind='scan' AND key='s'").fetchone()[0]>version
        save(db,'o','v1',{'reason':'no_person'})
        assert read(db,'o','v1')['status']=='unavailable'
        with pytest.raises(ValueError):save(db,'o','v1',{},np.zeros(4096))
        with pytest.raises(ValueError):save(db,'o','v1',{},np.ones(512))

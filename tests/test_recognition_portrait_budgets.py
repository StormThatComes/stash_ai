"""New encoding limits never obstruct historical acknowledgement or cached pixels."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import yaml

from stash_ai.recognition import models
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.visibility import LazyVisibility, VisibilityError
from stash_ai.recognition.expression import LazyExpression, ExpressionError
from tests.helpers.recognition import FACE, textured_image
from tests.test_recognition_image_retry import run, image_operation, lose_images
from tests.test_recognition_metadata import setup as video_setup
from tests.test_recognition_vtt_portraits import setup as vtt_setup


def reduced(config, *, disabled=False):
    data = yaml.safe_load(config.source_path.read_text())
    data['storage'] = {'max_crop_bytes':1024}
    if disabled: data['portraits']['enabled'] = False
    config.source_path.write_text(yaml.safe_dump(data))
    return load(config.source_path)


def setup(tmp_path, monkeypatch, source):
    return vtt_setup(tmp_path,monkeypatch)[:3] if source=='vtt' else video_setup(tmp_path,monkeypatch)


@pytest.mark.parametrize('source',['vtt','video'])
def test_smaller_budget_reencodes_verified_cached_pixels_without_inference(tmp_path,monkeypatch,source):
    config,stash,services=setup(tmp_path,monkeypatch,source)
    assert run(config,services,'1')==0
    performer=stash.links['1'][-1]
    old=stash.images[performer]
    previous=image_operation(config)
    old_id=json.loads(previous['payload'])['portrait']
    assert len(old)>1024
    changed=reduced(config)
    services.backend=lambda *args: pytest.fail('budget changes must not rerun face inference')
    services.decoder=lambda *args: pytest.fail('budget changes must not decode video')
    assert run(changed,services,'1')==0
    assert len(stash.images[performer])<=1024 and stash.images[performer]!=old
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT image FROM portraits WHERE id=?',(old_id,)).fetchone()[0]==old
        operation=db.connection.execute("SELECT * FROM publications WHERE kind='image'").fetchone()
        row=db.connection.execute('SELECT * FROM portraits WHERE id=?',
                                  (json.loads(operation['payload'])['portrait'],)).fetchone()
        metadata=json.loads(row['metadata'])
        assert metadata['encoding']['reencoded_from_cached_jpeg'] is True
        assert row['digest']==hashlib.sha256(row['image']).hexdigest()
        assert operation['status']=='confirmed'
        assert db.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0]==1
    writes=list(stash.writes)
    assert run(changed,services,'1')==0 and stash.writes==writes


@pytest.mark.parametrize('source',['vtt','video'])
def test_lower_encoding_budget_still_allows_explicit_owned_image_removal(tmp_path,monkeypatch,source):
    config,stash,services=setup(tmp_path,monkeypatch,source)
    assert run(config,services,'1')==0
    performer=stash.links['1'][-1]
    old=stash.images[performer]
    assert len(old)>1024
    changed=reduced(config,disabled=True)
    assert run(changed,services,'1')==0 and performer not in stash.images
    operation=image_operation(config)
    assert operation['status']=='confirmed' and json.loads(operation['payload'])['action']=='remove'
    assert json.loads(operation['precondition'])['image_sha256']==hashlib.sha256(old).hexdigest()


def test_unknown_applied_old_image_is_observed_before_new_budget_update(tmp_path,monkeypatch):
    config,stash,services=setup(tmp_path,monkeypatch,'vtt')
    original,attempts=lose_images(stash,applied=True)
    assert run(config,services,'1')==1 and len(attempts)==1
    old=attempts[0][1]
    assert len(old)>1024 and image_operation(config)['status']=='unknown'
    stash.update_person=original
    services.backend=lambda *args: pytest.fail('observed old bytes must not rerun inference')
    changed=reduced(config)
    assert run(changed,services,'1')==0
    assert len(attempts)==1 and image_operation(config)['status']=='confirmed'
    assert len(stash.images[stash.links['1'][-1]])<=1024


def test_lower_budget_preserves_manual_replacement(tmp_path,monkeypatch):
    config,stash,services=setup(tmp_path,monkeypatch,'vtt')
    assert run(config,services,'1')==0
    performer=stash.links['1'][-1]
    stash.images[performer]=b'manual replacement'*1000
    writes=list(stash.writes)
    assert run(reduced(config,disabled=True),services,'1')==0
    assert stash.images[performer]==b'manual replacement'*1000 and stash.writes==writes


@pytest.mark.parametrize('key,feature',[('faceocc','visibility'),('expression_mesh','expression'),('expression_blendshapes','expression')])
@pytest.mark.parametrize('corrupt',[False,True])
def test_actual_identity_loader_ignores_unavailable_optional_quality_paths(tmp_path,monkeypatch,key,feature,corrupt):
    import stash_ai.recognition.runtime as runtime
    config=load(root=tmp_path,device='cpu')
    paths={}
    for name in ('yunet','sface'):
        path=tmp_path/name
        path.write_bytes(b'pinned')
        monkeypatch.setitem(models.ASSETS,name,models.Asset('fixture','rev',name,hashlib.sha256(b'pinned').hexdigest(),6,'MIT'))
        paths[name]=path
    for name in ('faceocc','expression_mesh','expression_blendshapes'):
        paths[name]=tmp_path/name
    if corrupt: paths[key].write_bytes(b'custom invalid bytes')
    monkeypatch.setattr(runtime,'runtime',lambda *args: (None,[]))
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(__version__='fixture',
        FaceDetectorYN=SimpleNamespace(create=lambda *args: SimpleNamespace()),
        FaceRecognizerSF=SimpleNamespace(create=lambda *args: SimpleNamespace())))
    backend=models.OpenCVFaces(config,paths)
    assert backend.detector is not None and backend.recognizer is not None
    backend.close()
    optional=(LazyVisibility(config,paths) if feature=='visibility' else LazyExpression(config,paths))
    with textured_image() as image, pytest.raises(VisibilityError if feature=='visibility' else ExpressionError):
        optional.measure(image,FACE)
    optional.close()
    paths['yunet'].write_bytes(b'custom')
    with pytest.raises(ValueError,match='checksum'): models.OpenCVFaces(config,paths)

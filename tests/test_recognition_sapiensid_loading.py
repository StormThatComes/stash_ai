"""Offline loader failures and optional real, operator-supplied wb4m smoke."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import os
import sys

import pytest

from stash_ai.recognition import sapiensid_loading as loading
from tests.test_recognition_sapiensid_backend import settings


def test_relative_paths_follow_yaml_directory(tmp_path):
    c = settings(tmp_path)
    c = __import__('dataclasses').replace(c,source_path=tmp_path/'nested'/'face.yml')
    assert loading.local_path(c,'weights/model.pth') == tmp_path/'nested'/'weights'/'model.pth'
    assert loading.local_path(c,str(tmp_path/'weights')) == tmp_path/'weights'


def test_bad_local_asset_fails_on_hash_and_size(tmp_path):
    path = tmp_path/'weights'; path.write_bytes(b'weights')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert loading.checked_bytes(path,sha,limit=7) == b'weights'
    for checksum,limit in [('a'*64,7),(sha,6)]:
        with pytest.raises(ValueError,match='size/SHA256'):
            loading.checked_bytes(path,checksum,limit=limit)


def test_source_package_collision_does_not_change_import_path(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'src',SimpleNamespace(__path__=[str(tmp_path/'foreign')]))
    previous = list(sys.path)
    with pytest.raises(RuntimeError,match='foreign'):
        with loading.source_imports(tmp_path):
            pytest.fail('accepted foreign package')
    assert sys.path == previous


def test_pose_loader_receives_verified_private_bytes_and_no_download_name(tmp_path,monkeypatch):
    from dataclasses import replace
    from stash_ai.recognition.config import plain,_freeze
    c = settings(tmp_path)
    pose = tmp_path/'pose.pt';pose.write_bytes(b'local pose')
    data = plain(c.data)
    data['sapiensid']['pose']['path'] = str(pose)
    data['sapiensid']['pose']['sha256'] = hashlib.sha256(pose.read_bytes()).hexdigest()
    c = replace(c,data=_freeze(data))
    observed = []
    def yolo(path,**kwargs):
        observed.append(Path(path))
        assert Path(path).read_bytes() == b'local pose' and Path(path) != pose
        assert kwargs == {'task':'pose'}
        return 'model'
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=yolo))
    assert loading.load_pose(c) == 'model'
    assert not observed[0].exists()
    pose.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA256'):
        loading.load_pose(c)
    assert len(observed) == 1


@pytest.mark.parametrize('head,dirty', [('0'*40,''),(loading.UPSTREAM_COMMIT,' M tracked.py')])
def test_unpinned_or_dirty_checkout_is_rejected(tmp_path,monkeypatch,head,dirty):
    def git(args,**kwargs):
        if '--show-toplevel' in args: value=str(tmp_path)
        elif 'rev-parse' in args: value=head
        elif 'status' in args: value=dirty
        else: value='tracked'
        return SimpleNamespace(stdout=value)
    monkeypatch.setattr(loading.subprocess,'run',git)
    with pytest.raises(ValueError,match='clean checkout'):
        loading.trusted_checkout(tmp_path)


def test_real_local_sapiensid_wb4m_smoke():
    """Opt in with a normal complete YAML and real clean pinned source checkout."""
    path = os.getenv('STASH_AI_SAPIENSID_CONFIG')
    if not path:
        pytest.skip('Set STASH_AI_SAPIENSID_CONFIG to a local SapiensID YAML for real wb4m smoke.')
    from stash_ai.recognition.config import load
    from PIL import Image
    import numpy as np
    encoder = loading.load_encoder(load(path,device='cpu'))
    points = [[-1.,-1.] for _ in range(17)] + [[60,45],[100,45],[80,65],[65,90],[95,90]]
    points[0],points[5],points[6] = [80,65],[125,130],[35,130]
    try:
        with Image.new('RGB',(160,280),(80,120,160)) as image:
            vector = encoder.embed_person(image,points)
        assert vector.shape == (4096,) and np.isfinite(vector).all()
        assert np.linalg.norm(vector.astype(np.float64)) == pytest.approx(1.,abs=1e-6)
    finally:
        encoder.close()

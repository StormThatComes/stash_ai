import errno
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile
import zlib

import pytest
import requests

from stash_ai.recognition import sapiensid_assets as assets
from stash_ai.recognition import sapiensid_registry as registry
from stash_ai.recognition.config import defaults, validate
from stash_ai.recognition.sapiensid_downloads import install
from stash_ai.recognition.sapiensid_loading import load_pose, trusted_checkout


def sha(data):
    return hashlib.sha256(data).hexdigest()


def deflate(data):
    compressor = zlib.compressobj(wbits=-15)
    return compressor.compress(data) + compressor.flush()


class Response:
    def __init__(self, body, status=200, headers=None, failure=None):
        self.body, self.status_code = body, status
        self.headers, self.failure = headers or {}, failure
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), 97):
            yield self.body[start:start+97]
            if self.failure:
                raise self.failure


@pytest.mark.parametrize('ranged,ignore_range', [(False, False), (True, False), (True, True)])
def test_download_streams_verifies_and_atomically_installs(tmp_path, monkeypatch, ranged, ignore_range):
    data = b'model weights' * 200000  # Exercise bounded decompression across output chunks.
    packed = deflate(data)
    archive = b'prefix' + packed + b'wb12m must not be written'
    headers = {'Content-Range': f'bytes 6-{6+len(packed)-1}/{len(archive)}'}
    response = Response(archive if ignore_range else packed if ranged else data,
                        206 if ranged and not ignore_range else 200, headers)
    target = tmp_path / 'nested' / 'model.pth'
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        assert not target.exists()
        return response

    monkeypatch.setattr(requests, 'get', get)
    options = dict(byte_range=(6, len(packed)), archive_size=len(archive)) if ranged else {}
    install('https://example.test/asset', target, len(data), sha(data), **options)
    assert target.read_bytes() == data
    assert sorted(target.parent.iterdir()) == [target]
    assert response.closed
    assert calls[0][1]['stream'] is True
    if ranged:
        assert calls[0][1]['headers']['Range'] == f'bytes=6-{6+len(packed)-1}'


@pytest.mark.parametrize('problem', ['hash', 'truncated', 'range', 'deflate', 'oversize', 'network', 'interrupt', 'http'])
def test_failed_download_preserves_existing_file_and_removes_partial(tmp_path, monkeypatch, problem):
    data = b'verified weights' * 100
    body, checksum, size = data, sha(data), len(data)
    options, status, headers, failure = {}, 200, {}, None
    if problem == 'hash':
        checksum = '0' * 64
    elif problem == 'truncated':
        body = data[:-1]
    elif problem in ('range', 'deflate', 'oversize'):
        body = deflate(data)[:-1] if problem == 'deflate' else deflate(data)
        status = 206
        headers = {'Content-Range': f'bytes 0-{len(body)-1}/{len(body)}'}
        options = dict(byte_range=(0, len(body)), archive_size=len(body))
        if problem == 'range':
            headers['Content-Range'] = 'bytes 1-2/3'
        elif problem == 'oversize':
            size -= 1
    elif problem == 'network':
        failure = requests.ConnectionError('interrupted transfer')
    elif problem == 'interrupt':
        failure = KeyboardInterrupt()
    elif problem == 'http':
        status = 503
    response = Response(body, status, headers, failure)
    monkeypatch.setattr(requests, 'get', lambda *a, **k: response)
    target = tmp_path / 'model.pth'
    target.write_bytes(b'previous valid file')
    with pytest.raises((ValueError, requests.RequestException, KeyboardInterrupt)):
        install('https://example.test/asset', target, size, checksum, **options)
    assert target.read_bytes() == b'previous valid file'
    assert list(tmp_path.iterdir()) == [target]
    assert response.closed


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    root = tmp_path / 'upstream'
    root.mkdir()
    files = {'tasks/sapiensID/src/models/sapiens_id/__init__.py': b'# model source\n',
             'general_utils/img_utils.py': b'# utility source\n'}
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as zipped:
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            zipped.writestr('sapiensid-'+registry.UPSTREAM_COMMIT+'/'+name, content)
    source, pose, weights, yaml = archive.getvalue(), b'pose weights', b'wb4m weights', b'y' * 1198
    compressed_weights, compressed_yaml = deflate(weights), deflate(yaml)
    checkpoint = b'zip header' + compressed_weights + compressed_yaml + b'wb12m data'
    ranges = {'model.pth': (10, len(compressed_weights)),
              'model.yaml': (10+len(compressed_weights), len(compressed_yaml))}
    for name, value in dict(SOURCE_BYTES=len(source), SOURCE_SHA256=sha(source),
                            SOURCE_TREE_SHA256=assets.source_digest(root),
                            POSE_BYTES=len(pose), POSE_SHA256=sha(pose),
                            MODEL_YAML_SHA256=sha(yaml), CHECKPOINT_BYTES=len(checkpoint),
                            CHECKPOINT_RANGES=ranges).items():
        monkeypatch.setattr(registry, name, value)
    from stash_ai.recognition import sapiensid_v1
    monkeypatch.setattr(sapiensid_v1, 'WB4M_BYTES', len(weights))
    monkeypatch.setattr(sapiensid_v1, 'WB4M_SHA256', sha(weights))
    payloads = {registry.SOURCE_URL: source, registry.POSE_URL: pose,
                registry.CHECKPOINT_URL: checkpoint}
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        body = payloads[url]
        if 'Range' in kwargs['headers']:
            start, end = map(int, kwargs['headers']['Range'].removeprefix('bytes=').split('-'))
            return Response(body[start:end+1], 206,
                            {'Content-Range': f'bytes {start}-{end}/{len(body)}'})
        return Response(body)

    monkeypatch.setattr(requests, 'get', get)
    data = defaults()
    data['sapiensid']['enabled'] = True
    data['sapiensid']['model'].update(source_directory='custom/source',
                                     checkpoint_directory='custom/checkpoints', model_yaml_sha256=sha(yaml))
    data['sapiensid']['pose'].update(path='custom/pose/model.pt', sha256=sha(pose))
    config = SimpleNamespace(data=data, source_path=tmp_path/'recognition.yml', model_dir=tmp_path/'models')
    validate(data)
    return SimpleNamespace(config=config, calls=calls, payloads=payloads, pose=pose, weights=weights, yaml=yaml)


def test_missing_assets_follow_configured_paths_and_reuse_offline(upstream, tmp_path, monkeypatch):
    source, checkpoints = assets.encoder_assets(upstream.config)
    pose, checksum = assets.pose_asset(upstream.config)
    assert source == tmp_path/'custom/source'
    assert checkpoints == tmp_path/'custom/checkpoints'
    assert pose == tmp_path/'custom/pose/model.pt'
    assert (checkpoints/'model.pth').read_bytes() == upstream.weights
    assert (checkpoints/'model.yaml').read_bytes() == upstream.yaml
    assert pose.read_bytes() == upstream.pose
    assert checksum == sha(upstream.pose)
    assert trusted_checkout(source) == source
    assert len(upstream.calls) == 4
    assert not list(tmp_path.rglob('*.part'))
    upstream.config.data['models']['download_missing'] = False
    monkeypatch.setattr(requests, 'get', lambda *a, **k: pytest.fail('warm/offline setup attempted network'))
    assert assets.encoder_assets(upstream.config) == (source, checkpoints)
    assert assets.pose_asset(upstream.config) == (pose, checksum)


def test_old_empty_paths_use_model_directory(upstream):
    settings = upstream.config.data['sapiensid']
    settings['model'].update(source_directory='', checkpoint_directory='')
    settings['pose'].update(path='', sha256='')
    validate(upstream.config.data)
    source, checkpoints = assets.encoder_assets(upstream.config)
    pose, checksum = assets.pose_asset(upstream.config)
    root = upstream.config.model_dir/'sapiensid_v1'
    assert (source, checkpoints, pose) == (root/'source', root/'wb4m', root/'yolov8n-pose.pt')
    assert checksum == registry.POSE_SHA256


@pytest.mark.parametrize('kind', ['source', 'weights', 'pose'])
def test_offline_missing_assets_do_not_fetch_or_create_destination(upstream, tmp_path, kind):
    if kind == 'weights':
        assets.ensure_source(upstream.config, tmp_path/'custom/source')
    upstream.calls.clear()
    upstream.config.data['models']['download_missing'] = False
    with pytest.raises(FileNotFoundError, match='models.download_missing is disabled'):
        (assets.pose_asset if kind == 'pose' else assets.encoder_assets)(upstream.config)
    assert not upstream.calls
    assert not (tmp_path/'custom/checkpoints').exists()
    assert not (tmp_path/'custom/pose').exists()


def test_pose_only_path_does_not_download_source_or_identity_weights(upstream, monkeypatch):
    import sys
    loaded = []

    def yolo(path, *, task):
        assert task == 'pose'
        loaded.append(Path(path))
        assert loaded[-1].read_bytes() == upstream.pose
        return 'pose model'

    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=yolo))
    assert load_pose(upstream.config) == 'pose model'
    assert [url for url, _ in upstream.calls] == [registry.POSE_URL]
    assert not loaded[0].exists()  # Library receives verified bytes in a temporary file.


def test_corrupt_existing_pose_is_not_overwritten_or_loaded(upstream, tmp_path, monkeypatch):
    path = tmp_path/'custom/pose/model.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'corrupted local file')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_pose(upstream.config)
    assert path.read_bytes() == b'corrupted local file'
    assert not upstream.calls


@pytest.mark.parametrize('kind', ['pose', 'yaml'])
def test_missing_custom_hash_requires_matching_local_asset(upstream, tmp_path, kind):
    if kind == 'pose':
        upstream.config.data['sapiensid']['pose']['sha256'] = 'a' * 64
        prepare = assets.pose_asset
    else:
        assets.ensure_source(upstream.config, tmp_path/'custom/source')
        upstream.calls.clear()
        upstream.config.data['sapiensid']['model']['model_yaml_sha256'] = 'a' * 64
        prepare = assets.encoder_assets
    with pytest.raises(FileNotFoundError, match='requires an existing matching local'):
        prepare(upstream.config)
    assert not upstream.calls


@pytest.mark.parametrize('change', ['modified', 'extra', 'symlink', 'revision'])
def test_modified_downloaded_source_is_rejected_without_replacement(upstream, tmp_path, change):
    source = assets.ensure_source(upstream.config, tmp_path/'custom/source')
    original = source/'general_utils/img_utils.py'
    if change == 'modified':
        original.write_text('# changed source')
    elif change == 'extra':
        (source/'additional.py').write_text('# unexpected code')
    elif change == 'symlink':
        (source/'linked.py').symlink_to(original)
    else:
        (source/assets.MARKER).write_text('unexpected revision')
    upstream.calls.clear()
    with pytest.raises(ValueError, match='source'):
        assets.ensure_source(upstream.config, source)
    assert source.exists()
    assert not upstream.calls


def test_source_failure_leaves_empty_destination_recoverable(upstream, tmp_path, monkeypatch):
    target = tmp_path/'custom/source'
    target.mkdir(parents=True)
    expected = registry.SOURCE_TREE_SHA256
    monkeypatch.setattr(registry, 'SOURCE_TREE_SHA256', '0' * 64)
    with pytest.raises(ValueError, match='source tree checksum'):
        assets.ensure_source(upstream.config, target)
    assert list(target.iterdir()) == []
    assert list(target.parent.iterdir()) == [target]
    monkeypatch.setattr(registry, 'SOURCE_TREE_SHA256', expected)
    assert assets.ensure_source(upstream.config, target) == target


def test_source_archive_cannot_escape_configured_directory(upstream, tmp_path, monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('sapiensid-'+registry.UPSTREAM_COMMIT+'/../../escaped.py', '# bad')
    data = archive.getvalue()
    upstream.payloads[registry.SOURCE_URL] = data
    monkeypatch.setattr(registry, 'SOURCE_BYTES', len(data))
    monkeypatch.setattr(registry, 'SOURCE_SHA256', sha(data))
    with pytest.raises(ValueError, match='Unsafe SapiensID source archive'):
        assets.ensure_source(upstream.config, tmp_path/'custom/source')
    assert not list(tmp_path.rglob('escaped.py'))
    assert not (tmp_path/'custom/source').exists()


@pytest.mark.parametrize('conflict', [errno.EEXIST, errno.ENOTEMPTY])
def test_concurrent_source_install_accepts_only_verified_winner(upstream, tmp_path, monkeypatch, conflict):
    import shutil

    def completed_elsewhere(source, target):
        shutil.copytree(source, target)
        raise OSError(conflict, 'Another installer published this directory')

    monkeypatch.setattr(assets.os, 'rename', completed_elsewhere)
    target = tmp_path/'custom/source'
    assert assets.ensure_source(upstream.config, target) == target
    assert trusted_checkout(target) == target
    assert list(target.parent.iterdir()) == [target]

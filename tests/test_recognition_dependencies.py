"""Upgrade failures must explain repair before weights or scene media are read."""
from dataclasses import replace
import shlex
import sys
from types import SimpleNamespace

import pytest

from stash_ai.recognition.config import load
from stash_ai.recognition.models import ensure_assets
from stash_ai.recognition.runtime import check_dependencies


@pytest.mark.parametrize('device', ['cpu', 'cuda', 'auto'])
@pytest.mark.parametrize('recognizer', ['sface', 'auraface'])
def test_missing_opencv_stops_before_assets_and_explains_exact_python(tmp_path, monkeypatch, device, recognizer):
    import stash_ai.recognition.runtime as face_runtime
    config = load(root=tmp_path, device=device)
    config = replace(config, data={**config.data, 'models': {**config.data['models'], 'recognizer': recognizer}})
    monkeypatch.setitem(sys.modules, 'cv2', None)
    monkeypatch.setitem(sys.modules, 'onnx', SimpleNamespace())
    monkeypatch.setattr(face_runtime.metadata, 'version', lambda name: pytest.fail('ORT initialized before dependency check'))
    with pytest.raises(ValueError, match='dependencies missing: cv2') as error:
        ensure_assets(config)
    message = str(error.value)
    assert sys.executable in message
    assert 'opencv-python-headless>=4.10,<5' in message
    assert 'updater does not install Python packages' in message
    assert not config.model_dir.exists()


@pytest.mark.parametrize('device', ['cuda', 'auto'])
def test_gpu_model_adaptation_requires_onnx_before_assets(tmp_path, monkeypatch, device):
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'onnx', None)
    with pytest.raises(ValueError, match='dependencies missing: onnx') as error:
        ensure_assets(load(root=tmp_path, device=device))
    assert 'onnx>=1.21,<2' in str(error.value)
    assert not (tmp_path / 'recognition_models').exists()


def test_cpu_does_not_require_onnx_graph_adaptation(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'onnx', None)
    check_dependencies(load(root=tmp_path, device='cpu'))


@pytest.mark.parametrize('platform', ['posix', 'nt'])
def test_repair_command_quotes_interpreter_and_both_requirements(tmp_path, monkeypatch, platform):
    import stash_ai.recognition.runtime as face_runtime
    config = load(root=tmp_path)
    executable = "C:/Python install/O'Brien/python.exe" if platform == 'nt' else '/opt/Python install/bin/python'
    monkeypatch.setattr(face_runtime, 'os', SimpleNamespace(name=platform))
    monkeypatch.setattr(sys, 'executable', executable)
    monkeypatch.setitem(sys.modules, 'cv2', None)
    monkeypatch.setitem(sys.modules, 'onnx', None)
    with pytest.raises(ValueError, match='dependencies missing: cv2, onnx') as error:
        check_dependencies(config)
    command = str(error.value).split('this command: ', 1)[1]
    if platform == 'nt':
        assert command.startswith("& 'C:/Python install/O''Brien/python.exe' ")
        assert "'opencv-python-headless>=4.10,<5' 'onnx>=1.21,<2'" in command
    else:
        assert shlex.split(command) == [executable, '-m', 'pip', 'install',
                                      'opencv-python-headless>=4.10,<5', 'onnx>=1.21,<2']


def test_broken_opencv_internal_import_is_not_misreported_as_missing_cv2(tmp_path, monkeypatch):
    import stash_ai.recognition.runtime as face_runtime
    error = ModuleNotFoundError("No module named 'numpy'", name='numpy')
    def broken_import(name):
        raise error
    monkeypatch.setattr(face_runtime, 'import_module', broken_import)
    with pytest.raises(ModuleNotFoundError) as raised:
        check_dependencies(load(root=tmp_path))
    assert raised.value is error


def test_run_with_missing_opencv_fails_once_before_scene_reads(tmp_path, monkeypatch, capsys):
    import stash_ai.cli as cli
    import stash_ai.recognition.cli as recognition
    from stash_ai.recognition.database import FaceDatabase
    from tests.test_recognition_run import setup, run
    config, client, services, backends = setup(tmp_path, monkeypatch)
    services.assets = ensure_assets
    monkeypatch.setitem(sys.modules, 'cv2', None)
    monkeypatch.setitem(sys.modules, 'onnx', SimpleNamespace())
    monkeypatch.setattr(client, 'scene', lambda identifier: pytest.fail('Scene fetched without OpenCV'))
    monkeypatch.setattr(sys, 'argv', ['stash_ai.py', 'recognition', 'run', '--scene-id', '1', '--scene-id', '2'])
    monkeypatch.setattr(recognition, 'main', lambda argv: run(config, services, '1', '2'))
    assert cli.run() == 1
    output = capsys.readouterr().err
    assert output.count('dependencies missing: cv2') == 1
    assert 'opencv-python-headless>=4.10,<5' in output
    assert client.media_calls == [] and client.writes == [] and backends == []
    with FaceDatabase(config.database) as database:
        assert database.connection.execute('SELECT status FROM runs').fetchone()[0] == 'failed'
        assert database.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 0


def test_recognition_help_does_not_require_opencv_or_onnx(monkeypatch, capsys):
    from stash_ai.cli import main
    monkeypatch.setitem(sys.modules, 'cv2', None)
    monkeypatch.setitem(sys.modules, 'onnx', None)
    with pytest.raises(SystemExit) as error:
        main(['recognition', 'run', '--help'])
    assert error.value.code == 0
    output = capsys.readouterr()
    assert '--device' in output.out and output.err == ''

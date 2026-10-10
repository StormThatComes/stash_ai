"""Publish complete JSON atomically, preserving exclusive output ownership."""
from contextlib import contextmanager
import json

import pytest

from stash_ai.reports import io


@pytest.mark.parametrize('failure', [OSError('disk full'), KeyboardInterrupt('interrupted')])
def test_failed_partial_write_leaves_no_destination_or_temporary(tmp_path, monkeypatch, failure):
    original = io.tempfile.NamedTemporaryFile
    @contextmanager
    def temporary(**kwargs):
        with original(**kwargs) as stream:
            write = stream.write
            def fail(payload):
                write(payload[:8])
                raise failure
            stream.write = fail
            yield stream
    destination = tmp_path / 'report.json'
    with monkeypatch.context() as patch:
        patch.setattr(io.tempfile, 'NamedTemporaryFile', temporary)
        with pytest.raises(type(failure)):
            io.write_json_report({'complete': True}, destination)
    assert not list(tmp_path.iterdir())
    io.write_json_report({'complete': True}, destination)
    assert json.loads(destination.read_text()) == {'complete': True}


def test_racing_destination_is_not_overwritten(tmp_path, monkeypatch):
    original = io.os.link
    destination = tmp_path / 'report.json'
    def race(source, target):
        assert not destination.exists()
        destination.write_text('other writer')
        return original(source, target)
    monkeypatch.setattr(io.os, 'link', race)
    with pytest.raises(FileExistsError):
        io.write_json_report({'complete': True}, destination)
    assert destination.read_text() == 'other writer'
    assert list(tmp_path.iterdir()) == [destination]


def test_invalid_json_does_not_create_output_directory(tmp_path):
    destination = tmp_path / 'new' / 'report.json'
    with pytest.raises(ValueError):
        io.write_json_report({'value': float('nan')}, destination)
    assert not destination.parent.exists()

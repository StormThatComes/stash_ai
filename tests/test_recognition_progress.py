"""Live recognition progress is separate from durable evidence checkpoints."""
from dataclasses import replace
from io import StringIO
import json
from threading import current_thread
from types import SimpleNamespace

import pytest
from tqdm import tqdm

from stash_ai.recognition.config import plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.progress import WorkProgress
from stash_ai.recognition.run import execute
from stash_ai.recognition.scanning import scan_scene
from stash_ai.runtime.console import ConsolePresenter
from stash_ai.runtime.report import RunReport
from tests.helpers.recognition import Backend, MediaClient
from tests.test_recognition_run import setup


class Terminal(StringIO):
    def isatty(self):
        return True


def args(*ids, style='auto'):
    return SimpleNamespace(scene_id=list(ids), all_scenes=False, progress_style=style)


def test_status_row_can_follow_two_work_rows_without_empty_buffer_rows():
    owner = ConsolePresenter(lambda _: None, stream=Terminal(), status_position=2).start()
    try:
        owner.call(owner._render)
        assert abs(owner._status.pos) == 2
    finally:
        owner.close()


def test_terminal_has_scene_frame_publication_and_performer_bars_with_one_owner(tmp_path, monkeypatch, capsys):
    import stash_ai.recognition.progress as ui
    config, _, services, _ = setup(tmp_path, monkeypatch)
    stream, bars, owners = Terminal(), [], set()
    class Bar(tqdm):
        def __init__(self, *args, **kwargs):
            owners.add(current_thread().name)
            super().__init__(*args, **kwargs)
            bars.append(self)
        def update(self, count=1):
            owners.add(current_thread().name)
            return super().update(count)
        def close(self):
            owners.add(current_thread().name)
            return super().close()
    monkeypatch.setattr(ui, 'tqdm', Bar)
    monkeypatch.setattr('sys.stderr', stream)
    assert execute(args('1', '2'), config, services=services) == 0
    rows = {bar.desc: bar for bar in bars}
    assert rows['Recognition scenes'].n == rows['Recognition scenes'].total == 2
    assert rows['Publish recognition'].n == rows['Publish recognition'].total == 2
    assert rows['Names and portraits'].n == rows['Names and portraits'].total == 1
    for scene in ('1', '2'):
        assert rows['VTT scene ' + scene].n == rows['VTT scene ' + scene].total == 3
        assert rows['VTT scene ' + scene].initial == 0
    assert owners == {'stash-console'}
    assert all(bar.disable for bar in bars), 'rows must be closed when execution finishes'
    output = capsys.readouterr().out
    assert output.startswith('Recognition success:')
    assert 'Face references scene' not in output and 'Performer names starter' not in output
    assert stream.getvalue().count('Phase: Recognition VTT') == 2
    assert stream.getvalue().count('Phase: Recognition metadata') == 2


def test_frame_updates_precede_checkpoint_and_resume_excludes_failed_frame(tmp_path, monkeypatch):
    config, _, _, _ = setup(tmp_path, monkeypatch)
    data = plain(config.data)
    data['execution']['checkpoint_cues'] = 100
    config = replace(config, data=data)
    client, events, checkpoints = MediaClient(), [], []
    with FaceDatabase(config.database) as db, SpriteInput(client, client.scene, config) as source:
        def live(**values):
            cursor = db.connection.execute('SELECT cursor FROM scans').fetchone()[0]
            events.append((values, cursor))
        with pytest.raises(RuntimeError, match='synthetic'):
            scan_scene(db, 'scope', client.scene, config, 'ns', source, lambda: Backend(fail_at=2),
                       progress=lambda **v: checkpoints.append(v), frame_progress=live)
        assert [(e['completed'], cursor) for e, cursor in events] == [(0, 0), (1, 0)]
        assert checkpoints == [{'scanned': 1, 'total': 3, 'scene': '1'}]
        assert db.connection.execute('SELECT cursor,status FROM scans').fetchone()[:] == (1, 'partial')
        events.clear()
        model = Backend()
        scan_scene(db, 'scope', client.scene, config, 'ns', source, lambda: model, frame_progress=live)
        assert [e['completed'] for e, _ in events] == [1, 2, 3]
        assert model.calls == 2
        events.clear()
        scan_scene(db, 'scope', client.scene, config, 'ns', source,
                   lambda: pytest.fail('cached scans must not load models'), frame_progress=live)
        assert events == [({'completed': 3, 'total': 3, 'reused': True}, 3)]


def test_resumed_eta_uses_only_new_frames_and_rendering_is_throttled(monkeypatch):
    import stash_ai.recognition.progress as ui
    clock = [10.]
    monkeypatch.setattr(ui, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    report = RunReport()
    report.console = ConsolePresenter(lambda _: None, stream=StringIO())
    with WorkProgress(report, 'VTT scene 1', 100, initial=90, unit='frame', key='frames', transient=True) as work:
        assert report.console.state['frames'] == '90/100'
        assert report.console.state['frames_eta'] == 'warming up'
        work.set_completed(91)
        assert report.console.state['frames'] == '90/100'
        clock[0] = 12.
        work.set_completed(92)
        assert report.console.state['frames_rate'] == '1.00 frame/s'
        assert report.console.state['frames_eta'] == '00:08'
    assert 'frames' not in report.console.state


@pytest.mark.parametrize('style', ['auto', 'compact', 'off', 'verbose'])
def test_styles_preserve_summary_and_history_without_default_yaml_spam(tmp_path, monkeypatch, capsys, style):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    assert execute(args('1', '2', style=style), config, services=services) == 0
    output = capsys.readouterr()
    assert 'Recognition success:' in output.out
    assert ('Performer names starter' in output.out) == (style == 'verbose')
    assert ('Face references scene 2' in output.out) == (style == 'verbose')
    assert 'VTT scene' not in output.err
    if style == 'off':
        assert output.err == ''
    else:
        assert 'Recognition scenes: inspected 2/2 scene' in output.err
    with FaceDatabase(config.database) as db:
        starter = json.loads(db.connection.execute("SELECT data FROM progress WHERE phase='name_starter'").fetchone()[0])
        assert starter['yaml']


def test_failed_input_advances_inspected_scenes_and_groups_repeated_errors(tmp_path, monkeypatch, capsys):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    def unavailable(*args):
        raise ValueError('synthetic unavailable VTT')
    services.source = unavailable
    assert execute(args('1', '2', '3'), config, services=services) == 1
    output = capsys.readouterr()
    assert output.out.count('ERROR recognition scan:') == 1
    assert 'Recognition partial:' in output.out and 'scenes_failed=3' in output.out
    assert 'Recognition scenes: inspected 3/3 scene' in output.err
    assert 'Phase: Recognition VTT partial' in output.err


def test_interruption_closes_live_rows_without_claiming_unfinished_scene(tmp_path, monkeypatch):
    import stash_ai.recognition.progress as ui
    config, client, services, models = setup(tmp_path, monkeypatch)
    bars = []
    class Bar(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            bars.append(self)
    original = services.backend
    def interrupted(*args):
        model = original(*args)
        embed = model.embed
        def infer(*values):
            if model.calls == 1:
                raise KeyboardInterrupt()
            return embed(*values)
        model.embed = infer
        return model
    services.backend = interrupted
    monkeypatch.setattr(ui, 'tqdm', Bar)
    monkeypatch.setattr('sys.stderr', Terminal())
    with pytest.raises(KeyboardInterrupt):
        execute(args('1', '2'), config, services=services)
    rows = {bar.desc: bar for bar in bars}
    assert rows['Recognition scenes'].n == 0 and rows['VTT scene 1'].n == 1
    assert all(bar.disable for bar in bars)
    assert client.closed and all(model.closed for model in models)
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT status FROM runs').fetchone()[0] == 'interrupted'
        assert db.connection.execute('SELECT cursor FROM scans').fetchone()[0] == 1

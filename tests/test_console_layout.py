"""Activity stays visible at ordinary widths and rows survive terminal resizing."""
from io import StringIO
from threading import current_thread

import pytest
from tqdm import tqdm

from stash_ai.runtime.console import ConsolePresenter
from stash_ai.runtime.console_layout import status_lines, terminal_width


class Terminal(StringIO):
    def isatty(self):
        return True


def snapshot():
    # Insert stage after the counters, as the recognition run actually does.
    return dict(phase='Recognition VTT', elapsed=12.4, last_progress_age=.3), dict(
        scenes='100/1000', scenes_eta='15:43', scenes_rate='1.05 scene/s', scene='6306',
        frames='55/120', frames_eta='00:05', frames_rate='12.25 frame/s', frame_cache='fresh',
        progress='scanned=100, reused=3, failed=0', stage='detect and embed VTT faces')


@pytest.mark.parametrize('width', [60, 79, 119])
def test_stage_has_its_own_early_row_before_work_metrics(width):
    row, fields = snapshot()
    before = dict(fields)
    lines = status_lines(row, fields, width)
    assert lines[0] == 'Status: Recognition VTT'
    assert lines[1] == 'Stage: detect and embed VTT faces'
    assert all(len(line) <= width for line in lines)
    assert 'scene=6306' in '\n'.join(lines)
    assert 'frames=55/120' in '\n'.join(lines)
    assert 'frames_eta=00:05' in '\n'.join(lines)
    assert 'last completed=0.3s ago' in '\n'.join(lines)
    assert fields == before


def test_long_stage_wraps_without_newline_injection_and_status_height_is_bounded():
    row, _ = snapshot()
    lines = status_lines(row, {'stage': 'decode\nverified\r\ninputs ' + 'x' * 100}, 40)
    assert lines[1].startswith('Stage: decode verified inputs')
    assert all('\n' not in line and '\r' not in line and len(line) <= 40 for line in lines)
    assert ''.join(lines).count('x') == 100
    fields = {f'field{index}': 'x' * 180 for index in range(32)}
    fields['stage'] = 'detect faces'
    lines = status_lines(row, fields, 40)
    assert len(lines) == 12 and lines[-1].startswith('... ')
    assert lines[1] == 'Stage: detect faces'


def test_width_uses_output_terminal_then_environment_fallback(monkeypatch):
    from types import SimpleNamespace
    import stash_ai.runtime.console_layout as layout
    monkeypatch.setattr(layout.os, 'get_terminal_size', lambda descriptor: SimpleNamespace(columns=61, lines=24))
    assert terminal_width(SimpleNamespace(fileno=lambda: 9)) == 60
    monkeypatch.setenv('COLUMNS', '80')
    assert terminal_width(Terminal()) == 79


@pytest.mark.parametrize('position', [2, 4])
def test_resize_and_phase_change_move_resources_below_status_and_close_every_row(monkeypatch, position):
    import stash_ai.runtime.console_layout as layout
    import tqdm as package
    width, bars, owners = [60], [], set()
    class Bar(tqdm):
        def __init__(self, *args, **kwargs):
            owners.add(current_thread().name)
            super().__init__(*args, **kwargs)
            bars.append(self)
        def set_description_str(self, *args, **kwargs):
            owners.add(current_thread().name)
            return super().set_description_str(*args, **kwargs)
        def close(self):
            owners.add(current_thread().name)
            return super().close()
    monkeypatch.setattr(package, 'tqdm', Bar)
    monkeypatch.setattr(layout, 'terminal_width', lambda _: width[0])
    stream = Terminal()
    owner = ConsolePresenter(lambda _: None, stream=stream, status_position=position, refresh=60).start()
    resources = dict(cache=dict(retained_bytes=0, effective_bytes=1024, configured_bytes=1024),
                     parent_rss=None, evictions_delta=0, pressure_trims_delta=0)
    owner.register_sampler('cache', lambda: resources)
    try:
        work = [owner.bar(Bar, total=1, position=i, file=stream) for i in range(position)]
        row, fields = snapshot()
        owner.phase_name = row['phase']
        owner.update(**fields)
        owner.call(owner._render)
        narrow_rows, old_resource = tuple(owner._status_rows), owner._resource_row
        assert owner._status_rows[1].desc == 'Stage: detect and embed VTT faces'
        assert abs(old_resource.pos) == position + len(narrow_rows)
        width[0] = 119
        owner.call(owner._render)
        assert len(owner._status_rows) < len(narrow_rows)
        assert old_resource.disable
        active = [bar.target for bar in work] + owner._status_rows + [owner._resource_row]
        assert len({abs(bar.pos) for bar in active}) == len(active)
        assert abs(owner._resource_row.pos) == position + len(owner._status_rows)
        with owner.phase('Recognition assets'):
            owner.call(owner._render)
            assert len(owner._status_rows) == 2
            assert not any('Stage:' in bar.desc for bar in owner._status_rows)
            assert abs(owner._resource_row.pos) == position + 2
        owner.finish_rendering()
        assert all(bar.disable for bar in bars)
        assert owners == {'stash-console'}
    finally:
        owner.close()
    assert not owner.thread.is_alive() and owner._status_rows == []


def test_redirected_heartbeat_remains_one_plain_line_with_no_terminal_rows():
    stream = StringIO()
    owner = ConsolePresenter(lambda _: None, stream=stream, heartbeat=0).start()
    try:
        _, fields = snapshot()
        owner.update(**fields)
        owner.call(owner._render)
        assert owner._status_rows == []
        lines = stream.getvalue().splitlines()
        assert len(lines) == 1 and lines[0].startswith('Still working: Status:')
        assert 'stage=detect and embed VTT faces' in lines[0]
        assert '\x1b' not in lines[0]
    finally:
        owner.close()

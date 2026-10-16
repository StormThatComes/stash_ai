"""Every recognition phase exposes actual work, including cold/cached replay."""
from io import StringIO
import json
import sqlite3
from threading import current_thread
from types import SimpleNamespace

import pytest
from tqdm import tqdm

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.progress import tracking, work
from stash_ai.recognition.run import execute
from stash_ai.runtime.console import ConsolePresenter, present
from stash_ai.runtime.report import RunReport
from tests.test_recognition_progress import args
from tests.test_recognition_rebuild import changed
from tests.test_recognition_run import setup, run


class Terminal(StringIO):
    def isatty(self):
        return True


@pytest.fixture
def terminal(monkeypatch):
    import stash_ai.recognition.progress as progress
    bars, owners = [], set()
    class Bar(tqdm):
        def __init__(self, *args, **kwargs):
            owners.add(current_thread().name)
            super().__init__(*args, **kwargs)
            assert not any(not other.disable and other.pos == self.pos for other in bars), 'overlapping work rows'
            bars.append(self)
        def update(self, count=1):
            owners.add(current_thread().name)
            return super().update(count)
        def close(self):
            owners.add(current_thread().name)
            return super().close()
    clock = [0.]
    def now():
        clock[0] += .3
        return clock[0]
    monkeypatch.setattr(progress, 'time', SimpleNamespace(monotonic=now))
    monkeypatch.setattr(progress, 'tqdm', Bar)
    initialize = ConsolePresenter.__init__
    def initialize_terminal(self, write, **options):
        initialize(self, write, **{**options, 'stream': Terminal()})
    monkeypatch.setattr(ConsolePresenter, '__init__', initialize_terminal)
    return bars, owners


def test_cold_rebuild_has_scene_observation_binding_and_committed_transaction_progress(tmp_path, monkeypatch, terminal):
    import stash_ai.recognition.clustering as clustering
    config, _, services, _ = setup(tmp_path, monkeypatch)
    bars, owners = terminal
    seen = []
    original = clustering.vector
    def vector(row):
        active = next((bar for bar in bars if bar.desc == 'Match face observations' and not bar.disable), None)
        if active is not None:
            seen.append(active.n)
        return original(row)
    monkeypatch.setattr(clustering, 'vector', vector)
    assert execute(args('1', '2'), config, services=services) == 0
    rows = {bar.desc: bar for bar in bars}
    assert rows['Copy selected evidence'].n == rows['Copy selected evidence'].total == 2
    assert rows['Rebuild selected identities'].n == rows['Rebuild selected identities'].total == 2
    assert rows['Copy face observations'].n == rows['Copy face observations'].total == 3
    assert rows['Match face observations'].n == rows['Match face observations'].total == 3
    assert rows['Bind rebuilt identities'].n == rows['Bind rebuilt identities'].total == 1
    assert rows['Apply rebuilt identities'].n == rows['Apply rebuilt identities'].total == 1
    assert rows['Save rebuilt memberships'].n == rows['Save rebuilt memberships'].total == 6
    assert rows['Collect rebuilt identities'].n == rows['Collect rebuilt identities'].total == 2
    assert seen == [0, 1, 2, 0, 1, 2], 'matching must advance during each scene, not only after it'
    assert all(bar.disable for bar in bars) and owners == {'stash-console'}


def test_cached_rebuild_validates_observations_without_fabricating_replay_work(tmp_path, monkeypatch, terminal):
    config, _, services, models = setup(tmp_path, monkeypatch)
    assert run(config, services, '1', '2') == 0
    loaded = len(models)
    bars, owners = terminal
    assert execute(args('1', '2'), config, services=services) == 0
    rows = {bar.desc: bar for bar in bars}
    assert rows['Validate cached identity rebuild'].n == rows['Validate cached identity rebuild'].total == 2
    assert rows['Validate cached face observations'].n == rows['Validate cached face observations'].total == 3
    assert 'Copy selected evidence' not in rows and 'Match face observations' not in rows
    assert 'Apply rebuilt identities' not in rows
    assert len(models) == loaded and owners == {'stash-console'}


def test_same_scene_context_reports_scene_support_and_assignment_work(tmp_path, monkeypatch, terminal):
    import math
    from tests.test_recognition_rebuild import library
    vectors = {'1': [[1., 0., 0., 0.]] * 3 + [[.63, math.sqrt(1 - .63**2), 0., 0.]]}
    config, _, services, _ = library(tmp_path, monkeypatch, vectors)
    bars, owners = terminal
    assert execute(args('1'), config, services=services) == 0
    rows = {bar.desc: bar for bar in bars}
    for phase, count in [('Apply same-scene context', 1), ('Validate same-scene support', 3),
                         ('Match same-scene observations', 1)]:
        assert rows[phase].n == rows[phase].total == count
    assert all(bar.disable for bar in bars) and owners == {'stash-console'}


def test_selection_preflight_publication_and_names_starter_also_have_progress(tmp_path, monkeypatch, terminal):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    bars, _ = terminal
    assert execute(args('1', '2'), config, services=services) == 0
    rows = {bar.desc: bar for bar in bars}
    for label, expected in (
        ('Read recognition scene IDs', 2), ('Prepare recognition assets', 1),
        ('Reconcile scene publications', 2), ('Prepare selected names', 1),
        ('Validate selected names', 1), ('Prepare scene identities', 1),
        ('Link scene identities', 1), ('Resolve metadata targets', 1), ('Read names starter', 1),
    ):
        assert rows[label].n == rows[label].total == expected, label


def test_real_asset_preparation_reports_each_verified_asset(tmp_path, monkeypatch, terminal):
    import stash_ai.recognition.models as models
    import stash_ai.recognition.runtime as runtime
    import stash_ai.backends.downloads as downloads
    config, _, services, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, 'runtime', lambda _: None)
    def install(repo, filename, revision, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic asset')
    monkeypatch.setattr(downloads, 'install_hf_asset', install)
    monkeypatch.setattr(models, 'verify', lambda path, asset: path.read_bytes())
    services.assets = models.ensure_assets
    assert execute(args('1'), config, services=services) == 0
    row = next(bar for bar in terminal[0] if bar.desc == 'Verify recognition assets')
    assert row.n == row.total == 2 and row.disable


@pytest.mark.parametrize('kind', ['vtt', 'video'])
def test_portrait_candidates_advance_within_a_single_performer(tmp_path, monkeypatch, terminal, kind):
    if kind == 'vtt':
        from tests.test_recognition_vtt_portraits import setup as prepare
        config, client, services, _ = prepare(tmp_path, monkeypatch, weights={'visibility': 0})
        label, expected = 'Rank VTT portrait observations', 3
    else:
        from tests.test_recognition_metadata import setup as prepare
        config, client, services = prepare(tmp_path, monkeypatch)
        label, expected = 'Check video portrait frames', 2
    assert execute(args('1'), config, services=services) == 0
    row = next(bar for bar in terminal[0] if bar.desc == label)
    assert row.n == row.total == expected and row.disable
    assert client.images


def test_all_scene_selection_keeps_unknown_total_instead_of_a_fake_eta(tmp_path, monkeypatch, terminal):
    config, _, services, _ = setup(tmp_path, monkeypatch)
    invocation = args()
    invocation.all_scenes = True
    assert execute(invocation, config, services=services) == 0
    row = next(bar for bar in terminal[0] if bar.desc == 'Read recognition scene IDs')
    assert row.n == 2 and row.total is None
    report = RunReport(write=lambda _: None)
    with present(report, 'auto'):
        with work('Unknown source', unit='scene') as progress:
            progress.advance()
            assert report.console.state['detail'] == '1/?'
            assert report.console.state['detail_eta'] == 'unknown'


def test_failed_apply_keeps_transaction_progress_zero_and_rolls_back_assignments(tmp_path, monkeypatch, terminal, capsys):
    config, client, services, _ = setup(tmp_path, monkeypatch)
    assert run(config, services, '1') == 0
    with FaceDatabase(config.database) as db:
        before = [tuple(row) for row in db.connection.execute('SELECT * FROM members ORDER BY observation')]
        candidates = [tuple(row) for row in db.connection.execute('SELECT * FROM candidates ORDER BY id')]
    # Fail a real SQLite write after constructor validation, without persisting
    # an unsupported trigger that must now be refused before recovery.
    class RejectRebuild(FaceDatabase):
        def __init__(self, path):
            super().__init__(path)
            self.connection.set_authorizer(lambda action, table, *args:
                sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_INSERT and table == 'metadata'
                else sqlite3.SQLITE_OK)
    monkeypatch.setattr('stash_ai.recognition.run.FaceDatabase', RejectRebuild)
    capsys.readouterr()
    tuned = changed(config, recognition={'match_threshold': .95})
    writes = list(client.writes)
    assert execute(args('1'), tuned, services=services) == 1
    row = next(bar for bar in terminal[0] if bar.desc == 'Apply rebuilt identities')
    assert row.n == 0 and row.total == 1 and row.disable
    assert client.writes == writes
    assert 'Recognition partial:' in capsys.readouterr().out
    with FaceDatabase(config.database) as db:
        assert [tuple(row) for row in db.connection.execute('SELECT * FROM members ORDER BY observation')] == before
        assert [tuple(row) for row in db.connection.execute('SELECT * FROM candidates ORDER BY id')] == candidates
        assert json.loads(db.connection.execute('SELECT summary FROM runs ORDER BY started DESC').fetchone()[0])['counts']['scenes_failed'] == 1


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_interrupted_matching_closes_bars_without_completing_the_active_scene(tmp_path, monkeypatch, terminal, failure):
    import stash_ai.recognition.clustering as clustering
    config, _, services, _ = setup(tmp_path, monkeypatch)
    bars, _ = terminal
    original = clustering.vector
    calls = []
    def vector(row):
        if any(bar.desc == 'Match face observations' and not bar.disable for bar in bars):
            calls.append(row['id'])
            if len(calls) == 2:
                raise failure('synthetic interrupted matching')
        return original(row)
    monkeypatch.setattr(clustering, 'vector', vector)
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            execute(args('1'), config, services=services)
    else:
        assert execute(args('1'), config, services=services) == 1
    rows = {bar.desc: bar for bar in bars}
    assert rows['Rebuild selected identities'].n == 0
    assert rows['Match face observations'].n == 1
    assert all(bar.disable for bar in bars)
    with FaceDatabase(config.database) as db:
        assert db.connection.execute('SELECT COUNT(*) FROM members').fetchone()[0] == 0


@pytest.mark.parametrize('style', ['off', None])
def test_disabled_or_standalone_tracking_does_not_count_or_materialize_inputs(style):
    from contextlib import nullcontext
    seen = []
    def source():
        for value in range(3):
            seen.append(value)
            yield value
    report = RunReport(write=lambda _: None)
    with present(report, style) if style else nullcontext():
        with tracking(source(), 'Unused UI', total=lambda: pytest.fail('extra total query')) as items:
            assert seen == []
            assert next(items) == 0 and seen == [0]


def test_early_success_counts_the_processed_item_and_closes_its_row(terminal):
    report = RunReport(write=lambda _: None)
    def choose():
        with tracking([1, 2, 3], 'Choose candidate') as items:
            for value in items:
                return value
    with present(report, 'auto'):
        assert choose() == 1
    row = next(bar for bar in terminal[0] if bar.desc == 'Choose candidate')
    assert row.n == 1 and row.total == 3 and row.disable


def test_failed_optional_total_falls_back_to_indeterminate_work(terminal):
    report = RunReport(write=lambda _: None)
    def count():
        raise RuntimeError('optional count unavailable')
    with present(report, 'auto'):
        with tracking([1, 2], 'Optional total', total=count) as values:
            assert list(values) == [1, 2]
    row = next(bar for bar in terminal[0] if bar.desc == 'Optional total')
    assert row.n == 2 and row.total is None
    assert report.counts['recognition_progress_total_failures'] == 1


def test_lazy_asset_subphase_suspends_and_restores_detail_row(terminal):
    report = RunReport(False)
    with present(report, 'auto', status_position=2):
        with work('Outer candidate phase', 2, position=1) as outer:
            outer.advance()
            with work('Nested asset phase', 1, position=1) as inner:
                inner.advance()
            outer.advance()
    bars, owners = terminal
    assert outer.n == 2 and inner.n == 1
    assert max(bar.n for bar in bars if bar.desc == 'Outer candidate phase') == 2
    assert next(bar for bar in bars if bar.desc == 'Nested asset phase').n == 1
    assert owners == {'stash-console'}

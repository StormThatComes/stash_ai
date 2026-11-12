"""Numeric scene ranges select a finite union of existing Stash scenes."""
import json

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.run import _scene_ids, execute
from stash_ai.recognition.stash import RecognitionStash
from tests.test_recognition_run import setup


def test_ranges_accept_multiple_normalized_inclusive_bounds_and_explicit_ids():
    args = parser().parse_args(['run', '--scene-range', '0010-0020',
        '--scene-id', '0007', '--scene-range', ' 30 - 30 '])
    assert args.scene_range == [(10, 20), (30, 30)]
    assert args.scene_id == ['7']
    assert not args.all_scenes


@pytest.mark.parametrize('value', ['', '1', '1-', '-2', '1-2-3', '1:2', '1,2',
    '0-2', '1-0', '-1-2', '2-1', 'a-2', '1-2.5', '1-2e3', '１-２', '+1-2'])
def test_malformed_ranges_fail_during_argument_parsing(value, capsys):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(['run', '--scene-range=' + value])
    assert error.value.code == 2
    assert '--scene-range' in capsys.readouterr().err


@pytest.mark.parametrize('argv', [
    ['run'],
    ['run', '--all-scenes', '--scene-range', '1-2'],
    ['run', '--scene-range', '1-2', '--all-scenes'],
    ['run', '--scene-id', '1', '--all-scenes'],
    ['portraits', '--scene-range', '1-2'],
])
def test_scope_validation_is_preserved(argv):
    with pytest.raises(SystemExit) as error:
        parser().parse_args(argv)
    assert error.value.code == 2


def test_wide_ranges_filter_paginated_ids_without_expanding_integers():
    client = RecognitionStash.__new__(RecognitionStash)
    pages = []

    def gql(query, variables):
        page = variables['filter']['page']
        pages.append(page)
        assert page in (1, 2)
        assert variables['filter']['per_page'] == 1000
        values = range(1, 1001) if page == 1 else [1001, 10**30]
        return {'findScenes': {'scenes': [{'id': str(value)} for value in values]}}

    client.gql = gql
    args = parser().parse_args(['run', '--scene-range', '999-1001',
        '--scene-range', f'1000000-{10**30}'])
    assert list(_scene_ids(args, client)) == ['999', '1000', '1001', str(10**30)]
    assert pages == [1, 2]


@pytest.mark.parametrize('selectors,expected,listing_calls', [
    (['--scene-range', '10-12', '--scene-range', '12-20', '--scene-range', '30-30',
      '--scene-id', '12', '--scene-id', '2'], ['2', '10', '12', '20', '30'], 1),
    (['--scene-range', '10-12', '--scene-range', '30-30'], ['10', '12', '30'], 1),
    (['--scene-range', '40-50'], [], 1),
    (['--scene-range', '40-50', '--scene-id', '2'], ['2'], 1),
    (['--scene-id', '2', '--scene-id', '2'], ['2'], 0),
    (['--all-scenes'], ['2', '9', '10', '12', '20', '30', '31'], 1),
])
def test_run_snapshots_and_processes_only_the_unique_selected_scenes(
        tmp_path, monkeypatch, selectors, expected, listing_calls):
    import stash_ai.recognition.run as runner

    config, client, services, backends = setup(tmp_path, monkeypatch)
    # Deliberately not numerically sorted, and with gaps inside the ranges.
    client.links = {value: [] for value in ['31', '12', '2', '20', '10', '9', '30']}
    listings = []

    def scene_ids():
        listings.append(True)
        return iter(client.links)

    monkeypatch.setattr(client, 'scene_ids', scene_ids)
    scanned = []
    scan_scene = runner.scan_scene

    def scan(database, scope, scene, *args, **kwargs):
        scanned.append(scene['id'])
        return scan_scene(database, scope, scene, *args, **kwargs)

    monkeypatch.setattr(runner, 'scan_scene', scan)
    args = parser().parse_args(['run', *selectors, '--progress-style', 'off'])
    assert execute(args, config, services=services) == 0
    assert len(listings) == listing_calls
    assert scanned == expected
    assert {scene for kind, scene in client.reads if kind == 'scene'} == set(expected)
    assert {url.rsplit('/', 1)[-1].split('.')[0] for url in client.media_calls} == set(expected)
    assert all(not links for scene, links in client.links.items() if scene not in expected)
    with FaceDatabase(config.database) as database:
        history = database.connection.execute('SELECT * FROM runs').fetchone()
        selection = json.loads(history['selection'])
        assert sorted(selection, key=int) == expected
        assert len(selection) == len(set(selection))
        assert history['status'] == 'success'
        assert json.loads(history['summary'])['counts'].get('scenes_processed', 0) == len(expected)
    if not expected:
        assert not client.writes and not backends

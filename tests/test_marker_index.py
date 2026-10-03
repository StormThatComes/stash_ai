"""Indexed native marker lookup must preserve the old full-scan reconciliation."""
from copy import deepcopy
import math
import random

import pytest

from stash_ai import cli as app, merge, merge_sync
from stash_ai.marker_index import MarkerIntervalIndex, PreviousMarkerRanges
from tests.helpers.evidence import span
from tests.helpers.stash import Stash

pytestmark = pytest.mark.usefixtures('reset_merge')


def marker(identifier, start, end, primary='core'):
    return dict(id=str(identifier), seconds=start, end_seconds=end,
                primary_tag=dict(id=primary, name='ai:' + primary))


def old_temporal(value, ranges):
    ends = [row['end'] for row in ranges if row['end'] is not None
            and 'ai:' + row['tag'] == value['primary_tag']['name']
            and abs(round(row['start'], 3) - value['seconds']) <= .0005]
    return {**value, 'end_seconds': max(ends)} if ends else value


@pytest.mark.parametrize('tolerance', [0, .0005, .25, 1])
@pytest.mark.parametrize('by_name', [False, True])
def test_lookup_matches_full_scan_for_nested_duplicates_points_and_boundaries(tolerance, by_name):
    rng = random.Random(381)
    markers = [marker(i, float(rng.randrange(-20, 40)), rng.choice([None, float(rng.randrange(-20, 60))]),
                      rng.choice(['core', 'other'])) for i in range(250)]
    markers += [marker('long', -100, 100), marker('touch', 10, 20), marker('duplicate', 10, 20),
                marker('rounding', -1e-17, None), marker('point', 20, None)]
    rng.shuffle(markers)
    saved = deepcopy(markers)
    index = MarkerIntervalIndex(markers, PreviousMarkerRanges([]), by_name=by_name)
    queries = [(start, end) for start in (-100, 0, 1, 10, 20, 100) for end in (None, 0, 10, 20, 100)]
    queries += [(rng.uniform(-25, 65), rng.choice([None, rng.uniform(-25, 65)])) for _ in range(100)]
    for label in ('core', 'other', 'missing'):
        primary = 'ai:' + label if by_name else label
        for start, end in queries:
            interval = dict(seconds=start, end_seconds=end)
            expected = [m for m in markers if m['primary_tag']['name'] == 'ai:' + label
                        and merge.overlaps(m, interval, tolerance)]
            assert index.overlapping(primary, interval, tolerance) == expected
    assert markers == saved


def test_start_tolerance_does_not_extend_interval_ends():
    index = MarkerIntervalIndex([marker('old', 0, 5), marker('point', 10, None)], PreviousMarkerRanges([]))
    assert index.overlapping('core', dict(seconds=5.1, end_seconds=6), .25) == []
    assert [m['id'] for m in index.overlapping('core', dict(seconds=10.2, end_seconds=None), .25)] == ['point']
    assert [m['id'] for m in index.overlapping('core', dict(seconds=5, end_seconds=6))] == ['old']


def test_saved_point_extents_match_original_rounding_and_maximum_selection():
    ranges = [dict(tag='core', start=10.0004, end=15), dict(tag='core', start=10.0003, end=30),
              dict(tag='core', start=10.001, end=25), dict(tag='core', start=10, end=None),
              dict(tag='other', start=10, end=50)]
    ranges += [dict(tag='core', start=i * 3, end=i * 3 + 1) for i in range(200)]
    saved = deepcopy(ranges)
    previous = PreviousMarkerRanges(ranges)
    for start in (10, 10.0005, math.nextafter(10.0005, -math.inf), math.nextafter(10.0005, math.inf),
                  10.001, 11, 12, 999):
        for label in ('core', 'other', 'missing'):
            value = marker('id', start, None, label)
            assert previous.temporal(value) == old_temporal(value, ranges)
    value = marker('id', 10, None)
    index = MarkerIntervalIndex([value], previous)
    assert index.temporal(value)['end_seconds'] == 30
    assert index.overlapping('core', dict(seconds=20, end_seconds=22)) == [value]
    assert value['end_seconds'] is None and ranges == saved


def test_nested_long_range_does_not_force_scanning_unrelated_short_ranges(monkeypatch):
    markers = [marker(i, i*3, i*3 + 1) for i in range(10000)]
    markers.append(marker('outer', 0, 40000))
    index = MarkerIntervalIndex(markers, PreviousMarkerRanges([]))
    calls, overlaps = [], merge.overlaps
    def counted(*args):
        calls.append(args[0]['id'])
        return overlaps(*args)
    monkeypatch.setattr(merge, 'overlaps', counted)
    assert {m['id'] for m in index.overlapping('core', dict(seconds=15000, end_seconds=15001))} == {'5000', 'outer'}
    assert len(calls) == 2


class ScanningIndex:
    """Reference semantics: visit every marker of the selected primary tag."""
    def __init__(self, markers, previous_ranges, *, by_name=False):
        self.markers, self.previous, self.by_name = markers, previous_ranges, by_name

    def temporal(self, marker):
        return self.previous.temporal(marker)

    def overlapping(self, primary, interval, tolerance=0):
        field = 'name' if self.by_name else 'id'
        return [m for m in self.markers if str(m['primary_tag'][field]) == primary
                and merge.overlaps(self.temporal(m), interval, tolerance)]


@pytest.mark.parametrize('use_end', [True, False])
@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('cleanup', [False, True])
def test_publication_split_duplicates_cleanup_and_retry_match_scanner(
        tmp_path, monkeypatch, use_end, dry, cleanup):
    monkeypatch.setattr(app, 'MARKER_USE_END_TIME', use_end)
    variants = []
    for implementation in (ScanningIndex, MarkerIntervalIndex):
        with monkeypatch.context() as patch:
            patch.setattr(merge_sync, 'MarkerIntervalIndex', implementation)
            stash = Stash()
            manual = stash.marker(merge.AI, 'location', 'manual', manual=True)
            manual.update(seconds=75, end_seconds=None)
            point = {**span('wd', 75, None, 'location'), 'derived': 'occurrence'}
            path = tmp_path/(implementation.__name__ + '.sqlite3')
            resolver = app.TagResolver(stash, source=merge.AI)
            # Seed a normal publication before testing cleanup and preview.
            base = [span('wd', 10, 20), span('dtq', 30, 40), point]
            with merge.StateStore(path, 'scope') as store:
                merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), [], base)
            first = next(m for m in stash.scene['scene_markers'] if m['primary_tag']['name'] == 'ai:common')
            first['tags'].append(stash.tag('manual A'))
            duplicate = deepcopy(first)
            duplicate['id'] = 'duplicate'
            duplicate['tags'].append(stash.tag('manual B'))
            stash.scene['scene_markers'].append(duplicate)
            stash.dry_run = dry
            source = path.read_bytes()
            with merge.StateStore(path, 'scope', dry_run=dry, cache_bytes=2**20) as store:
                sync = lambda spans: merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'),
                                                     [], spans, cleanup=cleanup, migrate=False)
                bridged = sync([span('wd', 11, 38), point])
                split = [span('wd', 12, 16), span('dtq', 34, 37), point]
                if not dry:
                    update = stash.update_marker
                    def lost_response(payload):
                        update(payload)
                        raise RuntimeError('lost response')
                    stash.update_marker = lost_response
                    with pytest.raises(RuntimeError, match='lost response'):
                        sync(split)
                    assert store.has_pending_publications()
                    stash.update_marker = update
                finished = sync(split)
                repeated = sync(split)
                assert not store.has_pending_publications()
                assert stash.scene['scene_markers'][0] == manual
                if not dry:
                    common = [m for m in stash.scene['scene_markers'] if m['primary_tag']['name'] == 'ai:common']
                    assert len(common) == 2 and repeated['markers_unchanged'] == 2
                    assert all({'manual A', 'manual B'} <= {t['name'] for t in m['tags']} for m in common)
                variants.append((deepcopy(stash.scene), deepcopy(stash.writes), bridged, finished, repeated,
                                 store.snapshot('1', 'markers')))
            if dry:
                assert path.read_bytes() == source
    assert variants[0] == variants[1]

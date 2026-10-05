"""Indexed native marker lookup must preserve the old full-scan reconciliation."""
from copy import deepcopy
import random

import pytest

from tests.helpers.runtime import runtime as app
import stash_ai.evidence.interval_geometry as merge_intervals
import stash_ai.evidence.labels as merge_labels
import stash_ai.storage.store as merge_store
from stash_ai.publication import merged as merge_sync
from stash_ai.publication.index import MarkerIntervalIndex
from tests.helpers.evidence import span
from tests.helpers.stash import Stash

pytestmark = pytest.mark.usefixtures('reset_merge')


def marker(identifier, start, end, primary='core'):
    return dict(id=str(identifier), seconds=start, end_seconds=end,
                primary_tag=dict(id=primary, name='ai:' + primary))


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
    index = MarkerIntervalIndex(markers, by_name=by_name)
    queries = [(start, end) for start in (-100, 0, 1, 10, 20, 100) for end in (None, 0, 10, 20, 100)]
    queries += [(rng.uniform(-25, 65), rng.choice([None, rng.uniform(-25, 65)])) for _ in range(100)]
    for label in ('core', 'other', 'missing'):
        primary = 'ai:' + label if by_name else label
        for start, end in queries:
            interval = dict(seconds=start, end_seconds=end)
            expected = [m for m in markers if m['primary_tag']['name'] == 'ai:' + label
                        and merge_intervals.overlaps(m, interval, tolerance)]
            assert index.overlapping(primary, interval, tolerance) == expected
    assert markers == saved


def test_start_tolerance_does_not_extend_interval_ends():
    index = MarkerIntervalIndex([marker('old', 0, 5), marker('point', 10, None)])
    assert index.overlapping('core', dict(seconds=5.1, end_seconds=6), .25) == []
    assert [m['id'] for m in index.overlapping('core', dict(seconds=10.2, end_seconds=None), .25)] == ['point']
    assert [m['id'] for m in index.overlapping('core', dict(seconds=5, end_seconds=6))] == ['old']


def test_nested_long_range_does_not_force_scanning_unrelated_short_ranges(monkeypatch):
    markers = [marker(i, i*3, i*3 + 1) for i in range(10000)]
    markers.append(marker('outer', 0, 40000))
    index = MarkerIntervalIndex(markers)
    calls, overlaps = [], merge_intervals.overlaps
    def counted(*args):
        calls.append(args[0]['id'])
        return overlaps(*args)
    monkeypatch.setattr(merge_intervals, 'overlaps', counted)
    assert {m['id'] for m in index.overlapping('core', dict(seconds=15000, end_seconds=15001))} == {'5000', 'outer'}
    assert len(calls) == 2


class ScanningIndex:
    """Reference semantics: visit every marker of the selected primary tag."""
    def __init__(self, markers, *, by_name=False):
        self.markers, self.by_name = markers, by_name

    def overlapping(self, primary, interval, tolerance=0):
        field = 'name' if self.by_name else 'id'
        return [m for m in self.markers if str(m['primary_tag'][field]) == primary
                and merge_intervals.overlaps(m, interval, tolerance)]


@pytest.mark.parametrize('dry', [False, True])
@pytest.mark.parametrize('cleanup', [False, True])
def test_publication_split_duplicates_cleanup_and_retry_match_scanner(
        tmp_path, monkeypatch, dry, cleanup):
    variants = []
    for implementation in (ScanningIndex, MarkerIntervalIndex):
        with monkeypatch.context() as patch:
            from stash_ai.publication.planning import merged_evidence, merged_markers
            patch.setattr(merged_evidence, 'MarkerIntervalIndex', implementation)
            patch.setattr(merged_markers, 'MarkerIntervalIndex', implementation)
            stash = Stash()
            manual = stash.marker(merge_labels.AI, 'location', 'manual', manual=True)
            manual.update(seconds=75, end_seconds=None)
            point = {**span('wd', 75, None, 'location'), 'derived': 'occurrence'}
            path = tmp_path/(implementation.__name__ + '.sqlite3')
            resolver = app.services.TagResolver(stash, source=merge_labels.AI)
            # Seed a normal publication before testing cleanup and preview.
            base = [span('wd', 10, 20), span('dtq', 30, 40), point]
            with merge_store.StateStore(path, 'scope') as store:
                merge_sync.sync_merged(app, stash, resolver, store, stash.find_scene('1'), [], base)
            first = next(m for m in stash.scene['scene_markers'] if m['primary_tag']['name'] == 'ai:common')
            first['tags'].append(stash.tag('manual A'))
            duplicate = deepcopy(first)
            duplicate['id'] = 'duplicate'
            duplicate['tags'].append(stash.tag('manual B'))
            stash.scene['scene_markers'].append(duplicate)
            stash.dry_run = dry
            source = path.read_bytes()
            with merge_store.StateStore(path, 'scope', dry_run=dry, cache_bytes=2**20) as store:
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



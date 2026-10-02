"""Wide retained caches must preserve publication and label validation."""
import builtins
from copy import deepcopy
from dataclasses import replace
import math

import pytest

from stash_ai import cli as app, fusion, merge, temporal, thresholds
from tests.helpers.evidence import frames, policies, segment

pytestmark = pytest.mark.usefixtures('reset_temporal')


def group(records):
    return {'members': [(source, item, source + '-proof')
                        for source, items in records.items() for item in items]}


def test_shared_observations_sort_only_eligible_scores(monkeypatch):
    rows = [[('core', .9), ('support', .7)] + [(f'weak_{i}', .2) for i in range(100)]] * 4
    records = {'wd': [segment('wd', rows)]}
    sorted_sizes = []
    def observe_sort(values, **kwargs):
        values = list(values)
        if kwargs.get('key') is not None:
            sorted_sizes.append(len(values))
        return builtins.sorted(values, **kwargs)
    monkeypatch.setattr(fusion, 'sorted', observe_sort, raising=False)
    result = fusion.observations(app, group(records), policies())
    assert sorted_sizes == [2] * 4
    assert all(set(event['cores']) == {'core', 'support'} for event in result)
    assert all(set(event['extras']) == {'core'} for event in result)


def test_temporal_markers_sort_only_eligible_scores(monkeypatch):
    rows = [[('core', .9), ('support', .7)] + [(f'weak_{i}', .2) for i in range(100)]] * 4
    sorted_sizes = []
    def observe_sort(values, **kwargs):
        values = list(values)
        if values and isinstance(values[0], tuple):
            sorted_sizes.append(len(values))
        return builtins.sorted(values, **kwargs)
    monkeypatch.setattr(temporal, 'sorted', observe_sort, raising=False)
    markers, _ = temporal.build_markers(app, frames(rows), frozenset(), threshold=.8,
                                       collect_extras=False)
    assert sorted_sizes == [2] * 4
    assert [marker.tag for marker in markers] == ['core']


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('cap', [0, 1, 2])
def test_wide_cache_preserves_observations_ties_exclusions_and_empty_timestamps(monkeypatch, enabled, cap):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, enabled=enabled))
    monkeypatch.setattr(app, 'MARKER_MAX_TAGS_PER_FRAME', cap)
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {'blocked'})
    rows = [[('a', .95), ('c', .9), ('b', .9), ('support', .7), ('blocked', 1.)], [],
            [('b', .9), ('detail', .9)]]
    core_policies, tag_policies = policies({'a', 'detail'}), policies({'c'})
    lean = {source: [segment(source, rows)] for source in ('wd', 'dtq')}
    wide = deepcopy(lean)
    for items in wide.values():
        for row in items[0]['items']:
            row['tags'] += [(f'weak_{i}', .2) for i in range(100)]
    before = deepcopy(wide)
    expected = fusion.observations(app, group(lean), core_policies, tag_policies)
    assert fusion.observations(app, group(wide), core_policies, tag_policies) == expected
    assert wide == before
    assert expected[1] == dict(start=1, end=2, cores={}, extras={})
    if cap == 1:
        assert set(expected[0]['cores']) == {'b'}
        assert set(expected[0]['extras']) == {'a'}


def test_filter_uses_effective_per_source_threshold_and_preserves_exact_boundaries():
    floor = temporal.marker_floor(app.TEMPORAL_POLICY, .8)
    rows = [[('below', math.nextafter(floor, -math.inf)), ('edge', floor),
             ('above', math.nextafter(floor, math.inf)), ('high', .8),
             ('under_high', math.nextafter(.8, -math.inf))]]
    wd, dtq = segment('wd', rows), segment('dtq', [[('dtq_core', .5), ('dtq_support', .425)]])
    dtq[thresholds.OVERRIDE] = .5
    event = fusion.observations(app, group({'wd': [wd], 'dtq': [dtq]}), policies())[0]
    assert set(event['cores']) == {'edge', 'above', 'high', 'under_high', 'dtq_core', 'dtq_support'}
    assert set(event['extras']) == {'high', 'dtq_core'}
    assert not event['cores']['edge']['wd']['strong']
    assert event['cores']['high']['wd']['strong']


def test_zero_threshold_keeps_zero_score_observations():
    saved = segment('wd', [[('zero', 0.)]])
    saved['meta']['marker_threshold'] = 0.
    event = fusion.observations(app, group({'wd': [saved]}), policies())[0]
    assert set(event['cores']) == set(event['extras']) == {'zero'}


@pytest.mark.parametrize('label', ['', ' bad', 'category:bad', 'marker:stash-taggers', None])
@pytest.mark.parametrize('shared', [False, True])
def test_low_score_invalid_labels_still_fail_merge_validation(label, shared, monkeypatch):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY,
        shared_marker_frames=shared, shared_marker_extras=shared))
    records = {'wd': [segment('wd', [[('core', .9), (label, .01)]] * 2)]}
    with pytest.raises(ValueError, match='Invalid or reserved merged label'):
        merge.marker_contributions(app, records, policies())


def test_direct_observation_validation_checks_low_labels_but_keeps_exclusion_semantics(monkeypatch):
    records = {'wd': [segment('wd', [[('core', .9), (' bad', .01)]])]}
    with pytest.raises(ValueError, match='Invalid or reserved merged label'):
        fusion.observations(app, group(records), policies())
    monkeypatch.setattr(app, 'MARKER_TAG_BLOCKLIST', {' bad'})
    assert set(fusion.observations(app, group(records), policies())[0]['cores']) == {'core'}


@pytest.mark.parametrize('ratio', [.5, .85, 1.])
@pytest.mark.parametrize('cap', [0, 1])
@pytest.mark.parametrize('collect_extras', [False, True])
def test_temporal_markers_keep_weak_continuations_gaps_extras_and_proofs(monkeypatch, ratio, cap, collect_extras):
    monkeypatch.setattr(app, 'TEMPORAL_POLICY', replace(app.TEMPORAL_POLICY, marker_continue_ratio=ratio))
    monkeypatch.setattr(app, 'MARKER_MAX_TAGS_PER_FRAME', cap)
    monkeypatch.setattr(app, 'MARKER_MERGE_GAP_SECONDS', 1.)
    rows = [[('core', .9)], [('core', .7)], [('detail', .9)],
            [('core', .9), ('detail', .9)], [('core', .7)]]
    lean = frames(rows)
    wide = frames([row + [(f'weak_{i}', .01) for i in range(100)] for row in rows])
    before = deepcopy(wide)
    options = dict(threshold=.8, identity='input', collect_extras=collect_extras)
    expected = temporal.build_markers(app, lean, frozenset({'detail'}), **options)
    assert temporal.build_markers(app, wide, frozenset({'detail'}), **options) == expected
    assert wide == before
    if ratio < 1:
        assert expected[0]
        assert (expected[0][0].start, expected[0][0].end) == (0, 5)
        assert expected[0][0].hits == 4
        assert expected[0][0].strong_hits == 2
    else:
        # Without weaker continuation, the two strong hits form separate runs.
        assert expected == ([], 0)


def test_consensus_at_support_floor_survives_wider_retention():
    records = {source: [segment(source, [[('core', .7)] +
                [(f'weak_{i}', .2) for i in range(100)]] * 3)] for source in ('wd', 'dtq')}
    markers = merge.union_intervals(merge.marker_contributions(app, records, policies()))
    assert len(markers) == 1
    assert markers[0]['tag'] == 'core'
    assert (markers[0]['start'], markers[0]['end']) == (0, 3)
    assert markers[0]['sources'] == ['wd', 'dtq']

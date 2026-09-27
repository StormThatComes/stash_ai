"""Marker-policy replay must never treat unretained detections as absence."""
from copy import deepcopy
import sqlite3

import pytest

from stash_ai import cli as app, merge
from tests.test_thresholds import configure, raw

pytestmark = pytest.mark.usefixtures('reset_temporal')


def marker_ranges(stash):
    return sorted((m['primary_tag']['name'], m['seconds'], m.get('end_seconds'))
                  for m in stash.scene['scene_markers'])


def stored_outputs(path):
    with sqlite3.connect(path) as db:
        return {table: db.execute(f'SELECT * FROM {table} ORDER BY 1,2,3').fetchall()
                for table in ('results', 'snapshots')}


def offline(monkeypatch, stash):
    def unexpected(*args, **kwargs):
        pytest.fail('accessed inference inputs')
    for name in ('load_tagger', 'load_vocabulary', 'fetch_cues'):
        monkeypatch.setattr(app, name, unexpected)
    monkeypatch.setattr(stash, 'get_media', unexpected)


@pytest.mark.parametrize('markers_only', [False, True])
@pytest.mark.parametrize('dry_run', [False, True])
def test_lower_ratio_rejects_incomplete_cache_before_publication(
        pipeline, monkeypatch, tmp_path, capsys, markers_only, dry_run):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = 0
    p.scores['wd'][:2, 0] = .8
    p.scores['wd'][2:6, 0] = .55
    p.scores['wd'][6:8, 0] = .8
    args = [*p.args, *(['--markers-only'] if markers_only else [])]
    assert app.main(args) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 2), ('ai:common', 6, 8)]
    policy = tmp_path/'relaxed.yml'
    policy.write_text('version: 1\nmarker_continue_ratio: 0.7\n')
    before = stored_outputs(p.path), deepcopy(p.stash.scene)
    database_bytes = p.path.read_bytes()
    p.stash.writes.clear()
    p.stash.dry_run = dry_run
    capsys.readouterr()
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        assert app.main([*args, '--merge-only', '--temporal-policy', str(policy),
                         *(['--dry-run'] if dry_run else [])]) == 1
    assert (stored_outputs(p.path), p.stash.scene) == before
    if dry_run:
        assert p.path.read_bytes() == database_bytes
    assert not p.stash.writes
    message = capsys.readouterr().out
    assert 'wd' in message and 'cache floor' in message and 'run inference' in message
    # A normal run refreshes the incomplete evidence and connects the range.
    monkeypatch.setattr(app, 'DRY_RUN', False)
    p.stash.dry_run = False
    assert app.main([*args, '--temporal-policy', str(policy)]) == 0
    assert marker_ranges(p.stash) == [('ai:common', 0, 8)]


@pytest.mark.parametrize('marker,ratio,expected', [
    (.7, 1.0, [('ai:common', 0, 2), ('ai:common', 6, 8)]),
    (.9, .7, [('ai:common', 0, 8)]),
])
def test_sufficient_cache_matches_fresh_inference_with_effective_threshold(
        pipeline, monkeypatch, tmp_path, marker, ratio, expected):
    p = pipeline
    configure(monkeypatch)
    p.scores['wd'][:] = 0
    p.scores['wd'][:2, 0] = .95
    p.scores['wd'][2:6, 0] = .64
    p.scores['wd'][6:8, 0] = .95
    args = [*p.args, '--markers-only']
    assert app.main(args) == 0
    configure(monkeypatch, marker=marker)
    assert app.main([*args, '--merge-only', '--rethreshold-markers']) == 0
    previous = marker_ranges(p.stash)
    observations = raw(p.path)
    policy = tmp_path/'changed.yml'
    policy.write_text(f'version: 1\nmarker_continue_ratio: {ratio}\n')
    with monkeypatch.context() as blocked:
        offline(blocked, p.stash)
        assert app.main([*args, '--merge-only', '--temporal-policy', str(policy)]) == 0
        assert marker_ranges(p.stash) == expected
        assert raw(p.path) == observations
        assert app.main([*args, '--merge-only']) == 0
        assert marker_ranges(p.stash) == previous
    p.stash.scene['scene_markers'] = []
    assert app.main([*args, '--state-db', str(tmp_path/'fresh.sqlite3'), '--restart',
                     '--temporal-policy', str(policy)]) == 0
    assert marker_ranges(p.stash) == expected


def test_scene_tags_only_does_not_require_marker_cache(pipeline, monkeypatch, tmp_path):
    p = pipeline
    configure(monkeypatch)
    assert app.main(p.args) == 0
    policy = tmp_path/'relaxed.yml'
    policy.write_text('version: 1\nmarker_continue_ratio: 0.7\n')
    before = deepcopy(p.stash.scene)
    offline(monkeypatch, p.stash)
    assert app.main([*p.args, '--merge-only', '--scene-tags-only', '--temporal-policy', str(policy)]) == 0
    assert p.stash.scene == before


def test_legacy_frames_and_imported_ranges_remain_usable():
    from tests.helpers.evidence import segment, policies
    records = {'wd': [segment('wd', [[('core', .9)]] * 2)], 'joytag': [
        dict(mode='spans', meta={'origin': 'legacy'}, items=[
            dict(tag='legacy', start=3, end=5, extra_tags=[])])]}
    markers = merge.union_intervals(merge.marker_contributions(app, records, policies()))
    assert [(m['tag'], m['start'], m['end']) for m in markers] == [('core', 0, 2), ('legacy', 3, 5)]


@pytest.mark.parametrize('source', ['wd', 'dtq'])
def test_incomplete_history_cannot_contribute_consensus_votes(source):
    from tests.helpers.evidence import segment, policies
    records = {key: [segment(key, [[('core', .7)]] * 3)] for key in ('wd', 'dtq')}
    for segments in records.values():
        segments[0]['meta']['marker_score_floor'] = .6
    incomplete = deepcopy(records[source][0])
    incomplete['meta']['marker_score_floor'] = .75
    records[source].insert(0, incomplete)
    with pytest.raises(ValueError, match=source + ': marker cache floor.*run inference'):
        merge.marker_contributions(app, records, policies())

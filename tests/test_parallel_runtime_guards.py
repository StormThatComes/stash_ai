"""Spawned replay respects caller settings and the parent shares the RAM budget."""
from dataclasses import replace

import pytest

from tests.helpers.runtime import runtime as app
from tests.helpers.evidence import inference_record, segment, empty_policies
from stash_ai.storage.store import StateStore
from stash_ai.reports.model.report import build_report
from stash_ai.reports.model.workers import build_report_parallel

pytestmark = pytest.mark.usefixtures('reset_temporal')


@pytest.mark.parametrize('settings', [
    {'MARKER_MIN_HITS': 3},
    {'MAX_TAGS': 1, 'MAX_MARKERS_PER_SCENE': 1, 'MARKER_MAX_TAGS_PER_FRAME': 1},
    {'MARKER_MERGE_GAP_SECONDS': 2.0},
    {'APPLY_SCENE_TAGS': False, 'CREATE_MARKERS': False},
])
def test_spawned_model_report_matches_serial_with_custom_settings(tmp_path, settings):
    runtime = replace(app, TEMPORAL_POLICY=replace(app.TEMPORAL_POLICY,
                      occurrence_markers=False, shared_scene_tags=False, shared_marker_frames=False), **settings)
    path = tmp_path/'state.db'
    policies = empty_policies()
    marker = segment('wd', [[('a', .95), ('b', .9)]]*2 + [[]] + [[('a', .95), ('b', .9)]]*2)
    scene = inference_record(mode='scene', items=[{'tag': 'a', 'score': .95}, {'tag': 'b', 'score': .9}])
    with StateStore(path, 's') as state:
        for identifier in ('1', '2'):
            state.put(identifier, 'wd', {'scene': scene, 'markers': marker})
    with StateStore(path, 's', True) as state:
        serial = build_report(runtime, state, ['1', '2'], policies, policies)
    before = path.read_bytes()
    parallel = build_report_parallel(runtime, path, 's', ['1', '2'], policies, policies,
                                     workers=2, cache_bytes=0, memory_budget_bytes=2**30)
    assert parallel == serial
    assert path.read_bytes() == before
    if settings == {'MARKER_MIN_HITS': 3}:
        assert serial['totals']['markers'] == 0

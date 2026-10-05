"""Current evidence cannot fall through to historical qualification or coverage."""
from copy import deepcopy
import pytest

from stash_ai.evidence.formats import ANALYSIS_FORMAT, validate_current
from stash_ai.evidence.qualification import scene_qualification
from stash_ai.evidence.temporal.policy import TemporalPolicy
from stash_ai.evidence.temporal.extras import merged_extras
from stash_ai.storage.store import StateStore


def record():
    return dict(mode='scene', meta=dict(origin='inference', analysis_format=ANALYSIS_FORMAT,
        inference_signature='run', frame_input_signature='input', scene_frames_format=1,
        scene_score_floor=.1, scene_support_floor=.34, marker_score_floor=.1, scene_threshold=.4, marker_threshold=.4), frames=[],
        items=[dict(tag='core', score=.95, statistics=None)])


@pytest.mark.parametrize('field', ['inference_signature', 'frame_input_signature'])
def test_current_evidence_requires_provenance_before_storage(tmp_path, field):
    saved = record()
    saved['meta'].pop(field)
    with StateStore(tmp_path/'state.sqlite3', 'scope') as state:
        with pytest.raises(ValueError, match='requires both input and inference signatures'):
            state.put('1', 'wd', {'scene': saved})
        assert not state.records('1', 'scene')


def test_current_missing_statistics_never_uses_aggregate_fallback():
    saved = record()
    item = saved['items'][0]
    validate_current(saved)
    assert scene_qualification(item, saved['meta'], .4, TemporalPolicy()) is None
    old = {**saved['meta'], 'analysis_format': 3}
    assert scene_qualification(item, old, .4, TemporalPolicy()) == 'aggregate'
    imported = dict(tag='core', score=None)
    assert scene_qualification(imported, {'origin': 'legacy'}, .4, TemporalPolicy()) == 'aggregate'


def test_current_unfiltered_coverage_needs_no_retired_sentinel():
    saved = record()
    saved.update(mode='frames', items=[])
    validate_current(saved)
    saved['meta']['marker_tag_blocklist'] = []
    with pytest.raises(ValueError, match='retired marker blocklist'):
        validate_current(saved)
    saved['meta'].update(analysis_format=3)
    saved['meta'].pop('marker_tag_blocklist')
    with pytest.raises(ValueError, match='Obsolete active inference'):
        validate_current(saved)


def test_current_plain_extras_are_not_imported_evidence():
    contribution = dict(tag='core', meta=record()['meta'], extra_tags=['unproved'])
    component = dict(contributors=[contribution], start=0, end=3, extra_tags=['unproved'])
    with pytest.raises(ValueError, match='temporal extra_support'):
        merged_extras(component)
    contribution['meta'] = {'origin': 'legacy'}
    merged_extras(component)
    assert component['extra_tags'] == ['unproved']

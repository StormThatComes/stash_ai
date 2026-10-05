"""New writes reject incomplete evidence before changing authoritative state."""
from copy import deepcopy

import pytest

from stash_ai.storage.store import StateStore
from tests.test_result_thresholds import outputs


@pytest.mark.parametrize('field,value', [
    ('mode', None), ('mode', 'frames'), ('meta', None), ('meta', []),
    ('items', None), ('items', {}),
])
@pytest.mark.parametrize('dry_run', [False, True])
def test_incomplete_replacement_preserves_evidence_policy_metadata_and_index(tmp_path, field, value, dry_run):
    path = tmp_path / 'state.sqlite3'
    initial = outputs()['scene']
    initial['scene_threshold_override'] = .8
    with StateStore(path, 'scope') as store:
        store.put('1', 'wd', {'scene': initial})
    before_file = path.read_bytes()
    with StateStore(path, 'scope', dry_run=dry_run) as store:
        before = store.records('1', 'scene')
        tables = ('results', 'result_thresholds', 'result_metadata', 'result_index')
        def rows():
            return [store.connection.execute('SELECT * FROM ' + store.table(table)).fetchall()
                    for table in tables]
        original = rows()
        invalid = deepcopy(initial)
        if value is None:
            invalid.pop(field)
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match='Incomplete result record'):
            store.replace_record('1', 'wd', 'scene', invalid)
        assert rows() == original
        assert store.records('1', 'scene') == before
    if dry_run:
        assert path.read_bytes() == before_file


def test_zero_detection_is_current_inference_and_imported_ranges_are_rejected(tmp_path):
    with StateStore(tmp_path / 'state.sqlite3', 'scope') as store:
        empty = outputs()['scene']
        empty['items'] = empty['frames'] = []
        imported = dict(mode='spans', meta={'origin': 'legacy'}, items=[
            dict(tag='core', start=1., end=None, score=None, hits=None)])
        store.put('1', 'wd', {'scene': empty})
        with pytest.raises(ValueError, match='Incomplete result record'):
            store.put('1', 'wd', {'markers': imported})
        assert store.records('1', 'scene')['wd']['items'] == []
        assert store.records('1', 'markers') == {}
        assert store.connection.execute('SELECT COUNT(*) FROM result_metadata').fetchone()[0] == 1

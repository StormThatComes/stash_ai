from copy import deepcopy
import json
import pickle

import pytest

from stash_ai.storage.segments import StoredSegment, FrozenList, segment_digest
from stash_ai.evidence.formats import validate_current
from stash_ai.evidence.eligibility import FrameMemo, selections, eligible
from stash_ai.runtime.cache import RamCache
from tests.helpers.evidence import segment


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(extra=True),
    lambda s: s.pop('meta'),
    lambda s: s.clear(),
    lambda s: s.setdefault('extra', 1),
    lambda s: s.__ior__({'extra': 1}),
    lambda s: s.__init__({}),
    lambda s: s['meta'].__setitem__('marker_threshold', .2),
    lambda s: s['items'].append({}),
    lambda s: s['items'].sort(),
    lambda s: s['items'].__imul__(2),
    lambda s: s['items'].__iadd__([{}]),
    lambda s: s['items'].__init__([]),
    lambda s: s['items'][0]['tags'][0].__setitem__(1, .1),
    lambda s: s['items'][0]['tags'].reverse(),
    lambda s: s['items'][0].__delitem__('start'),
])
def test_nested_borrow_cannot_mutate_digest(mutate):
    value = StoredSegment(json.loads(json.dumps(segment('wd', [[('core', .9)]]))))
    before = segment_digest(value)
    with pytest.raises(TypeError, match='immutable'):
        mutate(value)
    assert segment_digest(value) == before
    assert json.loads(json.dumps(value)) == segment('wd', [[('core', .9)]]) | {'items': [{'start': 0, 'end': 1, 'tags': [['core', .9]]}]}


def test_detached_copy_and_spawn_pickle_keep_ownership():
    value = StoredSegment(segment('wd', [[('core', .9)]]))
    value._digest = segment_digest(value)
    public = deepcopy(value)
    public['meta']['marker_threshold'] = .1
    public['items'][0]['tags'].append(('other', .5))
    assert value['meta']['marker_threshold'] == .8
    assert len(value['items'][0]['tags']) == 1
    restored = pickle.loads(pickle.dumps(value))
    assert restored == value and isinstance(restored, StoredSegment)
    with pytest.raises(TypeError):
        restored['items'][0]['tags'].clear()


def test_only_immutable_successful_validation_is_memoized(monkeypatch):
    import stash_ai.evidence.formats as formats
    validate = formats._validate_current
    calls = []
    def counted(record):
        calls.append(record)
        return validate(record)
    monkeypatch.setattr(formats, '_validate_current', counted)
    value = StoredSegment(segment('wd', [[('core', .9)]]))
    validate_current(value)
    validate_current(value)
    assert len(calls) == 1
    mutable = deepcopy(value)
    validate_current(mutable)
    mutable['items'][0]['tags'].append(('core', .5))
    with pytest.raises(ValueError):
        validate_current(mutable)
    assert len(calls) == 3
    bad = StoredSegment(mutable)
    for _ in range(2):
        with pytest.raises(ValueError):
            validate_current(bad)
    assert len(calls) == 5


def test_shared_order_preserves_thresholds_caps_sources_and_exclusions():
    values = FrozenList([['z', .8], ['a', .8], ['low', .699999988079071], ['high', .9]])
    cache = RamCache(10000)
    memo = FrameMemo(cache)
    for source in ('wd', 'dtq'):
        for high, floor, cores, exclusions, limit in (
            (.8, .7, {'high'}, (), 1), (.7, .6, (), {'a'}, 2), (.9, .8, (), (), 0)):
            assert selections(values, high, floor, cores, exclusions, limit, validate=True,
                              memo=memo, source=source) == selections(deepcopy(values), high, floor,
                                                                    cores, exclusions, limit, validate=True)
    assert memo.sorts == 2
    memo.close()
    assert cache.size == 0


def test_frame_memo_is_bounded_and_cleans_evicted_keys():
    cache = RamCache(4000)
    memo = FrameMemo(cache)
    for index in range(500):
        eligible(FrozenList([['tag', .9]]), .8, memo=memo, source=str(index))
    assert cache.peak <= cache.limit
    assert not hasattr(memo, 'keys')
    memo.close()
    assert cache.size == 0

from types import SimpleNamespace
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor

import pytest

from stash_ai.recognition.retained_memory import Budget, scope
from stash_ai.recognition.resident_models import Pool
from stash_ai.recognition.sprite_cache import SpriteCache
from stash_ai.recognition.profiling import pipeline_scope, pipeline_snapshot


def config(limit=1024, cache=256):
    return SimpleNamespace(data={'execution': {'retained_memory_bytes': limit,
                                               'model_bytes_cache_bytes': cache}})


def test_shared_admission_is_atomic_and_releases_owners():
    budget = Budget(100)
    barrier = Barrier(2)
    def take(owner):
        barrier.wait(timeout=5)
        return budget.replace(owner, 60)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(take, ['a', 'b']))
    assert sum(results) == 1 and budget.used == 60
    for owner in ['a', 'b']:
        budget.replace(owner, 0)
    assert budget.used == 0


def test_models_and_sprites_share_capacity_without_cross_owner_deadlocks():
    class Model:
        def resident_modules(self):return [self]
        def parameters(self):return [self]
        def buffers(self):return []
        def numel(self):return 400
        def element_size(self):return 1
        def to(self,device):self.device=device
    with pipeline_scope(True), scope(config()) as budget:
        sprites = SpriteCache(4096)
        sprites.remember('a', b'x'*400)  # 912 including cache overhead.
        models = Pool(4096)
        models.put('model', Model(), 'cuda')
        assert models.used == 0 and budget.used == 912
        sprites.close()
        models.put('model', Model(), 'cuda')
        assert models.used == budget.used == 400
        restored = models.take('model', None, 'cuda')
        assert restored.device == 'cuda' and budget.used == 0
        assert pipeline_snapshot()['counts']['retained_memory_peak_bytes'] <= 1024


def test_verified_model_bytes_are_immutable_bounded_and_counted():
    loads = []
    with pipeline_scope(True), scope(config()) as budget:
        def load():loads.append(1);return b'x'*200
        a = budget.verified('a', load)
        assert budget.verified('a', load) is a
        budget.verified('b', load)
        assert list(budget.models) == ['b'] and budget.used == 200
        assert len(loads) == 2
        assert pipeline_snapshot()['counts']['verified_model_bytes_hits'] == 1
    assert budget.used == 0


def test_failed_model_restore_releases_shared_reservation():
    class Model:
        def resident_modules(self):return [self]
        def parameters(self):return [self]
        def buffers(self):return []
        def numel(self):return 100
        def element_size(self):return 1
        def to(self,device):
            if device == 'cuda':raise RuntimeError('restore failed')
    with scope(config()) as budget:
        models = Pool(1024)
        models.put('a', Model(), 'cuda')
        with pytest.raises(RuntimeError, match='restore failed'):
            models.take('a', None, 'cuda')
        assert models.used == budget.used == 0


def test_sam_residency_discards_image_features():
    from stash_ai.recognition.sam_pose2seg import Segmenter
    reset = []
    predictor = SimpleNamespace(model=object(), reset_predictor=lambda: reset.append(True))
    segmenter = Segmenter(predictor, 'cuda')
    assert segmenter.resident_modules() == [predictor.model]
    segmenter.prepare_residency()
    assert reset == [True]

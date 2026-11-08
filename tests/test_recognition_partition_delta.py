import random
import pytest

from stash_ai.recognition.replay_simulation import impact
from stash_ai.recognition.partition_delta import PartitionImpact, Predictions


@pytest.mark.parametrize('fraction', [0, .01, .1, 1])
def test_delta_metrics_match_full_partition(fraction):
    rng = random.Random(14871)
    people = [None, 'a', 'b', 'c']
    retained = {i: rng.choice(people) for i in range(401)}
    baseline = {i: rng.choice(people) for i in range(400)}
    changes = {i: rng.choice(people + ['new']) for i in baseline if rng.random() < fraction}
    after = Predictions(baseline, changes)
    for excluded in (set(), {0, 1, 50}):
        expected = impact({k:v for k,v in retained.items() if k not in excluded},
                          {k:v for k,v in after.items() if k not in excluded})
        assert PartitionImpact(retained, baseline, excluded).measure(changes) == expected

"""Canonical digests must preserve persisted inference/publication identities."""
import pytest

from stash_ai.signatures import digest


@pytest.mark.parametrize('value,expected', [
    ({}, '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'),
    ({'żółć': ['猫', None, True], 'float': .1234, 'zero': -0.0},
     '8d9123915477c1b3a0b1839c661f85d177c72fb4a40c5f9fc66b274f261d81ae'),
    ([1, '1', True, None, 0.0],
     '1bceef4da6288158cf10a2b6457cfc69f4131446eacdd1663b7d13fea5ac8af1'),
])
def test_digest_matches_both_previous_implementations(value, expected):
    # Captured from progress.digest and performance.digest before consolidation.
    assert digest(value) == expected
    if isinstance(value, dict):
        assert digest(dict(reversed(list(value.items())))) == expected


@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_digest_rejects_nonfinite_values_in_nested_evidence(value):
    with pytest.raises(ValueError):
        digest({'frames': [dict(score=value)]})

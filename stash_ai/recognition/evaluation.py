"""Developer-only local face-pair evaluation; no public stage command."""
import math


def evaluate_pairs(pairs, threshold):
    """Pairs contain normalized embeddings and explicit same-person ground truth.

    This measures the supplied dataset only. It must never turn detector scores
    or synthetic-vector checks into an accuracy claim about user video media.
    """
    import numpy as np
    from .models import normalized_embedding
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Face evaluation threshold must be a finite cosine value in [0,1].')
    counts = dict(same=0, different=0, false_match=0, false_miss=0)
    for first, second, same in pairs:
        if type(same) is not bool:
            raise ValueError('Face pair ground truth must be explicit boolean.')
        a = normalized_embedding(first, len(first))
        b = normalized_embedding(second, len(first))
        match = float(np.dot(a, b)) >= threshold
        counts['same' if same else 'different'] += 1
        counts['false_miss' if same else 'false_match'] += int(match != same)
    return {'scope': 'supplied-local-pairs-only', 'threshold': threshold,
            'samples': counts['same'] + counts['different'], **counts,
            'default_policy': 'conservative-v1-not-domain-calibrated'}

"""Route current temporal qualification and explicit historical aggregate replay."""
from stash_ai.evidence.formats import current
from stash_ai.evidence.temporal.statistics import scene_reason
from stash_ai.evidence.historical_evidence import aggregate_reason


def scene_qualification(entry, meta, threshold, policy):
    statistics = entry.get('statistics')
    if statistics is None and not current(meta):
        return aggregate_reason(entry['score'], threshold)
    return scene_reason(entry['score'], statistics, threshold, policy)

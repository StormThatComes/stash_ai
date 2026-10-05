"""Scene qualification uses the complete temporal statistics contract."""
from stash_ai.evidence.temporal.statistics import scene_reason


def scene_qualification(entry, meta, threshold, policy):
    return scene_reason(entry['score'], entry['statistics'], threshold, policy)

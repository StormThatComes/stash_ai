"""Select retained scene labels and qualify temporal aggregate candidates."""
from stash_ai.evidence.temporal.statistics import scene_reason


def retained_scores(scores, labels, scene_frames):
    """Preserve aggregate scores for every label witnessed in retained frames."""
    candidates = {tag for row in scene_frames for tag, _ in row['tags']}
    predictions = [(tag, float(scores[i])) for i, tag in enumerate(labels)
                   if tag in candidates]
    predictions.sort(key=lambda item: (-item[1], item[0]))
    return predictions


def temporal_scores(scores, labels, indexes, statistics, threshold, policy,
                    *, raw_scene, exclusions):
    """Raw evidence keeps candidates; direct callers apply qualification."""
    predictions = [(labels[index], float(scores[index])) for index in indexes]
    predictions = [(tag, score) for tag, score in predictions
                   if raw_scene or (tag not in exclusions and
                                    scene_reason(score, statistics[tag], threshold, policy))]
    predictions.sort(key=lambda item: (-item[1], item[0]))
    return predictions

"""Select raw scene candidates retained by inference."""


def retained_scores(scores, labels, scene_frames):
    """Preserve aggregate scores for every label witnessed in retained frames."""
    candidates = {tag for row in scene_frames for tag, _ in row['tags']}
    predictions = [(tag, float(scores[i])) for i, tag in enumerate(labels)
                   if tag in candidates]
    predictions.sort(key=lambda item: (-item[1], item[0]))
    return predictions


def candidate_scores(scores, labels, indexes):
    """Keep aggregate candidates without applying publication policy."""
    predictions = [(labels[index], float(scores[index])) for index in indexes]
    predictions.sort(key=lambda item: (-item[1], item[0]))
    return predictions

"""Public checked prediction boundary for native and injected backend adapters."""
from stash_ai import models


def checked_prediction(tagger, inputs, count, *, collated):
    from stash_ai.backends.loader import JoyTagger
    from stash_ai.backends.onnx import OnnxTagger
    scores = (tagger.predict_collated_batch(inputs, count) if collated
              else tagger.predict_batch(inputs))
    # Exact native adapters validate their public result. Subclasses/external
    # adapters can override that method, so always verify their output here.
    if type(tagger) in (JoyTagger, OnnxTagger):
        return scores
    return models.validate_scores(scores, count, tagger.labels, tagger.settings.source.title)

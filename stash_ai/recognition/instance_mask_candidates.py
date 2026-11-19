"""Bounded instance selection after the shared mask cleanup and admission gates."""
import numpy as np

from .mask_cleanup import inside


def select_instances(source, assess, positive, negative, *, limit=3):
    """Inspect every proposal, retaining only the best few full-frame masks.

    `assess` supplies the same cleanup, geometry and pose-ownership checks used
    for prompted masks. Stable model query IDs survive ranking and pair choice
    so retained-mask replay does not need the other people's geometry.
    """
    retained = []
    count = 0
    for instance, variant in source.iter_instances():
        count += 1
        rank, mask, box, reason, edits = assess(variant)
        if reason is None:
            # Even a zero coverage threshold cannot associate an unrelated
            # detector instance. Never join instances to satisfy pose prompts.
            if not inside(mask, positive).any():
                reason = 'missing_prompt_support'
            elif inside(mask, negative).any():
                reason = 'negative_prompt_in_mask'
        rank = (reason is None, *rank[1:])
        assessment = (rank, mask, box, reason, dict(edits, mask_instance=int(instance)))
        retained.append((variant, assessment))
        retained.sort(key=lambda item: item[1][0], reverse=True)
        del retained[limit:]
    if not retained:
        variant = (np.zeros(source.size[::-1], dtype=bool), 0.)
        rank, mask, box, _, edits = assess(variant)
        retained.append((variant, (rank, mask, box, 'no_person_instances', edits)))
    return [item[0] for item in retained], [item[1] for item in retained], count

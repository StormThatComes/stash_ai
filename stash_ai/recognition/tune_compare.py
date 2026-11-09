"""Sequential comparison on one original cue. Each panel can succeed independently."""
import os

from stash_ai.stash_api.transport import safe_error
from .detection_session import same_settings


def detect_variants(detector, frame, plans, reusable, stop):
    """Reuse a matching result from this exact frame, including an identical A/B pair."""
    results = {}
    for name, config in plans.items():
        if stop.is_set():
            break
        for previous, output in reusable:
            if same_settings(previous, config):
                results[name] = output
                break
        else:
            try:
                output = detector.render(frame, config)
            except Exception as error:
                # Release exception tracebacks (and their model tensors) on this thread.
                results[name] = safe_error(error, os.getenv('STASH_API_KEY', ''))
            else:
                reusable.append((config, output))
                results[name] = output
    return results

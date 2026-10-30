"""Optional search hints have their own bounded budget, never replay authority."""
from dataclasses import replace
from .config import plain
from .resource_errors import ResourceLimit

LIMITS = dict(max_candidate_pairs=4096, max_witness_comparisons=1000000, plan_bytes=16 * 1024**2)

def settings(config):
    data = plain(config.data)
    # These are diagnostic ceilings, independent of production acceptance.
    data['sapiensid']['execution'].update(LIMITS)
    return replace(config, data=data)


def measure(function, fallback, coverage):
    try:
        result = function()
    except ResourceLimit as error:
        coverage.update(complete=False, reason='diagnostic_resource_limit', detail=str(error))
        return fallback
    coverage.update(complete=True)
    return result

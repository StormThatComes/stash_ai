"""Stage-specific controls scoped to one explicitly requested private replay."""
from contextlib import contextmanager
from contextvars import ContextVar

STAGES = {
    'hybrid': 'Hybrid merge and query stages',
    'hybrid_merge': 'Hybrid identity consolidation',
    'hybrid_query': 'Hybrid query attachment',
    'confirmed_views': 'Confirmed View passes (global, anchor, auxiliary)',
    'global_view_merge': 'Confirmed View consolidation',
    'global_whole_view': 'Global whole-View attachment',
    'global_view_query': 'Global View query attachment',
    'strict_partition': 'Strict core consolidation',
    'anchor_view_query': 'Scene anchor View attachment',
    'auxiliary_view_query': 'Auxiliary View attachment',
}

_GROUPS = {
    'hybrid': ('hybrid_merge', 'hybrid_query'),
    'confirmed_views': ('global_view_merge', 'global_whole_view', 'global_view_query',
                        'anchor_view_query', 'auxiliary_view_query'),
}
_active = ContextVar('pair_replay_intervention', default=None)


def members(stage):
    return _GROUPS.get(stage, (stage,))


@contextmanager
def trial(stage, counts):
    if stage is not None and stage not in STAGES:
        raise ValueError('Unknown replay intervention.')
    token = _active.set((members(stage), counts))
    try:
        yield
    finally:
        _active.reset(token)


def controlled():
    return _active.get() is not None


def allow(stage):
    value = _active.get()
    if value is None:
        return True
    omitted, counts = value
    counts[stage] = counts.get(stage, 0) + 1
    return stage not in omitted

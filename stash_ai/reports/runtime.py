"""Shared isolated settings and read-only store lifecycle for evidence reports."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from stash_ai.evidence.temporal.policy import load_policy
from stash_ai.storage.store import StateStore, scope_for


def report_context(runtime, args, *, both_outputs=False):
    settings = dict(TEMPORAL_POLICY=load_policy(args.temporal_policy))
    if both_outputs:
        settings.update(APPLY_SCENE_TAGS=True, CREATE_MARKERS=True)
    return replace(runtime, **settings)


def report_policies(runtime, args):
    return (
        runtime.services.load_exclusion_policy(args.marker_core_exclusions, 'excluded_core_tags'),
        runtime.services.load_exclusion_policy(args.scene_tag_exclusions, 'excluded_scene_tags'),
    )


def report_memory(args):
    return dict(cache_bytes=int(args.ram_cache_gb * 2**30),
                memory_budget_bytes=int(args.ram_budget_gb * 2**30))


@contextmanager
def report_store(runtime, args, *, cache_bytes=0, memory_budget_bytes=0):
    require_database(args)
    with StateStore(args.state_db, scope_for(runtime.STASH_URL), True,
                    cache_bytes=cache_bytes, memory_budget_bytes=memory_budget_bytes,
                    allow_cleanup=True) as store:
        yield store


def require_database(args):
    if not Path(args.state_db).expanduser().is_file():
        raise FileNotFoundError(f'State database does not exist: {args.state_db}')

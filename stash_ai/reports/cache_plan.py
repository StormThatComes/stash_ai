"""Advisory cache backfill actions and command batching; never executes commands."""

from stash_ai.reports.cache_assessment import RESTART, REUSABLE


def backfill_action(statuses, checkpoint):
    if statuses <= REUSABLE:
        return 'rethreshold'
    # Gaps and refreshes let inference verify checkpoint compatibility; an
    # orphan checkpoint must not prevent replacing unusable completed evidence.
    restart = bool(statuses & RESTART) and not (
        checkpoint and statuses & {'missing', 'partial', 'refresh_required'})
    return 'restart_inference' if restart else 'resume_or_infer'


def _batches(scenes):
    for start in range(0, len(scenes), 50):
        yield scenes[start:start + 50]


def _argv(command_context, batch, *flags):
    argv = ['python', 'stash_ai.py', *command_context, *flags]
    for scene in batch:
        argv.extend(('--scene-id', scene))
    return argv


def build_plan(models, operations, blocked_scenes, recovery_scenes, command_context=()):
    """Order model backfills before final publication and exclude blocked scenes."""
    plan = []
    for action in ('resume_or_infer', 'restart_inference', 'rethreshold'):
        for model in models:
            targets = [scene for scene in operations.get((action, model), ())
                       if scene not in blocked_scenes]
            for batch in _batches(targets):
                argv = _argv(command_context, batch, '--models', model)
                if action == 'restart_inference':
                    argv.append('--restart')
                elif action == 'rethreshold':
                    argv.extend(('--merge-only', '--rethreshold-scenes', '--rethreshold-markers'))
                else:
                    argv.append('--resume')
                plan.append(dict(action=action, model=model, scene_ids=batch, argv=argv,
                                 uses_inference=action != 'rethreshold'))
    actionable = sorted(({scene for targets in operations.values() for scene in targets}
                         | recovery_scenes) - blocked_scenes)
    # Publish after all backfills, including cross-model votes and Occurrence.
    for batch in _batches(actionable):
        plan.append(dict(action='merge_after_backfill', scene_ids=batch,
                         argv=_argv(command_context, batch, '--merge-only'), uses_inference=False))
    return plan

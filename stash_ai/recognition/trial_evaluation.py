"""One independent production replay; SQLite never escapes its owner."""
from contextlib import closing, nullcontext
from types import SimpleNamespace
from .offline_evaluation import _ram
from .evaluation_scenarios import copy_history
from .body_reports import members as body_members, report as body_report
from .person_signal import coverage


def settings_for_replay(settings, runtime):
    from dataclasses import replace
    from .config import plain
    data = plain(settings.data)
    data['execution'].update(rebuild_checkpoint_scenes=0,
                             profile_rebuild=bool(runtime and runtime.profile))
    if runtime and runtime.inner_workers >= 2:
        data['execution']['rebuild_prefetch_scenes'] = True
    if runtime and runtime.trials > 1:
        data['execution']['rebuild_cohort_backend'] = 'threads'
    return replace(settings, data=data)


def evaluate(name, settings, *, seed, source, scope, namespace, installation, scans,
             observations, rebuild, inspect_trial=None, progress=None, copy_inputs=copy_history):
    from .progress import observe_updates
    from .solver_progress import ReplayProgress
    reporter = ReplayProgress(name, progress) if progress and getattr(progress, 'telemetry_enabled', True) else None
    with observe_updates(reporter.update) if reporter else nullcontext():
        return _evaluate(name, settings, seed=seed, source=source, scope=scope,
            namespace=namespace, installation=installation, scans=scans,
            observations=observations, rebuild=rebuild, inspect_trial=inspect_trial,
            copy_inputs=copy_inputs)


def _evaluate(name, settings, *, seed, source, scope, namespace, installation, scans,
              observations, rebuild, inspect_trial, copy_inputs):
    from .progress import observe_stages, stage
    active_stage = 'Prepare replay'
    def stage_update(label):
        nonlocal active_stage
        active_stage = label
    try:
        with observe_stages(stage_update), closing(_ram(settings)) as target:
            with stage('Prepare replay'):
                if seed is None:
                    target.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
                    copy_inputs(source, target, [], scope, namespace)
                else:
                    seed.backup(target)
                    pages = settings.data['execution']['rebuild_memory_bytes']//target.execute('PRAGMA page_size').fetchone()[0]
                    target.execute(f'PRAGMA max_page_count={pages}')
            state = SimpleNamespace(connection=target, installation=installation)
            with stage('Rebuild selected identities'):
                rebuild(state, [dict(scene=s['scene'], scan=s['id']) for s in scans], settings)
            rebuild_metrics = getattr(state, "last_rebuild_profile", None)
            with stage('Collect trial predictions'):
                predictions = dict.fromkeys(observations)
                predictions.update((r[0], r[1]) for r in target.execute('SELECT observation,person FROM members')
                                   if r[0] in observations)
                predictions.update((k,v) for k,v in body_members(target,settings).items() if k in observations)
                bodies=body_report(target,settings,[s['id'] for s in scans])
                hybrid=coverage(target,settings,[s['id'] for s in scans])
            with stage('Collect solver feedback'):
                evidence = inspect_trial(target, settings) if inspect_trial else None
    except Exception as exc:
        # The last printed stage can lag behind. Preserve the original
        # exception and expose locations, never locals (also during feedback).
        from pathlib import Path
        from traceback import extract_tb, extract_stack
        frames = extract_stack(limit=3)[:-1] + extract_tb(exc.__traceback__, limit=-4)
        location = ' -> '.join(f'{Path(f.filename).name}:{f.lineno} ({f.name})' for f in frames)
        exc.replay_stage = active_stage
        exc.add_note(f'Recognition replay {name}; stage: {active_stage}; location: {location}')
        import sqlite3
        if isinstance(exc, sqlite3.OperationalError) and getattr(exc, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL:
            from .resource_errors import ResourceLimit
            limit = ResourceLimit('Trial replay exceeds execution.rebuild_memory_bytes.')
            limit.replay_stage = active_stage
            limit.add_note(exc.__notes__[-1])
            raise limit from exc
        raise
    return dict(predictions=predictions, independent_person=bodies, hybrid=hybrid,
                decision_evidence=evidence, rebuild_profile=rebuild_metrics)

"""One independent production replay; SQLite never escapes its owner."""
from contextlib import closing
from types import SimpleNamespace
import time
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
    active_stage = 'Prepare replay'
    ns = namespace
    db = source
    with closing(_ram(settings)) as target:
        if seed is None:
            target.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
            copy_inputs(db, target, [], scope, ns)
        else:
            seed.backup(target)
            pages = settings.data['execution']['rebuild_memory_bytes']//target.execute('PRAGMA page_size').fetchone()[0]
            target.execute(f'PRAGMA max_page_count={pages}')
        state = SimpleNamespace(connection=target, installation=installation)
        from .progress import observe_stages
        last_stage = [0.]
        active_stage = 'Rebuild selected identities'
        def stage_update(label):
            nonlocal active_stage
            active_stage = label
            now = time.monotonic()
            if progress and now-last_stage[0] >= 1.:
                last_stage[0] = now
                progress(name, 'stage: '+label, None)
        try:
            with observe_stages(stage_update):
                rebuild(state, [dict(scene=s['scene'], scan=s['id']) for s in scans], settings)
        except Exception as exc:
            # The last printed stage can lag behind. Preserve the
            # original exception and expose locations, never locals.
            from pathlib import Path
            from traceback import extract_tb
            frames = extract_tb(exc.__traceback__, limit=-5)
            location = ' -> '.join(f'{Path(f.filename).name}:{f.lineno} ({f.name})' for f in frames)
            exc.replay_stage = active_stage
            exc.add_note(f'Recognition replay {name}; stage: {active_stage}; location: {location}')
            raise
        rebuild_metrics = getattr(state, "last_rebuild_profile", None)
        predictions = dict.fromkeys(observations)
        predictions.update((r[0], r[1]) for r in target.execute('SELECT observation,person FROM members')
                           if r[0] in observations)
        predictions.update((k,v) for k,v in body_members(target,settings).items() if k in observations)
        bodies=body_report(target,settings,[s['id'] for s in scans])
        hybrid=coverage(target,settings,[s['id'] for s in scans])
        evidence = inspect_trial(target, settings) if inspect_trial else None
    return dict(predictions=predictions, independent_person=bodies, hybrid=hybrid,
                decision_evidence=evidence, rebuild_profile=rebuild_metrics)

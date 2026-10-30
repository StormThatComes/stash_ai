"""Full production retuning on a consistent, read-only retained VTT snapshot."""
from .resource_errors import ResourceLimit
from collections import Counter, defaultdict
from contextlib import closing, nullcontext, contextmanager
from dataclasses import replace
from types import SimpleNamespace
import time
import sqlite3
import tempfile
from contextvars import ContextVar

_prepared_sources = ContextVar('recognition_prepared_sources', default=())

from .config import plain
from .database import FaceDatabase, encoded
from .evaluation_scenarios import copy_history
from .fragmentation_report import _open_readonly
from .models import namespace
from .offline_evaluation import _ram, _selection
from .rebuild import rebuild_selected
from .scanning import scan_identity
from .person_signal import coverage, same_contract


def impact(before, after):
    """Linear partition comparison, independent of UUID spelling; no truth claim."""
    common = before.keys() & after.keys()
    cells, left, right = Counter(), Counter(), Counter()
    for key in common:
        a, b = before[key], after[key]
        if a is not None and b is not None:
            cells[a, b] += 1
            left[a] += 1
            right[b] += 1
    def pairs(counts):
        return sum(n*(n-1)//2 for n in counts.values())
    shared = pairs(cells)
    return dict(common_observations=len(common),
        newly_joined_pairs=pairs(right)-shared, split_pairs=pairs(left)-shared,
        coverage_gained=sum(before[k] is None and after[k] is not None for k in common),
        coverage_lost=sum(before[k] is not None and after[k] is None for k in common),
        assigned=sum(p is not None for p in after.values()),
        identities=len(set(after.values())-{None}))


@contextmanager
def prepared_history(db, config, scope, ns, installation):
    """One disk-backed immutable seed; each experiment still gets fresh RAM."""
    if any(source is db and key == (scope, ns, installation) for source, key in _prepared_sources.get()):
        yield db
        return
    from .database import SCHEMA
    with tempfile.TemporaryDirectory(prefix='recognition-replay-') as directory:
        with closing(sqlite3.connect(directory+'/seed.sqlite3')) as seed:
            seed.execute('PRAGMA cache_size=-2048')
            page_size = seed.execute('PRAGMA page_size').fetchone()[0]
            seed.execute(f"PRAGMA max_page_count={config.data['execution']['replay_snapshot_bytes']//page_size}")
            seed.executescript(SCHEMA)
            seed.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
            try:
                copy_history(db, seed, [], scope, ns)
            except sqlite3.OperationalError as error:
                if getattr(error, 'sqlite_errorcode', None) == sqlite3.SQLITE_FULL:
                    raise ResourceLimit('Retained snapshot exceeds execution.replay_snapshot_bytes.') from error
                raise
            statistics = dict(embedding_bytes=sum(seed.execute(
                f'SELECT COALESCE(SUM(length(embedding)),0) FROM {table}').fetchone()[0]
                for table in ('observations','person_evidence','body_observations')))
            seed.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('replay-input-statistics', encoded(statistics)))
            seed.commit()
            size = seed.execute('PRAGMA page_count').fetchone()[0]*seed.execute('PRAGMA page_size').fetchone()[0]
            if size > config.data['execution']['replay_snapshot_bytes']:
                raise ResourceLimit('Retained snapshot exceeds execution.replay_snapshot_bytes.')
            seed.execute('PRAGMA query_only=ON')
            token = _prepared_sources.set((*_prepared_sources.get(), (seed, (scope, ns, installation))))
            try:
                yield seed
            finally:
                _prepared_sources.reset(token)


def replay(config, variants, **kwargs):
    from .run_cache import scope
    with scope(config):
        return _replay(config,variants,**kwargs)


@contextmanager
def prepared_replay(db, config, scope, ns, installation, reuse_history, progress):
    from contextlib import ExitStack
    from .solver_progress import heartbeat
    with ExitStack() as stack:
        with heartbeat('initialization', progress, 'Prepare immutable replay inputs'):
            seed = stack.enter_context(prepared_history(db, config, scope, ns, installation)) if reuse_history else None
        yield seed


def _replay(config, variants, *, scenes=None, scope=None, source=None, consume=None, variant_stream=None, inspect_trial=None,
            progress=None, allow_partial_selection=False, reuse_history=True, restore_trial=None, record_trial=None,
            prepared_selection=None, trial_error=None, parallel_request=None, parallel_accept=None, journal=None):
    """Stream one trial at a time; retain only baseline and compact summaries.

    Existing bindings/history seed each independent trial. Production rebuild
    recomputes selected assignments; no candidate is seeded from another trial.
    A consumer can reduce predictions without retaining N copies of the corpus.
    """
    ns = namespace(config)
    for name, variant in variants.items():
        if not isinstance(name, str) or not name or name == 'baseline':
            raise ValueError('Simulation names must be nonempty and distinct from baseline.')
        if namespace(variant) != ns or scan_identity(variant, ns) != scan_identity(config, ns) or not same_contract(config,variant):
            raise ValueError('Simulation variants must preserve model, person and VTT scan contracts.')
    with (closing(_open_readonly(config.database)) if source is None else nullcontext(source)) as db:
        inputs = {}
        if prepared_selection is None:
            scope, scans = _selection(db, config, scenes, scope, allow_partial=allow_partial_selection, diagnostics=inputs)
        else:
            scope, scans, inputs = prepared_selection
        installation = FaceDatabase.installation.fget(SimpleNamespace(connection=db))
        observations = {r['id']: r['scene'] for r in db.execute(
            'SELECT o.id,s.scene FROM observations o JOIN scans s ON s.id=o.scan '
            'WHERE o.scan IN (SELECT value FROM json_each(?))', (encoded([s['id'] for s in scans]),))}
        from .body_reports import observations as body_observations, members as body_members, report as body_report
        observations.update(body_observations(db,config,[s['id'] for s in scans]))
        retained = dict.fromkeys(observations)
        retained.update((r[0], r[1]) for r in db.execute('SELECT observation,person FROM members') if r[0] in observations)
        retained.update((k,v) for k,v in body_members(db,config).items() if k in observations)
        result = dict(format=1, kind='recognition_replay_simulation', source_read_only=True,
            source='retained_vtt_embeddings', scope=scope, selection=[s['scene'] for s in scans],
            starting_state='frozen_retained_history_and_bindings', input_selection=inputs, trials={},
            limitations=['Partition changes are not false-merge or accuracy estimates without explicit truth.',
                'Results apply to this retained snapshot and joint selection, not future media or other import orders.',
                'Runtime excludes model inference and media loading; each trial uses fresh RAM databases.'])
        if not inputs['complete_scope']:
            result['limitations'].append('Excluded scenes were not replayed. Recommendations cover only selection; '
                'a run including excluded or recollected scenes requires revalidation.')
        if not scans:
            result['stop_reason'] = 'no_compatible_scans'
            return result
        baseline = None
        wall_started = time.perf_counter()
        from itertools import chain
        stream = variant_stream() if variant_stream is not None else variants.items()
        seen_names = {'baseline'}
        preparation_started = time.perf_counter()
        with prepared_replay(db, config, scope, ns, installation, reuse_history, progress) as seed:
            from .solver_execution import current as solver_execution
            from .solver_pool import Trials
            runtime = solver_execution()
            if runtime and runtime.profile:
                result['performance'] = dict(input_preparation_seconds=time.perf_counter()-preparation_started,
                                             consume_seconds=0.,journal_seconds=0.)
            context = dict(seed_path=seed.execute('PRAGMA database_list').fetchone()[2],
                installation=installation, scans=scans, observations=observations) if parallel_request else None
            manager = Trials(runtime, parallel_request, context, journal, progress) if parallel_request else nullcontext(None)
            with manager as pool:
                yield_entries = pool.entries(stream) if pool else stream
                result['execution'] = runtime.document() if runtime else None
                try:
                    if progress and not inputs['complete_scope']:
                        progress('inputs', f"partial scope: {inputs['selected_scene_count']}/{inputs['latest_scene_count']} "
                                 'compatible complete scenes; excluded scenes will be listed in the report', None)
                    for name, settings in chain([('baseline', config)], yield_entries):
                        if name != 'baseline':
                            if name in seen_names:
                                raise ValueError('Duplicate simulation name.')
                            seen_names.add(name)
                        if namespace(settings) != ns or scan_identity(settings, ns) != scan_identity(config, ns) or not same_contract(config,settings):
                            raise ValueError('Simulation variants must preserve model, person and VTT scan contracts.')
                        from .trial_evaluation import settings_for_replay
                        settings = settings_for_replay(settings, runtime)
                        if progress and (not pool or name == 'baseline'):
                            progress(name, 'start', None)
                        restored = restore_trial(name,settings) if restore_trial else None
                        if restored is not None:
                            item,restored_baseline = restored
                            if pool: pool.ready.pop(name, None)
                            if baseline is None:baseline=restored_baseline
                            if context is not None:context['baseline_predictions']=baseline
                            if item.get('failed'):
                                result.setdefault('failed_trials', {})[name] = item
                                if progress:progress(name, 'failed', item)
                                continue
                            result['trials'][name]=item
                            if progress:progress(name,'complete',item)
                            continue
                        started = time.perf_counter()
                        cpu_started = time.process_time()
                        active_stage = 'Prepare replay'
                        try:
                            if pool and name != 'baseline':
                                evaluated = pool.ready.pop(name)
                                if evaluated.get('failed'):
                                    error = ResourceLimit(evaluated['error'])
                                    error.replay_stage = evaluated['stage']
                                    error.worker_elapsed_seconds = evaluated['elapsed_seconds']
                                    error.trial_profile = evaluated.get('trial_profile')
                                    for key in ('budget','limit','requested'):
                                        setattr(error,key,evaluated.get('resource',{}).get(key))
                                    for note in evaluated.get('diagnostics', ()):
                                        error.add_note(note)
                                    raise error
                                if parallel_accept:
                                    parallel_accept(settings, evaluated)
                            else:
                                from .trial_evaluation import evaluate
                                from .solver_execution import scope as execution_scope
                                baseline_runtime = replace(runtime, trials=1, inner_workers=max(1, runtime.cpus-1)) if runtime and name == 'baseline' else runtime
                                with execution_scope(baseline_runtime):
                                    evaluated = evaluate(name, settings, seed=seed, source=db, scope=scope,
                                        namespace=ns, installation=installation, scans=scans, observations=observations,
                                        rebuild=rebuild_selected, inspect_trial=inspect_trial, progress=progress, copy_inputs=copy_history)
                            if evaluated.get('prediction_encoding') == 'baseline-delta-v1':
                                if baseline is None or not evaluated['prediction_delta'].keys() <= baseline.keys():
                                    raise ValueError('Invalid baseline prediction delta.')
                                predictions = dict(baseline, **evaluated['prediction_delta'])
                            else:
                                predictions = evaluated['predictions']
                            bodies, hybrid = evaluated['independent_person'], evaluated['hybrid']
                            evidence, rebuild_metrics = evaluated['decision_evidence'], evaluated['rebuild_profile']
                        except Exception as exc:
                            failure = trial_error(name, settings, exc, getattr(exc, 'replay_stage', active_stage)) if trial_error else None
                            if failure is None:
                                raise
                            failure.update(failed=True, elapsed_seconds=getattr(exc, 'worker_elapsed_seconds', time.perf_counter()-started))
                            result.setdefault('failed_trials', {})[name] = failure
                            if record_trial:record_trial(name, settings, failure, inputs)
                            if progress:progress(name, 'failed', failure)
                            continue
                        if baseline is None:
                            baseline = predictions
                        if context is not None:context['baseline_predictions']=baseline
                        item = dict(hybrid=hybrid,independent_person=bodies,elapsed_seconds=evaluated.get('elapsed_seconds', time.perf_counter()-started),
                            versus_retained=impact(retained, predictions), versus_baseline=impact(baseline, predictions))
                        if runtime and runtime.profile:
                            item["performance"] = dict(cpu_seconds=evaluated.get("cpu_seconds", time.process_time()-cpu_started),
                                rebuild=rebuild_metrics, trial=evaluated.get('trial_profile'))
                            item['performance']['result_wait_seconds'] = max(0.,time.perf_counter()-evaluated.get('finished_monotonic_seconds',time.perf_counter()))
                        if runtime:
                            item['worker'] = {k: evaluated[k] for k in ('worker_pid', 'started_monotonic_seconds', 'finished_monotonic_seconds') if k in evaluated}
                        if consume is not None:
                            consume_started = time.perf_counter()
                            item.update(consume(name, observations, retained, predictions))
                            if runtime and runtime.profile:
                                result['performance']['consume_seconds'] += time.perf_counter()-consume_started
                        if evidence is not None:
                            item['decision_evidence'] = evidence
                        item['completed_wall_seconds'] = time.perf_counter()-wall_started
                        result['trials'][name] = item
                        if record_trial:
                            journal_started = time.perf_counter()
                            record_trial(name,settings,item,inputs)
                            if runtime and runtime.profile:
                                result['performance']['journal_seconds'] += time.perf_counter()-journal_started
                        if progress:
                            progress(name, 'complete', item)
                except KeyboardInterrupt:
                    if variant_stream is None:
                        raise
                    result['interrupted'] = True
        result['history_preparations'] = 1 if reuse_history else len(result['trials'])
        return result


def merge_impact(config, report, source):
    from .merge_report_policy import patched
    retained_totals = dict(source.execute('SELECT person,COUNT(*) FROM members GROUP BY person'))
    from .body_reports import members as body_members
    for person,count in Counter(body_members(source,config).values()).items():retained_totals[person]=retained_totals.get(person,0)+count
    variants, names = {}, {(): 'baseline'}
    for index, item in enumerate(report['recommendations']):
        changes = {r['parameter']: r['proposed'] for r in item['changes']}
        key = tuple(sorted(changes.items()))
        if key not in names:
            name = 'proposal-' + str(index+1)
            names[key] = name
            variants[name] = patched(config, changes)
        item['simulation_trial'] = names[key]
    def pair_status(name, observations, retained, predicted):
        peers = defaultdict(set)
        counts, lost = Counter(), Counter()
        for key, old in retained.items():
            counts[old] += 1
            if predicted[key] is None:
                lost[old] += 1
            else:
                peers[old].add(predicted[key])
        wanted = report['identity_ids']
        all_peers = set().union(*(peers[p] for p in wanted))
        complete = all(counts[p] and not lost[p] for p in wanted)
        return dict(requested_pair=dict(observation_scope='selected_latest_scans',
            fully_merged=complete and len(all_peers)==1,
            fully_assigned=complete, resulting_identities=sorted(all_peers),
            members=[dict(identity=p, observations=counts[p], withheld=lost[p],
                          outside_latest_selection=retained_totals.get(p, 0)-counts[p],
                          resulting_identities=sorted(peers[p])) for p in wanted]))
    result = replay(config, variants, scope=report['scope'], source=source, consume=pair_status)
    report['next_action'] = dict(scene_ids=result['selection'], requires_identity_rebuild=True,
                                 portraits_only_is_sufficient=False)
    for item in report['recommendations']:
        item['replay_verified'] = result['trials'][item['simulation_trial']]['requested_pair']['fully_merged']
    report['rejected_recommendations'] = [i for i in report['recommendations'] if not i['replay_verified']]
    report['recommendations'] = [i for i in report['recommendations'] if i['replay_verified']]
    if len(set(report['identity_ids'])) == 1:
        baseline = result['trials'].get('baseline')
        report['status'] = ('already_merged_replay_incomplete' if baseline is None else
                            'already_merged' if baseline['requested_pair']['fully_merged'] else 'retained_merge_not_reproduced')
    else:
        report['status'] = 'replay_verified' if report['recommendations'] else 'no_replay_verified_proposal'
    return result

"""Full production retuning on a consistent, read-only retained VTT snapshot."""
from collections import Counter, defaultdict
from contextlib import closing, nullcontext
from dataclasses import replace
from types import SimpleNamespace
import time

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


def replay(config, variants, **kwargs):
    from .run_cache import scope
    with scope(config):
        return _replay(config,variants,**kwargs)


def _replay(config, variants, *, scenes=None, scope=None, source=None, consume=None, variant_stream=None, inspect_trial=None,
            progress=None, allow_partial_selection=False):
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
        scope, scans = _selection(db, config, scenes, scope, allow_partial=allow_partial_selection, diagnostics=inputs)
        installation = FaceDatabase.installation.fget(SimpleNamespace(connection=db))
        observations = {r['id']: r['scene'] for r in db.execute(
            'SELECT o.id,s.scene FROM observations o JOIN scans s ON s.id=o.scan '
            'WHERE o.scan IN (SELECT value FROM json_each(?))', (encoded([s['id'] for s in scans]),))}
        retained = dict.fromkeys(observations)
        retained.update((r[0], r[1]) for r in db.execute('SELECT observation,person FROM members') if r[0] in observations)
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
        from itertools import chain
        stream = variant_stream() if variant_stream is not None else variants.items()
        seen_names = {'baseline'}
        try:
            if progress and not inputs['complete_scope']:
                progress('inputs', f"partial scope: {inputs['selected_scene_count']}/{inputs['latest_scene_count']} "
                         'compatible complete scenes; excluded scenes will be listed in the report', None)
            for name, settings in chain([('baseline', config)], stream):
                if name != 'baseline':
                    if name in seen_names:
                        raise ValueError('Duplicate simulation name.')
                    seen_names.add(name)
                if namespace(settings) != ns or scan_identity(settings, ns) != scan_identity(config, ns) or not same_contract(config,settings):
                    raise ValueError('Simulation variants must preserve model, person and VTT scan contracts.')
                data = plain(settings.data)
                data['execution'].update(rebuild_checkpoint_scenes=0, profile_rebuild=False)
                settings = replace(settings, data=data)
                if progress:
                    progress(name, 'start', None)
                started = time.perf_counter()
                with closing(_ram(settings)) as target:
                    target.execute('INSERT INTO metadata VALUES (?,?)', ('installation', installation))
                    copy_history(db, target, [], scope, ns)
                    state = SimpleNamespace(connection=target, installation=installation)
                    from .progress import observe_stages
                    last_stage = [0.]
                    def stage_update(label):
                        now = time.monotonic()
                        if progress and now-last_stage[0] >= 1.:
                            last_stage[0] = now
                            progress(name, 'stage: '+label, None)
                    with observe_stages(stage_update if progress else None):
                        rebuild_selected(state, [dict(scene=s['scene'], scan=s['id']) for s in scans], settings)
                    predictions = dict.fromkeys(observations)
                    predictions.update((r[0], r[1]) for r in target.execute('SELECT observation,person FROM members')
                                       if r[0] in observations)
                    hybrid=coverage(target,settings,[s['id'] for s in scans])
                    evidence = inspect_trial(target, settings) if inspect_trial else None
                if baseline is None:
                    baseline = predictions
                item = dict(hybrid=hybrid,elapsed_seconds=time.perf_counter()-started,
                    versus_retained=impact(retained, predictions), versus_baseline=impact(baseline, predictions))
                if consume is not None:
                    item.update(consume(name, observations, retained, predictions))
                if evidence is not None:
                    item['decision_evidence'] = evidence
                result['trials'][name] = item
                if progress:
                    progress(name, 'complete', item)
        except KeyboardInterrupt:
            if variant_stream is None:
                raise
            result['interrupted'] = True
        return result


def merge_impact(config, report, source):
    from .merge_report_policy import patched
    retained_totals = dict(source.execute('SELECT person,COUNT(*) FROM members GROUP BY person'))
    variants, names = {}, {}
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
    report['status'] = 'replay_verified' if report['recommendations'] else 'no_replay_verified_proposal'
    return result

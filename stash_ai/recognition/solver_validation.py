"""Labelled, scene-disjoint evaluation of one frozen solver finalist."""
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path

from .database import encoded
from .evaluation import evaluate_identities
from .fragmentation_report import _open_readonly
from .merge_report_policy import patched
from .offline_evaluation import evaluate, _selection


def _metrics(labels, samples, locations, decisions, exclude=(), partner_sets=(), timing=None, unbound=None):
    from .scene_strategy_validation import physical_owner
    records = [dict(observation=oid, scene=locations[oid][0], truth=truth,
                    predicted=decisions[oid]['accepted_identity']) for oid, truth in labels.items() if oid not in exclude]
    all_metrics = evaluate_identities(records)
    physical = defaultdict(list)
    for row in records:
        physical[(*locations[row['observation']],row['truth'])].append(row)
    projections = []
    conflicts = 0
    for (scene,cue,truth), rows in physical.items():
        owners = {row['predicted'] for row in rows if row['predicted'] is not None}
        conflicts += len(owners)>1
        projections.append(dict(observation=encoded([scene,cue,truth]),scene=scene,truth=truth,
            predicted=physical_owner(owners),
            slices={s for row in rows for s in samples[row['observation']]['slices']}))
    for (scene,cue,truth),info in (unbound or {}).items():
        if (scene,cue,truth) not in physical:
            projections.append(dict(observation=encoded([scene,cue,truth]),scene=scene,truth=truth,
                                    predicted=None,slices=set(info.get('slices',()))))
    provisional = [dict(row, predicted=decisions[row['observation']]['identity'])
                   for row in records if row['predicted'] is None
                   and decisions[row['observation']].get('status') == 'provisional']
    metrics = {'all': all_metrics, 'provisional_only': evaluate_identities(provisional),
               'partners': partner_metrics(partner_sets, labels, locations, decisions, exclude)}
    from .sequence_validation import sequence_metrics
    metrics['physical'] = dict(evaluate_identities(projections),projection_conflicts=conflicts,
        slices={name:evaluate_identities([row for row in projections if name in row['slices']])
                for name in ('back','entangled','entrant')})
    metrics['sequences'] = sequence_metrics(labels,samples,locations,decisions,exclude,timing=timing,unbound=unbound)
    for category in ('back','entangled'):
        within = [r for r in records if category in samples[r['observation']]['slices']]
        outside = [r for r in records if category not in samples[r['observation']]['slices']]
        values = evaluate_identities(within)
        # Count a back/front pair once, even when only one endpoint is in the slice.
        def pair_counts(rows):
            truth, predicted, both = Counter(), Counter(), Counter()
            for row in rows:
                truth[row['truth']] += 1
                if row['predicted'] is not None:
                    predicted[row['predicted']] += 1
                    both[row['truth'],row['predicted']] += 1
            return [sum(n*(n-1)//2 for n in counts.values()) for counts in (truth,predicted,both)]
        true, joined, correct = (a-b for a,b in zip(pair_counts(records),pair_counts(outside)))
        values['pairs_touching_slice'] = dict(true_pairs=true, joined_pairs=joined,
            false_merge_pairs=joined-correct, recall=correct/true if true else None,
            precision=correct/joined if joined else None)
        metrics[category] = values
    return metrics


def partner_metrics(triples, labels, locations, decisions, exclude):
    """Each performer/cue relation votes once, including face and body projections."""
    from collections import defaultdict
    from .scene_strategy_validation import physical_owner
    projections = defaultdict(set)
    for oid,truth in labels.items():
        if oid not in exclude:
            owner = decisions[oid]['accepted_identity']
            if owner is not None:
                projections[(truth,*locations[oid])].add(owner)
    positive, negative = {}, {}
    for face, back, partner in triples:
        for first, second, same in ((face, back, True), (back, partner, False), (face, partner, False)):
            if first in exclude or second in exclude:
                continue
            addresses = tuple(sorted((labels[oid], *locations[oid]) for oid in (first, second)))
            if addresses[0] == addresses[1]:
                continue
            left,right = (projections[key] for key in addresses)
            if same:
                owner = physical_owner(left)
                positive[addresses] = owner is not None and owner==physical_owner(right)
            else:
                negative[addresses] = bool(left & right)
    counts = defaultdict(Counter)
    for key, recovered in positive.items():
        counts[key[0][0]].update(target_pairs=1, recovered=int(recovered))
    for key, joined in negative.items():
        for performer in {address[0] for address in key}:
            counts[performer].update(partner_pairs=1, false_joins=int(joined))
    return dict(target_pairs=len(positive), recovered=sum(positive.values()),
                partner_pairs=len(negative), false_joins=sum(negative.values()),
                per_performer=dict(sorted(counts.items())))


def _partition_delta(original, stressed, identifiers):
    # Count joins and lost relations without materialising observation pairs.
    left, right, shared = Counter(), Counter(), Counter()
    for oid in identifiers:
        a, b = original[oid]['accepted_identity'], stressed[oid]['accepted_identity']
        if a is not None:
            left[a] += 1
        if b is not None:
            right[b] += 1
        if a is not None and b is not None:
            shared[a,b] += 1
    def pairs(counts):
        return sum(n*(n-1)//2 for n in counts.values())
    common = pairs(shared)
    return dict(lost_pairs=pairs(left)-common, joined_pairs=pairs(right)-common)


def _locations(source, selected, ids, *, timing=None):
    result = {}
    args = (encoded(sorted(ids)),encoded([r['id'] for r in selected]))
    for table in ('observations','body_observations'):
        if not source.execute('SELECT 1 FROM sqlite_master WHERE type=\'table\' AND name=?',(table,)).fetchone():
            continue
        shot = '$.view_shot' if table == 'observations' else '$.quality.view_shot'
        for oid, scene, cue, seconds, shot_id in source.execute(f"SELECT o.id,s.scene,o.cue,o.seconds,json_extract(o.quality,'{shot}') FROM {table} o "
                'JOIN scans s ON s.id=o.scan WHERE o.id IN (SELECT value FROM json_each(?)) '
                'AND o.scan IN (SELECT value FROM json_each(?))',args):
            result[oid] = (scene,cue)
            if timing is not None:
                timing[scene,cue] = seconds, shot_id if shot_id is not None else timing.get((scene,cue),(None,None))[1]
    return result


def read_manifest(manifest_path):
    path = Path(manifest_path)
    def unique(pairs):
        values = dict(pairs)
        if len(values) != len(pairs):
            raise ValueError('Duplicate field in solver validation manifest.')
        return values
    document = json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=unique)
    scenes = document.get('scenes')
    samples = document.get('observations')
    if (not isinstance(scenes,list) or not scenes or
            not all(isinstance(s,str) and s.isascii() and s.isdecimal() and int(s)>0 for s in scenes)
            or not isinstance(samples,dict) or (not samples and not (document.get('visible_people') or document.get('unbound_manual_labels')))):
        raise ValueError('Validation needs scene IDs and labelled observations.')
    labels = {}
    for oid, row in samples.items():
        if (not isinstance(row,dict) or not isinstance(row.get('truth'),str) or not row['truth'].strip()
                or not isinstance(row.get('slices'),list) or any(s not in ('back','entangled','entrant') for s in row['slices'])):
            raise ValueError('Each validation observation needs truth and a list of back/entangled/entrant slices.')
        if type(row.get('reference', False)) is not bool:
            raise ValueError('Validation reference must be a boolean.')
        labels[oid] = row['truth']
    database = document.get('database')
    if database is not None and (not isinstance(database,str) or not database):
        raise ValueError('Validation database must be a filename.')
    triples = document.get('partner_sets', [])
    if not isinstance(triples, list):
        raise ValueError('partner_sets must be a list of [face, back, partner] triples.')
    for triple in triples:
        if (not isinstance(triple, list) or len(triple) != 3
                or any(not isinstance(oid, str) or oid not in labels for oid in triple)
                or len(set(triple)) != 3 or labels[triple[0]] != labels[triple[1]]
                or labels[triple[1]] == labels[triple[2]]):
            raise ValueError('Each partner set needs two observations of one target and a different partner.')
    from .crowd_labels import validate
    visible = document.get('visible_people',{})
    # Older pedestrian manifests recorded unbound people in a single-scene list.
    if len(scenes)==1 and not visible and document.get('unbound_manual_labels'):
        from collections import defaultdict
        cues = defaultdict(list)
        for cue,truth in document['unbound_manual_labels']:
            cues[str(cue)].append(dict(truth=truth,binding='unbound'))
        visible = {scenes[0]:dict(cues)}
    validate(visible,scenes)
    return (scenes, samples, labels, (path.parent/database).resolve() if database else None,
            document.get('series', []), triples, visible)


def validate_winner(config, training, result, manifest_path, *, manifest=None, progress=None, compare_strategies=False):
    scenes, samples, labels, database, series, triples, visible = manifest or read_manifest(manifest_path)
    conflict = training.execute('SELECT DISTINCT scene FROM scans WHERE scope=? AND namespace=? AND '
                                'scene IN (SELECT value FROM json_each(?)) LIMIT 1',
                                (result['scope'],result['namespace'],encoded(scenes))).fetchone()
    if conflict:
        raise ValueError(f'Holdout scene {conflict[0]} occurs in the training snapshot.')
    settings = replace(config,database=database or config.database)
    changes = {item['parameter']:item['proposed'] for item in result['best_result']['changes']}
    winner = patched(settings,changes)
    if not isinstance(series,list) or len(series)>3:
        raise ValueError('At most three validation cue series are supported.')
    with closing(_open_readonly(settings.database)) as source:
        scope, scans = _selection(source, settings, scenes, result['scope'])
        counts = {scan['scene']:scan['total'] for scan in scans}
        windows = []
        for item in series:
            if (not isinstance(item,list) or len(item)!=3 or not isinstance(item[0],str) or item[0] not in counts or
                any(type(i) is not int for i in item[1:]) or not 0<=item[1]<=item[2]<counts[item[0]]):
                raise ValueError('Validation series must be [scene, first_cue, last_cue] inside a selected scene.')
            if tuple(item) not in windows:
                windows.append(tuple(item))
        options = dict(scenes=scenes,scope=scope,batch_sizes=(0,),source_snapshot=source)
        def announce(name):
            if progress:
                progress(f'Holdout {name} · {len(scans)} scenes · {len(labels)} labels')
        from .contact_holdout import physical_inputs, visible_scores, spread
        timing = {}
        locations = _locations(source,scans,labels,timing=timing)
        groups,cases = physical_inputs(source,settings,scans,visible,labels,locations)
        bindings = {(scene,int(cue),p['truth']):p for scene,cues in visible.items()
                    for cue,people in cues.items() for p in people}
        from .body_identity import enabled as body_enabled
        replay_labels = dict(labels)
        if body_enabled(settings):
            for (_,_,truth),addresses in groups.items():
                for oid in addresses:
                    replay_labels.setdefault(oid,truth)
        else:
            groups = {k:[oid for oid in ids if oid in labels] for k,ids in groups.items()}
            cases = []
        unbound = {key:bindings[key] for key,ids in groups.items() if not ids}
        if not replay_labels:
            from .crowd_labels import detection_counts
            from .body_observations import contract
            measured = _metrics({}, {}, {}, {}, unbound=unbound)
            measured['visible'] = visible_scores(groups,{}, {},bindings=bindings)
            return dict(verdict=dict(status='insufficient_coverage',measures={}),scenes=scenes,labelled=0,
                        detection=detection_counts(source,scans,visible,contract(settings)),
                        metrics={'baseline/joint':measured,'winner/joint':measured},omissions=[],incremental={},
                        scene_strategies={},omission_status='no_bound_observations')
        from .scene_strategy_validation import variants, strategy_summary, omission_damage
        strategies = variants(winner) if compare_strategies else {}
        selected = {}
        trained = result.get('search',{}).get('scene_finalists',{})
        if strategies and trained:
            for name,patch in trained.items():
                strategies[name] = patched(settings,patch)
                selected[name] = strategies[name]
        elif strategies:
            if result.get('groups'):
                from .merge_goals import evaluate as train, objective
                training_trials = train(winner, result, training, variants=strategies)['trials']
                order = sorted(strategies, key=lambda name:objective(training_trials[name]['goal_result']), reverse=True)
            else:
                # Direct evaluation without labelled training goals uses a fixed paired comparison.
                order = ['paths-only-evidence', 'tracks-1-evidence']
            from .config import digest
            distinct = {}
            for name in order:
                key = digest(strategies[name].data)
                distinct.setdefault(key,name)
                if len(distinct)==2:
                    break
            selected = {name:strategies[name] for name in distinct.values()}
        if compare_strategies:
            from .scene_strategy_validation import holdout_ablations
            controls = holdout_ablations(winner)
            strategies.update(controls)
            for name in ('order-0-extended-1-turns-1','order-1-extended-1-turns-1'):
                selected[name] = controls[name]
        stress_variants = {'winner':winner, **selected}
        tested = ('baseline', *stress_variants)
        full = evaluate(settings,replay_labels,variants={'winner':winner,**strategies},variant_started=announce,**options)
        predicted = full['trials']['winner/joint']['decisions']
        references = [oid for oid,row in samples.items() if row.get('reference')]
        metrics = {name:dict(_metrics(labels,samples,locations,trial['decisions'],partner_sets=triples,timing=timing,unbound=unbound),
                             visible=visible_scores(groups,trial['decisions'],labels,references,bindings)) for name,trial in full['trials'].items()}
        strategy_results = {name:strategy_summary(cfg,metrics.pop(name+'/joint'),labels,locations,
                            full['trials'][name+'/joint']['decisions']) for name,cfg in strategies.items()}
        contamination = dict(status='no_labelled_contact_pair')
        if cases:
            from .contact_holdout import stress_cases
            all_locations = _locations(source,scans,replay_labels)
            episodes = []
            for kind,case in stress_cases(cases,visible,scans,samples,locations):
                scene = next(s['scene'] for s in scans if s['id']==case[0])
                stressed = evaluate(settings,replay_labels,variants=stress_variants,swapped_bodies=case,**options)
                item = dict(kind=kind,scene=scene,cue=case[1],bodies=list(case[2:]))
                for name in tested:
                    key = name+'/joint'
                    decisions = stressed['trials'][key]['decisions']
                    before = full['trials'][key]['decisions']
                    item[name] = dict(visible=visible_scores(groups,decisions,labels,references,bindings),
                        **spread(before,decisions,all_locations,scene,case[1],replay_labels))
                episodes.append(item)
            first = episodes[0]
            contamination = dict(status='tested',episodes=episodes,scene=first['scene'],cue=first['cue'],
                                 bodies=first['bodies'],**first['winner'])
        incremental = {}
        if len(scans)>1:
            # Front-first is based on usable measured face inputs, including
            # unlabeled roots. A source with stronger face coverage wins a tie.
            weights = {}
            for scan in scans:
                faces = source.execute("SELECT COUNT(DISTINCT cue) FROM observations WHERE scan=? "
                    "AND rejected IS NULL AND embedding IS NOT NULL",(scan['id'],)).fetchone()[0]
                backs = {loc[1] for oid,loc in locations.items() if loc[0]==scan['scene'] and 'back' in samples[oid]['slices']}
                weights[scan['scene']] = (bool(faces), faces, -len(backs))
            order = sorted(counts,key=lambda scene:(weights[scene],int(scene)), reverse=True)
            for name, sequence in [('face_first',order),('rear_first',list(reversed(order)))]:
                if progress:
                    progress('Holdout incremental '+name)
                staged = evaluate(settings,labels,variants=stress_variants,
                                  **dict(options,batch_sizes=(1,),ordered_scenes=sequence))
                by_variant = {}
                for variant in tested:
                    joint = full['trials'][variant+'/joint']['decisions']
                    replay = staged['trials'][variant+'/batch-1']['decisions']
                    by_variant[variant] = dict(**_partition_delta(joint,replay,labels),
                        coverage_changes=sum((joint[k]['accepted_identity'] is None) !=
                                             (replay[k]['accepted_identity'] is None) for k in labels))
                decisions = staged['trials']['winner/batch-1']['decisions']
                incremental[name] = dict(order=sequence, strategies={s:by_variant[s] for s in selected}, baseline=by_variant['baseline'], **by_variant['winner'],
                    metrics=_metrics(labels,samples,locations,decisions,partner_sets=triples,timing=timing,unbound=unbound))
        if not windows:
            faceless = {r[0] for r in source.execute(
                "SELECT b.id FROM body_observations b LEFT JOIN observations o ON o.id=b.face "
                "WHERE b.id IN (SELECT value FROM json_each(?)) AND (o.embedding IS NULL OR o.rejected IS NOT NULL)",
                (encoded(sorted(labels)),))} if source.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='body_observations'").fetchone() else set()
            for scenario in ('entangled','back','faceless'):
                candidates = []
                for oid,(scene,cue) in locations.items():
                    hard = samples[oid]['slices']
                    belongs = (scenario in hard if scenario!='faceless' else
                               oid in faceless)
                    if belongs:
                        candidates.append((scene,max(0,cue-1),min(counts[scene]-1,cue+1)))
                window = next((candidate for candidate in sorted(candidates) if candidate not in windows),None)
                if window:
                    windows.append(window)
            # Derive omission windows from the actual winner's recorded face roots.
            tracks = sorted({tuple(track) for row in predicted.values()
                             for track in (row.get('evidence') or {}).get('source_tracks', [])})
            addresses = _locations(source,scans,{oid for track in tracks for oid in track})
            for start,end,_ in tracks:
                if len(windows)>=3:
                    break
                if start in addresses and end in addresses:
                    scene, first = addresses[start]
                    last_scene, last = addresses[end]
                    window = (scene,first,last)
                    if last_scene==scene and window not in windows:
                        windows.append(window)
                if len(windows)==3:
                    break
        from .crowd_labels import detection_counts
        from .body_observations import contract
        detections = detection_counts(source,scans,visible,contract(settings))
        stress = []
        for index, window in enumerate(windows, 1):
            if progress:
                progress(f'Holdout omission {index}/{len(windows)} · scene {window[0]} cues {window[1]}–{window[2]}')
            scene,first,last = window
            excluded = {oid for oid,(s,c) in locations.items() if s==scene and first<=c<=last}
            group = sorted(labels.keys()-excluded)
            remaining = {key:p for key,p in unbound.items() if key[0]!=scene or not first<=key[1]<=last}
            before = _metrics(labels,samples,locations,predicted,excluded,triples,timing,remaining)
            for mode in ('rejected','empty'):
                omitted = evaluate(settings,labels,variants=stress_variants,omitted_cues=window,omission_mode=mode,**options)['trials']
                trial = omitted['winner/joint']['decisions']
                damage = {name:omission_damage(labels,locations,group,full['trials'][name+'/joint']['decisions'],
                                               omitted[name+'/joint']['decisions']) for name in tested}
                delta = _partition_delta(predicted,trial,group)
                after = _metrics(labels,samples,locations,trial,excluded,triples,timing,remaining)
                stress.append(dict(scene=scene, first_cue=first, last_cue=last, mode=mode,scored=len(group),
                                   excluded_labels=len(excluded), damage=damage, **delta, reference=before, omitted=after))
    return dict(verdict=holdout_verdict(metrics, contamination, incremental, stress, require_slices=compare_strategies or any(
                k.startswith('sapiensid.identities.scene_') or k.endswith(('tracklet_matching','endpoint_matching','body_first_matching')) for k in changes)),
                scene_strategies=strategy_results, stress_strategies=list(selected), selected_trial=result.get('selected_trial'), scenes=scenes, metrics=metrics, detection=detections, association_stress=contamination, omissions=stress, incremental=incremental,
                omission_status='tested' if stress else 'no_series_identified', labelled=len(labels))


def holdout_verdict(metrics, contamination=None, incremental=None, omissions=(), *, require_slices=False):
    base, final = (metrics[name+'/joint'] for name in ('baseline','winner'))
    results = {}
    for label,category,field,total,higher in (
            ('own_turns','sequences','own_id_turns','turns',True),
            ('correct_contacts','sequences','correct','crossings',True),
            ('endpoint_returns','sequences','endpoint_returns','crossings',True),
            ('both_recovered','sequences','both_recovered','crossings',True),
            ('partner_swaps','sequences','partner','crossings',False),
            ('false_joins','partners','false_joins','partner_pairs',False),
            ('split_references','visible','split_identities','reference_identities',False)):
        values = []
        for trial in (base, final):
            group = trial[category]
            counts = group.get('contact_interior',{}) if field in ('correct','partner') else group
            values.append((counts.get(field,0),group.get(total,group.get('anchored_turns',0))))
        (a,n),(b,m) = values
        if label=='own_turns':
            n=m=max(n,m)
        state = 'no_coverage' if not n or not m else 'unchanged' if a*m==b*n else (
            'improved' if (b*n>a*m)==higher else 'regressed')
        results[label] = dict(baseline=a,winner=b,baseline_total=n,winner_total=m,status=state)
    def risk(label,before,after,covered):
        status = ('no_coverage' if not covered else 'regressed' if after>before else
                  'improved' if after<before else 'unchanged')
        results[label] = dict(baseline=before,winner=after,status=status)
    physical = [trial.get('physical', {}) for trial in (base, final)]
    covered = all(p.get('observations', 0) for p in physical)
    for field in ('withheld', 'extra_fragments', 'false_merge_pairs', 'projection_conflicts'):
        risk('physical_' + field, physical[0].get(field, 0), physical[1].get(field, 0), covered)
    sequences = [trial['sequences'] for trial in (base,final)]
    for key in ('scene_id_switches','unrecovered_at_end','wrong_reacquisitions','entrant_takeovers'):
        risk(key, sequences[0].get(key,0), sequences[1].get(key,0),
             all(key in seq for seq in sequences) and (covered if key!='entrant_takeovers' else all(seq.get('entrant_episodes') for seq in sequences)))
    first,last = (set(map(tuple,row.get('rooted_turns',()))) for row in sequences)
    risk('lost_turn_references',0,len(first-last),bool(first))
    before,after = (set(map(tuple,row.get('returned_people',()))) for row in sequences)
    risk('lost_person_endpoints',0,len(before-after),bool(before))
    episodes = (contamination or {}).get('episodes',[])
    for field in ('later_cues','other_scenes'):
        for i,item in enumerate(episodes):
            a,b = (item[name][field] for name in ('baseline','winner'))
            if field=='other_scenes':
                a,b = set(a),set(b)
                risk(f'contact_{i}_new_foreign_scenes',0,len(b-a),True)
            else:
                risk(f'contact_{i}_{field}',a,b,True)
    if not episodes:
        risk('contact_stress',0,0,False)
    for order,record in (incremental or {}).items():
        for field in ('lost_pairs','joined_pairs','coverage_changes'):
            risk(order+'_'+field,record['baseline'][field],record[field],True)
    if not incremental:
        risk('import_order',0,0,False)
    for row in omissions:
        for field in ('lost_correct_pairs','new_false_joins','new_projection_conflicts','new_partner_claims'):
            damage = row['damage']
            name = f"omission_{row['scene']}_{row['first_cue']}_{row['last_cue']}_{row['mode']}_{field}"
            risk(name,damage['baseline'][field],damage['winner'][field],row['scored']>=2)
    coverage = {}
    for name in ('back','entangled','entrant'):
        slices = [p.get('slices',{}).get(name,{}) for p in physical]
        count = min(s.get('observations',0) for s in slices)
        coverage[name] = dict(observations=count,performers=min(s.get('truth_identities',0) for s in slices))
        for field in ('withheld','extra_fragments','false_merge_pairs'):
            risk(f'{name}_{field}',slices[0].get(field,0),slices[1].get(field,0),count>0)
    states = {r['status'] for r in results.values()}
    status = next(s for s in ('regressed','improved','unchanged','no_coverage') if s in states)
    episodes = {key: min(row.get(key, 0) for row in sequences)
                for key in ('turns', 'crossings', 'entrant_episodes')}
    if require_slices and status!='regressed' and (not all(episodes.values()) or
            any(not c['observations'] for c in coverage.values())):
        status = 'insufficient_coverage'
    return dict(status=status,measures=results,slice_coverage=coverage,episode_coverage=episodes)

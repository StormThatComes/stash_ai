"""Labelled, scene-disjoint evaluation of one frozen solver finalist."""
from collections import Counter
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path

from .database import encoded
from .evaluation import evaluate_identities
from .fragmentation_report import _open_readonly
from .merge_report_policy import patched
from .offline_evaluation import evaluate, _selection


def _metrics(labels, samples, locations, decisions, exclude=(), partner_sets=()):
    records = [dict(observation=oid, scene=locations[oid][0], truth=truth,
                    predicted=decisions[oid]['accepted_identity']) for oid, truth in labels.items() if oid not in exclude]
    all_metrics = evaluate_identities(records)
    provisional = [dict(row, predicted=decisions[row['observation']]['identity'])
                   for row in records if row['predicted'] is None
                   and decisions[row['observation']].get('status') == 'provisional']
    metrics = {'all': all_metrics, 'provisional_only': evaluate_identities(provisional),
               'partners': partner_metrics(partner_sets, labels, locations, decisions, exclude)}
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
    positive = defaultdict(list)
    negative = defaultdict(list)
    for face, back, partner in triples:
        for first, second, same in ((face, back, True), (back, partner, False), (face, partner, False)):
            if first in exclude or second in exclude:
                continue
            a, b = (decisions[oid]['accepted_identity'] for oid in (first, second))
            addresses = tuple(sorted((labels[oid], *locations[oid]) for oid in (first, second)))
            if addresses[0] == addresses[1]:
                continue
            (positive if same else negative)[addresses].append(a is not None and a == b)
    counts = defaultdict(Counter)
    for key, values in positive.items():
        counts[key[0][0]].update(target_pairs=1, recovered=int(all(values)))
    for key, values in negative.items():
        for performer in {address[0] for address in key}:
            counts[performer].update(partner_pairs=1, false_joins=int(any(values)))
    return dict(target_pairs=len(positive), recovered=sum(map(all, positive.values())),
                partner_pairs=len(negative), false_joins=sum(map(any, negative.values())),
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


def _locations(source, selected, ids):
    result = {}
    args = (encoded(sorted(ids)),encoded([r['id'] for r in selected]))
    for table in ('observations','body_observations'):
        if not source.execute('SELECT 1 FROM sqlite_master WHERE type=\'table\' AND name=?',(table,)).fetchone():
            continue
        for oid, scene, cue in source.execute(f'SELECT o.id,s.scene,o.cue FROM {table} o '
                'JOIN scans s ON s.id=o.scan WHERE o.id IN (SELECT value FROM json_each(?)) '
                'AND o.scan IN (SELECT value FROM json_each(?))',args):
            result[oid] = (scene,cue)
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
            or not isinstance(samples,dict) or not samples):
        raise ValueError('Validation needs scene IDs and labelled observations.')
    labels = {}
    for oid, row in samples.items():
        if (not isinstance(row,dict) or not isinstance(row.get('truth'),str) or not row['truth'].strip()
                or not isinstance(row.get('slices'),list) or any(s not in ('back','entangled') for s in row['slices'])):
            raise ValueError('Each validation observation needs truth and a list of back/entangled slices.')
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
    return (scenes, samples, labels, (path.parent/database).resolve() if database else None,
            document.get('series', []), triples)


def validate_winner(config, training, result, manifest_path, *, manifest=None, progress=None):
    scenes, samples, labels, database, series, triples = manifest or read_manifest(manifest_path)
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
        full = evaluate(settings,labels,variants={'winner':winner},variant_started=announce,**options)
        locations = _locations(source,scans,labels)
        predicted = full['trials']['winner/joint']['decisions']
        metrics = {name: _metrics(labels,samples,locations,trial['decisions'], partner_sets=triples)
                   for name, trial in full['trials'].items()}
        if not windows:
            # Derive omission windows from the actual winner's recorded face roots.
            tracks = sorted({tuple(track) for row in predicted.values()
                             for track in (row.get('evidence') or {}).get('source_tracks', [])})
            addresses = _locations(source,scans,{oid for track in tracks for oid in track})
            for start,end,_ in tracks:
                if start in addresses and end in addresses:
                    scene, first = addresses[start]
                    last_scene, last = addresses[end]
                    window = (scene,first,last)
                    if last_scene==scene and window not in windows:
                        windows.append(window)
                if len(windows)==3:
                    break
        stress = []
        for index, window in enumerate(windows, 1):
            if progress:
                progress(f'Holdout omission {index}/{len(windows)} · scene {window[0]} cues {window[1]}–{window[2]}')
            trial = evaluate(winner,labels,omitted_cues=window,**options)['trials']['baseline/joint']['decisions']
            scene,first,last = window
            excluded = {oid for oid,(s,c) in locations.items() if s==scene and first<=c<=last}
            group = sorted(labels.keys()-excluded)
            delta = _partition_delta(predicted,trial,group)
            before = _metrics(labels,samples,locations,predicted,excluded,triples)
            after = _metrics(labels,samples,locations,trial,excluded,triples)
            stress.append(dict(scene=scene, first_cue=first, last_cue=last, scored=len(group),
                               excluded_labels=len(excluded), **delta, reference=before, omitted=after))
    return dict(selected_trial=result.get('selected_trial'), scenes=scenes, metrics=metrics, omissions=stress,
                omission_status='tested' if stress else 'no_series_identified', labelled=len(labels))

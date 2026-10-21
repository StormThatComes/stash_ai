"""Developer-only local face-pair evaluation; no public stage command."""
import math


def evaluate_pairs(pairs, threshold):
    """Pairs contain normalized embeddings and explicit same-person ground truth.

    This measures the supplied dataset only. It must never turn detector scores
    or synthetic-vector checks into an accuracy claim about user video media.
    """
    import numpy as np
    from .models import normalized_embedding
    if type(threshold) not in (float, int) or not 0 <= threshold <= 1 or not math.isfinite(threshold):
        raise ValueError('Face evaluation threshold must be a finite cosine value in [0,1].')
    counts = dict(same=0, different=0, false_match=0, false_miss=0)
    for first, second, same in pairs:
        if type(same) is not bool:
            raise ValueError('Face pair ground truth must be explicit boolean.')
        a = normalized_embedding(first, len(first))
        b = normalized_embedding(second, len(first))
        match = float(np.dot(a, b)) >= threshold
        counts['same' if same else 'different'] += 1
        counts['false_miss' if same else 'false_match'] += int(match != same)
    return {'scope': 'supplied-local-pairs-only', 'threshold': threshold,
            'samples': counts['same'] + counts['different'], **counts,
            'default_policy': 'conservative-v1-not-domain-calibrated'}


def evaluate_identities(records):
    """Evaluate explicit local ground truth for complete identity assignments.

    Each record supplies observation, scene, truth and predicted (ID or None).
    Withheld samples reduce coverage/recall; they never form a shared identity.
    UUID spelling and model scores are not treated as ground truth.
    """
    from collections import Counter, defaultdict
    rows=list(records)
    observed=set()
    truth=defaultdict(Counter)
    scenes=defaultdict(Counter)
    predicted=defaultdict(Counter)
    truth_totals=Counter()
    for row in rows:
        if not isinstance(row,dict) or any(not isinstance(row.get(k),str) or not row[k]
            for k in ('observation','scene','truth')) or 'predicted' not in row:
            raise ValueError('Identity evaluation requires observation, scene, explicit truth and predicted.')
        if row['observation'] in observed:
            raise ValueError('Identity evaluation observations must be unique.')
        observed.add(row['observation'])
        person=row['predicted']
        if person is not None and (not isinstance(person,str) or not person):
            raise ValueError('Predicted identity must be a nonempty ID or None.')
        truth_totals[row['truth']]+=1
        if person is None:
            continue
        truth[row['truth']][person]+=1
        scenes[(row['scene'],row['truth'])][person]+=1
        predicted[person][row['truth']]+=1
    def pairs(n):
        return n*(n-1)//2
    correct=sum(pairs(n) for counts in predicted.values() for n in counts.values())
    assigned_pairs=sum(pairs(sum(counts.values())) for counts in predicted.values())
    true_pairs=sum(pairs(n) for n in truth_totals.values())
    false_splits=sum(pairs(sum(counts.values()))-sum(pairs(n) for n in counts.values()) for counts in truth.values())
    assigned=sum(sum(counts.values()) for counts in predicted.values())
    return dict(scope='supplied-local-identity-ground-truth-only',observations=len(rows),assigned=assigned,
        withheld=len(rows)-assigned,coverage=assigned/len(rows) if rows else None,
        truth_identities=len(truth_totals),predicted_identities=len(predicted),
        fragmented_identities=sum(len(counts)>1 for counts in truth.values()),
        extra_fragments=sum(max(0,len(counts)-1) for counts in truth.values()),
        fragmented_scene_identities=sum(len(counts)>1 for counts in scenes.values()),
        extra_scene_fragments=sum(max(0,len(counts)-1) for counts in scenes.values()),
        merged_identities=sum(len(counts)>1 for counts in predicted.values()),
        false_merge_pairs=assigned_pairs-correct,false_split_pairs=false_splits,
        pair_precision=correct/assigned_pairs if assigned_pairs else None,
        pair_recall=correct/true_pairs if true_pairs else None)


def compare_identity_runs(records, predictions):
    """Compare run partitions on common observations, independent of label IDs.

    A missing key means unprocessed; None means explicitly withheld. Comparisons
    restrict peer groups to the common selection and preserve that distinction.
    """
    from itertools import combinations
    from collections import defaultdict
    rows=list(records)
    # Validate ground truth even when there are no runs.
    evaluate_identities([dict(row,predicted=None) for row in rows])
    known={row['observation']:row for row in rows}
    metrics={}
    for name,run in predictions.items():
        if not isinstance(name,str) or not name or not isinstance(run,dict) or set(run)-set(known):
            raise ValueError('Named identity runs must map known observation IDs to predictions.')
        metrics[name]=evaluate_identities([dict(known[key],predicted=value) for key,value in run.items()])
    comparisons=[]
    for left,right in combinations(sorted(predictions),2):
        a,b=predictions[left],predictions[right]
        common=set(a).intersection(b)
        canonical={}
        def partition(run):
            groups=defaultdict(set)
            for key in common:
                if run[key] is not None:
                    groups[run[key]].add(key)
            labels={person:canonical.setdefault(frozenset(peers),len(canonical)) for person,peers in groups.items()}
            return {key:labels[run[key]] if run[key] is not None else None for key in common}
        pa,pb=partition(a),partition(b)
        coverage=sum((pa[key] is None)!=(pb[key] is None) for key in common)
        changed=sum(pa[key] is not None and pb[key] is not None and pa[key]!=pb[key] for key in common)
        comparisons.append(dict(left=left,right=right,common_observations=len(common),
            coverage_changes=coverage,partition_changes=changed,equivalent=coverage==0 and changed==0))
    return dict(scope='supplied-local-identity-ground-truth-only',runs=metrics,comparisons=comparisons)

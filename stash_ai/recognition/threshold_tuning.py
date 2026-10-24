"""Bounded offline threshold search; recommendations require independent truth."""
from collections import Counter
from itertools import product
import hashlib
import json
import math

from .config import digest
from .evaluation import evaluate_identities
from .merge_report_policy import patched, proposal
from .replay_simulation import replay


PARAMETERS = frozenset({
    'recognition.match_threshold', 'recognition.ambiguity_margin',
    'recognition.min_identity_support', 'recognition.min_merge_support',
    'recognition.cooccurrence_merge_threshold',
    'clustering.cohesion_threshold', 'clustering.view_match_threshold',
    'clustering.view_cohesion_threshold', 'clustering.view_ambiguity_margin',
    'clustering.view_min_support', 'clustering.view_attach_min_support',
    'clustering.same_scene_match_bonus', 'clustering.same_scene_min_support',
})


def configurations(config, grid, maximum):
    if type(maximum) is not int or not 1 <= maximum <= 1024:
        raise ValueError('max_trials must be an integer from 1 to 1024, including baseline.')
    if not isinstance(grid, dict) or not grid or set(grid)-PARAMETERS:
        raise ValueError('Grid must contain supported recognition/clustering threshold parameters.')
    if any(not isinstance(v, list) or not v or any(type(x) not in (int, float) or not math.isfinite(x)
            for x in v) for v in grid.values()):
        raise ValueError('Each grid parameter requires a nonempty array of finite numbers.')
    # Refuse oversized Cartesian products before materializing or running trials.
    if math.prod(len(v) for v in grid.values()) + 1 > maximum:
        raise ValueError('Grid exceeds max_trials (including baseline).')
    result, patches, skipped, seen = {}, {}, [], set()
    keys = sorted(grid)
    for values in product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, values))
        patch = proposal(config, changes)
        fingerprint = digest(patch['yaml_patch'])
        if not patch['changes'] or fingerprint in seen:
            continue
        seen.add(fingerprint)
        try:
            variant = patched(config, changes)
        except ValueError as error:
            skipped.append(dict(yaml_patch=patch['yaml_patch'], reason=str(error)))
            continue
        name = 'trial-' + str(len(result)+1)
        result[name], patches[name] = variant, patch
    return result, patches, skipped


def truth_split(labels):
    if not isinstance(labels, dict) or any(not isinstance(k, str) or not k or
            not isinstance(v, str) or not v.strip() for k, v in labels.items()):
        raise ValueError('Labels must map observation IDs to explicit nonempty truth labels.')
    people = sorted(set(labels.values()), key=lambda p: (hashlib.sha256(p.encode()).hexdigest(), p))
    train = set(people[::2])
    return ({k: v for k, v in labels.items() if v in train},
            {k: v for k, v in labels.items() if v not in train})


def _objective(metrics):
    return (metrics['pair_recall'] or 0., metrics['coverage'] or 0.,
            -metrics['false_merge_pairs'], -metrics['extra_fragments'])


def tune(config, grid, *, labels=None, scenes=None, scope=None, max_trials=64, max_false_merge_pairs=0):
    if type(max_false_merge_pairs) is not int or max_false_merge_pairs < 0:
        raise ValueError('max_false_merge_pairs must be a nonnegative integer.')
    variants, patches, skipped = configurations(config, grid, max_trials)
    labels = {} if labels is None else labels
    train, holdout = truth_split(labels)
    # Both partitions need positive and negative pairs. One identity per side
    # cannot expose false merges, even when thousands of frames are labelled.
    def usable(part):
        counts = Counter(part.values())
        return len(counts) >= 2 and sum(n*(n-1)//2 for n in counts.values()) > 0
    validation_ready = usable(train) and usable(holdout)
    selected, selected_predictions, baseline_predictions = 'baseline', {}, {}
    training_metrics, labelled_scenes = {}, {}
    def consume(name, observations, retained, predictions):
        nonlocal selected, selected_predictions, baseline_predictions
        if set(labels)-observations.keys():
            raise ValueError('Every truth label must address a selected latest observation.')
        if name == 'baseline':
            labelled_scenes.update((k, observations[k]) for k in labels)
        rows = [dict(observation=k, scene=observations[k], truth=v, predicted=predictions[k]) for k, v in train.items()]
        metric = evaluate_identities(rows) if train else None
        training_metrics[name] = metric
        if name == 'baseline':
            selected_predictions = baseline_predictions = {k: predictions[k] for k in labels}
        elif validation_ready and metric['false_merge_pairs'] <= max_false_merge_pairs:
            current = training_metrics[selected]
            if current['false_merge_pairs'] > max_false_merge_pairs or _objective(metric) > _objective(current):
                selected = name
                selected_predictions = {k: predictions[k] for k in labels}
        return dict(training_metrics=metric)
    result = replay(config, variants, scenes=scenes, scope=scope, consume=consume)
    result.update(kind='recognition_threshold_tuning', labels_sha256=digest(labels) if labels else None,
        labelled_observations=len(labels), skipped_invalid_combinations=skipped,
        max_false_merge_pairs=max_false_merge_pairs, recommendation=None,
        split=dict(unit='truth_identity', training_identities=len(set(train.values())),
                   holdout_identities=len(set(holdout.values())), usable=validation_ready),
        status='unlabelled_comparison' if not labels else 'insufficient_independent_truth')
    for name, patch in patches.items():
        result['trials'][name].update(patch)
    if validation_ready:
        # Holdout never participates in candidate selection. If it rejects the
        # winner, stop instead of testing runners-up against the same holdout.
        def metrics(predictions, truth):
            return evaluate_identities([dict(observation=k, scene=labelled_scenes[k], truth=v,
                predicted=predictions[k]) for k, v in truth.items()])
        held = metrics(selected_predictions, holdout)
        base = metrics(baseline_predictions, holdout)
        overall = metrics(selected_predictions, labels)
        valid = (held['false_merge_pairs'] <= max_false_merge_pairs and
                 overall['false_merge_pairs'] <= max_false_merge_pairs and _objective(held) >= _objective(base))
        result.update(selected_on_training=selected, holdout_metrics=held, baseline_holdout_metrics=base,
                      selected_labelled_metrics=overall,
                      status='baseline_retained' if selected == 'baseline' else
                             'validated_proposal' if valid else 'holdout_rejected')
        if selected != 'baseline' and valid:
            result['recommendation'] = dict(trial=selected, **patches[selected])
    result['limitations'].append('Existing automatic assignments are never ground truth; no source YAML is changed.')
    return result


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate tuning JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def execute(args, config):
    targets = [p.expanduser().resolve() for p in (args.output, args.yaml_output) if p is not None]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        raise FileExistsError('Tuning outputs must be distinct new files.')
    result = tune(config, read_json(args.grid), labels=read_json(args.labels) if args.labels else None,
        scenes=args.scene_id, scope=args.scope, max_trials=args.max_trials,
        max_false_merge_pairs=args.max_false_merge_pairs)
    result['yaml_exported'] = bool(args.yaml_output and result['recommendation'])
    if result['yaml_exported']:
        with args.yaml_output.expanduser().open('x', encoding='utf-8') as file:
            file.write(result['recommendation']['yaml'])
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        with args.output.expanduser().open('x', encoding='utf-8') as file:
            file.write(output)
    else:
        print(output, end='')
    return 0

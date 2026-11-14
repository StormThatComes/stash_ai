"""Measure person detection, pose and SAM mask ownership on five labelled dance duets."""
import argparse
from collections import Counter
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment

from benchmarks.pedestrian_vtt import overlap
from stash_ai.recognition.body_overlap import foreground_mask
from stash_ai.recognition.config import load
from stash_ai.recognition.detection_parameters import apply_edits
from stash_ai.recognition.pose_mask_capture import capture_masks
from stash_ai.recognition.sapiensid_backend import SapiensIDFaces


DATA = Path(__file__).with_name('data') / 'dance_contact'
JOINTS = {'elbow': (7, 8), 'wrist': (9, 10), 'knee': (13, 14), 'ankle': (15, 16)}
BASELINE = 'stock/stock'
COUNTS = ('people', 'matched', 'extra', 'joints', 'correct_joints', 'masks', 'clean_masks',
          'own_hits', 'own_probes', 'partner_hits', 'partner_probes', 'background_hits', 'background_probes')


def assign_people(labels, people):
    # One dummy column per label permits any number of missed people.
    scores = np.zeros((len(labels), len(people) + len(labels)))
    for i, label in enumerate(labels):
        left, top, right, bottom = label['box']
        diagonal = math.hypot(right-left, bottom-top)
        for j, person in enumerate(people):
            box = person.get('pose_refinement', {}).get('detector_bounds', person['bounds'])
            iou = overlap(label['box'], box)
            if iou < 0.25:
                continue
            face = np.asarray(person['keypoints'][:5])
            valid = face[(face >= 0).all(axis=1)]
            # Person boxes can almost coincide in lifts. Use a separate head label to break ties.
            distance = math.dist(np.median(valid, axis=0), label['head']) / diagonal if len(valid) else 0.5
            scores[i, j] = len(labels) + 1 + 0.3 * iou - min(distance, 1)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return {i: j for i, j in zip(rows, cols) if scores[i, j] > 0}


def correct_joints(label, keypoints):
    left, top, right, bottom = label['box']
    tolerance = 0.08 * math.hypot(right-left, bottom-top)
    correct = 0
    for kind, reference in label['joints'].items():
        estimates = [keypoints[i] for i in JOINTS[kind] if min(keypoints[i]) >= 0]
        if estimates:
            distances = np.linalg.norm(np.array(reference)[:, None] - estimates, axis=2)
            hits = distances <= tolerance
            rows, cols = linear_sum_assignment(hits.astype(int), maximize=True)
            correct += int(hits[rows, cols].sum())
    return correct


def count_hits(points, mask, bounds):
    left, top, right, bottom = bounds
    return sum(int(mask[y-top, x-left]) for x, y in points if left <= x < right and top <= y < bottom)


def mask_samples(mask, bounds, own, partner, background):
    hits = count_hits(own, mask, bounds)
    foreign = count_hits(partner, mask, bounds)
    stray = count_hits(background, mask, bounds)
    return dict(own_hits=hits, partner_hits=foreign, background_hits=stray,
                clean_masks=int(hits >= 0.75 * len(own) and foreign == 0 and stray == 0))


def score(photo, people, capture=None):
    labels = photo['people']
    assigned = assign_people(labels, people)
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(people=len(labels), matched=len(assigned), extra=len(people)-len(assigned))
    reasons = Counter()
    candidates = []
    for index, label in enumerate(labels):
        own = label['foreground']
        counts['own_probes'] += len(own)
        counts['joints'] += sum(map(len, label['joints'].values()))
        if index not in assigned:
            reasons['person_missing'] += 1
            continue
        person = people[assigned[index]]
        counts['correct_joints'] += correct_joints(label, person['keypoints'])
        partner = [p for i, other in enumerate(labels) if i != index for p in other['foreground']]
        admitted = foreground_mask(person)
        if admitted is None:
            details = person.get('pose_refinement', {})
            reason = details.get('reason') or details.get('status') or 'mask_unavailable'
            reasons[reason] += 1
            if capture is not None:
                stages = capture.stages(details)
                stage = 'cleaned' if 'cleaned' in stages else 'raw'
                if stage in stages:
                    mask = capture.mask(details, stage)
                    samples = mask_samples(mask, [0, 0, *photo['size']], own, partner, photo['background'])
                    candidates.append(dict(person=index, reason=reason, stage=stage, **samples))
            continue
        mask, _ = admitted
        samples = mask_samples(mask, person['bounds'], own, partner, photo['background'])
        counts['masks'] += 1
        counts['partner_probes'] += len(partner)
        counts['background_probes'] += len(photo['background'])
        for key, value in samples.items():
            counts[key] += value
    return counts, dict(reasons), candidates


def save_overlay(image, people, capture, path):
    from stash_ai.recognition.detection_audit import _person_color
    from stash_ai.recognition.detection_masks import MaskOptions, append_details, draw_masks, draw_prompts
    from stash_ai.recognition.person_audit import _draw

    options = MaskOptions()
    with image.convert('RGB') as annotated:
        draw_masks(annotated, people, capture, options, _person_color)
        draw_prompts(annotated, people, options)
        for index, person in enumerate(people):
            color = _person_color(index)
            _draw(annotated, person['bounds'], person['keypoints'], labels=False, box_color=color, point_color=color)
        with append_details(image, annotated, people, capture, options, _person_color) as (output, _):
            output.save(path)


def evaluate(config, edits, photos, overlays=None):
    adjusted, _ = apply_edits(config, edits)
    totals = dict.fromkeys(COUNTS, 0)
    frames = {}
    rejected = Counter()
    if overlays is not None:
        overlays.mkdir(parents=True, exist_ok=True)
    with closing(SapiensIDFaces(adjusted, {}, load_detector=False, load_identity=False)) as backend:
        for name, photo in photos.items():
            with Image.open(DATA / f'{name}.jpg') as image:
                if list(image.size) != photo['size']:
                    raise ValueError(f'Image and annotation dimensions differ for {name}')
                with capture_masks(image) as capture:
                    backend.detect(image, faces=[], force_pose=True)
                    counts, reasons, rejected_masks = score(photo, backend.frame_people, capture)
                    if overlays is not None:
                        save_overlay(image, backend.frame_people, capture, overlays / f'{name}.png')
            frames[name] = {'counts': counts, 'missing': reasons, 'rejected_candidates': rejected_masks}
            for key, value in counts.items():
                totals[key] += value
            rejected.update(reasons)
    return {'totals': totals, 'missing': dict(rejected), 'frames': frames}


def rank(counts, edits):
    return (counts['clean_masks'], -counts['partner_hits'], -counts['background_hits'],
            counts['correct_joints'], counts['matched'], -counts['extra'], -len(edits))


def read_profiles(path):
    groups = yaml.safe_load(path.read_text())
    for kind in ('pose', 'mask'):
        if groups[kind].get('stock') != {}:
            raise ValueError(f'{kind}.stock must be an empty mapping for the baseline')
    profiles = {}
    for pose_name, pose in groups['pose'].items():
        for mask_name, mask in groups['mask'].items():
            if pose.keys() & mask.keys():
                raise ValueError('A setting appears in both profile groups')
            profiles[f'{pose_name}/{mask_name}'] = {**pose, **mask}
    return profiles


def run_split(config, split, profiles, photos, overlays):
    results = {}
    for index, (name, edits) in enumerate(profiles.items()):
        print(f'{split} {name}', file=sys.stderr, flush=True)
        directory = overlays / f'{split}-{index:02d}' if overlays else None
        result = evaluate(config, edits, photos, directory)
        if directory:
            result['overlay_directory'] = directory.name
        results[name] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DATA.parents[2] / 'facial_recognition.yml')
    parser.add_argument('--model-dir', type=Path, help='Override execution.models.directory')
    parser.add_argument('--profiles', type=Path, default=DATA / 'profiles.yml', help='Pose and mask sweep YAML')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--overlays', type=Path, help='Optional directory for detection tuner-style PNGs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load(args.config, device=args.device)
    if args.model_dir:
        config = replace(config, model_dir=args.model_dir.resolve())
    photos = json.loads((DATA / 'labels.json').read_text())
    tune = {k: v for k, v in photos.items() if v['split'] == 'tune'}
    holdout = {k: v for k, v in photos.items() if v['split'] == 'holdout'}
    profiles = read_profiles(args.profiles)
    results = run_split(config, 'tune', profiles, tune, args.overlays)
    winner = max(profiles, key=lambda name: rank(results[name]['totals'], profiles[name]))
    finalists = {name: profiles[name] for name in (BASELINE, winner)}
    validation = run_split(config, 'holdout', finalists, holdout, args.overlays)
    report = dict(device=args.device, versions={p: version(p) for p in ('torch', 'rfdetr', 'rfdetr-plus')},
                  config_sha256=sha256(args.config.read_bytes()).hexdigest(), profiles=profiles,
                  winner=winner, tune=results, holdout=validation)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

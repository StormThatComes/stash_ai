"""Measure person detection, pose and SAM mask ownership on labelled dance and skating photographs."""
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
COUNTS = ('people', 'matched', 'extra', 'joints', 'correct_joints', 'masks', 'clean_masks',
          'partner_joint_hits','maskless_partner_joints','maskless_correct_joints','maskless_labelled_joints', 'sided_joints', 'correct_side', 'wrong_side',
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
            distance = 0.5
            if len(valid) and label['head'] is not None:
                distance = math.dist(np.median(valid, axis=0), label['head']) / diagonal
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


def joint_ownership(label, others, keypoints):
    """Score unambiguous partner matches and manually sided anatomical landmarks."""
    left, top, right, bottom = label['box']
    tolerance = .08 * math.hypot(right-left, bottom-top)
    partner = right_side = wrong_side = 0
    for kind, reference in label['joints'].items():
        pair = JOINTS[kind]
        foreign = [xy for other in others for xy in other['joints'].get(kind, [])]
        for joint in pair:
            xy = keypoints[joint]
            if min(xy) >= 0 and foreign and reference:
                own_distance = min(math.dist(xy, p) for p in reference)
                other_distance = min(math.dist(xy, p) for p in foreign)
                partner += int(other_distance <= tolerance and own_distance > tolerance)
        sides = label.get('sides', {}).get(kind, [])
        for point, side in zip(reference, sides):
            expected, opposite = pair if side == 'left' else pair[::-1]
            distances = [math.dist(point,keypoints[k]) if min(keypoints[k])>=0 else math.inf
                         for k in (expected,opposite)]
            right_side += int(distances[0] <= tolerance and distances[1] > tolerance)
            wrong_side += int(distances[1] <= tolerance and distances[0] > tolerance)
    return partner, right_side, wrong_side


def count_hits(points, mask, bounds):
    left, top, right, bottom = bounds
    return sum(int(mask[y-top, x-left]) for x, y in points if left <= x < right and top <= y < bottom)


def mask_samples(mask, bounds, own, partner, background):
    hits = count_hits(own, mask, bounds)
    foreign = count_hits(partner, mask, bounds)
    stray = count_hits(background, mask, bounds)
    return dict(own_hits=hits, partner_hits=foreign, background_hits=stray,
                clean_masks=int(hits >= 0.75 * len(own) and foreign == 0 and stray == 0))


def contour_pixels(photo, people, contours):
    from PIL import Image, ImageDraw
    assigned = assign_people(photo['people'],people)
    full = {}
    for owner, index in assigned.items():
        person = people[index]
        admitted = foreground_mask(person)
        if admitted is not None:
            mask = np.zeros(photo['size'][::-1],bool)
            l,t,r,b = person['bounds']
            mask[t:b,l:r] = admitted[0]
            full[owner] = mask
    results = []
    for outline in contours:
        owner = outline['person']
        with Image.new('1',tuple(photo['size'])) as region:
            ImageDraw.Draw(region).polygon([tuple(p) for p in outline['points']],fill=1)
            pixels = np.asarray(region)
            rivals = [mask for key,mask in full.items() if key!=owner]
            results.append(dict(person=owner,limb=outline['limb'],labelled=int(pixels.sum()),
                covered=int(np.count_nonzero(pixels & full[owner])) if owner in full else 0,
                claimed_by_partner=int(np.count_nonzero(pixels & np.logical_or.reduce(rivals))) if rivals else 0))
    return results


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
        counts['sided_joints'] += sum(map(len, label.get('sides', {}).values()))
        if index not in assigned:
            reasons['person_missing'] += 1
            continue
        person = people[assigned[index]]
        counts['correct_joints'] += correct_joints(label, person['keypoints'])
        others = [other for j,other in enumerate(labels) if j != index]
        hits = joint_ownership(label, others, person['keypoints'])
        for key,value in zip(('partner_joint_hits', 'correct_side', 'wrong_side'), hits):
            counts[key] += value
        partner = [p for i, other in enumerate(labels) if i != index for p in other['foreground']]
        admitted = foreground_mask(person)
        if admitted is None:
            counts['maskless_partner_joints'] += hits[0]
            counts['maskless_correct_joints'] += correct_joints(label,person['keypoints'])
            counts['maskless_labelled_joints'] += sum(map(len,label['joints'].values()))
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
    device = adjusted.data['models']['device']
    cuda = str(device).startswith('cuda')
    if cuda:
        import torch
        torch.cuda.reset_peak_memory_stats(device)
    totals = dict.fromkeys(COUNTS, 0)
    frames = {}
    rejected = Counter()
    if overlays is not None:
        overlays.mkdir(parents=True, exist_ok=True)
    contours = json.loads((DATA/'limb_contours.json').read_text())
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
            if name in contours:
                frames[name]['limb_contours'] = contour_pixels(photo, backend.frame_people, contours[name])
            for key, value in counts.items():
                totals[key] += value
            rejected.update(reasons)
    return {'totals': totals, 'missing': dict(rejected), 'frames': frames,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None}


def read_profiles(path):
    profiles = yaml.safe_load(path.read_text())
    if isinstance(profiles, dict) and set(profiles) == {'pose', 'mask'}:
        # Accept sweep files created for the original dance-contact CLI.
        pose, mask = profiles['pose'], profiles['mask']
        if any(not isinstance(group, dict) or group.get('stock') != {} for group in (pose, mask)):
            raise ValueError('Both sweep axes must include an empty stock mapping')
        crossed = {}
        for pose_name, pose_edits in pose.items():
            for mask_name, mask_edits in mask.items():
                if pose_edits.keys() & mask_edits.keys():
                    raise ValueError('A setting appears in both profile groups')
                name = 'stock' if pose_name == mask_name == 'stock' else f'{pose_name}/{mask_name}'
                crossed[name] = {**pose_edits, **mask_edits}
        return crossed
    if not isinstance(profiles, dict) or profiles.get('stock') != {}:
        raise ValueError('Profiles must include an empty stock mapping for the supplied recognition YAML')
    for name, edits in profiles.items():
        if not isinstance(name, str) or not isinstance(edits, dict):
            raise ValueError('Expected named mappings of recognition YAML overrides')
    return profiles


def coverage(frames):
    counts = [f['counts'] for f in frames.values()]
    return dict(photos=len(counts), with_clean_mask=sum(c['clean_masks'] > 0 for c in counts),
                fully_clean=sum(c['clean_masks'] == c['people'] for c in counts))


def select_profile(results, profiles):
    def rank(name):
        result = results[name]
        counts = result['totals']
        # Compare the worst photo first. A solo and a trio each get one vote.
        per_photo = sorted(f['counts']['clean_masks'] / f['counts']['people'] for f in result['frames'].values())
        return (-counts.get('partner_joint_hits',0), -counts.get('maskless_partner_joints',0),
                -counts.get('wrong_side',0), per_photo, counts['clean_masks'], counts['matched'],
                counts['correct_joints'], counts.get('correct_side',0),
                counts['own_hits'], -len(profiles[name]))

    eligible = [name for name, result in results.items()
                if all(result['totals'][key] == 0 for key in ('partner_hits', 'background_hits', 'extra'))]
    return max(eligible, key=rank, default=None)


def run(config, profiles, photos, overlays=None):
    results = {}
    for index, (name, edits) in enumerate(profiles.items()):
        print(name, file=sys.stderr, flush=True)
        directory = overlays / f'profile-{index:02d}' if overlays else None
        from stash_ai.recognition.profiling import pipeline_scope, pipeline_snapshot
        with pipeline_scope(True):
            result = evaluate(config, edits, photos, directory)
            stats = pipeline_snapshot()
        result['runtime'] = dict(seconds=stats['elapsed_seconds'],
            sam_encoder_forwards=stats['stages'].get('sam_image_embedding', {}).get('calls',0),
            sam_decoder_forwards=stats['counts'].get('sam_decoder_forwards',
                stats['stages'].get('sam_mask_prediction', {}).get('calls',0)))
        result['coverage'] = coverage(result['frames'])
        if directory:
            result['overlay_directory'] = directory.name
        results[name] = result
    return dict(selection='all_photos', winner=select_profile(results, profiles), results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DATA.parents[2] / 'facial_recognition.yml')
    parser.add_argument('--model-dir', type=Path, help='Override execution.models.directory')
    parser.add_argument('--profiles', type=Path, default=DATA / 'profiles.yml', help='Named sets of recognition YAML overrides')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--fit-all', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--overlays', type=Path, help='Optional directory for detection tuner-style PNGs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load(args.config, device=args.device)
    if args.model_dir:
        config = replace(config, model_dir=args.model_dir.resolve())
    photos = json.loads((DATA / 'labels.json').read_text())
    profiles = read_profiles(args.profiles)
    results = run(config, profiles, photos, args.overlays)
    report = dict(device=args.device, versions={p: version(p) for p in ('torch', 'rfdetr', 'rfdetr-plus')},
                  config_sha256=sha256(args.config.read_bytes()).hexdigest(), profiles=profiles, **results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

"""Compare dance-contact profiles across the original fixtures, new tuning photos and held-out sessions."""
import argparse
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import yaml

from benchmarks.benchmark_dance_contact import DATA, read_profiles, run_split
from stash_ai.recognition.config import load


EXTENDED = DATA / 'extended'


def reference_profiles():
    prior = read_profiles(DATA / 'calibration_profiles.yml')
    return {name: prior[key] for name, key in (
        ('stock', 'stock/stock'), ('nms-only', 'close-boxes/stock'),
        ('five-photo-fit', 'close-boxes/four-and-expand'),
    )}


def select_profile(profiles, tune, original):
    stock = tune['stock']['totals']

    def rank(name):
        new = tune[name]['totals']
        old = original[name]['totals']
        return (
            all(new[k] <= stock[k] for k in ('partner_hits', 'background_hits', 'extra')),
            new['clean_masks'], -new['partner_hits'], -new['background_hits'],
            new['matched'], new['correct_joints'], -new['extra'], new['own_hits'],
            old['partner_hits'] == old['background_hits'] == old['extra'] == 0,
            old['clean_masks'], old['matched'], old['correct_joints'], old['own_hits'],
            -len(profiles[name]),
        )

    return max(profiles, key=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DATA.parents[2] / 'facial_recognition.yml')
    parser.add_argument('--profiles', type=Path, default=EXTENDED / 'profiles.yml',
                        help='Additional named sets of public YAML overrides')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--overlays', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = load(args.config, device=args.device)
    if args.model_dir:
        config = replace(config, model_dir=args.model_dir.resolve())
    references = reference_profiles()
    candidates = yaml.safe_load(args.profiles.read_text())
    if references.keys() & candidates.keys():
        parser.error('Candidate profile names must be distinct from the three reference profiles')
    profiles = {**references, **candidates}
    original = json.loads((DATA / 'labels.json').read_text())
    new = json.loads((EXTENDED / 'labels.json').read_text())
    results = {}
    splits = {
        split: {f'extended/{name}': photo for name, photo in new.items() if photo['split'] == split}
        for split in ('tune', 'holdout')
    }
    results['tune'] = run_split(config, 'tune', profiles, splits['tune'], args.overlays)
    results['original'] = run_split(config, 'original', profiles, original, args.overlays)
    winner = select_profile(profiles, results['tune'], results['original'])
    finalists = {**references, winner: profiles[winner]}
    results['holdout'] = run_split(config, 'holdout', finalists, splits['holdout'], args.overlays)
    report = dict(device=args.device, config_sha256=sha256(args.config.read_bytes()).hexdigest(),
                  versions={p: version(p) for p in ('torch', 'rfdetr', 'rfdetr-plus')},
                  profiles=profiles, winner=winner, **results)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

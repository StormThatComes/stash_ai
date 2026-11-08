"""Run disjoint Windows lifecycle partitions on separate Actions runners.

Full runs interleave the module manifest across two shards. The PR manifest
runs the Windows-specific boundaries and representative lifecycle cases together.
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def selected(suite, shard, root=ROOT, *, profile='full'):
    if suite not in {'core', 'video'} or shard not in {1, 2}:
        raise ValueError('Expected core/video lifecycle suite and shard 1/2')
    if profile not in {'full', 'pr'} or (profile == 'pr' and shard != 1):
        raise ValueError('The Windows PR profile uses only shard 1')
    suffix = '-pr' if profile == 'pr' else ''
    paths = (root / '.github/ci' / f'windows-{suite}{suffix}.txt').read_text().splitlines()
    if len(paths) != len(set(paths)) or len(paths) < 2:
        raise ValueError('Lifecycle manifest must contain distinct test modules')
    return paths if profile == 'pr' else paths[shard - 1::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('core', 'video'))
    parser.add_argument('shard', type=int, choices=(1, 2))
    parser.add_argument('--profile', choices=('full', 'pr'), default='full')
    args, pytest_options = parser.parse_known_args()
    import pytest
    # Verbose node IDs make an interrupted/slow lifecycle visible in live logs.
    paths = selected(args.suite, args.shard, profile=args.profile)
    raise SystemExit(pytest.main(['-v', *paths, *pytest_options]))


if __name__ == '__main__':
    main()

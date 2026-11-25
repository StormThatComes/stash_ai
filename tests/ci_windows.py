"""Run disjoint Windows lifecycle partitions on separate Actions runners.

Full core runs keep the first partition for model smoke and inspector replay,
and split the former second partition across two runners. This separates the
long rebuild and checkpoint suites within the 30-minute job budget. Video runs
still use two shards. The PR manifest runs representative boundaries together.
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARDS = {'core': (1, 2, 3), 'video': (1, 2)}


def selected(suite, shard, root=ROOT, *, profile='full'):
    if suite not in SHARDS or shard not in SHARDS[suite]:
        raise ValueError('Expected core shards 1/2/3 or video shards 1/2')
    if profile not in {'full', 'pr'} or (profile == 'pr' and shard != 1):
        raise ValueError('The Windows PR profile uses only shard 1')
    suffix = '-pr' if profile == 'pr' else ''
    paths = (root / '.github/ci' / f'windows-{suite}{suffix}.txt').read_text().splitlines()
    if len(paths) != len(set(paths)) or len(paths) < 2:
        raise ValueError('Lifecycle manifest must contain distinct test modules')
    if profile == 'pr':
        return paths
    if suite == 'core' and shard != 1:
        # Leave the smoke/inspector partition unchanged. Alternate the former
        # second partition so rebuild and replay checkpoints run independently.
        return paths[1::2][shard - 2::2]
    return paths[shard - 1::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('core', 'video'))
    parser.add_argument('shard', type=int, choices=sorted({s for shards in SHARDS.values() for s in shards}))
    parser.add_argument('--profile', choices=('full', 'pr'), default='full')
    args, pytest_options = parser.parse_known_args()
    import pytest
    # Verbose node IDs make an interrupted/slow lifecycle visible in live logs.
    paths = selected(args.suite, args.shard, profile=args.profile)
    raise SystemExit(pytest.main(['-v', *paths, *pytest_options]))


if __name__ == '__main__':
    main()

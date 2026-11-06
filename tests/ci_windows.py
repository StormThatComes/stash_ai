"""Run disjoint Windows lifecycle partitions on separate Actions runners.

Manifest lines are interleaved to balance module totals measured in Actions run
42243582071. Keep whole modules together; append new files to the shared manifest
and rebalance from the JUnit artifacts when suite timings drift.
"""
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def selected(suite, shard, root=ROOT):
    if suite not in {'core', 'video'} or shard not in {1, 2}:
        raise ValueError('Expected core/video lifecycle suite and shard 1/2')
    paths = (root / '.github/ci' / f'windows-{suite}.txt').read_text().splitlines()
    if len(paths) != len(set(paths)) or len(paths) < 2:
        raise ValueError('Lifecycle manifest must contain distinct test modules')
    return paths[shard - 1::2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=('core', 'video'))
    parser.add_argument('shard', type=int, choices=(1, 2))
    args, pytest_options = parser.parse_known_args()
    import pytest
    # Verbose node IDs make an interrupted/slow lifecycle visible in live logs.
    raise SystemExit(pytest.main(['-v', *selected(args.suite, args.shard), *pytest_options]))


if __name__ == '__main__':
    main()

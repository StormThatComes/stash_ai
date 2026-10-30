"""Compare time-to-quality and throughput under a shared CPU budget."""
import argparse
import json
from pathlib import Path
import tempfile
from .benchmark_merge_solver import CASES, fixture, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus', nargs='+', type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument('--trials', nargs='+', type=int, default=[1])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=['face', 'disjoint', 'body_modes'])
    parser.add_argument('--max-trials', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for repeat in range(args.repeats):
            for case in args.cases:
                config, goals = fixture(Path(directory)/str(repeat)/case, case)
                for cpus in args.cpus:
                    for trials in args.trials:
                        if trials > cpus:
                            continue
                        result = measure(config, goals, args.max_trials,
                            execution=dict(cpus=cpus, trials=trials, profile=True))
                        rows.append(dict(repeat=repeat, case=case, cpus=cpus, trials=trials, **result))
    print(json.dumps(dict(format=1, scope='synthetic retained evidence; excludes inference', results=rows), indent=2))


if __name__ == '__main__':
    main()

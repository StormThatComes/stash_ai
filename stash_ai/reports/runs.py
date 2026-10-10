"""Offline run history and weighted profiling aggregates for comparable runs."""
from collections import Counter

from stash_ai.reports.io import write_json_report
from stash_ai.reports.runtime import offline_report, report_history
from stash_ai.runtime.profile import combine, describe, empty_sample
from stash_ai.signatures import digest
from stash_ai.storage.store import scope_for


def work_profile(run):
    """Legacy profiles remain readable, with explicit unknown frame accounting."""
    counts = run.get('counts', {})
    return dict(
        outcomes={model: dict(sorted(values.items()))
                  for model, values in sorted(run.get('outcomes', {}).items())},
        processed_frames=counts.get('inference_processed_frames'),
        restored_frames=counts.get('checkpoint_restored_frames'),
    )


def build_report(rows):
    rows, groups = list(rows), {}
    for run in rows:
        # Failed/interrupted/running work must not lower successful-run averages.
        partial = run.get('partial', run.get('finished_at') is None and run['status'] == 'running')
        workload = run.get('workload', {})
        work = work_profile(run)
        key = digest([run['command'], run['status'], partial, run['configuration']['fingerprint'], run['models'], workload, work])
        group = groups.setdefault(key, dict(command=run['command'], status=run['status'],
            configuration_fingerprint=run['configuration']['fingerprint'], models=run['models'],
            partial=partial, workload=workload, work_profile=work, runs=0, wall_seconds=0.0, cpu_seconds=0.0, counts=Counter(), outcomes={}, samples={}))
        group['runs'] += 1
        group['wall_seconds'] += run['wall_seconds'] or 0.0
        group['cpu_seconds'] += run['cpu_seconds'] or 0.0
        group['counts'].update(run['counts'])
        for model, counts in run['outcomes'].items():
            group['outcomes'].setdefault(model, Counter()).update(counts)
        for metric in run['metrics']:
            metric_key = tuple(metric[name] for name in ('kind', 'name', 'model', 'work', 'outcome', 'unit'))
            combine(group['samples'].setdefault(metric_key, empty_sample()), metric)
    for group in groups.values():
        group['mean_wall_seconds'] = group['wall_seconds'] / group['runs']
        group['mean_cpu_seconds'] = group['cpu_seconds'] / group['runs']
        group['metrics'] = [dict(zip(('kind', 'name', 'model', 'work', 'outcome', 'unit'), key), **describe(value))
                            for key, value in sorted(group.pop('samples').items())]
    return dict(format=1, runs=rows, groups=list(groups.values()),
        notes=['Means use total elapsed seconds / total calls, not a mean of run means.',
               'p50/p95 are upper-bucket estimates from bounded logarithmic histograms.',
               'Worker/thread time can overlap; sums of stages are not run wall time.',
               'CPU seconds cover the parent process; worker CPU is reported separately when measured.',
               'A partial running record has no recorded finish; last_progress_at marks its last saved aggregates. It may be active or killed.',
               'Selection digests, code/package provenance, observed execution metadata and verified-input workloads separate comparison groups when available.',
               'Fresh, checkpoint resume and reuse outcomes and performed/restored frame counts separate averages.',
               'Legacy records missing provenance/workload/frame accounting remain readable; their comparability is limited.',
               'Groups describe only the selected, most recent matching runs.'])


@offline_report('Run history report')
def run(app, args):
    with report_history(args) as store:
        report = build_report(store.rows(scope_for(app.STASH_URL), limit=args.run_limit,
            command=args.run_command, status=args.run_status, model=args.run_model))
    destination = write_json_report(report, args.run_report)
    print('Run history: ' + str(len(report['runs'])) + ' runs')
    for item in report['runs']:
        print(f"  {item['started_at']} {item['command']} {item['status']}: "
              + (f"{item['wall_seconds']:.3f}s" if item['wall_seconds'] is not None else 'unfinished'))
    for group in report['groups']:
        print(f"  Profile {group['command']}/{group['status']} ({group['runs']} runs): "
              f"mean wall={group['mean_wall_seconds']:.3f}s")
        for metric in sorted(group['metrics'], key=lambda row: -row['inclusive_seconds'])[:12]:
            print(f"    {metric['kind']}/{metric['model'] or 'all'}/{metric['name']}/{metric['work'] or 'all'}/{metric['outcome']}: "
                  f"calls={metric['calls']}, total={metric['inclusive_seconds']:.3f}s, "
                  f"mean={metric['mean_seconds']:.6f}s, p95~={metric['p95_seconds']:.6f}s")
    print('Report: ' + str(destination))
    return 0

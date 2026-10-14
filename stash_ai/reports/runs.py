"""Offline run history and weighted profiling aggregates for comparable runs."""
from collections import Counter

from stash_ai.reports.io import write_json_report
from stash_ai.reports.runtime import offline_report, report_history
from stash_ai.runtime.profile import combine, describe, empty_sample
from stash_ai.signatures import digest
from stash_ai.storage.store import scope_for

METRIC_KEYS = ('kind', 'name', 'model', 'work', 'outcome', 'unit')


from .run_validation import finite_nonnegative, sanitize


def valid_metric(metric):
    if not isinstance(metric, dict):
        return False
    for key in METRIC_KEYS:
        value = metric.get(key)
        if not isinstance(value, str) and not (key in ('model', 'work', 'unit') and key in metric and value is None):
            return False
    for key in ('calls', 'errors', 'units'):
        if type(metric.get(key)) is not int or metric[key] < 0:
            return False
    for key in ('seconds', 'inclusive_seconds', 'max_seconds'):
        value = metric.get(key)
        if not finite_nonnegative(value):
            return False
    minimum, maximum = metric.get('min_seconds'), metric['max_seconds']
    if 'min_seconds' not in metric or (minimum is None and metric['calls']):
        return False
    if minimum is not None and (not finite_nonnegative(minimum) or minimum > maximum):
        return False
    histogram = metric.get('histogram')
    return (isinstance(histogram, list) and len(histogram) == 64
            and all(type(value) is int and value >= 0 for value in histogram)
            and sum(histogram) == metric['calls'] and metric['errors'] <= metric['calls'])


def work_profile(run):
    """Legacy profiles remain readable, with explicit unknown frame accounting."""
    counts = run.get('counts', {})
    counts = counts if isinstance(counts, dict) else {}
    outcomes = run.get('outcomes', {})
    outcomes = outcomes if isinstance(outcomes, dict) else {}
    return dict(
        outcomes={model: dict(sorted(values.items()))
                  for model, values in sorted(outcomes.items()) if isinstance(values, dict)},
        processed_frames=counts.get('inference_processed_frames'),
        restored_frames=counts.get('checkpoint_restored_frames'),
    )


def build_report(rows):
    rows, groups = list(rows), {}
    warnings, warning_counts, reported_rows = [], Counter(), []
    def warn(code, index, metric_index=None):
        warning_counts[code] += 1
        if len(warnings) < 20:
            warnings.append(dict(code=code, run_index=index,
                                 **({'metric_index': metric_index} if metric_index is not None else {})))
    for index, raw in enumerate(rows):
        run, invalid_record = sanitize(raw, lambda code: warn(code, index))
        if run is None:
            continue
        # Failed/interrupted/running work must not lower successful-run averages.
        partial = run.get('partial', run.get('finished_at') is None and run['status'] == 'running')
        workload = run.get('workload', {})
        work = work_profile(run)
        configuration = run.get('configuration')
        fingerprint = configuration.get('fingerprint') if isinstance(configuration, dict) else None
        comparable = isinstance(fingerprint, str) and bool(fingerprint)
        if not comparable:
            fingerprint = None
            warn('unknown_configuration_fingerprint', index)
        comparable = comparable and not invalid_record
        # Unknown fingerprints describe isolated runs, not a verified cohort.
        identity = fingerprint if comparable else ['unverified-run', index]
        key = digest([run['command'], run['status'], partial, identity, run['models'], workload, work])
        if key in groups and any(run[field] is not None and not finite_nonnegative(
                groups[key][field] + run[field]) for field in ('wall_seconds', 'cpu_seconds')):
            warn('elapsed_aggregate_overflow', index)
            comparable = False
            invalid_record = True
            key = digest([key, 'unverified-run', index])
        group = groups.setdefault(key, dict(command=run['command'], status=run['status'],
            configuration_fingerprint=fingerprint, configuration_comparable=comparable, models=run['models'],
            partial=partial, workload=workload, work_profile=work, runs=0, wall_seconds=0.0, cpu_seconds=0.0,
            counts=Counter(), outcomes={}, samples={}, invalid_metrics=0,
            wall_samples=0, cpu_samples=0))
        group['runs'] += 1
        for field, sample_key in (('wall_seconds','wall_samples'), ('cpu_seconds','cpu_samples')):
            if run[field] is not None:
                group[field] += run[field]
                group[sample_key] += 1
        group['counts'].update(run['counts'])
        for model, counts in run['outcomes'].items():
            group['outcomes'].setdefault(model, Counter()).update(counts)
        metrics = run.get('metrics', [])
        if not isinstance(metrics, (list, tuple)):
            metrics = [None]
        valid_metrics = []
        invalid_metrics = 0
        for metric_index, metric in enumerate(metrics):
            if not valid_metric(metric):
                group['invalid_metrics'] += 1
                invalid_metrics += 1
                warn('invalid_metric', index, metric_index)
                continue
            valid_metrics.append(metric)
            metric_key = tuple(metric[name] for name in METRIC_KEYS)
            combine(group['samples'].setdefault(metric_key, empty_sample()), metric)
        reported_rows.append({**run, 'metrics': valid_metrics,
                              **({'comparison_valid': False} if invalid_record else {}),
                              **({'invalid_metrics': invalid_metrics} if invalid_metrics else {})})
    for group in groups.values():
        group['mean_wall_seconds'] = group['wall_seconds'] / group['wall_samples'] if group['wall_samples'] else None
        group['mean_cpu_seconds'] = group['cpu_seconds'] / group['cpu_samples'] if group['cpu_samples'] else None
        group['metrics'] = [dict(zip(METRIC_KEYS, key), **describe(value))
                            for key, value in sorted(group.pop('samples').items(),
                                key=lambda item: tuple('' if value is None else value for value in item[0]))]
    return dict(format=1, runs=reported_rows, groups=list(groups.values()), warnings=warnings, warning_counts=dict(warning_counts),
        notes=['Means use total elapsed seconds / total calls, not a mean of run means.',
               'p50/p95 are upper-bucket estimates from bounded logarithmic histograms.',
               'Worker/thread time can overlap; sums of stages are not run wall time.',
               'CPU seconds cover the parent process; worker CPU is reported separately when measured.',
               'A partial running record has no recorded finish; last_progress_at marks its last saved aggregates. It may be active or killed.',
               'Selection digests, code/package provenance, observed execution metadata and verified-input workloads separate comparison groups when available.',
               'Fresh, checkpoint resume and reuse outcomes and performed/restored frame counts separate averages.',
               'Legacy records missing provenance/workload/frame accounting remain readable; their comparability is limited.',
               'Missing/invalid comparison fields form isolated, explicitly incomparable per-run groups. Invalid counters and metrics are excluded with bounded warnings; elapsed means use only available valid samples.',
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
              + (f"{item['wall_seconds']:.3f}s" if item['wall_seconds'] is not None else 'elapsed unknown'))
    for group in report['groups']:
        print(f"  Profile {group['command']}/{group['status']} ({group['runs']} runs): "
              + (f"mean wall={group['mean_wall_seconds']:.3f}s" if group['mean_wall_seconds'] is not None else 'mean wall=unknown'))
        if not group['configuration_comparable']:
            print('    This run has unverified comparison fields and is kept in a separate group.')
        if group['invalid_metrics']:
            print(f"    Excluded invalid metrics: {group['invalid_metrics']}")
        for metric in sorted(group['metrics'], key=lambda row: -row['inclusive_seconds'])[:12]:
            print(f"    {metric['kind']}/{metric['model'] or 'all'}/{metric['name']}/{metric['work'] or 'all'}/{metric['outcome']}: "
                  f"calls={metric['calls']}, total={metric['inclusive_seconds']:.3f}s, "
                  f"mean={metric['mean_seconds']:.6f}s, p95~={metric['p95_seconds']:.6f}s")
    print('Report: ' + str(destination))
    return 0

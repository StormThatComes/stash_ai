"""Render the existing grouped counters, errors and timing summary."""
from collections import Counter


def write_summary(report, title, counts, cache=None):
    groups = (
        ('Scenes/models', ('processed', 'reused', 'skipped', 'errors', 'scenes_synced', 'merge_reused', 'bootstrap_reused', 'checkpoint_resumes')),
        ('Scene tags', ('merged_scene_tags', 'scene_tags_updated', 'shared_scene_tags', 'scene_cache_partial', 'scene_cache_missing')),
        ('Markers', ('merged_markers', 'markers_created', 'markers_updated', 'markers_deleted', 'markers_unchanged')),
        ('Occurrence points', ('occurrence_markers', 'occurrence_missing_timestamps', 'occurrence_limited',
                               'occurrence_manual_locations')),
        ('Cleanup', ('cleanup_fast_path', 'cleanup_resumed', 'cleanup_candidate_scenes', 'blacklisted_tags_deleted',
                     'cleanup_markers_changed_or_gone', 'cleanup_tags_changed_or_gone')),
        ('Shared evidence', ('shared_marker_runs', 'shared_marker_extensions', 'short_extras', 'cross_model_extras')),
    )
    combined = Counter(counts)
    combined.update(report.counts)
    report.write(title)
    if report.threshold_preflight is not None:
        threshold = report.threshold_preflight
        report.write(
            f"  Threshold preflight: checked={threshold['checked']}, "
            f"failed={threshold['failed']}, applied={threshold['applied']}, "
            f"selected={threshold['selected']}, "
            f"skipped_no_evidence={threshold['skipped_no_evidence']}"
        )
        if threshold['reasons']:
            ordered = sorted(threshold['reasons'].items(), key=lambda item: (-item[1], item[0]))
            report.write('  Threshold blockers: ' + ', '.join(
                f'{reason}={count}' for reason, count in ordered))
    printed = set()
    for label, keys in groups:
        present = [key for key in keys if key in combined]
        if present:
            report.write('  ' + label + ': ' + ', '.join(f'{key}={combined[key]}' for key in present))
            printed.update(present)
    rest = sorted(set(combined) - printed)
    if rest:
        report.write('  Other: ' + ', '.join(f'{key}={combined[key]}' for key in rest))
    if cache is not None:
        report.write(f"  RAM cache: peak={cache.peak / 2**30:.2f} GiB, hits={cache.stats['hits']}, "
                   f"misses={cache.stats['misses']}, evictions={cache.stats['evictions']}, "
                   f"RSS sampled peak={cache.rss_peak / 2**30:.2f} GiB, pressure trims={cache.stats['pressure_trims']}")
    if report.timings:
        report.write('  Timings (exclusive): ' + ', '.join(
            f'{key}={value:.3f}s/{report.timing_calls[key]} calls' for key, value in sorted(report.timings.items())))
    for (stage, source, kind), group in report.groups.items():
        report.write(f"  Errors {stage}/{source}/{kind}: {group['count']}; " + ' | '.join(group['examples']))
    if report.overflow:
        report.write(f'  Other errors: {report.overflow}; use --verbose for full diagnostics.')

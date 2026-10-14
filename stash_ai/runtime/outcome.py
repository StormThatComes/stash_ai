"""Content-free terminal outcome from already observed execution boundaries."""
from collections import Counter
from time import monotonic

from stash_ai.commands.modes import Command, CLEANUP_COMMANDS
from stash_ai.signatures import digest


class CommandOutcome:
    def __init__(self, app, args):
        self.app, self.args, self.report = app, args, args.report
        self.started, self.result, self.export = monotonic(), None, None

    def __enter__(self):
        destination = getattr(self.args, 'run_summary_json', None)
        if destination is not None:
            from stash_ai.reports.io import ReservedJsonReport
            self.export = ReservedJsonReport(destination).__enter__()
        return self

    def snapshot(self, kind):
        report, args = self.report, self.args
        with report._timing._timing_lock:
            counts = Counter(report.counts)
        for source in report.count_sources:
            counts.update(source)
        ui = getattr(report, 'console', None)
        status = ('interrupted' if kind is not None and issubclass(kind, KeyboardInterrupt) else
                  'failed' if kind is not None or self.result else 'completed')
        phases, skipped_reasons = {}, {}
        current_phase = 'setup'
        if ui is not None:
            with ui.lock:
                phases, skipped_reasons = dict(ui.phase_outcomes), dict(ui.skipped_reasons)
                current_phase = ui.failure_phase or (ui.phase_name if kind is not None else ui.last_phase) or 'setup'
        if kind is not None and (ui is None or ui.failure_phase is None):
            phases[current_phase] = 'interrupted' if issubclass(kind, KeyboardInterrupt) else 'failed'
        merge = report.workload.get('merge', {})
        telemetry = report.workload.get('merge_telemetry') or {}
        selection = sorted(set(map(str, args.scene_id))) if args.scene_id else None
        preview = bool(self.app.DRY_RUN or args.dry_run)
        threshold = dict(report.threshold_preflight) if report.threshold_preflight is not None else None
        if preview and threshold is not None:
            threshold['planned'] = threshold['applied']
            threshold['applied'] = 0
        recovery = dict(publication=bool(counts['publication_pending'] or getattr(report, 'publication_pending_observed', False)),
                        cleanup=bool(getattr(report, 'cleanup_pending_observed', False)))
        guidance = None
        if status != 'completed':
            if preview:
                guidance = 'Rerun the preview.'
            elif args.command in CLEANUP_COMMANDS and recovery['cleanup']:
                guidance = 'Rerun the same cleanup mode and exclusion policies with the same state DB.'
            elif recovery['publication']:
                guidance = 'Resume pending publication with --merge-only and the same state DB before changing policies.'
            elif args.rethreshold_scenes or args.rethreshold_markers:
                guidance = 'Resolve the reported preflight/application error and rerun the same threshold command.'
            elif args.command is Command.INFERENCE and counts['inference_processed_frames']:
                guidance = 'Rerun with the same state DB and models; default --resume reuses valid checkpoints.'
        return dict(schema_version=1, status=status, command=args.command.value,
                    mode='planned' if preview else 'apply',
                    phase=current_phase, phases=phases, skipped_phase_reasons=skipped_reasons,
                    elapsed_seconds=max(0., monotonic()-self.started),
                    scope=dict(mode='explicit' if selection is not None else 'scope',
                               selected_scene_count=len(selection) if selection is not None else None,
                               selection_fingerprint=digest(selection) if selection is not None else None),
                    provenance=dict(configuration_hashes=dict(args.configuration_snapshot.hashes)),
                    counts=dict(counts),
                    confirmed=dict(inference_frames=counts['inference_processed_frames'],
                                   published_scenes=counts['publication_committed'],
                                   threshold_pairs=counts['threshold_pairs_committed']),
                    mutations=dict(unit='logical API batches', attempted=counts['mutation_attempted'],
                                   confirmed=counts['mutation_confirmed'], failed=counts['mutation_failed'],
                                   outcome_unknown=counts['mutation_unknown'],
                                   pending=max(0, counts['mutation_attempted']-counts['mutation_confirmed']-
                                               counts['mutation_failed']-counts['mutation_unknown']),
                                   unknown_affected_rows=None),
                    known_not_started=dict(merge_decisions=max(0, merge.get('selected', 0)-merge.get('checked', 0)),
                                           worker_chunks=telemetry.get('queued_chunks')),
                    unresolved=dict(merge_scenes=merge.get('unresolved', max(0, merge.get('pending', 0)-
                                    merge.get('published', 0)-merge.get('planned', 0)-merge.get('failed_publications', 0)))),
                    recovery_pending=recovery, resume_guidance=guidance,
                    threshold_preflight=threshold,
                    observations={key: report.workload[key] for key in (
                        'rethreshold_apply', 'rethreshold_models', 'rethreshold_spool', 'rethreshold_unvisited',
                        'merge', 'merge_telemetry', 'cleanup', 'shutdown', 'resources') if key in report.workload},
                    failures=dict(report.issue_counts),
                    exception=kind.__name__ if kind is not None else None,
                    timings_seconds=dict(report.timings))

    def __exit__(self, kind, error, traceback):
        ui = getattr(self.report, 'console', None)
        try:
            if ui is not None:
                if kind is not None and issubclass(kind, KeyboardInterrupt):
                    ui.begin_stopping()
                ui.finish_rendering()
            row = self.snapshot(kind)
            self.report.outcome = row
            if row['status'] != 'completed':
                confirmed, mutations = row['confirmed'], row['mutations']
                messages = [f'Outcome: {row["status"]}; mode={row["mode"]}; '
                    f'published={confirmed["published_scenes"]} scene; thresholds={confirmed["threshold_pairs"]} pair; '
                    f'mutations confirmed={mutations["confirmed"]} failed={mutations["failed"]} unknown={mutations["outcome_unknown"]} batch.']
                if row['resume_guidance']:
                    messages.append(row['resume_guidance'])
                for message in messages:
                    try:
                        self.report.write(message)
                    except Exception as failure:
                        self.report.counts['outcome_console_failures'] += 1
                        if error is not None:
                            error.add_note('Outcome console delivery failed: ' + type(failure).__name__)
                # Include failed console delivery without suppressing the
                # independent durable summary export or changing work counts.
                row = self.snapshot(kind)
                self.report.outcome = row
            if self.export is not None:
                self.export.publish(row)
        except BaseException as failure:
            if error is not None:
                error.add_note('Outcome reporting failed: ' + type(failure).__name__)
            elif isinstance(failure, Exception):
                self.report.optional('outcome', lambda: (_ for _ in ()).throw(failure))
            else:
                raise
        finally:
            if self.export is not None:
                from .cleanup import cleanup
                self.report.optional('outcome_cleanup', cleanup, [('release outcome reservation', self.export.close)],
                                     report=self.report, primary=error)

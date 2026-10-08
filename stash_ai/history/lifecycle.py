"""Non-fatal run telemetry; dry runs and inspections preserve read-only behavior."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import time
import uuid

from stash_ai.commands.modes import Command
from stash_ai.runtime.profile import ProfileMetrics
from stash_ai.runtime.report import RunReport
from stash_ai.signatures import digest
from stash_ai.storage.store import scope_for
from .store import HistoryStore, history_path

AUTOMATIC_COMMANDS = frozenset((Command.INFERENCE, Command.MERGE, Command.CLEANUP,
    Command.CLEANUP_MARKERS, Command.CLEANUP_SCENES, Command.SYNC_HIERARCHY,
    Command.DOWNLOAD, Command.MARKER_GROUPS, Command.PRUNE_TAGS, Command.ABANDON,
    Command.DB_UPDATE, Command.DB_COMPACT))


def now():
    return datetime.now(timezone.utc).isoformat()


def configuration(app, args):
    options = {name: getattr(args, name) for name in (
        'models', 'device', 'offline', 'resume', 'restart', 'force_fetch_vtt', 'db_init', 'markers_only', 'scene_tags_only',
        'checkpoint_batches', 'prefetch_batches', 'prepare_workers', 'merge_workers', 'model_report_workers',
        'ram_cache_gb', 'ram_budget_gb', 'rethreshold_scenes', 'rethreshold_markers', 'force_merge',
        'cleanup_strategy', 'cleanup_page_size', 'cleanup_write_batch_size', 'no_hierarchy',
        'prune_empty_tags', 'prune_prefixes', 'create_all_tags')}
    selected = sorted(set(map(str, args.scene_id))) if args.scene_id else None
    options['selected_scene_count'] = len(selected) if selected is not None else None
    options['selection'] = dict(mode='explicit' if selected is not None else 'scope',
                                digest=digest(selected) if selected is not None else None)
    options['profile_merge'] = args.profile_merge is not None
    options['runtime'] = {name: getattr(app, name) for name in (
        'AGGREGATION', 'TOP_K_FRAMES', 'MAX_FRAMES_PER_SCENE', 'MARKER_MAX_TAGS_PER_FRAME',
        'MARKER_MIN_HITS', 'MARKER_MERGE_GAP_SECONDS', 'MARKER_MATCH_TOLERANCE_SECONDS')}
    files = {}
    for name in ('model_config', 'temporal_policy', 'hierarchy', 'marker_core_exclusions', 'scene_tag_exclusions'):
        try:
            with Path(getattr(args, name)).expanduser().open('rb') as stream:
                files[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        except OSError:
            files[name] = None
    options['files'] = files
    options['environment'] = dict(python=platform.python_version(), system=platform.system(),
                                   machine=platform.machine(), processor=platform.processor())
    from stash_ai.history.provenance import capture
    try:
        options['provenance'] = capture(app.INSTALL_ROOT)
    except Exception as exc:
        options['provenance'] = dict(unavailable=type(exc).__name__)
    options['fingerprint'] = digest(options)
    return options


class RunHistory:
    progress_seconds = 30.0
    progress_units = 32

    def __init__(self, app, args):
        self.app, self.args, self.store = app, args, None
        self.identifier, self.result = uuid.uuid4().hex, None
        self.enabled = (not (app.DRY_RUN or args.dry_run or args.no_run_history)
                        and args.command not in (Command.RUN_REPORT, Command.REUSE_REPORT)
                        and (args.record_run or args.command in AUTOMATIC_COMMANDS))

    def warn(self, exc):
        # Never include URLs, exception payloads or secrets in persisted history.
        try:
            self.args.report.write('Run history unavailable: ' + type(exc).__name__ + '; command continues.')
        except Exception:
            pass

    def close(self):
        store, self.store = self.store, None
        if store is not None:
            try:
                store.close()
            except Exception as exc:
                self.warn(exc)

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            from tqdm import tqdm
            self.args.report = RunReport(self.args.verbose, tqdm.write)
            self.args.report.profile = ProfileMetrics()
            self.args.report.history = self
            import threading
            self.owner_thread = threading.get_ident()
            self.start, self.cpu_start = time.perf_counter(), time.process_time()
            self.last_progress, self.last_units = self.start, 0
            self.store = HistoryStore(history_path(self.args))
            self.store.start(self.identifier, scope_for(self.app.STASH_URL), self.args.command.value,
                             now(), configuration(self.app, self.args))
        except Exception as exc:
            self.close()
            self.warn(exc)
        return self

    def __exit__(self, kind, exc, traceback):
        if not self.enabled:
            return
        if self.store is None:
            return
        try:
            report = self.args.report
            status = ('interrupted' if kind is not None and not issubclass(kind, Exception)
                      else 'error' if kind is not None or self.result else 'finished')
            issues = dict(report.issue_counts)
            if kind is not None:
                issues['command/' + kind.__name__] = issues.get('command/' + kind.__name__, 0) + 1
            counts = Counter(report.counts)
            for source in report.count_sources:
                counts.update(source)
            self.store.finish(self.identifier, now(), status,
                max(0.0, time.perf_counter() - self.start), max(0.0, time.process_time() - self.cpu_start),
                models=report.profile.models, counts=dict(counts), outcomes=report.profile.outcomes,
                metrics=report.profile.export(), issues=issues, workload=report.workload)
        except Exception as failure:
            self.warn(failure)
        finally:
            self.close()

    def progress(self):
        if self.store is None:
            return
        import threading
        if threading.get_ident() != self.owner_thread:
            return  # Preparation workers never touch the parent's SQLite journal.
        try:
            report, moment = self.args.report, time.perf_counter()
            counts = Counter(report.counts)
            for source in report.count_sources:
                counts.update(source)
            units = sum(counts[name] for name in ('processed', 'reused', 'threshold_cache_reused', 'scenes_synced'))
            if moment - self.last_progress < self.progress_seconds and units - self.last_units < self.progress_units:
                return
            self.store.progress(self.identifier, now(), max(0., moment-self.start),
                max(0., time.process_time()-self.cpu_start), models=report.profile.models, counts=dict(counts),
                outcomes=report.profile.outcomes, metrics=report.profile.export(), issues=dict(report.issue_counts),
                workload=report.workload)
            self.last_progress, self.last_units = moment, units
        except Exception as exc:
            self.warn(exc)
            self.close()

    def call(self, operation, *args):
        """Optional diagnostic writes/reads must never affect state or inference."""
        if self.store is None:
            return None
        try:
            return getattr(self.store, operation)(*args)
        except Exception as exc:
            self.warn(exc)
            self.close()
            return None

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
        'models', 'device', 'offline', 'resume', 'restart', 'db_init', 'markers_only', 'scene_tags_only',
        'checkpoint_batches', 'prefetch_batches', 'prepare_workers', 'merge_workers', 'model_report_workers',
        'ram_cache_gb', 'ram_budget_gb', 'rethreshold_scenes', 'rethreshold_markers', 'force_merge',
        'cleanup_strategy', 'cleanup_page_size', 'cleanup_write_batch_size', 'no_hierarchy',
        'prune_empty_tags', 'prune_prefixes', 'create_all_tags')}
    options['selected_scene_count'] = len(set(args.scene_id)) if args.scene_id else None
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
    options['fingerprint'] = digest(options)
    return options


class RunHistory:
    def __init__(self, app, args):
        self.app, self.args, self.store = app, args, None
        self.identifier, self.result = uuid.uuid4().hex, None
        self.enabled = (not (app.DRY_RUN or args.dry_run or args.no_run_history)
                        and args.command is not Command.RUN_REPORT
                        and (args.record_run or args.command in AUTOMATIC_COMMANDS))

    def warn(self, exc):
        # Never include URLs, exception payloads or secrets in persisted history.
        self.args.report.write('Run history unavailable: ' + type(exc).__name__ + '; command continues.')

    def __enter__(self):
        if not self.enabled:
            return self
        from tqdm import tqdm
        self.args.report = RunReport(self.args.verbose, tqdm.write)
        self.args.report.profile = ProfileMetrics()
        self.start, self.cpu_start = time.perf_counter(), time.process_time()
        try:
            self.store = HistoryStore(history_path(self.args))
            self.store.start(self.identifier, scope_for(self.app.STASH_URL), self.args.command.value,
                             now(), configuration(self.app, self.args))
        except Exception as exc:
            if self.store is not None:
                self.store.close()
                self.store = None
            self.warn(exc)
        return self

    def __exit__(self, kind, exc, traceback):
        if not self.enabled:
            return
        report = self.args.report
        if self.store is None:
            return
        status = ('interrupted' if kind is not None and not issubclass(kind, Exception)
                  else 'error' if kind is not None or self.result else 'finished')
        issues = dict(report.issue_counts)
        if kind is not None:
            issues['command/' + kind.__name__] = issues.get('command/' + kind.__name__, 0) + 1
        counts = Counter(report.counts)
        for source in report.count_sources:
            counts.update(source)
        try:
            self.store.finish(self.identifier, now(), status,
                max(0.0, time.perf_counter() - self.start), max(0.0, time.process_time() - self.cpu_start),
                models=report.profile.models, counts=dict(counts), outcomes=report.profile.outcomes,
                metrics=report.profile.export(), issues=issues)
        except Exception as failure:
            self.warn(failure)
        finally:
            self.store.close()

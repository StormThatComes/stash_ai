"""One selected CLI command and declarative groups of command capabilities."""
from dataclasses import dataclass
from enum import Enum


class Command(Enum):
    INFERENCE = 'inference'
    SYNC_HIERARCHY = 'sync_hierarchy_only'
    VALIDATE_HIERARCHY = 'validate_hierarchy'
    CLEANUP = 'cleanup_only'
    CLEANUP_MARKERS = 'cleanup_markers_only'
    CLEANUP_SCENES = 'cleanup_scene_tags_only'
    DOWNLOAD = 'download_models'
    MERGE = 'merge_only'
    MARKER_GROUPS = 'sync_marker_groups_only'
    EXPLAIN = 'explain_tag'
    PRUNE_TAGS = 'prune_empty_tags_only'
    DB_REPORT = 'db_report'
    DB_CHECK = 'db_check'
    DB_UPDATE = 'db_update'
    DB_COMPACT = 'db_compact'
    DIAGNOSTICS = 'anonymous_diagnostics'
    MODEL_REPORT = 'model_report'
    REVIEW_REPORT = 'tag_review_report'
    CACHE_REPORT = 'cache_report'
    RUN_REPORT = 'run_report'
    REUSE_REPORT = 'reuse_report'
    ABANDON = 'abandon_pending'


HIERARCHY_COMMANDS = frozenset((Command.SYNC_HIERARCHY, Command.VALIDATE_HIERARCHY))
CLEANUP_COMMANDS = frozenset((Command.CLEANUP, Command.CLEANUP_MARKERS, Command.CLEANUP_SCENES))
DATABASE_COMMANDS = frozenset((Command.DB_REPORT, Command.DB_CHECK, Command.DB_UPDATE,
                               Command.DB_COMPACT, Command.DIAGNOSTICS))


def select_command(args):
    """Abandon is a modifier when another primary command was selected."""
    for command in Command:
        if command not in (Command.INFERENCE, Command.ABANDON) and getattr(args, command.value):
            return command
    return Command.ABANDON if args.abandon_pending else Command.INFERENCE


EXECUTION_OPTIONS = ('device', 'offline', 'resume', 'restart')
MODEL_OPTIONS = ('models', *EXECUTION_OPTIONS)
PUBLICATION_OPTIONS = ('db_init', 'markers_only', 'scene_tags_only', 'profile_merge', 'force_merge', 'dry_run_plan',
                       'rethreshold_scenes', 'rethreshold_markers', 'abandon_pending',
                       'prune_empty_tags', 'prune_prefixes', 'create_all_tags', 'no_hierarchy')


def supplied(args, fields):
    return any(getattr(args, name) is not None if name == 'resume' else bool(getattr(args, name))
               for name in fields)


@dataclass(frozen=True)
class ReportRule:
    command: Command
    inference_options: tuple[str, ...]
    inference_message: str
    modifiers_message: str
    expand_destination: bool = False

    @property
    def flag(self):
        return '--' + self.command.value.replace('_', '-')


CACHE_RULE = ReportRule(
    Command.CACHE_REPORT, EXECUTION_OPTIONS,
    '--cache-report does not use model execution/resume options.',
    '--cache-report is read-only and checks both outputs; tagging/recovery modifiers do not apply.', True)
REVIEW_RULE = ReportRule(
    Command.REVIEW_REPORT, MODEL_OPTIONS,
    '--tag-review-report uses all saved model evidence; model/inference options do not apply.',
    '--tag-review-report cannot be combined with tagging, threshold, pruning or dry-run modifiers.')
MODEL_RULE = ReportRule(
    Command.MODEL_REPORT, MODEL_OPTIONS,
    '--model-report uses all saved model evidence; model/inference options do not apply.',
    '--model-report cannot be combined with tagging, threshold, pruning or dry-run modifiers.')

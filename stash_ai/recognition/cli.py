"""Explicit finite recognition and retained-identity portrait refresh commands."""
import argparse
from pathlib import Path

from .config import load


def stash_id(value):
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise argparse.ArgumentTypeError('ID must be a positive Stash integer ID')
    return str(int(value))


class RecognitionParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        if result.command == 'run' and not (result.scene_id or result.all_scenes):
            self.error('recognition run requires --scene-id or --all-scenes')
        if result.command == 'report-merge':
            if result.yaml_output and not result.solve:
                self.error('--yaml-output requires --solve')
            if result.yaml_mode == 'diff' and (not result.solve or not (result.yaml_output or result.checkpoint)):
                self.error('--yaml-mode diff requires --solve and --yaml-output or --checkpoint')
            if result.solver_sampler!='heuristic' and not result.solve:
                self.error('--solver-sampler requires --solve')
            if (result.checkpoint or result.resume) and not result.solve:
                self.error('--checkpoint/--resume require --solve')
            if result.resume and not result.checkpoint:
                self.error('--resume requires --checkpoint')
            for selector in ('performer_id', 'identity_id'):
                values = getattr(result, selector)
                if values is not None and len(values) < 2:
                    self.error(f'recognition report-merge requires at least two --{selector.replace("_", "-")} values')
        return result


def _arguments(command, *, retained=False):
    scope = command.add_mutually_exclusive_group()
    scope.add_argument('--scene-id', action='append', type=stash_id,
                       help='Only this scene; repeat for explicit multiple IDs.' if not retained else
                            'Limit identities and portrait evidence to this scene; repeat for multiple IDs.')
    scope.add_argument('--all-scenes', action='store_true',
                       help='Explicit one-time full-library selection.' if not retained else
                            'Use all retained scenes in the face database (the default); never scan scenes.')
    command.add_argument('--face-config', type=Path, help='Face-only YAML; default facial_recognition.yml.')
    command.add_argument('--faces-db', type=Path, help='Override the separate face SQLite; default faces_state.sqlite3.')
    command.add_argument('--device', choices=('cuda', 'cpu', 'auto'), help='Override models.device; default CUDA, CPU is explicit.')
    command.add_argument('--retry-unresolved-images', action='store_true',
                         help='Retry uncertain portrait writes for selected owned performers after fresh image/identity checks.')
    command.add_argument('--progress-style', choices=('auto', 'compact', 'verbose', 'off'), default='auto',
                         help='Performer/candidate progress and ETA; verbose includes quality details; off keeps errors and summary.' if retained else
                              'Scene/frame progress and ETA; verbose includes quality details and the names YAML starter; off keeps errors and summary.')


def performer_set(value):
    return [stash_id(part.strip()) for part in value.split(',')]


def parser():
    root = RecognitionParser(prog='stash_ai.py recognition', allow_abbrev=False,
                                   description='On-demand facial recognition independent of tagging.')
    commands = root.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', allow_abbrev=False, help='Recognize selected scenes and publish eligible people.')
    _arguments(run)
    evaluation = commands.add_parser('evaluate', allow_abbrev=False,
        help='Evaluate labelled retained VTT embeddings offline in isolated RAM; never publish.')
    selection = evaluation.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scene-id', action='append', type=stash_id)
    selection.add_argument('--all-scenes', action='store_true')
    evaluation.add_argument('--labels', type=Path, required=True,
        help='JSON object mapping retained observation IDs to explicit human truth labels.')
    evaluation.add_argument('--compare-config', action='append', type=Path,
        help='Additional YAML with the same model/detection contract; repeatable.')
    evaluation.add_argument('--batch-size', action='append', type=int,
        help='Scenes per replay; 0 means joint. Repeatable; default compares 0 and 1.')
    evaluation.add_argument('--scene-order', action='append', choices=('numeric', 'reverse', 'shuffle'),
        help='Import order; repeatable. Default numeric. Joint rebuild remains canonical.')
    evaluation.add_argument('--shuffle-seed', type=int, default=0)
    evaluation.add_argument('--history', action='append', choices=('cold', 'retained'),
        help='Retained copies external evidence to RAM, excluding all selected scenes. Default cold.')
    evaluation.add_argument('--repeat-runs', type=int, default=1,
        help='Repeat each trial on its own resulting state, reporting each pass.')
    evaluation.add_argument('--retune-from-baseline', action='store_true',
        help='Run baseline first inside each comparison trial, then apply its changed policy.')
    evaluation.add_argument('--face-config', type=Path)
    evaluation.add_argument('--faces-db', type=Path)
    evaluation.add_argument('--scope')
    evaluation.add_argument('--output', type=Path, help='New JSON output file; stdout if omitted.')
    tuning = commands.add_parser('tune', allow_abbrev=False,
        help='Search a bounded threshold grid using isolated retained-VTT replays; never publish.')
    selection = tuning.add_mutually_exclusive_group(required=True)
    selection.add_argument('--scene-id', action='append', type=stash_id)
    selection.add_argument('--all-scenes', action='store_true')
    tuning.add_argument('--grid', type=Path, required=True,
        help='JSON object mapping dotted threshold names to arrays of trial values.')
    tuning.add_argument('--labels', type=Path,
        help='Optional observation-to-truth JSON; without labels only partition comparisons are reported.')
    tuning.add_argument('--max-trials', type=int, default=64, help='Cartesian grid limit including baseline (1..1024).')
    tuning.add_argument('--max-false-merge-pairs', type=int, default=0)
    tuning.add_argument('--face-config', type=Path)
    tuning.add_argument('--faces-db', type=Path)
    tuning.add_argument('--scope')
    tuning.add_argument('--output', type=Path)
    tuning.add_argument('--yaml-output', type=Path,
        help='New YAML patch file, written only for a proposal passing the independent truth holdout.')
    sync=commands.add_parser('sync-identities',allow_abbrev=False,
        help='Inspect queued recognition scene-link repairs; --apply publishes current retained assignments without inference.')
    sync.add_argument('--apply',action='store_true',help='Publish queued current assignments and acknowledge successful scenes.')
    sync.add_argument('--scene-id',action='append',type=stash_id,help='Only these queued scene IDs; omitted selects the current model queue.')
    sync.add_argument('--face-config',type=Path,help='Current recognition YAML; must match the queued matching policy.')
    sync.add_argument('--faces-db',type=Path,help='Existing face SQLite database.')
    sync.add_argument('--progress-style',choices=('auto','compact','verbose','off'),default='auto')
    audit = commands.add_parser('audit-alignment', allow_abbrev=False,
        help='Read-only YuNet landmark and aligned 112px crop audit; local biometric images, explicit scenes only.')
    audit.add_argument('--scene-id', action='append', type=stash_id, required=True,
        help='Scene to inspect; repeat for at most 16 explicit IDs.')
    audit.add_argument('--output-dir', type=Path, required=True,
        help='Nonexistent private local directory for sensitive PNGs and manifest.json.')
    audit.add_argument('--max-samples-per-scene', type=int,
        help='Optional positive limit on saved passing faces per scene; omitted exports all passing faces.')
    audit.add_argument('--cue-index', action='append', type=int,
        help='0-based index in selected VTT cues to inspect; repeat to target specific frames.')
    audit.add_argument('--face-config', type=Path, help='Recognition YAML used for detector and quality gates.')
    audit.add_argument('--device', choices=('cuda','cpu','auto'),
        help='YuNet detector execution device; recognizer weights are not loaded.')
    portraits = commands.add_parser('portraits', allow_abbrev=False,
        help="Refresh existing identities' portraits without scanning or rebuilding identities.",
        description='Refresh portraits from retained identity memberships and current portrait settings. '
                    'Defaults to all eligible published identities; does not change identities, scene links or names.')
    portraits.add_argument('--performer-id', action='append', type=stash_id,
        help='Only this Stash performer ID; repeat for multiple people. Uses all their retained scenes unless --scene-id limits the evidence.')
    _arguments(portraits, retained=True)
    cleanup = commands.add_parser('cleanup', allow_abbrev=False,
        help='Find recognition-owned Stash performers missing from the face database; deletion requires --apply.',
        description='Compare Stash recognition-owned performers with the face SQLite. '
                    'Report orphan candidates by default; --apply deletes only verified orphans. '
                    'Never delete unmarked/manual performers.')
    cleanup.add_argument('--apply', action='store_true',
        help='Explicitly destroy verified orphan performer profiles in Stash (default is read-only).')
    cleanup.add_argument('--face-config', type=Path, help='Face-only YAML; default facial_recognition.yml.')
    cleanup.add_argument('--faces-db', type=Path, help='Existing face SQLite to compare with Stash; must exist.')
    cleanup.add_argument('--progress-style', choices=('auto', 'compact', 'verbose', 'off'), default='auto')
    identity = commands.add_parser('report-identity', allow_abbrev=False,
        help='Export every retained member face of one identity as PNG with recorded assignment evidence (offline, read-only).')
    selector = identity.add_mutually_exclusive_group(required=True)
    selector.add_argument('--identity-id', help='Internal face identity ID, as shown by recognition report.')
    selector.add_argument('--performer-id', type=stash_id, help='Stash performer ID linked to the identity.')
    identity.add_argument('--output-dir', type=Path, required=True,
        help='New local directory for all retained face PNGs and manifest.json.')
    identity.add_argument('--face-config', type=Path, help='Recognition YAML selecting the recognizer namespace.')
    identity.add_argument('--faces-db', type=Path, help='Existing face SQLite database.')
    identity.add_argument('--scope', help='Stash scope; required if the performer ID occurs in multiple scopes.')
    merge_report = commands.add_parser('report-merge', allow_abbrev=False,
        help='Compare two Stash performers or internal identities and propose verified YAML changes for frozen-catalogue consolidation (offline, read-only).')
    selector = merge_report.add_mutually_exclusive_group(required=True)
    selector.add_argument('--performer-id', action='append', type=stash_id,
        help='Stash performer ID linked to an identity; provide at least twice. Merged predecessor IDs are resolved automatically.')
    selector.add_argument('--performer-set', action='append', type=performer_set,
        help='Comma-separated Stash IDs to merge; repeat for mutually disjoint sets.')
    selector.add_argument('--identity-id', action='append',
        help='Internal face identity ID; provide at least twice. Merged predecessor IDs are resolved automatically.')
    merge_report.add_argument('--face-config', type=Path, help='Recognition YAML selecting the recognizer and current thresholds.')
    merge_report.add_argument('--faces-db', type=Path, help='Existing face SQLite database.')
    merge_report.add_argument('--scope', help='Restrict both identities to this Stash scope; required if a performer ID occurs in multiple scopes.')
    merge_report.add_argument('--output', type=Path, help='New JSON report path; stdout if omitted. Includes copyable YAML patches.')
    merge_report.add_argument('--progress-style', choices=('auto','off'), default='auto')
    merge_report.add_argument('--yaml-output', type=Path, help='Export the best tested configuration, including best-effort or unchanged results; requires --solve.')
    merge_report.add_argument('--yaml-mode', choices=('full', 'diff'), default='full',
        help='YAML export: full configuration (default) or only changed keys relative to the input baseline; diff must be merged into that baseline before use.')
    merge_report.add_argument('--solve', action='store_true', help='Search configurations using full replay of merge goals.')
    merge_report.add_argument('--goal-policy', choices=('identity','strict'), default='identity',
        help='Goal evaluation: identity (default) tolerates bounded noisy membership losses; strict requires every original observation. Applies to solver and set evaluation.')
    merge_report.add_argument('--max-trials', type=int, default=500, help='Maximum replay budget including baseline (default: 500; may stop on stagnation).')
    merge_report.add_argument('--solver-sampler', choices=('heuristic','tpe'), default='heuristic', help='Optional learned proposal lane alongside heuristics; TPE requires requirements-solver.txt.')
    merge_report.add_argument('--solver-seed', type=int, default=0, help='Reproducible optional sampler seed (default: 0).')
    merge_report.add_argument('--checkpoint', type=Path, help='Durable solver journal; writes best YAML alongside it if --yaml-output is omitted.')
    merge_report.add_argument('--resume', action='store_true', help='Resume --checkpoint with identical inputs, code and search settings.')
    merge_report.add_argument('--simulate', action='store_true',
        help='Replay baseline and proposals on all latest compatible scenes in this scope, in private RAM.')
    report = commands.add_parser('report', allow_abbrev=False,
        help='Explain suspected identity fragmentation in one retained scene (offline, read-only).')
    report.add_argument('--scene-id', required=True, type=stash_id)
    report.add_argument('--face-config', type=Path, help='Recognition YAML used for the selected run.')
    report.add_argument('--faces-db', type=Path, help='Existing face SQLite database.')
    report.add_argument('--compare-scene-id', action='append', type=stash_id,
        help='Compare all retained anchor samples with another scene, including withheld observations; repeatable.')
    report.add_argument('--scope', help='Stash scope, required only when the same scene ID occurs in multiple scopes.')
    report.add_argument('--output', type=Path, help='New JSON report path; stdout if omitted.')
    report.add_argument('--limit', type=int, default=20, help='Maximum suspect pairs to include (1..500).')
    report.add_argument('--min-mean', type=float,
                        help='Minimum cross-group cosine mean to include; default match_threshold - 0.20.')
    return root


def main(argv=None, *, services=None):
    args = parser().parse_args(argv)
    config = load(args.face_config, database=getattr(args, 'faces_db', None),
                  device=getattr(args, 'device', None))
    if args.command == 'evaluate':
        from .offline_evaluation import execute
        return execute(args, config)
    if args.command == 'tune':
        from .threshold_tuning import execute
        return execute(args, config)
    if args.command == 'audit-alignment':
        from .alignment_audit import execute as audit
        return audit(args, config, services=services)
    if args.command == 'report':
        from .fragmentation_report import execute
        return execute(args, config)
    if args.command == 'report-identity':
        from .identity_report import execute
        return execute(args, config)
    if args.command == 'report-merge':
        if args.solve or args.performer_set or len(args.performer_id or args.identity_id) > 2:
            from .merge_goals import execute
            return execute(args, config)
        from .merge_report import execute
        return execute(args, config)
    if args.command == 'cleanup':
        from .cleanup_orphans import execute
        return execute(args, config, services=services)
    if args.command == 'sync-identities':
        from .sync_publications import execute
        return execute(args,config,services=services)
    from .run import execute
    return execute(args, config, services=services)

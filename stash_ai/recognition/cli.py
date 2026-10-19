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


def parser():
    root = RecognitionParser(prog='stash_ai.py recognition', allow_abbrev=False,
                                   description='On-demand facial recognition independent of tagging.')
    commands = root.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', allow_abbrev=False, help='Recognize selected scenes and publish eligible people.')
    _arguments(run)
    audit = commands.add_parser('audit-alignment', allow_abbrev=False,
        help='Read-only YuNet landmark and aligned 112px crop audit; local biometric images, explicit scenes only.')
    audit.add_argument('--scene-id', action='append', type=stash_id, required=True,
        help='Scene to inspect; repeat for at most 16 explicit IDs.')
    audit.add_argument('--output-dir', type=Path, required=True,
        help='Nonexistent private local directory for sensitive PNGs and manifest.json.')
    audit.add_argument('--max-samples-per-scene', type=int, default=32,
        help='Maximum saved faces per scene (1..128); automatic even cue coverage.')
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
    config = load(args.face_config, database=args.faces_db, device=getattr(args, 'device', None))
    if args.command == 'audit-alignment':
        from .alignment_audit import execute as audit
        return audit(args, config, services=services)
    if args.command == 'report':
        from .fragmentation_report import execute
        return execute(args, config)
    if args.command == 'cleanup':
        from .cleanup_orphans import execute
        return execute(args, config, services=services)
    from .run import execute
    return execute(args, config, services=services)

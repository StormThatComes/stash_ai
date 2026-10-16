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
    portraits = commands.add_parser('portraits', allow_abbrev=False,
        help="Refresh existing identities' portraits without scanning or rebuilding identities.",
        description='Refresh portraits from retained identity memberships and current portrait settings. '
                    'Defaults to all eligible published identities; does not change identities, scene links or names.')
    portraits.add_argument('--performer-id', action='append', type=stash_id,
        help='Only this Stash performer ID; repeat for multiple people. Uses all their retained scenes unless --scene-id limits the evidence.')
    _arguments(portraits, retained=True)
    return root


def main(argv=None, *, services=None):
    args = parser().parse_args(argv)
    config = load(args.face_config, database=args.faces_db, device=args.device)
    from .run import execute
    return execute(args, config, services=services)

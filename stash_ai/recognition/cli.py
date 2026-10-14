"""The only public recognition operation: explicit finite `recognition run`."""
import argparse
from pathlib import Path

from .config import load


def scene_id(value):
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise argparse.ArgumentTypeError('scene ID must be a positive Stash integer ID')
    return str(int(value))


def parser():
    root = argparse.ArgumentParser(prog='stash_ai.py recognition', allow_abbrev=False,
                                   description='On-demand facial recognition independent of tagging.')
    commands = root.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run', allow_abbrev=False, help='Recognize selected scenes and publish eligible people.')
    scope = run.add_mutually_exclusive_group(required=True)
    scope.add_argument('--scene-id', action='append', type=scene_id, help='Only this scene; repeat for explicit multiple IDs.')
    scope.add_argument('--all-scenes', action='store_true', help='Explicit one-time full-library selection.')
    run.add_argument('--face-config', type=Path, help='Face-only YAML; default facial_recognition.yml.')
    run.add_argument('--faces-db', type=Path, help='Override the separate face SQLite; default faces_state.sqlite3.')
    run.add_argument('--device', choices=('cuda', 'cpu', 'auto'), help='Override models.device; default CUDA, CPU is explicit.')
    run.add_argument('--retry-unresolved-images', action='store_true',
                     help='Retry uncertain portrait writes for selected owned performers after fresh image/identity checks.')
    run.add_argument('--progress-style', choices=('auto', 'compact', 'verbose', 'off'), default='auto',
                     help='Scene/frame progress and ETA; verbose includes quality details and the names YAML starter; off keeps errors and summary.')
    return root


def main(argv=None, *, services=None):
    args = parser().parse_args(argv)
    config = load(args.face_config, database=args.faces_db, device=args.device)
    from .run import execute
    return execute(args, config, services=services)

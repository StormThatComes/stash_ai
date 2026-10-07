"""Explicit offline migration of an interrupted cleanup's historical fingerprint."""
from stash_ai.cleanup.fingerprint import fingerprint
from stash_ai.models import SOURCES
from stash_ai.signatures import digest


def migrate(store, tags, cores, kinds, *, dry_run=False):
    """Verify the full original intent and change only its pending identity."""
    pending = store.cleanup_pending
    target = fingerprint(tags, cores, kinds)
    if pending is None or pending == target:
        return False
    legacy = {'tags': {s: sorted(tags[s]) for s in SOURCES},
              'cores': {s: sorted(cores[s]) for s in SOURCES}, 'marker_blocklist': []}
    if kinds != ['scene', 'markers']:
        legacy['output_kinds'] = kinds
    if pending != digest(legacy):
        raise RuntimeError('Pending cleanup does not match these exclusion policies and output scope.')
    if not dry_run:
        with store.connection:
            changed = store.connection.execute(
                'UPDATE cleanup_state SET pending=? WHERE scope=? AND pending=?',
                (target, store.scope, pending)).rowcount
            if changed != 1:
                raise RuntimeError('Pending cleanup changed during migration.')
        store.cleanup_pending = target
    return True


def main(argv=None, *, runtime=None):
    import argparse
    from pathlib import Path
    from stash_ai.runtime.context import RuntimeContext
    from stash_ai.storage.store import StateStore, scope_for
    app = runtime if runtime is not None else RuntimeContext()
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description='Migrate an interrupted cleanup fingerprint without connecting to Stash.')
    parser.add_argument('--state-db', type=Path, default=app.MERGE_STATE_FILE)
    parser.add_argument('--marker-core-exclusions', type=Path, default=app.MARKER_CORE_EXCLUSIONS_FILE)
    parser.add_argument('--scene-tag-exclusions', type=Path, default=app.SCENE_TAG_EXCLUSIONS_FILE)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--cleanup-only', action='store_true')
    modes.add_argument('--cleanup-markers-only', action='store_true')
    modes.add_argument('--cleanup-scene-tags-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    app.services.validate_config(scene_work=False)
    if not args.state_db.is_file():
        raise FileNotFoundError(args.state_db)
    cores = app.services.load_exclusion_policy(args.marker_core_exclusions, 'excluded_core_tags')
    tags = app.services.load_exclusion_policy(args.scene_tag_exclusions, 'excluded_scene_tags')
    kinds = (['scene', 'markers'] if args.cleanup_only else
             ['markers'] if args.cleanup_markers_only else ['scene'])
    dry_run = app.DRY_RUN or args.dry_run
    with StateStore(args.state_db, scope_for(app.STASH_URL), dry_run, allow_cleanup=True) as store:
        changed = migrate(store, tags, cores, kinds, dry_run=dry_run)
    print(('Would migrate' if dry_run else 'Migrated') + ' pending cleanup fingerprint.'
          if changed else 'No fingerprint migration needed.')
    return 0


if __name__ == '__main__':
    from ._entrypoint import run as entrypoint
    raise SystemExit(entrypoint(main))

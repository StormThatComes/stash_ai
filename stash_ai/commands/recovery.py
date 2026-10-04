"""Abandon interrupted plans using the ordinary state writer lifecycle."""
from stash_ai.storage.recovery import abandon


def run_only(app, args):
    from stash_ai.storage.store import StateStore, scope_for
    with StateStore(args.state_db, scope_for(app.STASH_URL), app.DRY_RUN, allow_cleanup=True) as store:
        stats = abandon(store, args.report.write)
    args.report.finish('Abandon pending' + (' (planned)' if app.DRY_RUN else ''), stats)
    return 0

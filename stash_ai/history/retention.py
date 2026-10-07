"""Atomic retention of completed diagnostics, independent from source state."""
from datetime import datetime, timedelta, timezone
import math
import sqlite3

from stash_ai.history.store import HistoryStore, history_path
from stash_ai.storage.store import scope_for


def prune(store, scope, *, keep_last=None, max_age_days=None, dry_run=True, now=None):
    if keep_last is not None and (type(keep_last) is not int or keep_last < 0):
        raise ValueError('keep_last must be an integer >= 0.')
    if max_age_days is not None and (type(max_age_days) not in (int, float) or
                                   not math.isfinite(max_age_days) or max_age_days <= 0):
        raise ValueError('max_age_days must be finite and positive.')
    if keep_last is None and max_age_days is None:
        raise ValueError('At least one retention limit is required.')
    moment = now or datetime.now(timezone.utc)
    cutoff = (moment - timedelta(days=max_age_days)).isoformat() if max_age_days is not None else None
    db = store.connection
    if db.in_transaction or store.pending:
        raise RuntimeError('History retention requires an idle connection.')
    # A writer cannot finish or add a candidate between selection and deletion.
    db.execute('BEGIN' if dry_run else 'BEGIN IMMEDIATE')
    try:
        rows = db.execute("SELECT id,finished_at FROM runs WHERE scope=? AND status<>'running' "
                          "AND finished_at IS NOT NULL ORDER BY julianday(finished_at) DESC,id DESC",
                          (scope,)).fetchall()
        old = set()
        if cutoff is not None:
            old = {row[0] for row in db.execute("SELECT id FROM runs WHERE scope=? "
                "AND status<>'running' AND finished_at IS NOT NULL AND julianday(finished_at)<julianday(?)",
                (scope, cutoff))}
        candidates = [identifier for position, (identifier, _) in enumerate(rows)
                      if identifier in old or (keep_last is not None and position >= keep_last)]
        decisions = sum(db.execute('SELECT COUNT(*) FROM reuse_decisions WHERE run_id=?',
                                   (identifier,)).fetchone()[0] for identifier in candidates)
        if not dry_run:
            for identifier in candidates:
                db.execute('DELETE FROM reuse_decisions WHERE run_id=?', (identifier,))
                db.execute("DELETE FROM runs WHERE id=? AND scope=? AND status<>'running' "
                           "AND finished_at IS NOT NULL", (identifier, scope))
            db.commit()
        else:
            db.rollback()
        return dict(dry_run=dry_run, candidate_runs=len(candidates), candidate_decisions=decisions,
                    retained_finished_runs=len(rows)-len(candidates), run_ids=candidates,
                    protected_active_runs=True, reuse_proofs_unchanged=True)
    except BaseException:
        db.rollback()
        raise


def run(app, args):
    with HistoryStore(history_path(args), readonly=bool(args.dry_run or app.DRY_RUN)) as store:
        result = prune(store, scope_for(app.STASH_URL), keep_last=args.history_keep_last,
                       max_age_days=args.history_max_age_days, dry_run=bool(args.dry_run or app.DRY_RUN))
    verb = 'Would remove' if result['dry_run'] else 'Removed'
    print(f"{verb} {result['candidate_runs']} finished runs and {result['candidate_decisions']} decisions; "
          f"retained {result['retained_finished_runs']} finished runs.")
    if result['run_ids']:
        print('Run IDs (first 20): ' + ', '.join(result['run_ids'][:20]))
    print('Active runs, reuse proofs and source state preserved.')
    return 0

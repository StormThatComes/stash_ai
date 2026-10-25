"""Bounded deterministic recovery over frozen references and derived decisions."""
import hashlib
from .database import encoded
from .profiling import count


def state(db):
    digest = hashlib.sha256()
    for row in db.execute('SELECT id,person,status FROM candidates ORDER BY id'):
        digest.update(encoded(tuple(row)).encode())
    for row in db.execute('SELECT source,target FROM identity_merges ORDER BY source'):
        digest.update(encoded(tuple(row)).encode())
    return digest.hexdigest()


def converge(db, step, *, maximum=8):
    previous = state(db)
    seen = {previous}
    for _ in range(maximum):
        from stash_ai.runtime.cancellation import current
        token = current()
        if token is not None:
            token.check()
        step()
        count('recovery_passes')
        current_state = state(db)
        if current_state == previous:
            return
        if current_state in seen:
            raise ValueError('Recognition recovery entered a decision cycle.')
        seen.add(current_state)
        previous = current_state
    raise ValueError('Recognition recovery did not converge within its bounded pass budget.')

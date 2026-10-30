"""Detach a scoped diagnostic snapshot before running long offline analysis."""
from contextlib import contextmanager, closing
import sqlite3


@contextmanager
def snapshot(config, identifier, *, scope=None, by_performer=False, other_identifiers=()):
    from .fragmentation_report import _open_readonly
    from .identity_report import _select
    from .models import namespace
    from .replay_simulation import prepared_history
    from .database import FaceDatabase
    from types import SimpleNamespace
    with closing(_open_readonly(config.database)) as source:
        _, person = _select(source, namespace(config),
            None if by_performer else identifier, identifier if by_performer else None, scope)
        for other in other_identifiers:
            _, compared = _select(source, namespace(config),
                None if by_performer else other, other if by_performer else None, scope)
            if compared['scope'] != person['scope']:
                raise ValueError('Identity comparison requires the same Stash scope.')
        installation = FaceDatabase.installation.fget(SimpleNamespace(connection=source))
        from contextlib import ExitStack
        from .report_metrics import phase
        with ExitStack() as retained:
            with phase('Copy retained snapshot'):
                seed=retained.enter_context(prepared_history(source, config, person['scope'], namespace(config), installation))
                seed.row_factory = sqlite3.Row
                # The seed owns all evidence and bindings needed by this report.
                source.close()
            yield seed

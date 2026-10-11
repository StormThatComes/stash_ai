"""Observe logical write acknowledgement without assuming uncertain server state."""
from .console import activity, status
from functools import wraps


def observed(name):
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            return write(getattr(self, 'report', None), name, lambda: method(self, *args, **kwargs))
        return wrapped
    return decorate


def write(report, name, callback, *, rows=1):
    if report is None or not rows:
        return callback()
    report.counts['mutation_attempted'] += 1
    status(report, write=name, write_outcome='pending')
    try:
        with activity(report, 'Stash write: ' + name):
            result = callback()
    except BaseException as error:
        outcome = getattr(error, 'write_outcome', 'unknown')
        if outcome not in ('failed', 'unknown'):
            outcome = 'unknown'
        report.counts['mutation_' + outcome] += 1
        status(report, write_outcome=outcome,
               mutations=_counts(report))
        raise
    report.counts['mutation_confirmed'] += 1
    report.counts['mutation_rows_confirmed'] += rows
    report.counts['mutation_' + name + '_confirmed'] += rows
    status(report, completed=1, write_outcome='confirmed', mutations=_counts(report))
    return result


def _counts(report):
    return ' '.join(f'{name}={report.counts["mutation_" + name]}'
                    for name in ('attempted', 'confirmed', 'failed', 'unknown'))

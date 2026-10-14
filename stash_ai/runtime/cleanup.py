"""Finish every cleanup step without replacing the original failure."""
import sys
from contextlib import contextmanager


@contextmanager
def closing(resource, *, report=None, name='close resource'):
    """Close a resource without replacing an exception from its body."""
    try:
        yield resource
    finally:
        cleanup([(name, resource.close)], report=report)


@contextmanager
def shutdown_step(report, name, primary=None):
    from .console import current_report, activity
    report = report or current_report()
    ui = getattr(report, 'console', None)
    if isinstance(primary or sys.exception(), KeyboardInterrupt) and ui is not None:
        ui.begin_stopping()
    stopping = ui is not None and ui.stopping and not ui.closed
    if not stopping:
        yield
        return
    from time import monotonic
    started = monotonic()
    ui.message('Stopping: ' + name, decorative=True)
    rows = report.workload.setdefault('shutdown', {})
    state = 'completed'
    try:
        with activity(report, name):
            yield
    except BaseException:
        state = 'failed'
        raise
    finally:
        if name in rows or len(rows) < 16:
            rows[name] = dict(status=state, seconds=monotonic()-started)


def cleanup(callbacks, *, report=None, primary=None):
    primary = primary if primary is not None else sys.exception()
    failures = []
    for item in callbacks:
        name, callback = item if isinstance(item, tuple) else (getattr(item, '__name__', 'close resource'), item)
        try:
            with shutdown_step(report, name, primary):
                callback()
        except BaseException as error:
            failures.append(error)
            if getattr(report, 'counts', None) is not None:
                report.counts['cleanup_failures'] += 1
    if not failures:
        return
    if primary is not None:
        for error in failures:
            primary.add_note(f'Cleanup also failed: {type(error).__name__}: {error}')
            if report is not None:
                try:
                    write = getattr(report, 'write', None) or report.detail
                    write(f'Cleanup also failed: {type(error).__name__}.')
                except BaseException as diagnostic_error:
                    primary.add_note(f'Cleanup diagnostic also failed: {type(diagnostic_error).__name__}: {diagnostic_error}')
        return
    for error in failures[1:]:
        failures[0].add_note(f'Additional cleanup failure: {type(error).__name__}: {error}')
    raise failures[0]

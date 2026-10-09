"""Finish every cleanup step without replacing the original failure."""
import sys


def cleanup(callbacks, *, report=None, primary=None):
    primary = sys.exception() or primary
    failures = []
    for callback in callbacks:
        try:
            callback()
        except BaseException as error:
            failures.append(error)
    if not failures:
        return
    if primary is not None:
        for error in failures:
            primary.add_note(f'Cleanup also failed: {type(error).__name__}: {error}')
            if report is not None:
                try:
                    report.detail(f'Cleanup also failed: {type(error).__name__}: {error}')
                except BaseException as diagnostic_error:
                    primary.add_note(f'Cleanup diagnostic also failed: {type(diagnostic_error).__name__}: {diagnostic_error}')
        return
    for error in failures[1:]:
        failures[0].add_note(f'Additional cleanup failure: {type(error).__name__}: {error}')
    raise failures[0]

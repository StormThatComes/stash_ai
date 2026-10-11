"""Optional phase progress and durable phase outcomes for offline maintenance."""
from stash_ai.runtime.console import progress as console_bar
from contextlib import contextmanager
import time


@contextmanager
def maintenance_progress(label, *, show=False, callback=None, report=None):
    bar, previous, active = None, None, None
    rows = []
    def finish(status):
        nonlocal active
        if active is not None:
            active['status'] = status
            active['seconds'] = max(0., time.perf_counter() - active.pop('_start'))
            active = None
    def record(phase, completed, total):
        nonlocal active
        if active is None or active['phase'] != phase:
            finish('completed')
            active = dict(phase=phase, completed=completed, total=total,
                          status='running', _start=time.perf_counter())
            rows.append(active)
        active.update(completed=completed, total=total)
    def notify(phase, completed, total):
        nonlocal bar, previous
        record(phase, completed, total)
        if callback is not None:
            callback(phase, completed, total)
        if not show:
            return
        if bar is None:
            from tqdm import tqdm
            bar = console_bar(tqdm, total=total, desc=label, unit='item', dynamic_ncols=True)
        elif phase != previous:
            bar.reset(total=total)
        if phase != previous:
            bar.set_postfix_str(phase, refresh=True)
        bar.update(completed - bar.n)
        previous = phase
    status = 'completed'
    try:
        yield notify
    except BaseException as error:
        status = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        raise
    finally:
        finish(status)
        if report is not None:
            def persist():
                report.workload.setdefault('maintenance', []).append(
                    dict(operation=label, status=status, phases=rows))
            report.optional('maintenance_history', persist)
        if bar is not None:
            from stash_ai.runtime.cleanup import cleanup
            cleanup([bar.close])

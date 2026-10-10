"""Optional phase progress for offline maintenance, without no-op scan costs."""
from contextlib import contextmanager


@contextmanager
def maintenance_progress(label, *, show=False, callback=None):
    bar, previous = None, None
    def notify(phase, completed, total):
        nonlocal bar, previous
        if callback is not None:
            callback(phase, completed, total)
        if not show:
            return
        if bar is None:
            from tqdm import tqdm
            bar = tqdm(total=total, desc=label, unit='item', dynamic_ncols=True)
        elif phase != previous:
            bar.reset(total=total)
        if phase != previous:
            bar.set_postfix_str(phase, refresh=True)
        bar.update(completed - bar.n)
        previous = phase
    try:
        yield notify
    finally:
        if bar is not None:
            bar.close()

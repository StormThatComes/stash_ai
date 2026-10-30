"""CLI report lifetime measurements and stage-aware progress outside the solver."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from types import SimpleNamespace
import json
import sys
import time

_active = ContextVar('recognition_report_metrics', default=None)


def command(function):
    @wraps(function)
    def wrapped(args, config):
        from .trial_metrics import Metrics
        state = SimpleNamespace(metrics=Metrics(getattr(args,'profile_solver',False)),
                                progress=getattr(args,'progress_style','off')!='off',depth=0)
        token = _active.set(state)
        outcome = 'completed'
        try:
            code=function(args,config)
            outcome='interrupted' if code==130 else 'completed' if code==0 else 'incomplete'
            return code
        except BaseException as error:
            outcome = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            if state.metrics.enabled:error.report_profile=state.metrics.snapshot()
            raise
        finally:
            _active.reset(token)
            if state.metrics.enabled:
                # Only this final summary includes serialization, fsync and
                # publication. The JSON report honestly identifies its cutoff.
                try:
                    print('Merge report performance: '+json.dumps(dict(state.metrics.snapshot(),outcome=outcome,
                        scope='complete report execution including output publication'),allow_nan=False),file=sys.stderr,flush=True)
                except OSError:
                    pass  # A closed diagnostic stream must not replace the primary error.
    return wrapped


@contextmanager
def phase(label, *, quiet=False):
    state = _active.get()
    if state is None:
        yield
        return
    from .solver_progress import heartbeat, ReplayProgress, worker_text
    from .progress import observe_updates
    nested = state.depth > 0
    state.depth += 1
    last = [0.,None]
    def progress(name, stage, item):
        now=time.monotonic()
        if now-last[0] >= 1 or last[1] != item['stage']:
            last[:]=[now,item['stage']]
            print('Merge report: '+worker_text(item,now),file=sys.stderr,flush=True)
    try:
        with state.metrics.stage(label):
            if quiet or nested or not state.progress:
                yield
            else:
                with heartbeat('report',progress,label) as live:
                    reporter=ReplayProgress('report',live)
                    with observe_updates(reporter.update):yield
    finally:
        state.depth -= 1


def attach(result):
    state=_active.get()
    if state is not None and state.metrics.enabled:
        result['report_performance']=dict(state.metrics.snapshot(),
            scope='before output serialization; complete execution profile is emitted to stderr')

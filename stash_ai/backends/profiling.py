"""Opt-in backend telemetry; profiling failures never change prediction results."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections import Counter
import json
from pathlib import Path

_current = ContextVar('backend_profiling', default=None)


def active():
    return _current.get()


@contextmanager
def use(report, *, enabled):
    token = _current.set(report if enabled else None)
    try:
        yield
    finally:
        _current.reset(token)


def gpu_events(report, torch):
    if report is None:
        return None
    return report.optional('backend_events', lambda: [torch.cuda.Event(enable_timing=True) for _ in range(3)])


def record_gpu(report, events, model, work, count):
    if report is None or events is None or report.profile is None:
        return
    def record():
        # Output .cpu() already waits for the preceding stream work. Never add
        # a device-wide synchronize solely for telemetry.
        if not events[-1].query():
            report.record_counts(backend_events_not_ready=1)
            return
        for name, first, last in (('gpu_input_transfer', 0, 1), ('gpu_model_compute', 1, 2)):
            seconds = events[first].elapsed_time(events[last]) / 1000.
            report.profile.record('backend', name, seconds, seconds, model=model,
                                  work=work, units=count, unit='frames')
    report.optional('backend_gpu_profile', record)


def summarize_trace(path):
    """Only observed node providers/durations are retained, never graph names."""
    events = json.loads(Path(path).read_text())
    nodes, microseconds = Counter(), Counter()
    for event in events:
        if event.get('cat') != 'Node':
            continue
        provider = event.get('args', {}).get('provider')
        if provider is None:
            continue
        # Provider strings are telemetry dimensions; keep this vocabulary bounded.
        name = provider if provider in ('CPUExecutionProvider', 'CUDAExecutionProvider') else 'other'
        nodes[name] += 1
        microseconds[name] += max(0., float(event.get('dur', 0.)))
    return dict(node_events=dict(nodes), seconds={k:v/1e6 for k,v in microseconds.items()})


def record_onnx(report, session, model):
    if report is None:
        return
    def record():
        observed = summarize_trace(session.end_profiling())
        if report.profile is not None:
            row = report.profile.models.setdefault(model, {})
            row['observed_node_providers'] = observed
            row['node_provider_observations'] = 'observed' if observed['node_events'] else 'unavailable'
        report.record_counts(**{f'ort_{key}_node_events': value for key,value in observed['node_events'].items()})
    report.optional('backend_onnx_profile', record)

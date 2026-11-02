"""Expected construction notices must not hide unrelated warnings or failures."""
from concurrent.futures import ThreadPoolExecutor
import logging

import pytest

from stash_ai.recognition import rfdetr_detector as rf
from stash_ai.runtime import console
from stash_ai.runtime.report import RunReport


def emit(logger, message, *, level=logging.WARNING, module='dinov2'):
    logger.handle(logging.LogRecord(logger.name, level, f'/rfdetr/{module}.py', 1, message, (), None))


@pytest.fixture
def logger(monkeypatch, caplog):
    selected = logging.getLogger('rf-detr')
    monkeypatch.setattr(selected, 'propagate', True)
    caplog.set_level(logging.DEBUG, logger=selected.name)
    caplog.set_level(logging.DEBUG, logger=rf.__name__)
    return selected


@pytest.mark.parametrize('verbose', [False, True])
def test_only_expected_warnings_use_verbose_report(monkeypatch, caplog, logger, verbose):
    messages = []
    report = RunReport(verbose=verbose, write=messages.append)
    monkeypatch.setattr(console, 'current_report', lambda: report)
    expected = sorted(rf._BACKBONE_NOTICES)
    original_filters, handlers, level = list(logger.filters), list(logger.handlers), logger.level
    with rf._backbone_diagnostics():
        for message in expected:
            emit(logger, message)
        emit(logger, 'Unexpected checkpoint warning')
        emit(logger, expected[0], level=logging.ERROR)
        emit(logger, expected[1], module='another_backend')
    assert messages == (['RF-DETR-XL initialization: '+m for m in expected] if verbose else [])
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.WARNING, 'Unexpected checkpoint warning'), (logging.ERROR, expected[0]),
        (logging.WARNING, expected[1])]
    assert logger.filters == original_filters and logger.handlers == handlers and logger.level == level
    emit(logger, expected[0])
    assert caplog.records[-1].levelno == logging.WARNING  # No filtering after construction.


def test_debug_fallback_and_cleanup_after_construction_failure(monkeypatch, caplog, logger):
    monkeypatch.setattr(console, 'current_report', lambda: None)
    before = list(logger.filters)
    message = next(iter(rf._BACKBONE_NOTICES))
    with pytest.raises(RuntimeError, match='model construction failed'):
        with rf._backbone_diagnostics():
            emit(logger, message)
            raise RuntimeError('model construction failed')
    assert logger.filters == before
    assert [(r.name, r.levelno) for r in caplog.records] == [(rf.__name__, logging.DEBUG)]
    emit(logger, message)
    assert caplog.records[-1].levelno == logging.WARNING


def test_same_warning_from_other_thread_stays_visible(monkeypatch, caplog, logger):
    monkeypatch.setattr(console, 'current_report', lambda: None)
    message = next(iter(rf._BACKBONE_NOTICES))
    with rf._backbone_diagnostics(), ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(emit, logger, message).result(timeout=5)
    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ('rf-detr', logging.WARNING, message)]

"""Lifecycle stages remain truthful during reuse, lookahead and failure."""
from io import StringIO

import pytest

from stash_ai.runtime.console import ConsolePresenter, activity, quiet_activity
from stash_ai.runtime.report import RunReport
from tests.helpers.runtime import runtime as app
pytestmark = pytest.mark.usefixtures('reset_progress')


def test_collected_errors_mark_phase_partial_and_lookahead_keeps_current_stage():
    stream, errors = StringIO(), []
    report = RunReport()
    ui = report.console = ConsolePresenter(report.write, stream=stream).start()
    try:
        with ui.phase('merge', errors=lambda: len(errors)):
            with activity(report, 'commit', scene='1'):
                with quiet_activity(), activity(report, 'prepare', scene='2'):
                    assert ui.snapshot()['values']['scene'] == '1'
                    assert ui.snapshot()['values']['stage'] == 'commit'
            errors.append('failed scene')
        assert 'Phase: merge partial' in stream.getvalue()
        assert 'Phase: merge complete' not in stream.getvalue()
    finally:
        ui.close()


def test_merge_only_has_skipped_inference_and_known_selected_total(pipeline, capsys):
    assert app.main([*pipeline.args, '--merge-only']) == 0
    output = capsys.readouterr()
    assert 'Phase: inference skipped (merge-only)' in output.err
    assert 'Read sources: inspected 1/1 scene' in output.err
    assert 'Phase: merge publication complete' in output.err


def test_interrupt_has_partial_frame_count_and_no_inference_completion(pipeline, capsys):
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main([*pipeline.args, '--prefetch-batches', '2'])
    output = capsys.readouterr().err
    assert 'Phase: inference interrupted' in output
    assert 'Phase: inference complete' not in output
    assert 'inspected 4/10 frame' in output

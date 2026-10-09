"""Comparable work, precise status and batch-only cache reuse telemetry."""
from dataclasses import replace
import json

import pytest

from stash_ai.runtime.cancellation import Cancelled
from stash_ai.runtime.diagnostics import MergeDiagnostics
from stash_ai.runtime.report import RunReport
from stash_ai.publication.dry_run import DryRunPlan
from stash_ai.reports.runs import build_report
from tests.helpers.runtime import runtime as app
from tests.test_run_history import history_rows
from tests.test_cached_input_reuse import settings

pytestmark = pytest.mark.usefixtures('reset_progress')


@pytest.mark.parametrize('failure,status', [(RuntimeError, 'error'), (KeyboardInterrupt, 'interrupted'), (Cancelled, 'interrupted')])
def test_profile_and_concrete_plan_share_failure_status(tmp_path, failure, status):
    profile_path, plan_path = tmp_path/'profile.jsonl', tmp_path/'plan.jsonl'
    with pytest.raises(failure):
        with MergeDiagnostics(RunReport(write=lambda _: None), profile_path) as profile:
            with profile.scene('1'):
                raise failure('failure')
    rows = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert rows[-2]['status'] == rows[-1]['status'] == status
    with pytest.raises(failure):
        with DryRunPlan(plan_path):
            raise failure('failure')
    assert json.loads(plan_path.read_text().splitlines()[-1])['type'] == status


def test_fresh_resume_and_reuse_do_not_share_wall_mean():
    def row(action, wall, frames, restored):
        return dict(command='inference', status='finished', configuration={'fingerprint': 'same'},
            models={'wd': {}}, workload={'verified_cues': 100}, wall_seconds=wall, cpu_seconds=0,
            counts={'inference_processed_frames': frames, 'checkpoint_restored_frames': restored},
            outcomes={'wd': {action: 1}}, metrics=[])
    report = build_report([row('fresh', 100, 100, 0), row('checkpoint_resumed', 10, 10, 90),
                           row('reused', 1, 0, 0)])
    assert len(report['groups']) == 3
    assert [group['mean_wall_seconds'] for group in report['groups']] == [100, 10, 1]
    legacy = row('fresh', 100, 100, 0)
    legacy['counts'] = {}
    assert build_report([legacy])['groups'][0]['work_profile']['processed_frames'] is None


@pytest.mark.parametrize('force_fetch', [False, True])
def test_batch_only_change_reports_reuse_without_threshold_updates(pipeline, monkeypatch, tmp_path, force_fetch):
    selected = settings(monkeypatch, tmp_path)
    assert app.main(pipeline.args) == 0
    selected[0] = replace(selected[0], batch_size=8)
    args = [*pipeline.args, *(['--force-fetch-vtt'] if force_fetch else [])]
    assert app.main(args) == 0
    row = history_rows(pipeline.path)[0]
    assert row['outcomes']['wd'] == {'reused': 1}
    assert row['counts']['inference_processed_frames'] == row['counts']['checkpoint_restored_frames'] == 0
    assert len(pipeline.loaded) == 1


def test_resume_counts_consumed_and_restored_frames_separately(pipeline):
    pipeline.fail['wd'] = 4
    with pytest.raises(KeyboardInterrupt):
        app.main(pipeline.args)
    pipeline.fail.clear()
    assert app.main(pipeline.args) == 0
    row = history_rows(pipeline.path)[0]
    assert row['counts']['inference_processed_frames'] == 6
    assert row['counts']['checkpoint_restored_frames'] == 4

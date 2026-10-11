"""Exclusive terminal JSON preserves outcome, evidence and secret boundaries."""
import json

import pytest

from tests.helpers.runtime import runtime as app
pytestmark = pytest.mark.usefixtures('reset_progress')


@pytest.mark.parametrize('style', ['auto', 'off'])
def test_execution_summary_has_confirmed_units_and_content_free_provenance(pipeline, tmp_path, style, monkeypatch):
    destination = tmp_path / 'outcome.json'
    monkeypatch.setattr(app, 'STASH_API_KEY', 'OUTCOME_SECRET')
    assert app.main([*pipeline.args, '--progress-style', style, '--no-run-history',
                     '--run-summary-json', str(destination)]) == 0
    row = json.loads(destination.read_text())
    assert row['schema_version'] == 1 and row['status'] == 'completed'
    assert row['confirmed']['inference_frames'] == 10
    assert row['confirmed']['published_scenes'] == 1
    assert row['mutations']['outcome_unknown'] == 0
    assert row['scope']['selected_scene_count'] == 1
    assert 'OUTCOME_SECRET' not in destination.read_text()
    assert str(tmp_path) not in destination.read_text()
    before = destination.read_bytes()
    with pytest.raises(SystemExit):
        app.main([*pipeline.args, '--run-summary-json', str(destination)])
    assert destination.read_bytes() == before


def test_interrupted_summary_matches_partial_frames_and_stopping(pipeline, tmp_path, capsys):
    pipeline.fail['wd'] = 4
    destination = tmp_path / 'interrupted.json'
    with pytest.raises(KeyboardInterrupt):
        app.main([*pipeline.args, '--run-summary-json', str(destination)])
    row = json.loads(destination.read_text())
    assert row['status'] == 'interrupted' and row['phase'] == 'inference'
    assert row['confirmed']['inference_frames'] == 4
    assert row['confirmed']['published_scenes'] == 0
    assert row['observations']['shutdown']['save partial checkpoint']['status'] == 'completed'
    assert row['resume_guidance']
    output = capsys.readouterr()
    assert output.out.count('Outcome: interrupted') == 1
    assert 'Stopping: save partial checkpoint' in output.err


def test_preview_export_keeps_source_absent_and_never_claims_publication(pipeline, tmp_path):
    pipeline.stash.dry_run = True
    destination = tmp_path / 'preview.json'
    assert app.main([*pipeline.args, '--dry-run', '--run-summary-json', str(destination)]) == 0
    row = json.loads(destination.read_text())
    assert row['status'] == 'completed' and row['mode'] == 'planned'
    assert row['confirmed']['published_scenes'] == 0
    assert row['mutations']['attempted'] == 0
    assert not pipeline.path.exists()


def test_cleanup_unknown_write_export_keeps_confirmed_work_and_recovery(tmp_path, monkeypatch):
    from tests.helpers.cleanup_sql import SQLStash
    from tests.helpers.cleanup import install
    from tests.test_cleanup_sql import args
    from stash_ai.evidence.labels import AI
    stash = SQLStash()
    stash.scene['tags'] = [stash.tag('ai:blocked')]
    stash.marker(AI, 'blocked', '1', manual=True)
    stash.fail_after = 'CleanupDeleteTags'
    install(monkeypatch, stash)
    destination = tmp_path / 'cleanup.json'
    assert app.main([*args(tmp_path, ['blocked']), '--run-summary-json', str(destination)]) == 1
    row = json.loads(destination.read_text())
    assert row['status'] == 'failed'
    assert row['mutations']['confirmed'] == 1 and row['mutations']['outcome_unknown'] == 1
    assert row['recovery_pending']['cleanup'] is True

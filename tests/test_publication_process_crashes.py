"""Recovery after process death does not rerun models or duplicate owned markers."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.helpers.runtime import runtime as app
from stash_ai.runtime.settings import evidence_settings
from stash_ai.storage.store import StateStore, scope_for
from tests.test_cache_readiness import blobs

pytestmark = pytest.mark.usefixtures('reset_merge')


@pytest.mark.parametrize('stage', ['prepare', 'create', 'update', 'scene', 'delete', 'before_commit', 'commit'])
def test_abrupt_exit_then_resume_keeps_results_and_unique_markers(pipeline, tmp_path, stage):
    p = pipeline
    assert app.main(p.args) == 0
    before = blobs(p.path)
    assert len(p.stash.scene['scene_markers']) >= 2
    p.stash.scene['scene_markers'].pop()
    p.stash.scene['scene_markers'][0]['title'] += ' outdated'
    from stash_ai.evidence.labels import AI
    p.stash.marker(AI, 'obsolete', 'stale-marker')
    p.stash.scene['tags'] = []
    scene_path, settings_path = tmp_path/'remote.json', tmp_path/'settings.json'
    scene_path.write_text(json.dumps(dict(tags=p.stash.tags, scene=p.stash.scene)))
    values = evidence_settings(app)
    values['TEMPORAL_POLICY'] = asdict(values['TEMPORAL_POLICY'])
    settings_path.write_text(json.dumps(values))
    result = subprocess.run([sys.executable, '-m', 'tests.helpers.crash_publication',
                             str(p.path), str(scene_path), str(settings_path),
                             scope_for(app.STASH_URL), stage],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert result.returncode == 73, result.stdout + result.stderr
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert store.has_pending_publications() == (stage != 'commit')
    remote = json.loads(scene_path.read_text())
    p.stash.tags, p.stash.scene = remote['tags'], remote['scene']
    loaded = len(p.loaded)
    assert app.main([*p.args, '--merge-only']) == 0
    markers = p.stash.scene['scene_markers']
    assert len({(m['primary_tag']['name'], m['seconds'], m['end_seconds']) for m in markers}) == len(markers)
    assert all(m['id'] != 'stale-marker' for m in markers)
    assert len(p.loaded) == loaded and blobs(p.path) == before
    with StateStore(p.path, scope_for(app.STASH_URL)) as store:
        assert not store.has_pending_publications()

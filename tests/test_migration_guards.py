"""Retired inputs require the bridge before inference or mutation."""
from copy import deepcopy
import importlib.util

import pytest
import yaml

from stash_ai.config import load_tag_hierarchy, vocabulary_digest
from stash_ai.models import WD
from stash_ai.publication.preconditions import require_native_ranges
from stash_ai.evidence.labels import AI
from stash_ai.evidence.marker_identity import OCCURRENCE_TAG
from stash_ai.tools import MIGRATION_BRIDGE_COMMIT
from tests.helpers.stash import Stash

pytestmark = pytest.mark.usefixtures('reset_merge')


def test_hierarchy_v1_requires_bridge_without_changing_file(tmp_path):
    path = tmp_path / 'hierarchy.yml'
    data = dict(version=1, root='WD', category_prefix='wd:category:',
                source=dict(tag_count=1, vocabulary_sha256=vocabulary_digest(['core'])),
                groups={'General': ['core']})
    path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    files = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match=MIGRATION_BRIDGE_COMMIT):
        load_tag_hierarchy(path)
    assert path.read_bytes() == before and set(tmp_path.iterdir()) == files
    data = {key: value for key, value in data.items() if key not in {'root', 'category_prefix'}}
    data['version'] = 2
    path.write_text(yaml.safe_dump(data))
    assert load_tag_hierarchy(path).leaf_parents['core'] == ('ai:category:General',)


def test_occurrence_manual_points_and_native_ranges_are_valid():
    stash = Stash()
    occurrence = stash.marker(AI, 'common', 'occurrence')
    occurrence.update(end_seconds=None)
    occurrence['tags'].append(stash.tag(OCCURRENCE_TAG))
    manual = stash.marker(AI, 'other', 'manual', manual=True)
    manual.update(end_seconds=None)
    native = stash.marker(AI, 'range', 'native')
    native.update(seconds=10, end_seconds=20)
    before = deepcopy(stash.scene)
    require_native_ranges(stash.find_scene('1'))
    assert stash.scene == before and not stash.writes


@pytest.mark.parametrize('module', ['stash_ai.tools.namespace_migration',
                                  'stash_ai.tools.point_migration',
                                  'stash_ai.tools.legacy_namespaces',
                                  'stash_ai.commands.hierarchy_migration'])
def test_retired_modules_are_absent(module):
    assert importlib.util.find_spec(module) is None


def test_ordinary_run_rejects_ai_points_before_inference_and_stash_writes(pipeline, monkeypatch, capsys):
    from tests.helpers.runtime import runtime as app
    p = pipeline
    assert app.main(p.args) == 0
    for marker in p.stash.scene['scene_markers']:
        marker['end_seconds'] = None
    before = deepcopy(p.stash.scene)
    p.stash.writes.clear()
    monkeypatch.setattr(app.services, 'load_tagger', lambda *args: pytest.fail('model loaded before range check'))
    assert app.main(p.args) == 1
    assert MIGRATION_BRIDGE_COMMIT in capsys.readouterr().out
    assert p.stash.scene == before and not p.stash.writes

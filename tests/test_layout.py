"""The package move must preserve entry points and existing data locations."""
from pathlib import Path

from tests.helpers.runtime import runtime as app


def test_only_cli_entry_points_remain_at_repository_root():
    root = Path(__file__).resolve().parents[1]
    assert {path.name for path in root.glob('*.py')} == {'stash_ai.py', 'update_stash_ai.py'}


def test_default_data_and_model_paths_stay_at_installation_root(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    for value in (app.MERGE_STATE_FILE, app.MODELS_CONFIG_FILE, app.TEMPORAL_POLICY_FILE,
                  app.JOYTAG_HIERARCHY_FILE, app.MARKER_CORE_EXCLUSIONS_FILE, app.SCENE_TAG_EXCLUSIONS_FILE):
        assert value.parent == root
    settings = app.services.load_model_settings(app.MODELS_CONFIG_FILE, ['all'], None)
    assert all(config.model_dir.parent == root / 'models' for config in settings)


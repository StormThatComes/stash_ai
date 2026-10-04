"""Model selection and prediction settings must be explicit in custom YAML."""
import pytest
import yaml

from stash_ai.config import load_model_settings
from stash_ai.models import SOURCES


@pytest.mark.parametrize('model', list(SOURCES))
@pytest.mark.parametrize('missing', ['scene_threshold', 'marker_threshold', 'batch_size'])
def test_selected_model_rejects_missing_prediction_settings(tmp_path, model, missing):
    values = dict(scene_threshold=.45, marker_threshold=.70, batch_size=4)
    del values[missing]
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, enabled_models=[model], **{model: values})))
    with pytest.raises(ValueError, match=f'Missing {model} settings: {missing}'):
        load_model_settings(path, None, None)


@pytest.mark.parametrize('model', list(SOURCES))
def test_selected_model_requires_its_section(tmp_path, model):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, enabled_models=[model])))
    with pytest.raises(ValueError, match=f'Missing {model} settings'):
        load_model_settings(path, None, None)


def test_model_selection_has_no_implicit_joytag_fallback(tmp_path):
    path = tmp_path / 'models.yml'
    path.write_text('version: 1\n')
    with pytest.raises(ValueError, match='enabled_models is required'):
        load_model_settings(path, None, None)


def test_cli_selection_uses_explicit_values_without_requiring_other_models(tmp_path):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(dict(version=1, wd=dict(
        scene_threshold=.52, marker_threshold=.73, batch_size=3))))
    settings = load_model_settings(path, ['wd'], 'cpu', offline=True)
    assert len(settings) == 1
    model = settings[0]
    assert (model.scene_threshold, model.marker_threshold, model.batch_size) == (.52, .73, 3)
    assert model.device == 'cpu' and not model.download_missing
    assert model.scene_cache_floor is model.marker_cache_floor is None

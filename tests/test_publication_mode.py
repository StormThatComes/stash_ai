"""One active output engine; historical frozen proof stays replayable."""
from dataclasses import asdict
import pytest

from stash_ai.commands.arguments import parse_args
from stash_ai.evidence.frozen_replay import snapshot_context
from stash_ai.evidence.source_markers import marker_contributions
from stash_ai.evidence.temporal.policy import TemporalPolicy, build_options, load_policy
from tests.helpers.evidence import segment, policies
from tests.helpers.runtime import runtime as app

pytestmark = pytest.mark.usefixtures('reset_temporal')


@pytest.mark.parametrize('argv', [
    ['--output-mode', 'merged'], ['--output-mode', 'separate'], ['--no-temporal'],
])
def test_retired_engine_options_fail_before_any_execution(argv):
    with pytest.raises(SystemExit) as error:
        parse_args(app, argv)
    assert error.value.code == 2


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_yaml_cannot_select_a_retired_engine(tmp_path, enabled):
    path = tmp_path / 'policy.yml'
    path.write_text(f'version: 1\nenabled: {enabled}\n')
    with pytest.raises(ValueError, match='Unknown temporal policy keys: enabled'):
        load_policy(path)


def test_frozen_classic_snapshot_preserves_first_frame_extras():
    options = build_options(app)
    options['policy'] = asdict(TemporalPolicy(enabled=False))
    options['marker_thresholds'] = {source: .8 for source in policies()}
    frozen = snapshot_context(app, options, 'markers')
    records = {'wd': [segment('wd', [
        [('core', .9), ('first', .9)],
        [('core', .9), ('later', .9)],
        [('core', .9), ('later', .9)],
    ])]}
    exclusions = policies(['first', 'later'])
    historical = marker_contributions(frozen, records, exclusions)
    current = marker_contributions(app, records, exclusions)
    assert [(s['start'], s['end'], s['hits'], s['extra_tags']) for s in historical] == [
        (0, 3, 3, ['first'])]
    assert current[0]['extra_tags'] == ['later']
    assert app.TEMPORAL_POLICY.enabled

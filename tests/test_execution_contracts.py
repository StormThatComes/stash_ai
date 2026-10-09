from argparse import Namespace
import hashlib
from pathlib import Path

import pytest

from stash_ai.config import load_unique_yaml
from stash_ai.runtime.configuration import EffectiveConfiguration, FILES
from stash_ai.runtime.reasons import EvidenceError, Reason, PendingPublication
from stash_ai.rethreshold.targets import threshold_failure_reason
from stash_ai.inference.identity_types import ProvenanceIdentity
from stash_ai.inference.provenance import predictive
from stash_ai.runs.model_scene import ModelScene, ScenePreparation, PreparationStatus


def test_configuration_bytes_hash_and_parser_share_snapshot(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('answer: 1\n')
    args = Namespace(**{name: str(path) for name in FILES})
    snapshot = EffectiveConfiguration.capture(args)
    path.write_text('answer: 2\n')
    with snapshot.use():
        first = load_unique_yaml(path, context='test')
        first['answer'] = 9
        path.unlink()
        assert load_unique_yaml(path, context='test') == {'answer': 1}
    assert snapshot.hashes['model_config'] == hashlib.sha256(b'answer: 1\n').hexdigest()
    with pytest.raises(TypeError):
        snapshot.hashes['model_config'] = 'changed'


def test_provenance_adapter_preserves_legacy_predictive_semantics():
    identity = dict(batch_size=8, files=[['/tmp/model', 'abc']],
                    scene_threshold=.5, marker_threshold=.4, retention={'x': 1}, model='m')
    typed = ProvenanceIdentity.from_legacy(identity)
    identity['model'] = 'mutated'
    assert typed.legacy()['model'] == 'm'
    assert typed.predictive().legacy() == predictive(typed.legacy())
    assert typed.predictive(checkpoint=True).legacy()['scene_threshold'] == .5


@pytest.mark.parametrize('code', list(Reason))
def test_reason_does_not_depend_on_message(code):
    from stash_ai.runtime.reasons import LABELS
    assert threshold_failure_reason(EvidenceError(code, 'completely different wording')) == LABELS[code]
    assert threshold_failure_reason(PendingPublication('different pending wording')) == 'pending publication'


def test_preparation_owns_exactly_one_ready_job():
    job = ModelScene('1', {}, [], None, 'signature')
    assert ScenePreparation(PreparationStatus.READY, job).job is job
    with pytest.raises(ValueError):
        ScenePreparation(PreparationStatus.REUSED, job)
    with pytest.raises(ValueError):
        ScenePreparation(PreparationStatus.FAILED)


def test_job_closes_all_resources_and_detaches_even_on_error():
    events = []
    class Resource:
        def __init__(self, name, fails=False):
            self.name, self.fails = name, fails
        def close(self):
            events.append(self.name)
            if self.fails:
                raise RuntimeError('close failed')
    job = ModelScene('1', {}, [], None, 'signature')
    job.batches, job.prepared = Resource('batches', True), Resource('prepared')
    with pytest.raises(RuntimeError, match='close failed'):
        job.close()
    assert events == ['batches', 'prepared']
    assert job.batches is job.prepared is None
    job.close()
    assert events == ['batches', 'prepared']

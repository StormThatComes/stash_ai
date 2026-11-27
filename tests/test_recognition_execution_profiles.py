from pathlib import Path
import shutil
import pytest
import yaml

from stash_ai.recognition.config import load


def test_policy_is_inherited_and_solver_only_overrides_execution(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for name in ('facial_recognition.yml','facial_recognition_execution.yml','facial_recognition_solver.yml'):
        shutil.copy(root/name,tmp_path/name)
    base=tmp_path/'facial_recognition.yml'
    doc=yaml.safe_load(base.read_bytes())
    doc['matching']['face']['match_threshold'] = .79
    base.write_text(yaml.safe_dump(doc))
    regular=load(base)
    solver=load(tmp_path/'facial_recognition_solver.yml')
    assert regular.data['recognition']['match_threshold']==solver.data['recognition']['match_threshold']==.79
    assert regular.data['execution']['rebuild_memory_bytes'] != solver.data['execution']['rebuild_memory_bytes']
    assert load(base,solver_profile=True).data==solver.data
    override=tmp_path/'facial_recognition_solver.yml'
    override.write_text('version: 2\nexecution: {}\nmatching: {}\n')
    with pytest.raises(ValueError):
        load(base,solver_profile=True)


def test_dense_delta_has_an_index_and_verifies_integrity(tmp_path,monkeypatch):
    from stash_ai.recognition import solver_delta_transport as transport
    monkeypatch.setattr(transport,'INLINE_ROWS',2)
    with transport.worker_target(tmp_path,1024**2):
        delta = transport.collect([('a','one'),('b',None),('c','one')])
    changes = delta.read(tmp_path,indexed=True)
    try:
        assert len(changes)==3 and changes['b'] is None
        assert sorted(changes.for_owners({'one'}))==[('a','one'),('c','one')]
    finally:
        changes.close()
    (tmp_path/delta.filename).write_bytes(b'')
    with pytest.raises(ValueError,match='corrupt'):
        delta.read(tmp_path,indexed=True)

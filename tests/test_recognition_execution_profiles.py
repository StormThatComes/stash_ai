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


def test_explicit_legacy_solver_next_to_a_split_policy_keeps_its_own_threshold(tmp_path):
    (tmp_path/'facial_recognition.yml').write_text('version: 2\nexecution_file: facial_recognition_execution.yml\n')
    solver = tmp_path/'facial_recognition_solver.yml'
    solver.write_text('version: 2\nmatching: {face: {match_threshold: 0.85}}\n')
    assert load(solver).data['recognition']['match_threshold'] == .85


def test_sparse_solver_requires_its_recognition_policy(tmp_path):
    solver = tmp_path/'facial_recognition_solver.yml'
    solver.write_text('version: 2\nexecution: {}\n')
    with pytest.raises(ValueError, match='sibling facial_recognition'):
        load(solver)


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


def test_seed_shares_exact_cross_revision_vectors_and_keeps_old_sources(tmp_path):
    from contextlib import closing
    from benchmarks.benchmark_merge_solver import fixture
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.replay_simulation import prepared_history
    from stash_ai.recognition.offline_evaluation import _ram
    from stash_ai.recognition.trial_inputs import prepare
    config,goals = fixture(tmp_path,'face')
    with FaceDatabase(config.database) as state:
        db = state.connection
        scan = db.execute("SELECT * FROM scans WHERE scene='1'").fetchone()
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',('later',*tuple(scan)[1:4],'changed-input',*tuple(scan)[5:]))
        db.execute("INSERT INTO observations SELECT 'new-'||id,'later',cue,seconds,source,rect,geometry,quality,"
                   "rejected,dimensions,embedding,crop FROM observations WHERE scan=?",(scan['id'],))
        vector = bytes(16384)
        for current,key in [(scan['id'],'body-old'),('later','body-new')]:
            db.execute('INSERT INTO body_observations VALUES (?,?,0,0,?,NULL,?,?,NULL,?,NULL)',
                       (key,current,'contract','{}','{}',vector))
        db.commit()
        with prepared_history(db,config,'scope',goals['namespace'],state.installation) as seed:
            assert seed.execute('SELECT COUNT(*) FROM snapshot_vector_links').fetchone()[0]==4
            assert seed.execute("SELECT embedding FROM main.body_observations WHERE id='body-old'").fetchone()[0] is None
            assert seed.execute("SELECT embedding FROM body_observations WHERE id='body-old'").fetchone()[0]==vector
            with closing(_ram(config)) as trial:
                prepare(trial,seed)
                assert trial.execute("SELECT embedding FROM body_observations WHERE id='body-old'").fetchone()[0]==vector
                expected = list(map(tuple,db.execute('SELECT id,embedding FROM observations ORDER BY id')))
                assert list(map(tuple,trial.execute('SELECT id,embedding FROM observations ORDER BY id')))==expected

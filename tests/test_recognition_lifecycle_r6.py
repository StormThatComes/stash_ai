from contextlib import closing
import json
import subprocess
import sys
import time

import pytest

from tests.test_recognition_merge_report import seeded
from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.merge_solver_checkpoint import Journal, AtomicFile


def test_journal_exclusive_across_processes_and_released_after_crash(tmp_path):
    path = tmp_path/'checkpoint.json'
    journal = Journal(path, {'test':1})
    code = ('import sys; from stash_ai.recognition.merge_solver_checkpoint import Journal; '
            'j=Journal(sys.argv[1], {"test":1}, resume=True)')
    child = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, timeout=10)
    assert child.returncode and b'Another solver owns' in child.stderr
    journal.close()
    child = subprocess.Popen([sys.executable, '-c', code+'; print("ready",flush=True); import time; time.sleep(60)', str(path)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert child.stdout.readline() == b'ready\n'
        with pytest.raises(RuntimeError, match='Another solver owns'):
            Journal(path, {'test':1}, resume=True)
    finally:
        child.kill(); child.communicate(timeout=10)
    restored = Journal(path, {'test':1}, resume=True)
    restored.close()


@pytest.mark.parametrize('progress', [None, lambda *args: None])
def test_interrupt_finalization_preserves_completed_trial(tmp_path, monkeypatch, progress):
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.merge_solver import solve
    from stash_ai.recognition.merge_solver_feasibility import Feasibility
    config, _ = seeded(tmp_path)
    def interrupt(*args):raise KeyboardInterrupt
    monkeypatch.setattr(Feasibility, 'report', interrupt)
    with closing(_open_readonly(config.database)) as db:
        result = solve(config, resolve(db,config,[['900','901']]), db, maximum=1,
                       checkpoint=tmp_path/'checkpoint.json', progress=progress)
    assert result['interrupted'] and result['selected_trial'] == 'baseline'
    assert 'baseline' in result['simulation']['trials']
    from stash_ai.recognition.merge_solver_feasibility import lines
    assert lines(result['feasibility'])


def test_full_yaml_retains_vitpose_asset_location(tmp_path):
    from stash_ai.recognition.merge_goal_output import configuration_document
    from stash_ai.recognition.merge_report_policy import proposal
    import yaml
    config,_ = seeded(tmp_path)
    from dataclasses import replace
    from stash_ai.recognition.config import plain
    config=replace(config,data=plain(config.data))
    config.data['sapiensid']['pose']['vitpose_directory']='custom/vitpose'
    exported = yaml.safe_load(configuration_document(config,proposal(config,{}),{}))
    assert exported['sapiensid']['pose']['vitpose_directory'] == str((config.source_path.parent/'custom/vitpose').resolve())


def test_artifact_only_fields_do_not_change_replay_signature(tmp_path):
    from tests.test_recognition_solver_feasibility import body_handoff
    from stash_ai.recognition.database import FaceDatabase
    from stash_ai.recognition.replay_signature import fingerprint
    from stash_ai.recognition.models import namespace
    config,_,_=body_handoff(tmp_path)
    with FaceDatabase(config.database) as state:
        db=state.connection; original=fingerprint(db,'scope',namespace(config))
        db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.crop_status','retained','$.crop_attempt_limit',42)")
        db.execute("UPDATE body_observations SET quality=json_set(quality,'$.crop_status','retained')")
        assert fingerprint(db,'scope',namespace(config)) == original
        db.execute("UPDATE person_evidence SET metadata=json_set(metadata,'$.sharpness',123)")
        assert fingerprint(db,'scope',namespace(config)) != original


def test_atomic_new_report_failure_leaves_no_partial_destination(tmp_path,monkeypatch):
    import os
    target=tmp_path/'report.json'
    def full(*args):raise OSError('disk full')
    monkeypatch.setattr(os,'fsync',full)
    with pytest.raises(OSError,match='disk full'):AtomicFile(target).write('{"complete":true}')
    assert not target.exists() and list(tmp_path.iterdir()) == []


def _uncooperative_worker():
    time.sleep(60)


def test_solver_stops_uncooperative_worker(monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from stash_ai.recognition import solver_shutdown
    monkeypatch.setattr(solver_shutdown,'GRACE_SECONDS',.5)
    pool=ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'))
    future=pool.submit(_uncooperative_worker)
    started=time.monotonic()
    solver_shutdown.shutdown(pool,[future])
    assert future.done() and time.monotonic()-started < 5

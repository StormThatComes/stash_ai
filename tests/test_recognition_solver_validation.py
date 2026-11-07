import json
import sqlite3
from contextlib import closing

import pytest

from stash_ai.recognition.fragmentation_report import _open_readonly
from stash_ai.recognition.solver_validation import validate_winner
from tests.test_recognition_offline_evaluation import fixture


def test_holdout_replays_one_winner_and_removes_only_private_cues(tmp_path):
    cfg, labels = fixture(tmp_path)
    with closing(_open_readonly(cfg.database)) as db:
        scope, namespace = db.execute('SELECT scope,namespace FROM scans LIMIT 1').fetchone()
    manifest = tmp_path/'holdout.json'
    manifest.write_text(json.dumps(dict(database=str(cfg.database), scenes=['1','2'],
        observations={oid:{'truth':label,'slices':['back','entangled']} for oid,label in labels.items()},
        series=[['1',0,0]])))
    goal = dict(scope=scope,namespace=namespace,selected_trial='trial1',best_result={'changes':[]})
    with closing(sqlite3.connect(':memory:')) as training:
        training.execute('CREATE TABLE scans(scope,namespace,scene)')
        result = validate_winner(cfg,training,goal,manifest)
        assert result['metrics']['winner/joint']['back']['observations']==9
        assert result['omissions'][0]['scored']==8
        assert result['omissions'][0]['excluded_labels']==1
        with closing(_open_readonly(cfg.database)) as source:
            assert not source.execute("SELECT 1 FROM observations WHERE rejected='validation_omission'").fetchone()
        training.execute('INSERT INTO scans VALUES (?,?,?)',(scope,namespace,'2'))
        with pytest.raises(ValueError,match='training snapshot'):
            validate_winner(cfg,training,goal,manifest)


def test_back_slice_counts_joins_to_frontal_partner():
    from stash_ai.recognition.solver_validation import _metrics
    labels = {'back':'A','face':'A','partner':'B'}
    tagged = {k:{'slices':['back'] if k=='back' else []} for k in labels}
    scores = _metrics(labels, tagged, dict.fromkeys(labels,('1',0)),
                      {k:{'identity':'mixed'} for k in labels})
    assert scores['back']['pairs_touching_slice'] == dict(true_pairs=1,joined_pairs=2,
        false_merge_pairs=1,recall=1.,precision=.5)

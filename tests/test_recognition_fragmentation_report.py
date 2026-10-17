"""Read-only scene fragmentation reporting must never run inference or change SQLite."""
import json
import math
import sqlite3
import uuid

import numpy as np
import pytest

from stash_ai.recognition.cli import main, parser
from stash_ai.recognition.clustering import matching_policy
from stash_ai.recognition.config import load
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.fragmentation_report import build
from stash_ai.recognition.models import namespace


def seeded(tmp_path):
    config = load(root=tmp_path)
    scan = str(uuid.uuid4())
    policy, ns = matching_policy(config), namespace(config)
    a, b, c = [str(uuid.uuid4()) for _ in range(3)]
    vectors = {
        a: np.asarray([1., 0., 0., 0.], dtype=np.float32),
        b: np.asarray([.7, math.sqrt(1 - .7 ** 2), 0., 0.], dtype=np.float32),
        c: np.asarray([0., 0., 1., 0.], dtype=np.float32),
    }
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute('INSERT INTO scans VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   (scan, 'scope', '123', ns, 'input', 'detection', '{}', '{}', 3, 3, 'complete'))
        for cue, (identity, face) in enumerate(vectors.items()):
            obs = str(uuid.uuid4())
            db.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
                       (identity, 'scope', ns, float(cue), str(cue + 900), 'token-' + identity))
            db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (obs, scan, cue, float(cue), 'vtt', '{}', str(cue),
                        '{}', None, 4, face.tobytes(), None))
            db.execute('INSERT INTO members VALUES (?,?)', (obs, identity))
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                       (str(uuid.uuid4()), scan, policy, identity, 'accepted',
                        json.dumps({'observation': obs, 'reason': 'candidate_group'})))
        db.commit()
    return config


def test_report_ranks_pairs_and_explains_cross_sample_blockers(tmp_path):
    config = seeded(tmp_path)
    result = build(config, '123')
    assert result['format'] == 1
    assert result['summary']['identities'] == 3
    assert result['summary']['pairs_checked'] == 3
    assert result['summary']['pairs_above_min_mean'] == 1
    pair = result['potential_fragments'][0]
    assert pair['cross_mean'] == pytest.approx(.7, abs=1e-5)
    assert pair['cross_minimum'] == pytest.approx(.7, abs=1e-5)
    assert pair['blockers'] == []
    assert pair['recorded_merge_support']['validated_nominations'] == 0
    assert pair['recorded_merge_support']['complete_replay_proof'] is False
    assert pair['can_merge'] is None
    assert pair['worst_representatives']['left']['scene'] == '123'
    assert result['decisions'][0]['count'] == 3


def test_report_thresholds_limits_and_config_mismatch(tmp_path):
    config = seeded(tmp_path)
    result = build(config, '123', min_mean=-1, limit=2)
    assert result['summary']['pairs_above_min_mean'] == 3
    assert len(result['potential_fragments']) == 2
    assert any('cross_mean_below_match_threshold' in pair['blockers']
               for pair in result['potential_fragments'])
    with pytest.raises(ValueError, match='Report limit'):
        build(config, '123', limit=0)
    with pytest.raises(ValueError, match='Report min-mean'):
        build(config, '123', min_mean=float('nan'))
    changed = tmp_path / 'different.yml'
    changed.write_text('version: 1\nrecognition: {match_threshold: 0.8}\n')
    with pytest.raises(ValueError, match='current face configuration'):
        build(load(changed, database=config.database), '123')


def test_cli_writes_json_without_modifying_database_and_does_not_call_run(tmp_path, monkeypatch):
    config = seeded(tmp_path)
    output = tmp_path / 'fragmentation.json'
    with sqlite3.connect(config.database) as db:
        before = tuple(db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                       for table in ('members', 'candidates', 'runs', 'identity_merges'))
    import stash_ai.recognition.run as run_module
    monkeypatch.setattr(run_module, 'execute', lambda *args, **kwargs:
                        pytest.fail('Report must not enter recognition run'))
    assert main(['report', '--scene-id', '123', '--faces-db', str(config.database),
                 '--output', str(output)]) == 0
    data = json.loads(output.read_text())
    assert data['scene_id'] == '123'
    with sqlite3.connect(config.database) as db:
        after = tuple(db.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0]
                      for table in ('members', 'candidates', 'runs', 'identity_merges'))
    assert before == after
    with pytest.raises(FileExistsError):
        main(['report', '--scene-id', '123', '--faces-db', str(config.database),
              '--output', str(output)])


def test_report_rejects_non_database_and_parser_requires_scene(tmp_path):
    config = load(root=tmp_path)
    with pytest.raises(ValueError, match='does not exist'):
        build(config, '123')
    assert not config.database.exists()
    with pytest.raises(SystemExit):
        parser().parse_args(['report'])

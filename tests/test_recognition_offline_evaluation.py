"""Offline metrics use the production replay and never change the source."""
import json

import pytest

from stash_ai.recognition.cli import parser
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.models import namespace
from stash_ai.recognition.scanning import scan_identity
from stash_ai.recognition.offline_evaluation import evaluate
from stash_ai.recognition.decision_evidence import summary
from stash_ai.recognition.clustering import matching_policy
from tests.test_recognition_anchor_views import config_for, A, X, C
from tests.test_recognition_clustering import scan
from tests.test_recognition_rebuild import changed


def fixture(tmp_path):
    config = config_for(tmp_path, view_prototypes_enabled=True)
    with FaceDatabase(config.database) as db:
        for scene, vectors in [('1', [A]*3+[X]*3), ('2', [C]*3)]:
            scan(db, config, scene, vectors, count=len(vectors))
        ns = namespace(config)
        db.connection.execute('UPDATE scans SET namespace=?,policy_sha=?', (ns, scan_identity(config, ns)))
        labels = {r[0]: 'one' for r in db.connection.execute('SELECT id FROM observations')}
        db.connection.commit()
    return config, labels


def test_joint_and_batched_replays_are_isolated_and_report_real_decisions(tmp_path, monkeypatch):
    config, labels = fixture(tmp_path)
    before = config.database.read_bytes()
    monkeypatch.setattr(FaceDatabase, '__init__', lambda *a, **k: pytest.fail('writer opened'))
    from stash_ai.recognition import models
    monkeypatch.setattr(models, 'load_backend', lambda *a, **k: pytest.fail('model loaded'))
    result = evaluate(config, labels)
    assert config.database.read_bytes() == before
    assert result['metrics']['comparisons'][0]['equivalent']
    assert all(m['false_merge_pairs'] == m['extra_fragments'] == m['withheld'] == 0
               for m in result['metrics']['runs'].values())
    assert all(len(t['decisions']) == 9 for t in result['trials'].values())
    assert all(t['reason_counts'] for t in result['trials'].values())
    assert all('embedding' not in d['evidence'] for t in result['trials'].values() for d in t['decisions'].values())


def test_variants_recompute_partitions_and_preserve_input_contract(tmp_path):
    config, labels = fixture(tmp_path)
    strict = changed(config, recognition={'match_threshold': .999})
    result = evaluate(config, labels, variants={'strict': strict}, batch_sizes=(0,))
    assert result['metrics']['runs']['strict/joint']['extra_fragments'] > 0
    assert result['metrics']['runs']['baseline/joint']['extra_fragments'] == 0
    with pytest.raises(ValueError, match='contracts'):
        evaluate(config, labels, variants={'bad': changed(config, detection={'min_face_pixels': 80})})


@pytest.mark.parametrize('change', ['missing', 'partial', 'stale', 'scope'])
def test_refuses_unavailable_or_incompatible_input(tmp_path, change):
    config, labels = fixture(tmp_path)
    with FaceDatabase(config.database) as db:
        if change == 'partial':
            db.connection.execute("UPDATE scans SET status='partial'")
        if change == 'stale':
            db.connection.execute("UPDATE scans SET policy_sha='old'")
        if change == 'scope':
            db.connection.execute("UPDATE scans SET scope=scene")
        db.connection.commit()
    if change == 'missing':
        labels['unknown'] = 'one'
    with pytest.raises(ValueError):
        evaluate(config, labels)


def test_cli_output_is_exclusive_and_duplicate_truth_keys_rejected(tmp_path):
    config, labels = fixture(tmp_path)
    from stash_ai.recognition.offline_evaluation import execute
    label_path, output = tmp_path/'labels.json', tmp_path/'evaluation.json'
    label_path.write_text(json.dumps(labels))
    args = parser().parse_args(['evaluate', '--all-scenes', '--labels', str(label_path), '--output', str(output)])
    assert execute(args, config) == 0
    assert json.loads(output.read_text())['source_read_only']
    with pytest.raises(FileExistsError):
        execute(args, config)
    label_path.write_text('{"x":"one","x":"two"}')
    with pytest.raises(ValueError, match='Duplicate'):
        execute(args, config)


def test_production_summary_projects_original_blocking_evidence(tmp_path):
    config, _ = fixture(tmp_path)
    with FaceDatabase(config.database) as db:
        sid = db.connection.execute('SELECT id FROM scans LIMIT 1').fetchone()[0]
        policy = matching_policy(config)
        for i in range(5):
            db.connection.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                (str(i), sid, policy, None, 'withheld', json.dumps(dict(observation=str(i), reason='ambiguous_identity',
                    decision_conflict={'rival': 'competitor'}, embedding='must not export'))))
        result = summary(db.connection, policy, [sid])
        group = result['groups'][0]
        assert group['count'] == 5 and len(group['examples']) == 3
        assert all(e['decision_conflict']['rival'] == 'competitor' and 'embedding' not in e for e in group['examples'])


def test_order_and_repeat_matrix_are_reproducible_and_read_only(tmp_path):
    config, labels = fixture(tmp_path)
    before = config.database.read_bytes()
    result = evaluate(config, labels, batch_sizes=(1,), scene_orders=('numeric', 'reverse', 'shuffle'),
                      shuffle_seed=27, repeat_runs=2)
    assert len(result['trials']) == 6
    assert result['trials']['baseline/batch-1/reverse/run-1']['selection'] == ['2', '1']
    for order in ('numeric', 'reverse', 'shuffle'):
        a = result['trials'][f'baseline/batch-1/{order}/run-1']
        b = result['trials'][f'baseline/batch-1/{order}/run-2']
        assert a['selection'] == b['selection']
        assert a['elapsed_seconds'] >= 0
    assert config.database.read_bytes() == before


def test_retained_history_excludes_selected_scenes_and_images(tmp_path, monkeypatch):
    from stash_ai.recognition.rebuild import rebuild_selected
    from stash_ai.recognition import offline_evaluation as module
    config, _ = fixture(tmp_path)
    with FaceDatabase(config.database) as db:
        selected = [dict(scene=r['scene'], scan=r['id']) for r in db.connection.execute('SELECT * FROM scans')]
        rebuild_selected(db, selected, config)
        labels = {r[0]: 'one' for r in db.connection.execute('SELECT o.id FROM observations o JOIN scans s ON s.id=o.scan WHERE s.scene=\'2\'')}
    before = config.database.read_bytes()
    original = module.rebuild_selected
    observed = []
    def inspect(state, selection, settings):
        observed.append(state.connection.execute('SELECT COUNT(*) FROM members m JOIN observations o ON o.id=m.observation '
            'JOIN scans s ON s.id=o.scan WHERE s.scene=\'1\'').fetchone()[0])
        assert not state.connection.execute('SELECT 1 FROM observations WHERE crop IS NOT NULL').fetchone()
        assert not state.connection.execute('SELECT 1 FROM members m JOIN observations o ON o.id=m.observation '
            'JOIN scans s ON s.id=o.scan WHERE s.scene=\'2\'').fetchone()
        return original(state, selection, settings)
    monkeypatch.setattr(module, 'rebuild_selected', inspect)
    result = evaluate(config, labels, scenes=['2'], history_modes=('retained',), batch_sizes=(0,))
    assert observed == [6]
    assert result['trials']['baseline/joint/retained']['history']['scenes'] == 1
    assert config.database.read_bytes() == before


def test_retune_reuses_baseline_state_but_runs_new_policy(tmp_path):
    config, labels = fixture(tmp_path)
    strict = changed(config, recognition={'match_threshold': .999})
    result = evaluate(config, labels, variants={'strict': strict}, batch_sizes=(0,), retune_from_baseline=True,
                      repeat_runs=2)
    for run in (1, 2):
        key = f'strict/joint/retuned/run-{run}'
        assert result['trials'][key]['retuned_from_baseline']
        assert result['trials'][key]['matching_policy'] == matching_policy(strict)
        assert result['metrics']['runs'][key]['extra_fragments'] > 0


@pytest.mark.parametrize('options', [dict(scene_orders=()), dict(scene_orders=('numeric','numeric')),
    dict(history_modes=('future',)), dict(repeat_runs=0), dict(shuffle_seed=True)])
def test_invalid_scenarios_fail_before_opening_database(tmp_path, options):
    config, labels = fixture(tmp_path)
    with pytest.raises(ValueError):
        evaluate(config, labels, **options)

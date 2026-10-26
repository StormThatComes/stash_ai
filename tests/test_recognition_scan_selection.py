"""Returning to a cached scan must select the evidence validated by this run."""
import uuid
import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.media import SpriteInput
from stash_ai.recognition.merge_report_policy import patched
from stash_ai.recognition.models import namespace
from stash_ai.recognition.offline_evaluation import _selection
from stash_ai.recognition.scanning import scan_scene
from tests.helpers.recognition import Backend, MediaClient
from tests.test_recognition_scanning import config


def scan(state, settings, client, factory=Backend):
    with SpriteInput(client, client.scene, settings) as source:
        return scan_scene(state, 'scope', client.scene, settings, namespace(settings), source, factory)


@pytest.mark.parametrize('interrupted', [False, True])
def test_returning_to_previous_scan_policy_reuses_and_selects_that_scan(tmp_path, interrupted):
    from stash_ai.recognition.merge_goals import resolve
    from stash_ai.recognition.merge_solver import solve
    settings = config(tmp_path)
    changed = patched(settings, {'detection.min_sharpness': 1.})
    client = MediaClient('7')
    with FaceDatabase(settings.database) as state:
        first, _ = scan(state, settings, client)
        person = str(uuid.uuid4())
        state.connection.execute('INSERT INTO people VALUES (?,?,?,?,?,?)',
            (person, 'scope', namespace(settings), 0, '47080', person))
        state.connection.execute('INSERT INTO members SELECT id,? FROM observations WHERE scan=?', (person, first))
        state.connection.commit()
        insertion_order = state.connection.execute('SELECT rowid FROM scans WHERE id=?', (first,)).fetchone()[0]
        original = [tuple(r) for r in state.connection.execute('SELECT * FROM observations WHERE scan=?', (first,))]
        if interrupted:
            with pytest.raises(RuntimeError, match='synthetic'):
                scan(state, changed, client, lambda: Backend(fail_at=1))
        else:
            scan(state, changed, client)
        # Read-only replay must not silently fall back before a real run.
        _, selected = _selection(state.connection, settings, None, 'scope', allow_partial=True)
        assert selected == []
        goals = resolve(state.connection, settings, [['47080']], scope='scope')
        assert solve(settings, goals, state.connection, maximum=1)['status'] == 'no_compatible_scans'
        assert scan(state, settings, client, lambda: pytest.fail('cached scan must not reload models')) == (first, True)
        _, selected = _selection(state.connection, settings, None, 'scope')
        assert [s['id'] for s in selected] == [first]
        assert state.connection.execute('SELECT rowid FROM scans WHERE id=?', (first,)).fetchone()[0] == insertion_order
        assert original == [tuple(r) for r in state.connection.execute('SELECT * FROM observations WHERE scan=?', (first,))]
        assert state.connection.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 2
        from stash_ai.recognition.publication_queue import latest
        from stash_ai.recognition.evidence_cache import Evidence
        assert latest(state.connection, 'scope', namespace(settings), '7')['id'] == first
        assert Evidence(state.connection, settings).heads('scope', namespace(settings)) == {'7': first}
        from stash_ai.recognition.replay_simulation import replay
        changes = state.connection.total_changes
        result = replay(settings, {}, source=state.connection, scope='scope')
        assert result['selection'] == ['7'] and 'baseline' in result['trials']
        assert solve(settings, goals, state.connection, maximum=1)['search']['executed'] == 1
        assert state.connection.total_changes == changes


def test_new_partial_scan_supersedes_reselected_cache_and_can_be_resumed(tmp_path):
    settings = config(tmp_path)
    second = patched(settings, {'detection.min_sharpness': 1.})
    third = patched(settings, {'detection.min_sharpness': 2.})
    client = MediaClient('7')
    with FaceDatabase(settings.database) as state:
        first, _ = scan(state, settings, client)
        scan(state, second, client)
        scan(state, settings, client, lambda: pytest.fail('unexpected inference'))
        with pytest.raises(RuntimeError, match='synthetic'):
            scan(state, third, client, lambda: Backend(fail_at=2))
        diagnostics = {}
        _, selected = _selection(state.connection, settings, None, 'scope', allow_partial=True, diagnostics=diagnostics)
        assert not selected
        assert diagnostics['excluded_scans'][0]['reasons'] == ['scan_policy_mismatch', 'incomplete_scan']
        resumed, _ = scan(state, third, client)
        _, selected = _selection(state.connection, third, None, 'scope')
        assert [s['id'] for s in selected] == [resumed] and resumed != first


def test_switching_between_cached_scans_preserves_heads_in_private_history(tmp_path):
    from stash_ai.recognition.evaluation_scenarios import copy_history
    from stash_ai.recognition.offline_evaluation import _ram
    from stash_ai.recognition.publication_queue import latest
    from stash_ai.recognition.merge_goals import fingerprint
    from stash_ai.recognition.hybrid_merge import signature
    settings = config(tmp_path)
    changed = patched(settings, {'detection.min_sharpness': 1.})
    client = MediaClient('7')
    with FaceDatabase(settings.database) as state:
        first, _ = scan(state, settings, client)
        second, _ = scan(state, changed, client)
        original_fingerprint = fingerprint(state.connection, settings, 'scope')
        selection = [dict(scene='7', scan=first)]
        original_signature = signature(state.connection, selection, settings)
        scan(state, settings, client)
        assert fingerprint(state.connection, settings, 'scope') != original_fingerprint
        assert signature(state.connection, selection, settings) != original_signature
        target = _ram(settings)
        try:
            copy_history(state.connection, target, [], 'scope', namespace(settings))
            assert latest(target, 'scope', namespace(settings), '7')['id'] == first
            assert [r[0] for r in target.execute('SELECT id FROM scans ORDER BY rowid')] == [first, second]
        finally:
            target.close()
        assert scan(state, changed, client, lambda: pytest.fail('unexpected inference')) == (second, True)
        assert latest(state.connection, 'scope', namespace(settings), '7')['id'] == second
        assert fingerprint(state.connection, settings, 'scope') == original_fingerprint
        assert signature(state.connection, selection, settings) == original_signature


def test_reselected_scan_does_not_leak_into_excluded_evaluation_history(tmp_path):
    from stash_ai.recognition.evaluation_scenarios import copy_history
    from stash_ai.recognition.offline_evaluation import _ram
    from stash_ai.recognition.scan_heads import PREFIX
    settings = config(tmp_path)
    changed = patched(settings, {'detection.min_sharpness': 1.})
    client = MediaClient('7')
    with FaceDatabase(settings.database) as state:
        scan(state, settings, client)
        scan(state, changed, client)
        scan(state, settings, client)
        target = _ram(settings)
        try:
            copy_history(state.connection, target, [{'scene': '7'}], 'scope', namespace(settings))
            assert target.execute('SELECT COUNT(*) FROM scans').fetchone()[0] == 0
            assert target.execute('SELECT COUNT(*) FROM metadata WHERE key LIKE ?', (PREFIX+'%',)).fetchone()[0] == 0
        finally:
            target.close()

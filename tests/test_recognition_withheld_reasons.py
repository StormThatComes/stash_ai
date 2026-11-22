"""Withheld reason groups partition current observations before pagination."""
import json

import pytest

from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.scene_inspection import SceneReader, UnassignedReader
from tests.test_recognition_identity_inspector import scene_fixture, add_grouped_unassigned


def reason_fixture(tmp_path):
    config, _ = scene_fixture(tmp_path)
    add_grouped_unassigned(config)
    with FaceDatabase(config.database) as state:
        db = state.connection
        db.execute("UPDATE body_members SET proof=? WHERE observation IN ('withheld-body','withheld-1')",
                   (json.dumps(dict(reason='body_anchor_keypoint_rules')),))
        db.execute("UPDATE body_members SET proof=? WHERE observation='withheld-2'",
                   (json.dumps(dict(reason='unconfirmed_contact_face')),))
        source = dict(db.execute("SELECT * FROM body_observations WHERE id='withheld-body'").fetchone())
        member = dict(db.execute("SELECT * FROM body_members WHERE observation='withheld-body'").fetchone())
        for index, proof in enumerate(('{}', '{"reason":null}', '{"reason":12}', '{"reason":[]}')):
            address = 'unknown-'+str(index)
            row = dict(source, id=address, cue=index+4)
            db.execute('INSERT INTO body_observations VALUES ('+','.join('?' for _ in row)+')', tuple(row.values()))
            row = dict(member, observation=address, proof=proof)
            db.execute('INSERT INTO body_members VALUES ('+','.join('?' for _ in row)+')', tuple(row.values()))
        db.commit()
    return config


def test_reason_counts_pages_and_keyset_cursors_match_without_loading_pixels(tmp_path, monkeypatch):
    from stash_ai.recognition import scene_inspection, identity_inspection
    config = reason_fixture(tmp_path)
    before = config.database.read_bytes()
    monkeypatch.setattr(scene_inspection, 'PAGE_SIZE', 2)
    monkeypatch.setattr(identity_inspection, 'cold_crop', lambda *_: pytest.fail('Grouping loaded pixels'))
    catalogue = SceneReader(config, '2').catalogue()
    group = next(g for g in catalogue['unassigned_groups'] if (g['kind'],g['state']) == ('body','withheld'))
    assert group['count'] == 7
    assert {r['reason']:r['count'] for r in group['reasons']} == {
        'body_anchor_keypoint_rules':2, 'unconfirmed_contact_face':1, 'unknown_reason':4}
    reader = UnassignedReader(config, catalogue['scan'])
    ids = []
    for reason in group['reasons']:
        after = None
        selected = []
        while True:
            rows, more = reader.page(kind='body', state='withheld', reason=reason['reason'], after=after)
            assert all(row['reason'] == reason['reason'] and row['state'] == 'withheld' for row in rows)
            selected.extend(row['id'] for row in rows)
            if not more:
                break
            after = rows[-1]
        assert len(selected) == reason['count']
        ids.extend(selected)
    assert len(set(ids)) == 7
    assert not {'rejected-withheld', 'provisional-body', 'accepted-body'} & set(ids)
    assert reader.page(kind='body', state='withheld', reason="' OR 1=1 --") == ([], False)
    assert reader.page(kind='body', state='withheld', reason='unconfirmed_contact_face', observation='withheld-body') == ([], False)
    assert reader.page(kind='body', state='withheld', reason='unknown_reason', offset=4) == ([], False)
    assert config.database.read_bytes() == before


def test_face_reasons_use_latest_current_policy_in_same_scan_and_refresh(tmp_path):
    from stash_ai.recognition.clustering import matching_policy
    config, _ = scene_fixture(tmp_path)
    source = SceneReader(config, '2')
    initial = source.catalogue()
    assert not any(g['kind']=='face' and g['state']=='withheld' for g in initial['unassigned_groups'])
    scan = initial['scan']['id']
    with FaceDatabase(config.database) as state:
        db = state.connection
        other = db.execute("SELECT id FROM scans WHERE scene='10'").fetchone()[0]
        for identifier, at, policy, reason in (
                ('current-old', scan, matching_policy(config), 'old_reason'),
                ('current-new', scan, matching_policy(config), 'insufficient_match_or_cohesion'),
                ('later-historical', scan, 'other-policy', 'historical_reason'),
                ('other-scan', other, matching_policy(config), 'wrong_scan_reason')):
            db.execute('INSERT INTO candidates VALUES (?,?,?,?,?,?)',
                (identifier, at, policy, None, 'withheld', json.dumps(dict(observation='unmatched-face', reason=reason))))
        db.commit()
    catalogue = source.catalogue()
    group = next(g for g in catalogue['unassigned_groups'] if (g['kind'],g['state']) == ('face','withheld'))
    assert group['count'] == 1
    assert group['reasons'] == [dict(reason='insufficient_match_or_cohesion', count=1)]
    reader = UnassignedReader(config, catalogue['scan'])
    rows, more = reader.page(kind='face', state='withheld', reason='insufficient_match_or_cohesion')
    assert not more and [r['id'] for r in rows] == ['unmatched-face']
    with FaceDatabase(config.database) as state:
        state.connection.execute("UPDATE candidates SET payload=? WHERE id='current-new'",
            (json.dumps(dict(observation='unmatched-face', reason='ambiguous_candidate_group')),))
        state.connection.commit()
    assert reader.page(kind='face', state='withheld', reason='insufficient_match_or_cohesion') == ([], False)
    group = next(g for g in source.catalogue()['unassigned_groups'] if (g['kind'],g['state']) == ('face','withheld'))
    assert group['reasons'] == [dict(reason='ambiguous_candidate_group', count=1)]


def test_reason_branch_worker_filters_and_history_keys_are_independent(tmp_path, monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    from stash_ai.recognition.scene_window import SceneInspector
    from stash_ai.recognition.identity_window import Inspector
    config = reason_fixture(tmp_path)
    reader = UnassignedReader(config, SceneReader(config, '2').catalogue()['scan'])
    ui = object.__new__(SceneInspector)
    output = []
    ui.tree_worker = SimpleNamespace(future=None, poll=lambda: None,
        request=lambda task, *_: output.append(task()))
    ui.branches = {reason:dict(reader=reader, unassigned=True, kind='body', state='withheld',
        reason=reason, level='observations', after=None) for reason in
        ('body_anchor_keypoint_rules', 'unconfirmed_contact_face')}
    ui.queue = deque(ui.branches)
    monkeypatch.setattr(Inspector, 'poll', lambda _: None)
    assert len({ui.branch_key(b) for b in ui.branches.values()}) == 2
    ui.poll()
    ui.poll()
    assert [[r['id'] for r in rows] for rows, _ in output] == [['withheld-body', 'withheld-1'], ['withheld-2']]


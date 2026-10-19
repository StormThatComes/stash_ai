"""Anchor-only multi-view matching is local, frozen and cannot merge people."""
from dataclasses import replace
import math

import numpy as np
import pytest

from stash_ai.recognition.clustering import Group, matching_policy
from stash_ai.recognition.anchor_views import _competing_anchor
from stash_ai.recognition.config import load, plain
from stash_ai.recognition.database import FaceDatabase
from stash_ai.recognition.rebuild import rebuild_selected
from stash_ai.recognition.scanning import scan_identity
from tests.test_recognition_pose_policy import configured
from tests.test_recognition_clustering import scan
from tests.test_recognition_scene_context import candidates

A = np.array([1., 0., 0., 0.])
X = np.array([.7, math.sqrt(1-.7**2), 0., 0.])
C = np.array([.35, math.sqrt(1-.35**2), 0., 0.])
RIVAL = np.array([0., 0., 1., 0.])


def config_for(tmp_path, **options):
    return configured(tmp_path, same_scene_match_bonus=0,
                      anchor_view_matching=True, **options)


def run(db, config, vectors, *, scene='1'):
    identifier, _ = scan(db, config, scene, vectors, count=len(vectors))
    selected = [{'scene': scene, 'scan': identifier}]
    rebuild_selected(db, selected, config)
    return identifier, selected


def statuses(db, scan, config):
    return candidates(db, scan, config)


def test_anchor_view_recovers_diverse_identity_without_weak_global_cohesion(tmp_path):
    config = config_for(tmp_path)
    assert float(np.dot(A, X)) >= .65
    assert float(np.dot(A, C)) < .60
    assert float(np.dot(X, C)) > .82
    with FaceDatabase(config.database) as db:
        identifier, selected = run(db, config, [A]*3 + [X]*3 + [C]*2)
        rows = statuses(db, identifier, config)
        assert len(rows) == 8
        assert {person for status, person, _ in rows if status == 'accepted'} == {rows[0][1]}
        assert [row[2]['reason'] for row in rows[-2:]] == ['anchor_view_identity']*2
        assert all(row[2]['anchor_view']['anchor_support'] >= 3 for row in rows[-2:])
        assert not db.connection.execute('SELECT 1 FROM identity_merges LIMIT 1').fetchone()
        # Turning off just the new policy invalidates matching, never scanning.
        data = plain(config.data)
        data['clustering']['anchor_view_matching'] = False
        disabled = replace(config, data=data)
        assert matching_policy(config) != matching_policy(disabled)
        assert scan_identity(config, 'ns') == scan_identity(disabled, 'ns')
        rebuild_selected(db, selected, disabled)
        assert [r[0] for r in statuses(db, identifier, disabled)[-2:]] == ['withheld']*2
        rebuild_selected(db, selected, config)
        assert [r[2]['reason'] for r in statuses(db, identifier, config)[-2:]] == ['anchor_view_identity']*2


def test_one_unsupported_candidate_must_remain_withheld(tmp_path):
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        identifier, _ = run(db, config, [A]*3 + [X]*3 + [C])
        last = statuses(db, identifier, config)[-1]
        assert last[0] == 'withheld'
        assert last[2]['reason'] == 'anchor_view_insufficient_support'


def test_incomplete_view_cannot_bootstrap_attachment(tmp_path):
    config = config_for(tmp_path, view_min_support=4)
    with FaceDatabase(config.database) as db:
        identifier, _ = run(db, config, [A]*3 + [X]*3 + [C]*2)
        assert [r[0] for r in statuses(db, identifier, config)[-2:]] == ['withheld']*2


def test_independent_timestamps_required_for_anchor_view(tmp_path):
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        identifier, _ = scan(db, config, '1', [A]*3 + [X]*3 + [C]*2, count=8)
        # Three embeddings at one timestamp do not make three votes.
        db.connection.execute('UPDATE observations SET seconds=3 WHERE scan=? AND cue IN (3,4,5)', (identifier,))
        db.connection.commit()
        rebuild_selected(db, [{'scene': '1', 'scan': identifier}], config)
        assert [r[0] for r in statuses(db, identifier, config)[-2:]] == ['withheld']*2


def test_anchor_view_matches_distant_timestamps_across_same_scene(tmp_path):
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        identifier, _ = scan(db, config, '1', [A]*3 + [X]*3 + [C]*2, count=8)
        # Valid supporting cues appear near the beginning of the scene,
        # whereas the matching views are more than an hour later.
        db.connection.execute(
            'UPDATE observations SET seconds=3600 WHERE scan=? AND cue=6',
            (identifier,))
        db.connection.execute(
            'UPDATE observations SET seconds=3620 WHERE scan=? AND cue=7',
            (identifier,))
        db.connection.commit()
        rebuild_selected(db, [{'scene': '1', 'scan': identifier}], config)
        assert [r[2]['reason'] for r in statuses(db, identifier, config)[-2:]] == [
            'anchor_view_identity', 'anchor_view_identity']
        assert all('max_gap_seconds' not in r[2]['anchor_view']
                   for r in statuses(db, identifier, config)[-2:])


def test_anchor_view_never_uses_cross_scene_only_core(tmp_path):
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        one, _ = scan(db, config, '1', [A]*3 + [X]*3, count=6)
        two, _ = scan(db, config, '2', [C]*2, count=2)
        rebuild_selected(db, [{'scene':'1','scan':one},{'scene':'2','scan':two}], config)
        assert [r[0] for r in statuses(db, two, config)] == ['withheld']*2


def test_anchor_view_does_not_override_co_present_identity(tmp_path):
    config = config_for(tmp_path)
    with FaceDatabase(config.database) as db:
        identifier, _ = scan(db, config, '1', [A]*3 + [X]*3 + [RIVAL]*3 + [C]*2, count=11)
        # Keep the rival's other two independent strict observations.
        db.connection.execute('UPDATE observations SET seconds=9,cue=9 WHERE scan=? AND cue=8', (identifier,))
        db.connection.commit()
        rebuild_selected(db, [{'scene':'1','scan':identifier}], config)
        rows = statuses(db, identifier, config)
        c_rows = [payload for status, _, payload in rows if payload.get('reason', '').startswith('anchor_view')]
        assert all(payload['reason'] != 'anchor_view_identity' for payload in c_rows)
        assert db.connection.execute('SELECT COUNT(*) FROM identity_merges').fetchone()[0] == 0



def test_sparse_rival_with_no_own_prototype_still_vetoes_match():
    query = np.array([1., 0., 0., 0.])
    favoured = Group('owner', (query,), True)
    # This rival has just one anchor, and therefore could not have passed
    # view_min_support, yet is strongly competitive at the person level.
    rival = Group('other', (np.array([.91, math.sqrt(1-.91**2), 0., 0.]),), True)
    assert _competing_anchor([favoured, rival], 'owner', query, .94, .12)
    distant = Group('distant', (np.array([0., 1., 0., 0.]),), True)
    assert not _competing_anchor([favoured, distant], 'owner', query, .94, .12)


def test_anchor_view_rejects_invalid_configuration(tmp_path):
    # The obsolete temporal gate must not reappear as a configurable knob.
    p = tmp_path/'obsolete.yml'
    p.write_text('version: 1\nclustering: {anchor_view_max_gap_seconds: 180}\n')
    with pytest.raises(ValueError, match='clustering: unknown keys or expected mapping'):
        load(p)
    path = tmp_path/'off.yml'
    path.write_text('version: 1\nclustering: {anchor_view_matching: true}\n')
    with pytest.raises(ValueError, match='anchor_view_matching'):
        load(path)


def test_validated_same_scene_cues_can_enrich_but_not_seed_a_view(tmp_path):
    from stash_ai.recognition.anchor_views import _confirmed
    from stash_ai.recognition.reference_usage import ReferenceUsage
    config = config_for(tmp_path, context_view_enrichment=True)
    with FaceDatabase(config.database) as db:
        identifier, _ = run(db, config, [A]*5)
        rows = list(db.connection.execute(
            'SELECT c.id FROM candidates c JOIN observations o '
            "ON o.id=json_extract(c.payload,'$.observation') "
            'WHERE c.scan=? ORDER BY o.cue', (identifier,)))
        for candidate in rows[-2:]:
            db.connection.execute(
                "UPDATE candidates SET payload=json_set(payload,'$.reason','same_scene_identity') "
                'WHERE id=?', (candidate['id'],))
        db.connection.commit()
        policy=matching_policy(config)
        enriched=_confirmed(db.connection,policy,config.data['clustering'],
                            ReferenceUsage(config.data['clustering']['max_reference_bytes']))
        assert len(enriched)==2
        assert all(v.support_count>=3 for v in enriched)
        # The strict seed was not promoted from contextual evidence alone.
        for candidate in rows[:3]:
            db.connection.execute(
                "UPDATE candidates SET payload=json_set(payload,'$.reason','same_scene_identity') "
                'WHERE id=?', (candidate['id'],))
        db.connection.commit()
        unsafe=_confirmed(db.connection,policy,config.data['clustering'],
                          ReferenceUsage(config.data['clustering']['max_reference_bytes']))
        assert unsafe==[]

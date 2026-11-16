import json
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from stash_ai.recognition import cli
from stash_ai.recognition.config import load
from stash_ai.recognition.config_check import report
from stash_ai.recognition.config_layout import put


SHIPPED = Path(__file__).resolve().parents[1] / 'facial_recognition.yml'


def profile(tmp_path, changes):
    doc = yaml.safe_load(SHIPPED.read_bytes())
    for path, value in changes.items():
        put(doc, path, value)
    dest = tmp_path / 'recognition.yml'
    dest.write_text(yaml.safe_dump(doc), encoding='utf-8')
    return dest


def rows(config, gap=.02):
    result = report(config, gap)
    checks = {r['rule']: r for r in result['checks']}
    assert len(checks) == sum(result['counts'].values())
    return checks


@pytest.mark.parametrize(('track', 'gap', 'expected'), [
    (.845, .02, 'advice'), (.83, .02, 'ok'), (.85, .02, 'advice'),
    (.845, 0., 'ok'), (.9, 0., 'advice'), (0., .02, 'ok'), (None, .02, 'ok'),
])
def test_adjacent_cue_width_and_inheritance(tmp_path, track, gap, expected):
    path = 'matching.body.identities.track_similarity_threshold'
    c = load(profile(tmp_path, {path: track}))
    row = rows(c, gap)['body-adjacent-cues']
    assert row['status'] == expected and row['settings'][path] == track
    assert row['measures']['lower'] == (.8 if track is None else track)


def test_composed_gates_and_unit_specific_counts(tmp_path):
    config = load(profile(tmp_path, {
        'matching.face.views.match_threshold': .62,
        'matching.face.views.ambiguity_margin': .04,
        'matching.face.cross_scene.cooccurrence_merge_threshold': .65,
        'matching.hybrid.face_floor': .645,
        'matching.body.identities.cohesion_threshold': .84,
        'matching.body.identities.scene_match_bonus': None,
        'matching.face.same_scene.match_bonus': .06,
        'matching.body.global.match_threshold': .84,
        'matching.body.identities.turn_similarity_discount': .95,
        'matching.face.same_scene.min_support': 5,
        'execution.scan.max_cues': 4,
    }))
    r = rows(config)
    assert r['view-match_threshold']['status'] == r['view-ambiguity_margin']['status'] == 'warning'
    assert r['face-reflection']['status'] == 'warning'
    assert r['hybrid-face-match_threshold']['status'] == 'advice'
    assert r['hybrid-face-cohesion_threshold']['status'] == 'warning'
    assert r['body-scene-bonus']['measures']['gap'] == pytest.approx(.01)
    assert r['body-scene-bonus']['settings']['matching.face.same_scene.match_bonus'] == .06
    assert r['body-turn-discount']['measures']['lower'] == 0
    assert r['body-turn-discount-cap']['status'] == 'warning'
    assert r['global-local-body-match_threshold']['status'] == 'advice'
    assert r['local-anchor-view-support']['status'] == r['local-view-cue-cap']['status'] == 'warning'
    assert r['body-roster-cue-cap']['measures']['required'] == 6
    assert 'hybrid-cue-cap' not in r  # Hybrid independent support can pool scenes.


def test_quality_pose_masks_and_portraits(tmp_path):
    config = load(profile(tmp_path, {
        'matching.face.anchors.quality.max_pose_deviation': .5,
        'matching.face.anchors.quality.min_visible_face_fraction': .5,
        'body.collection.min_keypoints': 7,
        'pose.refinement.min_recovery_mask_score': .49,
        'pose.refinement.min_prompt_coverage': .85,
        'pose.point_quality.min_presence': 1.,
        'pose.deduplication.min_presence': .4,
        'pose.deduplication.min_shared_points': 8,
        'portraits.selection.anchor_only': True,
        'portraits.quality.min_visible_face_fraction': .45,
        'portraits.weights.closed_mouth': 1.,
        'portraits.weights.open_mouth': 1.,
    }))
    r = rows(config)
    assert r['face-auxiliary-quality-band']['status'] == 'advice'
    assert r['body-anchor-min_keypoints']['status'] == 'advice'
    assert r['recovery-mask-score']['status'] == 'warning'
    assert r['mask-prompt-coverage']['status'] == r['duplicate-core-agreement']['status'] == 'advice'
    inherited = r['pose.refinement.min_presence-ceiling']
    assert inherited['status'] == 'warning'
    assert inherited['settings'] == {'pose.point_quality.min_presence': 1., 'pose.refinement.min_presence': None}
    assert r['pose.deduplication.min_presence-ceiling']['status'] == 'ok'
    assert r['portrait-anchor-min_visible_face_fraction']['status'] == r['portrait-mouth-weights']['status'] == 'advice'


def test_inactive_routes_and_affiliated_body_evidence(tmp_path):
    disabled = {
        'matching.face.views.anchor_view_matching': False, 'matching.face.views.prototypes_enabled': False,
        'matching.face.views.context_view_enrichment': False, 'matching.face.views.strict_appearance_matching': False,
        'matching.face.views.anchor_outlier_quarantine': False, 'portraits.enabled': False,
        'matching.body.identities.enabled': False, 'matching.body.global.enabled': False,
        'body.collection.enabled': False,
    }
    r = rows(load(profile(tmp_path, disabled)))
    assert 'hybrid-face-match_threshold' in r and 'body-anchor-min_keypoints' in r
    assert 'view-match_threshold' not in r and 'body-adjacent-cues' not in r
    assert 'portrait-weight-resolution' not in r
    r = rows(load(profile(tmp_path, {**disabled, 'body.enabled': False})))
    assert 'hybrid-face-match_threshold' not in r and 'recovery-mask-score' not in r


def test_v1_and_cli_offline_read_only(tmp_path, capsys):
    path = tmp_path / 'old.yml'
    initial = b'version: 1\nrecognition:\n  cooccurrence_merge_threshold: 0.66\n'
    path.write_bytes(initial)
    argv = ['check-config', '--face-config', str(path), '--json']
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['version'] == 1 and all(r['status'] != 'ok' for r in result['checks'])
    reflection = next(r for r in result['checks'] if r['rule'] == 'face-reflection')
    assert reflection['settings']['recognition.cooccurrence_merge_threshold'] == .66
    assert cli.main([*argv, '--all', '--min-score-gap', '0']) == 0
    all_rows = json.loads(capsys.readouterr().out)
    assert len(all_rows['checks']) == sum(all_rows['counts'].values())
    assert next(r for r in all_rows['checks'] if r['rule'] == 'face-reflection')['status'] == 'ok'
    assert list(tmp_path.iterdir()) == [path] and path.read_bytes() == initial
    for value in ('nan', 'inf', '-1', '1.01'):
        with pytest.raises(ValueError, match='min-score-gap'):
            cli.main([*argv, '--min-score-gap', value])
    path.write_text('version: 2\nmatching: {face: {match_threshold: 2}}')
    with pytest.raises(ValueError, match='matching.face.match_threshold'):
        cli.main(argv)


def test_opt_in_run_preflight_continues_with_same_config(tmp_path, monkeypatch, capsys):
    from stash_ai.recognition import work_locations
    path = profile(tmp_path, {'matching.face.views.match_threshold': .6})
    monkeypatch.setattr(work_locations, 'invocation', lambda config: nullcontext())
    seen = []
    def execute(args, config, services):
        seen.append(config.data)
        return 17
    monkeypatch.setattr(cli, '_execute', execute)
    argv = ['run', '--face-config', str(path), '--scene-id', '123']
    assert cli.main(argv) == 17 and capsys.readouterr().err == ''
    assert cli.main([*argv, '--check-config']) == 17
    captured = capsys.readouterr()
    assert 'view-match_threshold' in captured.err and not captured.out
    assert len(seen) == 2 and seen[0] == seen[1]

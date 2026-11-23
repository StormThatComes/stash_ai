"""Contact masks are independent consumers, with retained policy provenance."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from stash_ai.recognition.contact_face_evidence import measure, retained
from stash_ai.recognition.mask_usage import applications, contact_disabled_settings, uses
from tests.test_recognition_contact_single_shoulder import single_shoulder
from tests.test_recognition_contact_upper_body import masked_profile


def scene(route):
    if route == 'single':
        return single_shoulder()
    if route == 'upper':
        face, own = masked_profile()
        _, peer = masked_profile(220)
        return face, own, peer
    from tests.test_recognition_contact_confirmation import sample, mask
    face, own = sample(angle=np.pi/4)
    _, peer = sample(140, np.pi/4)
    for point in peer['keypoints']:
        if point[0] >= 0:
            point[0] += 15; point[1] -= 25
    mask(own); mask(peer)
    return face, own, peer


@pytest.mark.parametrize('route', ['full', 'upper', 'single'])
@pytest.mark.parametrize('overlap,admission', [(False,False),(False,True),(True,False),(True,True)])
def test_contact_masks_confirm_independently_of_other_consumers(route, overlap, admission):
    face, own, peer = scene(route)
    for person in (own, peer):
        person['pose_refinement']['mask_usage'] = applications(dict(
            mask_overlap=overlap, mask_body_admission=admission))
    before = deepcopy([own,peer])
    proof = measure(face, own, [own,peer], face_id='face')
    assert proof['confirmed'], proof
    assert proof['reason'] == {'full':'same_frame_head_torso', 'upper':'same_frame_upper_body',
                              'single':'same_frame_single_shoulder'}[route]
    assert proof['disabled_mask_settings'] == []
    assert json.loads(json.dumps(proof)) == proof
    assert [own,peer] == before


@pytest.mark.parametrize('route', ['full','upper','single'])
@pytest.mark.parametrize('owner', [0,1])
@pytest.mark.parametrize('failure', ['disabled','missing','invalid','conflict'])
def test_independence_keeps_missing_and_conflicting_masks_blocked(route, owner, failure):
    face, own, peer = scene(route)
    for person in (own,peer):
        person['mask_usage'] = dict(contact_confirmation=True, overlap=False, body_admission=False)
    target = (own,peer)[owner]
    if failure == 'disabled':
        target['mask_usage']['contact_confirmation'] = False
    elif failure == 'missing':
        target['pose_refinement'].pop('overlap_mask')
    elif failure == 'invalid':
        target['pose_refinement']['mask_valid'] = False
    else:
        target['pose_refinement']['mask_conflict'] = True
    proof = measure(face, own, [own,peer], face_id='face')
    assert not proof['confirmed'], proof
    node = proof if owner == 0 else (proof.get('upper_body') or proof)['peers'][0]
    assert node['disabled_mask_settings'] == (
        ['pose.mask.apply_to.contact_confirmation'] if failure == 'disabled' else [])


def test_retained_legacy_controls_are_not_reinterpreted_as_new_opt_in():
    person = dict(pose_refinement=dict(mask_usage=dict(overlap=False,body_admission=False)))
    assert not uses(person,'contact_confirmation')
    assert contact_disabled_settings(person) == ['pose.mask.apply_to.overlap','pose.mask.apply_to.body_admission']
    person['pose_refinement']['mask_usage']['contact_confirmation'] = True
    assert uses(person,'contact_confirmation') and contact_disabled_settings(person) == []
    assert not retained({'face':'face'},dict(contact_face_evidence=dict(
        version='same-cue-head-torso-v3-single-shoulder',face='face',confirmed=True)))['confirmed']


@pytest.mark.parametrize('version', [1,2])
@pytest.mark.parametrize('enabled', [False,True])
@pytest.mark.parametrize('value', [None,1,'true'])
def test_contact_consumer_requires_boolean_in_both_layouts(version, enabled, value):
    from stash_ai.recognition.config import effective_data
    document = (dict(version=2, pose=dict(mask=dict(enabled=enabled,apply_to=dict(contact_confirmation=value))))
                if version == 2 else dict(sapiensid=dict(pose=dict(refinement=dict(
                    enabled=enabled,mask_contact_confirmation=value)))))
    with pytest.raises(ValueError, match='boolean|true or false'):
        effective_data(document)


@pytest.mark.parametrize('backend', ['pmpose','vitpose_plus_large'])
def test_contracts_refresh_current_proofs_and_preserve_pre_switch_hashes(tmp_path, backend):
    from stash_ai.recognition.config import load, plain, digest
    from stash_ai.recognition.body_observations import contract, compatible_contracts
    from stash_ai.recognition.person_signal import contract as person_contract
    config = load(root=tmp_path)
    data = plain(config.data)
    data['sapiensid']['pose']['backend'] = backend
    data['sapiensid']['pose']['refinement']['enabled'] = True
    config = replace(config,data=data)
    old_data = deepcopy(data)
    old_data['sapiensid']['pose']['refinement'].pop('mask_contact_confirmation')
    old_person = person_contract(SimpleNamespace(data=old_data))
    collection = deepcopy(data['sapiensid']['collection']); collection.pop('enabled')
    old = digest(['independent-person-v7-single-shoulder-contact',old_person,collection])
    before = deepcopy(config.data)
    assert old in compatible_contracts(config) and old != contract(config)
    assert config.data == before
    current = contract(config)
    data['sapiensid']['pose']['refinement']['mask_contact_confirmation'] = False
    assert contract(config) != current and old in compatible_contracts(config)


def decision(*, current=True, peer=False, legacy=False):
    parameter = 'body_admission' if legacy else 'contact_confirmation'
    node = dict(reason='masks_disabled',disabled_mask_settings=['pose.mask.apply_to.'+parameter])
    local = dict(reason='insufficient_head_points',upper_body=(dict(peers=[node]) if peer else node))
    return dict(current=current,status='withheld',recorded=dict(
        reason='unconfirmed_contact_face',contact_confirmation=dict(same_frame=local)))


@pytest.mark.parametrize('version', [1,2])
@pytest.mark.parametrize('current', [False,True])
@pytest.mark.parametrize('peer', [False,True])
def test_inspector_advice_changes_only_contact_mask_controls(tmp_path, version, current, peer):
    import yaml
    from stash_ai.recognition.inspection_advice import suggest, describe
    from stash_ai.recognition.merge_report_policy import patched
    from tests.test_recognition_identity_inspector import advice_fixture, advice_admission
    config = replace(advice_fixture(tmp_path),document_version=version)
    config.data['sapiensid']['pose']['refinement'].update(
        mask_contact_confirmation=False,mask_body_admission=False,mask_overlap=False)
    result = suggest(config,'body',advice_admission(config,'body'),[decision(current=current,peer=peer)],None)
    path = result['paths'][0]
    changes = {c['parameter']:c['proposed'] for c in path['changes']}
    assert changes == {'sapiensid.pose.refinement.mask_contact_confirmation':True}
    patched(config, changes)
    document = yaml.safe_load(path['yaml'])
    assert (document['pose']['mask']['apply_to']['contact_confirmation'] if version == 2
            else document['sapiensid']['pose']['refinement']['mask_contact_confirmation']) is True
    assert 'fresh scan evidence' in path['next_action']
    assert ('historical attempt' in describe(result)) is (not current)


def test_inspector_legacy_blocker_requests_recollection_when_current_switch_is_enabled(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest, describe
    from tests.test_recognition_identity_inspector import advice_fixture, advice_admission
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['pose']['refinement']['mask_body_admission'] = False
    result = suggest(config,'body',advice_admission(config,'body'),[decision(current=False,legacy=True)],None)
    assert not any(p['changes'] for p in result['paths'])
    assert result['paths'][0]['title'] == 'Recollect contact-mask evidence'
    assert 'pose.mask.apply_to.body_admission' in describe(result)


def test_inspector_patch_enables_generation_when_also_disabled(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest
    from stash_ai.recognition.merge_report_policy import patched
    from tests.test_recognition_identity_inspector import advice_fixture, advice_admission
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['pose']['refinement'].update(
        enabled=False,mask_sharpness=False,mask_contact_confirmation=False,mask_body_admission=False)
    result = suggest(config,'body',advice_admission(config,'body'),[decision()],None)
    changes = {c['parameter']:c['proposed'] for c in result['paths'][0]['changes']}
    assert changes == {'sapiensid.pose.refinement.enabled':True,
                       'sapiensid.pose.refinement.mask_contact_confirmation':True}
    patched(config,changes)


@pytest.mark.parametrize('route', ['upper','single'])
def test_collection_retains_independent_mask_policy_and_peer_references(tmp_path, route):
    import uuid
    from stash_ai.recognition.body_collection import collect
    from stash_ai.recognition.config import load
    face, own, peer = scene(route)
    for person in (own,peer):
        person['pose_refinement']['mask_usage'] = applications(dict(mask_overlap=False,mask_body_admission=False))
    captured = []
    pipeline = SimpleNamespace(collect_items=lambda db,image,items:captured.extend(items))
    collect(pipeline,None,SimpleNamespace(person_geometry=lambda *args:None),load(root=tmp_path),
            str(uuid.uuid4()),0,0.,None,[dict(id='face',rejected=None)],[face],
            captured=([own,peer],[0],[(dict(width=320,height=320,keypoints=4),None) for _ in range(2)]))
    metadata = captured[0][0][1]
    proof = metadata['quality']['contact_face_evidence']
    assert proof['confirmed'] and proof['upper_body']['peers'][0]['observation'] == captured[1][0][0]
    assert metadata['person']['pose_refinement']['mask_usage']['contact_confirmation']
    assert not metadata['person']['pose_refinement']['mask_usage']['body_admission']


def test_missing_contact_evidence_does_not_suggest_enabling_masks(tmp_path):
    from stash_ai.recognition.inspection_advice import suggest
    from tests.test_recognition_identity_inspector import advice_fixture, advice_admission
    config = advice_fixture(tmp_path)
    item = decision(); item['recorded'].pop('contact_confirmation')
    result = suggest(config,'body',advice_admission(config,'body'),[item],None)
    assert not any(p['changes'] for p in result['paths'])
    assert result['paths'][0]['title'] == 'Recheck matching and independent support'


@pytest.mark.parametrize('reason', ['unused_mask','temporal_identity_conflict'])
def test_inspector_does_not_mislabel_disabled_but_unneeded_masks_as_blockers(tmp_path, reason):
    from stash_ai.recognition.inspection_advice import suggest
    from tests.test_recognition_identity_inspector import advice_fixture, advice_admission
    config = advice_fixture(tmp_path)
    config.data['sapiensid']['pose']['refinement']['mask_contact_confirmation'] = False
    item = decision()
    contact = item['recorded']['contact_confirmation']
    if reason == 'unused_mask':
        contact['same_frame']['upper_body'].update(reason='same_frame_upper_body',confirmed=True)
    else:
        contact['reason'] = reason
    result = suggest(config,'body',advice_admission(config,'body'),[item],None)
    assert not any(p['changes'] for p in result['paths'])


@pytest.mark.parametrize('owner', [0,1])
def test_diagnostics_names_the_disabled_control_and_owner(owner):
    from stash_ai.recognition.contact_anchors import questionable_links
    from stash_ai.recognition.contact_diagnostics import describe
    from stash_ai.recognition.body_identity import defaults
    from tests.test_recognition_contact_confirmation import row
    face, own, peer = scene('upper')
    for person in (own,peer):
        person['mask_usage'] = dict(contact_confirmation=True,body_admission=False)
    (own,peer)[owner]['mask_usage']['contact_confirmation'] = False
    candidate = row(local=False)
    quality = json.loads(candidate['quality'])
    quality['quality']['contact_face_evidence'] = measure(face,own,[own,peer],face_id='face')
    candidate['quality'] = json.dumps(quality)
    checks = {}
    questionable_links([candidate],defaults(),SimpleNamespace(face_similarity=np.dot),evidence=checks)
    text = '\n'.join(describe(checks['contact']))
    prefix = 'Own' if owner == 0 else 'Peer'
    assert prefix+' contact mask disabled by: pose.mask.apply_to.contact_confirmation' in text
    assert 'pose.mask.apply_to.body_admission' not in text

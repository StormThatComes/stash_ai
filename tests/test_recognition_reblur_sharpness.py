"""Blur ordering and foreground isolation, not just formula snapshots."""
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest

from stash_ai.recognition.blur_quality import ALGORITHM, batch, measure


def scene():
    y, x = np.indices((128, 128))
    pixels = np.full((128, 128), 100, np.float32)
    cv2.circle(pixels, (35, 40), 18, 200, -1)
    cv2.rectangle(pixels, (65, 18), (100, 55), 25, -1)
    cv2.ellipse(pixels, (64, 90), (40, 12), 20, 0, 360, 230, -1)
    return pixels + (((x//7+y//9) % 2)*8).astype(np.float32)


def motion(pixels, angle):
    kernel = np.zeros((19, 19), np.float32)
    delta = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])*8
    cv2.line(kernel, tuple(np.rint(9-delta).astype(int)), tuple(np.rint(9+delta).astype(int)), 1, 1)
    return cv2.filter2D(pixels, -1, kernel/kernel.sum())


@pytest.mark.parametrize('angle', [0, 30, 45, 60, 90, 135])
def test_motion_cannot_outrank_sharp_low_contrast_scene(angle):
    pixels = scene()
    sharp = measure(128+.15*(pixels-128))['sharpness']
    assert sharp > 10
    assert measure(motion(pixels, angle))['sharpness'] < sharp*.65


@pytest.mark.parametrize('radius', [2, 4, 6])
def test_optical_disk_blur_lowers_sharpness(radius):
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    kernel = (x*x+y*y <= radius*radius).astype(np.float32)
    pixels = scene()
    blurred = cv2.filter2D(pixels, -1, kernel/kernel.sum())
    assert measure(blurred)['sharpness'] < measure(pixels)['sharpness']*.85


def test_low_contrast_and_smooth_shading_do_not_change_resolved_detail_score():
    pixels = scene()
    score = measure(pixels)['sharpness']
    assert measure(128+.15*(pixels-128))['sharpness'] == pytest.approx(score, abs=.02)
    # A linear brightness ramp has no local derivative variance.
    ramp = np.indices(pixels.shape)[1].astype(np.float32)*.1
    assert measure(pixels+ramp)['sharpness'] == pytest.approx(score, abs=.02)


@pytest.mark.parametrize('sigma', [2, 5, 10])
def test_noise_does_not_rescue_blur_or_a_featureless_crop(sigma):
    pixels = scene()
    blur = cv2.GaussianBlur(pixels, (0, 0), 2)
    noise = np.random.default_rng(791).normal(0, sigma, pixels.shape).astype(np.float32)
    noisy = measure(blur+noise)['sharpness']
    assert noisy < measure(pixels)['sharpness']*.7
    assert noisy <= measure(blur)['sharpness']+4
    assert measure(128+noise)['sharpness'] == 0


def test_full_filter_footprint_excludes_background_holes_and_mask_edges():
    pixels = scene()
    mask = np.zeros(pixels.shape, bool)
    mask[15:115, 12:116] = True
    mask[45:65, 55:75] = False
    before = measure(pixels, mask)
    pixels[~mask] = np.random.default_rng(42).uniform(0, 255, (~mask).sum())
    after = measure(pixels, mask)
    assert before == after


@pytest.mark.parametrize('shape', [(0, 0), (1, 1), (8, 12), (14, 14), (32, 32)])
def test_empty_small_and_flat_samples_are_finite_and_cannot_pass(shape):
    for mask in (np.zeros(shape, bool), np.ones(shape, bool)):
        quality = measure(np.full(shape, 128, np.float32), mask)
        assert quality['sharpness'] == 0
        assert np.isfinite(quality['sharpness_contrast'])
        assert quality['sharpness_algorithm'] == ALGORITHM


def test_nonfinite_pixels_are_excluded_without_poisoning_nearby_measurements():
    pixels = scene()
    mask = np.ones(pixels.shape, bool)
    mask[64, 64] = False
    expected = measure(pixels, mask)
    for value in (np.nan, np.inf, -np.inf):
        pixels[64, 64] = value
        assert measure(pixels) == expected


def test_legacy_selectors_and_disabled_diagnostics_share_one_score():
    from stash_ai.recognition.quality import sharpness_failed
    quality = measure(scene())
    assert quality['sharpness_normalized'] == quality['sharpness']
    disabled = measure(scene(), diagnostics=False)
    assert disabled == dict(quality, sharpness_noise=None)
    for selector in ('raw', 'normalized'):
        policy = dict(sharpness_metric=selector, min_sharpness=quality['sharpness']+.01)
        assert sharpness_failed(quality, policy)
        assert not sharpness_failed(quality, dict(policy, min_sharpness=quality['sharpness']))


def test_absent_direction_does_not_penalize_sharp_stripes():
    stripes = np.repeat(((np.arange(128)//12) % 2)[None, :], 128, axis=0).astype(np.float32)*120+50
    sharp = measure(stripes)['sharpness']
    assert sharp > 20
    assert measure(stripes.T)['sharpness'] == pytest.approx(sharp, abs=1e-4)
    assert measure(cv2.blur(stripes, (17, 1)))['sharpness'] < sharp*.6


@pytest.mark.parametrize('size', [48, 64, 80, 96, 112, 128, 160, 192, 256])
def test_edges_crossing_tile_boundaries_remain_sharp(size):
    pixels = np.full((size, size), 127, np.float32)
    pixels[size//4:3*size//4, size//4:3*size//4] = 35
    sharp = measure(pixels)['sharpness']
    assert sharp > 20
    assert measure(cv2.GaussianBlur(pixels, (0, 0), 2))['sharpness'] < sharp*.75
    assert measure(motion(pixels, 0))['sharpness'] < sharp*.75


def test_measured_score_drives_body_collection_anchors_and_face_ranking():
    from PIL import Image
    from stash_ai.recognition.body_collection import quality
    from stash_ai.recognition.body_identity import defaults as body_defaults
    from stash_ai.recognition.person_anchor import anchor_gate
    from stash_ai.recognition.quality import sharpness_failed, portrait_components
    from stash_ai.recognition.config import defaults
    from stash_ai.recognition.reference_selection import quality as reference_quality
    from tests.test_recognition_body_mask_overlap import person
    sharp, blurred = scene(), motion(scene(), 0)
    floor = (measure(sharp)['sharpness']+measure(blurred)['sharpness'])/2
    policy = dict(min_person_pixels=1, min_keypoints=1, max_overlap=1, min_sharpness=floor)
    anchors = dict(body_defaults(), anchor_min_keypoints=1, anchor_min_sharpness=floor)
    scores = []
    for pixels, rejected in ((sharp, False), (blurred, True)):
        selected = person([0, 0, 128, 128], np.ones((128, 128), bool))
        with Image.fromarray(pixels.astype(np.uint8)).convert('RGB') as image:
            measured, reason = quality(image, selected, [selected], policy)
        assert reason == ('person_sharpness' if rejected else None)
        gate = anchor_gate(dict(quality=measured, person=selected), anchors)
        assert gate['reason'] == ('body_anchor_sharpness' if rejected else None)
        assert sharpness_failed(measured, policy) is rejected
        scores.append((portrait_components(measured, defaults()['portraits'])['sharpness'],
                       reference_quality(json.dumps(measured))))
    assert all(a > b for a, b in zip(*scores))


def test_torch_batch_matches_numpy_with_mask_holes_noise_and_empty_members():
    import torch
    pixels = scene()
    rng = np.random.default_rng(81)
    images = np.stack([pixels, 128+.15*(pixels-128), motion(pixels, 30),
                       cv2.GaussianBlur(pixels, (0, 0), 2)+rng.normal(0, 5, pixels.shape),
                       np.full_like(pixels, 128)]).astype(np.float32)
    masks = np.ones(images.shape, bool)
    masks[0, 40:65, 30:80] = False
    masks[-1] = False
    expected = batch(images, masks)
    for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        actual = batch(torch.from_numpy(images).to(device), torch.from_numpy(masks).to(device), torch)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=2e-4, atol=2e-4)
    for index in range(len(images)):
        np.testing.assert_allclose(batch(images[index:index+1], masks[index:index+1]),
                                   expected[index:index+1], rtol=1e-6, atol=1e-6)


def test_contracts_refresh_unmasked_bodies_and_preserve_old_frozen_banks():
    from stash_ai.recognition.config import defaults, plain, digest
    from stash_ai.recognition.person_signal import contract as person_contract
    from stash_ai.recognition.body_observations import contract, compatible_contracts
    data = defaults()
    data['sapiensid']['pose']['refinement'].update(enabled=False, mask_sharpness=False)
    config = SimpleNamespace(data=data)
    old_person = person_contract(config, legacy_sharpness=True)
    assert old_person != person_contract(config)
    collection = plain(data['sapiensid']['collection'])
    collection.pop('enabled')
    old = digest(['independent-person-v8-contact-mask-policy', old_person, collection])
    assert old in compatible_contracts(config)[1:]
    assert old != contract(config)

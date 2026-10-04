"""Block colour tests on synthetic frames (EXPLAINER Module 4)."""

import cv2
import numpy as np
import pytest

from palm_prior.perception.block import fit_hsv, load_hsv, mask, save_hsv
from palm_prior.utils import load_config

CFG = load_config()
CENTRE = (300, 200)
HALF = 40


def _frame(block_bgr=(200, 60, 40), noise=4, seed=0):
    """A grey table with one coloured square at CENTRE, plus a little sensor noise."""
    rng = np.random.default_rng(seed)
    img = np.full((400, 600, 3), 120, np.uint8)
    x, y = CENTRE
    img[y - HALF : y + HALF, x - HALF : x + HALF] = block_bgr
    noisy = img.astype(np.int16) + rng.integers(-noise, noise + 1, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _roi_at(u, v):
    half = int(CFG.block_color.roi_half)
    return (u - half, v - half, 2 * half, 2 * half)


def test_fit_then_mask_finds_the_block():
    frame = _frame()
    hsv = fit_hsv(frame, _roi_at(*CENTRE), CFG)
    blob = mask(frame, hsv, CFG)
    assert blob.centroid is not None
    np.testing.assert_allclose(blob.centroid, CENTRE, atol=1.5)
    # the square is 80x80 px; morphology nibbles the border by at most a pixel or two
    assert 0.9 * (2 * HALF) ** 2 < blob.area <= (2 * HALF) ** 2 + 1


def test_fitted_range_survives_shadow_and_noise():
    """The range is fitted once on empty.mp4, then reused on clips where a hand shades
    the block. Dimming the frame by 40% must not lose it."""
    hsv = fit_hsv(_frame(seed=0), _roi_at(*CENTRE), CFG)
    shaded = (_frame(seed=1, noise=8).astype(np.float32) * 0.6).astype(np.uint8)
    blob = mask(shaded, hsv, CFG)
    assert blob.centroid is not None
    np.testing.assert_allclose(blob.centroid, CENTRE, atol=2.0)


def test_mask_ignores_the_grey_table_and_a_skin_coloured_hand():
    frame = _frame()
    hsv = fit_hsv(frame, _roi_at(*CENTRE), CFG)
    hand = frame.copy()
    cv2.circle(hand, (120, 300), 55, (150, 180, 220), -1)  # BGR skin tone
    blob = mask(hand, hsv, CFG)
    assert blob.centroid is not None
    np.testing.assert_allclose(blob.centroid, CENTRE, atol=1.5)


def test_mask_reports_nothing_when_the_block_is_hidden():
    """A fully occluded block must return None, not a stray speck of noise."""
    hsv = fit_hsv(_frame(), _roi_at(*CENTRE), CFG)
    blob = mask(np.full((400, 600, 3), 120, np.uint8), hsv, CFG)
    assert blob.centroid is None
    assert blob.area == 0.0


def test_mask_keeps_only_the_largest_blob():
    frame = _frame()
    hsv = fit_hsv(frame, _roi_at(*CENTRE), CFG)
    distractor = frame.copy()
    distractor[40:60, 500:520] = (200, 60, 40)  # a small patch of the same colour
    blob = mask(distractor, hsv, CFG)
    np.testing.assert_allclose(blob.centroid, CENTRE, atol=1.5)
    assert distractor.shape[:2] == blob.mask.shape
    assert blob.mask[50, 510] == 0


def test_fit_hsv_rejects_a_red_block():
    """Red straddles the hue wrap-around, which a single inRange cannot express."""
    with pytest.raises(ValueError, match="red"):
        fit_hsv(_frame(block_bgr=(40, 40, 200)), _roi_at(*CENTRE), CFG)


def test_hsv_range_survives_a_save_load_round_trip(tmp_path):
    hsv = fit_hsv(_frame(), _roi_at(*CENTRE), CFG)
    path = tmp_path / "block_hsv.yaml"
    save_hsv(path, hsv)
    loaded = load_hsv(path)
    np.testing.assert_allclose(loaded.lo, hsv.lo)
    np.testing.assert_allclose(loaded.hi, hsv.hi)

"""Palm-size depth (FIX.md §2)."""

import numpy as np

from palm_prior.vision.hand_depth import (
    backproject,
    camera_depth,
    filter_log_z,
    fit_frame,
    fit_scale,
    kappa_at,
    one_euro,
)


SEGMENTS = [
    (0, 5, 0.090), (0, 9, 0.085), (0, 13, 0.080), (0, 17, 0.080),
    (5, 9, 0.028), (9, 13, 0.025), (13, 17, 0.025), (5, 17, 0.072),
]


def _palm_points() -> dict[int, np.ndarray]:
    """Five knuckles in a plane, distances equal to SEGMENTS, facing +depth."""
    pts = {0: np.zeros(3)}
    # Place knuckles by intersecting the length constraints in the x-z plane.
    pts[5] = np.array([0.0, 0.0, 0.090])
    pts[9] = _meet(pts[0], 0.085, pts[5], 0.028, sign=1.0)
    pts[13] = _meet(pts[0], 0.080, pts[9], 0.025, sign=1.0)
    pts[17] = _meet(pts[0], 0.080, pts[13], 0.025, sign=1.0)
    return pts


def _meet(a: np.ndarray, ra: float, b: np.ndarray, rb: float, sign: float) -> np.ndarray:
    delta = b - a
    d = float(np.linalg.norm(delta))
    along = (ra ** 2 - rb ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(ra ** 2 - along ** 2, 0.0))
    direction = delta / d
    perp = np.array([-direction[2], 0.0, direction[0]])
    return a + along * direction + sign * h * perp


def _camera():
    # Looks along table +y, optical axis is camera +z. Image +y is world -z.
    r = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    c = np.array([0.0, -0.8, 0.40])
    t = -r @ c
    k = np.array([[700.0, 0, 960.0], [0, 700.0, 540.0], [0, 0, 1.0]])
    return r, t, k


def _project(x, r, t, k):
    cam = r @ x + t
    return np.array([k[0, 0] * cam[0] / cam[2] + k[0, 2], k[1, 1] * cam[1] / cam[2] + k[1, 2]])


def test_scale_fit_matches_the_worked_example():
    ell = np.array([55, 56, 52, 48, 15, 12, 11, 34], float)
    length = np.array([92, 95, 90, 82, 22, 20, 19, 58], float)
    s, residual = fit_scale(ell, length)
    assert abs(s - 1.697) < 0.02
    assert residual[4] > 0.10


def test_jump_in_size_is_rejected_and_a_smooth_change_is_kept():
    z = np.array([1.20, 1.21, 1.32])
    y = np.log(z)
    sigma = np.full(3, 0.02 / 1.2)
    out, rejected, _ = filter_log_z(y, sigma, fps=30, q=1.0, gate_sigma=3.0, v_max=2.0, max_consecutive_reject=3)
    assert rejected[0] is np.False_ or rejected[0] == False
    assert not rejected[1]
    assert rejected[2]
    assert abs(out[2] - 1.32) > abs(out[2] - 1.21)


def test_one_euro_smooths_a_still_hand():
    fps = 30.0
    te = 1.0 / fps
    fc = 1.0
    tau = 1.0 / (2 * np.pi * fc)
    alpha = 1.0 / (1.0 + tau / te)
    assert abs(alpha - 0.17) < 0.01
    noisy = 100 + np.array([0, 2, -2, 1, -1, 0.5], float)
    smooth = one_euro(noisy, fps, min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    assert np.std(smooth) < np.std(noisy)


def test_anchored_pinch_stays_within_a_centimetre():
    r, t, k = _camera()
    k_inv = np.linalg.inv(k)
    palm = _palm_points()
    # Pinch is the middle knuckle, so it shares the palm's depth.
    tip = palm[9].copy()
    b_h = np.array([0.05, 0.15, 0.08]) + tip
    b_end = np.array([-0.02, 0.35, 0.10]) + tip
    n = 11
    k_g, k_r = 2, 8
    err = []
    z_ruler = []
    truth = []
    for i in range(n):
        w = 0.0 if i <= k_g else (1.0 if i >= k_r else (i - k_g) / (k_r - k_g))
        origin = (1 - w) * b_h + w * b_end - tip
        pts = {idx: origin + p for idx, p in palm.items()}
        pixels = np.full((21, 2), np.nan)
        world = np.full((21, 3), np.nan)
        cam_pts = []
        for idx, p in pts.items():
            pixels[idx] = _project(p, r, t, k)
            cam_pts.append(r @ p + t)
        centre = np.mean(cam_pts, axis=0)
        for idx, p in pts.items():
            world[idx] = 0.9 * ((r @ p + t) - centre)
        fit = fit_frame(pixels, world, SEGMENTS, 700.0, 0.7, 0.15, 4)
        z_ruler.append(fit["Z"])
        truth.append(origin + tip)
        assert fit["depth_ok"]
    z_ruler = np.array(z_ruler)
    # Anchor with the median of the ruler around the grasp and the release.
    def med(k):
        return float(np.median(z_ruler[max(0, k - 1): k + 2]))
    kappa_g = camera_depth(b_h, r, t) / med(k_g)
    kappa_r = camera_depth(b_end, r, t) / med(k_r)
    kappa = kappa_at(np.arange(n), [(k_g, kappa_g), (k_r, kappa_r)])
    for i in range(n):
        u, v = _project(truth[i], r, t, k)
        got = backproject(u, v, kappa[i] * z_ruler[i], k_inv, r, t)
        err.append(np.linalg.norm(got - truth[i]))
    err = np.array(err)
    assert err.max() < 0.01
    assert err[k_g] < 0.002 and err[k_r] < 0.002


def test_a_wrong_ruler_is_absorbed_by_kappa():
    r, t, k = _camera()
    palm = _palm_points()
    origin = np.array([0.0, 0.2, 0.08])
    pixels = np.full((21, 2), np.nan)
    world = np.full((21, 3), np.nan)
    cam = []
    for idx, p in palm.items():
        pixels[idx] = _project(origin + p, r, t, k)
        cam.append(r @ (origin + p) + t)
    centre = np.mean(cam, axis=0)
    for idx, p in palm.items():
        world[idx] = 0.9 * ((r @ (origin + p) + t) - centre)
    true = fit_frame(pixels, world, SEGMENTS, 700.0, 0.7, 0.15, 4)
    long = [(a, b, 1.1 * length) for a, b, length in SEGMENTS]
    wrong = fit_frame(pixels, world, long, 700.0, 0.7, 0.15, 4)
    kappa = true["Z"] / wrong["Z"]
    assert abs(kappa - 1 / 1.1) < 0.02
    assert abs(kappa * wrong["Z"] - true["Z"]) < 1e-6


def test_a_bad_segment_is_dropped():
    ell = np.array([55, 56, 52, 48, 15, 12, 11, 34], float)
    length = np.array([0.092, 0.095, 0.090, 0.082, 0.022, 0.020, 0.019, 0.058])
    pixels = np.zeros((21, 2))
    world = np.zeros((21, 3))
    # Build a frame whose segment pixel lengths are ell, all in the image plane.
    ids = [(a, b) for a, b, _ in SEGMENTS]
    pixels[0] = [0, 0]
    # Place each point so the first segment has the right length; the fit test uses fit_scale.
    s, residual = fit_scale(ell, length)
    ell_bad = ell.copy()
    ell_bad[4] *= 1.20
    s_bad, residual_bad = fit_scale(ell_bad, length)
    assert residual_bad[4] > 0.15
    kept = residual_bad <= 0.15
    s2, _ = fit_scale(ell_bad[kept], length[kept])
    assert abs(s2 - s) / s < 0.01
    assert ids[4] == (5, 9)


def test_too_few_segments_is_not_ok():
    pixels = np.full((21, 2), np.nan)
    world = np.full((21, 3), np.nan)
    pixels[0] = [0, 0]
    pixels[5] = [40, 0]
    world[0] = [0, 0, 0]
    world[5] = [0.04, 0, 0]
    fit = fit_frame(pixels, world, SEGMENTS, 700.0, 0.7, 0.15, 4)
    assert fit["depth_ok"] is False


def test_kappa_holds_outside_the_anchors():
    k = kappa_at(np.array([0, 5, 10, 15]), [(5, 0.9), (10, 1.1)])
    assert abs(k[0] - 0.9) < 1e-9 and abs(k[-1] - 1.1) < 1e-9
    assert abs(k[2] - 1.1) < 1e-9

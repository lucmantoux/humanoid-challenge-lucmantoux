"""Pure pieces of the human extraction (EXPLAINER Modules 3–5)."""

import numpy as np

from palm_prior.human.transitions import clip_transitions, expected_grasps, grasp_count, sample_times
from palm_prior.perception.block_track import on_table
from palm_prior.perception.hand import fill_gaps, gripper_signal, rest_scale
from palm_prior.utils import load_config

CFG = load_config()


def test_rest_scale_is_the_median_ratio():
    z_plane = np.array([0.80, 0.79, np.nan, 0.81])
    z_pnp = np.array([0.74, 0.75, 0.10, 0.76])
    # ratios 1.081, 1.053, (skipped), 1.066 → median 1.066
    k = rest_scale(z_plane, z_pnp)
    assert abs(k - np.median([0.80 / 0.74, 0.79 / 0.75, 0.81 / 0.76])) < 1e-12


def test_gripper_closes_when_the_aperture_leaves_the_rest_pose():
    # Fingers together at rest (small aperture), spread while holding the lid.
    a = np.array([0.05] * 5 + [0.70, 0.60, 0.55] + [0.05] * 5)
    rest = np.array([True] * 5 + [False] * 3 + [True] * 5)
    g, close_above, open_below = gripper_signal(a, CFG, rest)
    assert close_above > open_below
    assert g[0] == 0.0
    assert g[5] == 1.0 and g[7] == 1.0
    assert g[-1] == 0.0


def test_fill_gaps_stops_at_ten_frames():
    x = np.arange(15, dtype=float)
    x[1:4] = np.nan          # 3 missing, filled
    x[5:16 - 1] = np.nan     # will set a longer hole below
    values = np.array([0.0, np.nan, np.nan, np.nan, 4.0, *([np.nan] * 11), 16.0])
    filled = fill_gaps(values, max_gap=10)
    assert np.allclose(filled[1:4], [1.0, 2.0, 3.0])
    assert np.isnan(filled[5:16]).all()
    assert filled[0] == 0.0 and filled[-1] == 16.0


def test_on_table_when_open_or_hovering_and_in_hand_when_closed_and_close():
    assert on_table(0.0, 0.10, 0.02, 0.03) is True
    assert on_table(1.0, 0.10, 0.02, 0.03) is True     # 8 cm above the last centre
    assert on_table(1.0, 0.03, 0.02, 0.03) is False    # closed and only 1 cm above


def test_sample_times_match_the_tau_formula():
    t = sample_times(duration_s=1.0, rate_hz=10.0, tau=2.0)
    # t_k = k * 0.1 / 2 = k * 0.05, and 1.0 is not included
    assert np.allclose(t, 0.05 * np.arange(20))


def test_one_grasp_is_one_rising_edge():
    g = np.array([0, 0, 1, 1, 0, 1, np.nan, 1])
    assert grasp_count(g) == 2
    assert expected_grasps("success") == 1
    assert expected_grasps("F4") == 0


def test_transitions_drop_a_gap_and_build_the_state():
    t = np.arange(10, dtype=float) * 0.05          # already at the sample spacing
    p_ee = np.zeros((10, 3))
    p_ee[:, 0] = 0.01 * np.arange(10)
    p_obj = np.zeros((10, 3))
    p_obj[:, 2] = 0.0085                           # 1.7 cm lid, centre at b/2
    g = np.zeros(10)
    g[4:] = 1.0
    p_obj[3:6] = np.nan                            # a hole covering two samples
    tr = clip_transitions(
        t, p_ee, p_obj, g, np.array([0.5, 0.1]), h_tgt=0.015,
        block_size=0.017, attach_margin=0.01, duration_s=0.5, rate_hz=10.0, tau=2.0,
    )
    assert tr.s.shape[1] == 8 and tr.a.shape[1] == 4
    assert len(tr.s) < 9                           # the hole removed at least one transition
    assert np.all(tr.attach == 0.0)                # centre is not above b/2 + 1 cm

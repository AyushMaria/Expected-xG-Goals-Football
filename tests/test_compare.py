"""Tests for xg.compare (synthetic data)."""

import numpy as np
import pytest

from xg.compare import log_loss_diff_ci, per_shot_log_loss, ratio_ci


def _sim(matches=400, shots=25, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(matches), shots)
    p = rng.uniform(0.02, 0.4, matches * shots)
    y = rng.binomial(1, p)
    return g, y, p


def test_ratio_ci_covers_one_for_a_calibrated_model():
    g, y, p = _sim()
    r, lo, hi = ratio_ci(g, y, p, n_boot=500)
    assert lo < 1 < hi and lo < r < hi
    r2, lo2, hi2 = ratio_ci(g, y, p * 1.3, n_boot=500)    # over-predicting by 30% is detected
    assert lo2 > 1


def test_log_loss_diff_sign_and_ci():
    g, y, p = _sim()
    d, lo, hi = log_loss_diff_ci(g, y, p, np.full_like(p, y.mean()), n_boot=500)
    assert d < 0 and hi < 0                               # the informative model beats a constant
    d0, lo0, hi0 = log_loss_diff_ci(g, y, p, p, n_boot=200)
    assert d0 == 0 and lo0 == 0 and hi0 == 0


def test_per_shot_log_loss_values():
    assert per_shot_log_loss([1], [0.5])[0] == pytest.approx(np.log(2))
    assert np.isfinite(per_shot_log_loss([1, 0], [0.0, 1.0])).all()   # clipped, never infinite

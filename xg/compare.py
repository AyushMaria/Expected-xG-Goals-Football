"""Bootstrap confidence intervals over matches, for comparing xG models.

Shots within a match are not independent, so intervals resample whole matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_shot_log_loss(y, p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _match_sums(groups, **cols) -> pd.DataFrame:
    df = pd.DataFrame({"g": np.asarray(groups), **{k: np.asarray(v, dtype=float) for k, v in cols.items()}})
    df["n"] = 1.0
    return df.groupby("g").sum()


def _resample(m: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Bootstrap totals: each row is one resample of matches (with replacement)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(n_boot, len(m)))
    return pd.DataFrame({c: m[c].to_numpy()[idx].sum(axis=1) for c in m.columns})


def ratio_ci(groups, y, p, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Predicted ÷ actual goals with a match-bootstrap confidence interval: (estimate, low, high)."""
    m = _match_sums(groups, p=p, y=y)
    b = _resample(m, n_boot, seed)
    r = b["p"] / b["y"]
    a = (1 - level) / 2
    return float(m["p"].sum() / m["y"].sum()), float(r.quantile(a)), float(r.quantile(1 - a))


def log_loss_diff_ci(groups, y, p_a, p_b, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Mean per-shot log loss of A minus B (negative = A is better), with a paired match-bootstrap CI."""
    la, lb = per_shot_log_loss(y, p_a), per_shot_log_loss(y, p_b)
    m = _match_sums(groups, d=la - lb)
    b = _resample(m, n_boot, seed)
    d = b["d"] / b["n"]
    a = (1 - level) / 2
    return float(m["d"].sum() / m["n"].sum()), float(d.quantile(a)), float(d.quantile(1 - a))

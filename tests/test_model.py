"""Tests for xg.model (fast, synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest

from xg.model import (FEATURES, GROUP, TARGET, calibration_table, expected_calibration_error,
                      modelling_set, preprocessor, scores, split_by_match)


def _shots(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "shot id": [f"s{i}" for i in range(n)],
        "game_id": [f"g{i // 10}" for i in range(n)],            # 10 shots per match
        "season_id": rng.choice(["1990/1991", "2015/2016"], n, p=[0.1, 0.9]),
        "type of shot": rng.choice(["Open Play", "Penalty"], n, p=[0.9, 0.1]),
        "body part": rng.choice(["Right Foot", "Left Foot", "Head", "Other"], n),
        "play pattern": rng.choice(["Regular Play", "From Corner", "From Counter"], n),
        "first time": rng.choice([True, False], n),
        "Pass Type": rng.choice(["Ground Pass", "High Pass", "Low Pass", "None"], n),
        "x location shot": rng.uniform(90, 119, n), "y location shot": rng.uniform(20, 60, n),
        "x gk location": rng.uniform(114, 120, n), "y gk location": rng.uniform(36, 44, n),
        "shot_angle": rng.uniform(0, 90, n),
        "Number of opponents in 5 yards": rng.integers(0, 4, n),
        "Players between goal": rng.integers(0, 4, n),
        "outcome": rng.choice(["Goal", "Saved", "Off T"], n, p=[0.1, 0.5, 0.4]),
    })


def test_modelling_set_filters_and_encodes():
    raw = _shots()
    d = modelling_set(raw)
    assert d["type of shot"].eq("Open Play").all()
    assert (d["season_id"].str[:4].astype(int) >= 2000).all()
    assert set(d["body part"]) <= {"Foot", "Head"}
    assert set(d["play pattern"]) <= {"Regular Play", "Non Regular Play"}
    assert set(d[TARGET]) <= {0, 1}
    # distance from (100, 40) to the goal centre is 20 yards
    one = modelling_set(raw.assign(**{"x location shot": 100.0, "y location shot": 40.0}))
    assert one["shot distance"].iloc[0] == pytest.approx(20.0)


def test_split_keeps_each_match_on_one_side():
    d = modelling_set(_shots(1000))
    train, test = split_by_match(d, test_size=0.2, seed=1)
    assert set(train[GROUP]).isdisjoint(set(test[GROUP]))
    assert len(train) + len(test) == len(d)


def test_preprocessor_ignores_unseen_categories():
    d = modelling_set(_shots())
    pre = preprocessor().fit(d[FEATURES])
    new = d[FEATURES].head(3).copy()
    new["Pass Type"] = "Some New Pass Type"
    assert pre.transform(new).shape[1] == pre.transform(d[FEATURES].head(3)).shape[1]


def test_calibration_metrics():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.6, 200_000)
    y = rng.binomial(1, p)                                      # perfectly calibrated by construction
    assert expected_calibration_error(y, p) < 0.005
    assert scores(y, p)["xg_over_goals"] == pytest.approx(1.0, abs=0.01)
    assert expected_calibration_error(y, np.clip(p * 1.5, 0, 1)) > 0.05   # over-predicting is caught
    t = calibration_table(y, p, bins=5)
    assert len(t) == 5 and t["shots"].sum() == len(p)

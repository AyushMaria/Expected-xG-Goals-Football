"""Tests for xg.tables (synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest

from xg.tables import finishing_table, league_table


def _match(h, a, hs, as_):
    return {"home_team": {"home_team_name": h}, "away_team": {"away_team_name": a},
            "home_score": hs, "away_score": as_}


def test_league_table_points_and_order():
    t = league_table([_match("A", "B", 2, 0), _match("B", "C", 1, 1), _match("C", "A", 0, 3)])
    assert t.loc["A", ["points", "gf", "ga", "played"]].tolist() == [6, 5, 0, 2]
    assert t.loc["B", "points"] == 1 and t.loc["C", "points"] == 1
    # B and C level on points; B has the better goal difference (-1 vs -3)
    assert t.index.tolist() == ["A", "B", "C"] and t["position"].tolist() == [1, 2, 3]


def test_finishing_table_z_score():
    oof = pd.DataFrame({"player name": ["p"] * 100 + ["q"] * 10,
                        "xg": [0.1] * 110, "goal": [1] * 20 + [0] * 80 + [0] * 10})
    t = finishing_table(oof, ["player name"], min_shots=50)
    assert list(t.index) == ["p"]                              # q filtered out (10 shots)
    row = t.loc["p"]
    assert row["goals"] == 20 and row["xg"] == pytest.approx(10)
    assert row["z"] == pytest.approx(10 / np.sqrt(100 * 0.1 * 0.9))   # 10 goals above xG / sd 3.0

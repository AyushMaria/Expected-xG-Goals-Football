"""Tests for xg.features and xg.data. They need no network access."""

import gzip
import json
import math

import numpy as np
import pandas as pd
import pytest

from xg.data import OpenData, select_matches
from xg.features import add_angle_features, player_density, shots_from_events


# ----------------------------------------------------------------- helpers
def _player(x, y, teammate=False, position="Center Back"):
    return {"location": [x, y], "teammate": teammate, "position": {"name": position}}


def _shot(shot_id, x, y, freeze_frame=None, key_pass_id=None, first_time=None):
    shot = {
        "outcome": {"name": "Goal"},
        "technique": {"name": "Normal"},
        "type": {"name": "Open Play"},
        "body_part": {"name": "Right Foot"},
        "statsbomb_xg": 0.1,
    }
    if freeze_frame is not None:
        shot["freeze_frame"] = freeze_frame
    if key_pass_id is not None:
        shot["key_pass_id"] = key_pass_id
    if first_time is not None:
        shot["first_time"] = first_time
    return {
        "id": shot_id, "type": {"name": "Shot"}, "location": [x, y], "duration": 0.5,
        "play_pattern": {"name": "Regular Play"}, "possession_team": {"name": "Team A"},
        "player": {"name": "Player A"}, "shot": shot,
    }


def _pass(pass_id, height):
    return {"id": pass_id, "type": {"name": "Pass"}, "pass": {"height": {"name": height}}}


# ----------------------------------------------------------------- geometry
def test_penalty_spot_angle():
    # From the penalty spot (108, 40) the goal (8 yards wide, 12 yards away) subtends 2*atan(4/12) = 36.87 degrees
    df = add_angle_features(pd.DataFrame({"x location shot": [108.0], "y location shot": [40.0]}))
    assert df["shot_angle"].iloc[0] == pytest.approx(math.degrees(2 * math.atan(4 / 12)), abs=0.05)


def test_angle_on_goal_line_is_zero_not_nan():
    df = add_angle_features(pd.DataFrame({"x location shot": [120.0], "y location shot": [30.0]}))
    assert df["shot_angle"].iloc[0] == 0


def test_player_density_inside_and_outside_triangle():
    assert player_density(108, 40, 114, 40)          # straight between shot and goal centre
    assert not player_density(108, 40, 114, 50)      # wide of the far post
    assert not player_density(108, 40, 100, 40)      # behind the shooter
    assert not player_density(120, 40, 120, 40)      # shot from the goal line: no triangle


# ----------------------------------------------------------------- goalkeeper fix (card #5)
def test_goalkeeper_position_is_reset_between_shots():
    keeper = _player(118, 41, position="Goalkeeper")
    events = [
        _shot("s1", 100, 40, freeze_frame=[keeper]),
        _shot("s2", 100, 40, freeze_frame=[_player(110, 40)]),   # freeze frame without a keeper
        _shot("s3", 100, 40),                                    # no freeze frame at all
    ]
    df = shots_from_events(events).set_index("shot id")
    assert (df.loc["s1", "x gk location"], df.loc["s1", "y gk location"]) == (118, 41)
    # the old notebook reused s1's keeper here; now it falls back to the goal centre and is flagged
    assert (df.loc["s2", "x gk location"], df.loc["s2", "y gk location"]) == (120, 40)
    assert df.loc["s2", "freeze_frame_available"] and not df.loc["s2", "gk_in_freeze_frame"]
    assert not df.loc["s3", "freeze_frame_available"]


def test_opponent_counts_ignore_teammates_and_keeper_for_5_yards():
    frame = [
        _player(102, 40),                                  # opponent within 5 yards, in the triangle
        _player(103, 41, teammate=True),                   # teammate: never counted
        _player(118, 40, position="Goalkeeper"),           # keeper: in the triangle, not within 5 yards
    ]
    row = shots_from_events([_shot("s", 100, 40, freeze_frame=frame)]).iloc[0]
    assert row["Number of opponents in 5 yards"] == 1
    assert row["Players between goal"] == 2               # outfield opponent + keeper, as in the original


# ----------------------------------------------------------------- key pass
def test_key_pass_height_lookup():
    events = [_pass("p1", "Low Pass"), _shot("s1", 100, 40, key_pass_id="p1"), _shot("s2", 100, 40)]
    df = shots_from_events(events).set_index("shot id")
    assert (df.loc["s1", "Pass id"], df.loc["s1", "Pass Type"]) == ("p1", "Low Pass")
    assert (df.loc["s2", "Pass id"], df.loc["s2", "Pass Type"]) == ("No Pass", "None")


def test_first_time_defaults_to_false():
    df = shots_from_events([_shot("a", 100, 40), _shot("b", 100, 40, first_time=True)]).set_index("shot id")
    assert df.loc["a", "first time"] == False and df.loc["b", "first time"] == True  # noqa: E712


# ----------------------------------------------------------------- data cache
def _write_cache(root, relpath, obj):
    path = root / (relpath + ".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(obj, fh)


def test_select_matches_reads_cache_and_filters(tmp_path, monkeypatch):
    comps = [
        {"competition_id": 11, "season_id": 1, "competition_name": "La Liga", "season_name": "2015/2016", "competition_gender": "male"},
        {"competition_id": 37, "season_id": 2, "competition_name": "FA WSL", "season_name": "2020/2021", "competition_gender": "female"},
    ]
    matches = [
        {"match_id": 1, "home_team": {"home_team_name": "Barcelona"}, "away_team": {"away_team_name": "Sevilla"}},
        {"match_id": 2, "home_team": {"home_team_name": "Getafe"}, "away_team": {"away_team_name": "Sevilla"}},
    ]
    _write_cache(tmp_path, "competitions.json", comps)
    _write_cache(tmp_path, "matches/11/1.json", matches)
    # any network access would fail the test
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network used")))
    od = OpenData(tmp_path)
    assert [m["match_id"] for m in select_matches(od, "La Liga")] == [1, 2]
    picked = select_matches(od, "La Liga", ["2015/2016"], team="Barcelona")
    assert [m["match_id"] for m in picked] == [1] and picked[0]["season_name"] == "2015/2016"
    assert select_matches(od, gender="female", competition_name="La Liga") == []

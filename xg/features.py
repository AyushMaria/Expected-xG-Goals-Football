"""Turn StatsBomb match events into one row per shot.

This is a port of ``obtain_shots_match`` and the shot-angle cells from
``src/Cleaning_Data.ipynb``. It produces the same columns with the same
meaning, with one deliberate change (card #5):

* The original code only set the goalkeeper position when the freeze frame
  contained the opposing goalkeeper and never reset it between shots, so a
  shot whose freeze frame had no goalkeeper silently reused the previous
  shot's goalkeeper position. Here every shot starts with no goalkeeper
  position. If the freeze frame has no opposing goalkeeper, the position is
  set to the centre of the goal line (120, 40), the same default the original
  used for shots with no freeze frame, and ``gk_in_freeze_frame`` is False.

Two columns are added after the original 24:
    freeze_frame_available  - the shot has a StatsBomb freeze frame
    gk_in_freeze_frame      - the freeze frame contains the opposing goalkeeper
    competition             - competition name, e.g. "La Liga"

Pitch coordinates follow StatsBomb: 120 x 80 yards, attacking towards x = 120,
goal posts at y = 36 and y = 44.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOAL_X = 120.0
POST_LOW_Y, POST_HIGH_Y = 36.0, 44.0
GOAL_CENTRE = (120.0, 40.0)
GOAL_WIDTH = POST_HIGH_Y - POST_LOW_Y  # 8 yards

ORIGINAL_COLUMNS = [
    "shot id", "play pattern", "x location shot", "y location shot", "duration", "outcome",
    "technique used", "first time", "x gk location", "y gk location", "body part", "type of shot",
    "Number of opponents in 5 yards", "Players between goal", "player name", "Team Name",
    "official xg", "Pass id", "Pass Type", "game_id", "season_id",
    "distance_from_goalpost_a", "distance_from_goalpost_b", "shot_angle",
]
EXTRA_COLUMNS = ["freeze_frame_available", "gk_in_freeze_frame", "competition"]


def player_density(x_shot: float, y_shot: float, x_player: float, y_player: float) -> bool:
    """True if a player stands inside the triangle formed by the shot location and the two posts."""
    if GOAL_X - x_shot == 0:
        return False
    x_difference = x_player - x_shot
    slope_1 = (POST_LOW_Y - y_shot) / (GOAL_X - x_shot)
    slope_2 = (POST_HIGH_Y - y_shot) / (GOAL_X - x_shot)
    return (x_difference >= 0) and ((y_shot + slope_1 * x_difference) < y_player < (y_shot + slope_2 * x_difference))


def _key_pass(shot_event: dict, events_by_id: dict) -> tuple[str, str]:
    """Return (key pass id, pass height name), matching the original notebook's behaviour.

    No key pass -> ('No Pass', 'None'); key pass id not found in the match -> (id, '').
    """
    pass_id = shot_event["shot"].get("key_pass_id")
    if pass_id is None:
        return "No Pass", "None"
    ev = events_by_id.get(pass_id)
    if ev is None:
        return pass_id, ""
    try:
        return pass_id, ev["pass"]["height"]["name"]
    except KeyError:  # the original try/except treated this like "no key pass"
        return "No Pass", "None"


def shots_from_events(events: list[dict]) -> pd.DataFrame:
    """One row per shot in a match, with the original 19 per-shot columns plus the two flags."""
    events_by_id = {ev["id"]: ev for ev in events}
    rows = []
    for ev in events:
        if ev["type"]["name"] != "Shot":
            continue
        shot = ev["shot"]
        x_shot, y_shot = ev["location"][0], ev["location"][1]
        pass_id, pass_height = _key_pass(ev, events_by_id)

        frame = shot.get("freeze_frame")
        opponents_5_yards = 0
        opponents_between = 0
        gk_x, gk_y = GOAL_CENTRE          # reset for every shot (card #5)
        gk_found = False
        if frame is not None:
            for player in frame:
                if player["teammate"]:
                    continue
                x_p, y_p = player["location"][0], player["location"][1]
                is_gk = player["position"]["name"] == "Goalkeeper"
                if (x_shot - x_p) ** 2 + (y_shot - y_p) ** 2 <= 25 and not is_gk:
                    opponents_5_yards += 1
                if player_density(x_shot, y_shot, x_p, y_p):
                    opponents_between += 1   # note: includes the goalkeeper, as in the original
                if is_gk:
                    gk_x, gk_y, gk_found = x_p, y_p, True

        rows.append({
            "shot id": ev["id"],
            "play pattern": ev["play_pattern"]["name"],
            "x location shot": x_shot,
            "y location shot": y_shot,
            "duration": ev.get("duration"),
            "outcome": shot["outcome"]["name"],
            "technique used": shot["technique"]["name"],
            "first time": bool(shot.get("first_time", False)),
            "x gk location": gk_x,
            "y gk location": gk_y,
            "body part": shot["body_part"]["name"],
            "type of shot": shot["type"]["name"],
            "Number of opponents in 5 yards": opponents_5_yards,
            "Players between goal": opponents_between,
            "player name": ev["player"]["name"],
            "Team Name": ev["possession_team"]["name"],
            "official xg": shot.get("statsbomb_xg"),
            "Pass id": pass_id,
            "Pass Type": pass_height,
            "freeze_frame_available": frame is not None,
            "gk_in_freeze_frame": gk_found,
        })
    return pd.DataFrame(rows)


def add_angle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distances to each post and the shot angle in degrees (cells 20 and 24 of Cleaning_Data)."""
    dx = df["x location shot"] - GOAL_X
    df["distance_from_goalpost_a"] = np.sqrt(dx ** 2 + (df["y location shot"] - POST_LOW_Y) ** 2)
    df["distance_from_goalpost_b"] = np.sqrt(dx ** 2 + (df["y location shot"] - POST_HIGH_Y) ** 2)
    a, b = df["distance_from_goalpost_a"], df["distance_from_goalpost_b"]
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.degrees(np.arccos((a ** 2 + b ** 2 - GOAL_WIDTH ** 2) / (2 * a * b)))
    df["shot_angle"] = angle.round(1).fillna(0)
    return df


def build_shot_table(od, matches: list[dict], workers: int = 8, progress: bool = True) -> pd.DataFrame:
    """Download events for ``matches`` (see ``xg.data.select_matches``) and return the shot table."""
    od.prefetch_events([m["match_id"] for m in matches], workers=workers)
    frames = []
    for i, m in enumerate(matches, 1):
        shots = shots_from_events(od.events(m["match_id"]))
        if shots.empty:
            continue
        shots["game_id"] = str(m["match_id"])
        shots["season_id"] = m["season_name"]
        shots["competition"] = m["competition_name"]
        frames.append(shots)
        if progress and i % 100 == 0:
            print(f"  processed {i}/{len(matches)} matches")
    df = pd.concat(frames, ignore_index=True)
    df = add_angle_features(df)
    return df[ORIGINAL_COLUMNS + EXTRA_COLUMNS]

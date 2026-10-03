"""Team and player tables from out-of-fold xG (card #6).

Every shot's xG comes from ``results/oof_xg.parquet``: a prediction from a model that never saw
that shot's match. All xG and goal counts here are **open play only** (penalties and direct free
kicks are not modelled); league tables use full match results from StatsBomb's match data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xg.data import OpenData


def match_sides(od: OpenData, match: dict) -> dict:
    """Map the team names used in a match's events to 'home'/'away', using StatsBomb team ids.

    Event data and match metadata sometimes spell a team differently (e.g. 'Marseille' vs
    'Olympique Marseille'), so names are matched through the Starting XI events' team ids.
    """
    home_id = match["home_team"]["home_team_id"]
    sides = {}
    for ev in od.events(match["match_id"]):
        if ev["type"]["name"] != "Starting XI":
            continue
        sides[ev["team"]["name"]] = "home" if ev["team"]["id"] == home_id else "away"
        if len(sides) == 2:
            break
    return sides


def league_table(matches: list[dict]) -> pd.DataFrame:
    """Final league table (3 points for a win, 1 for a draw) from match results."""
    rows = []
    for m in matches:
        h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
        hs, as_ = m["home_score"], m["away_score"]
        rows.append((h, hs, as_)); rows.append((a, as_, hs))
    t = pd.DataFrame(rows, columns=["team", "gf", "ga"])
    t["points"] = np.select([t.gf > t.ga, t.gf == t.ga], [3, 1], 0)
    t["played"] = 1
    t = t.groupby("team").sum()
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["points", "gd", "gf"], ascending=False)
    t["position"] = np.arange(1, len(t) + 1)
    return t[["position", "played", "points", "gf", "ga", "gd"]]


def team_season_table(oof: pd.DataFrame, od: OpenData, competition: str, season: str) -> pd.DataFrame:
    """League table plus open-play xG and goals, for and against, for one competition-season."""
    comp = next(c for c in od.competitions()
                if c["competition_name"] == competition and c["season_name"] == season and c["competition_gender"] == "male")
    matches = od.matches(comp["competition_id"], comp["season_id"])
    table = league_table(matches)

    names = {}  # (match_id, event team name) -> canonical team name from match metadata
    for m in matches:
        canon = {"home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"]}
        for ev_name, side in match_sides(od, m).items():
            names[(str(m["match_id"]), ev_name)] = (canon[side], canon["away" if side == "home" else "home"])

    s = oof[oof["game_id"].isin({str(m["match_id"]) for m in matches})].copy()
    keys = list(zip(s["game_id"], s["Team Name"]))
    s["team"] = [names[k][0] for k in keys]
    s["opponent"] = [names[k][1] for k in keys]
    f = s.groupby("team").agg(op_shots=("xg", "size"), op_goals=("goal", "sum"), op_xg=("xg", "sum"))
    a = s.groupby("opponent").agg(op_goals_against=("goal", "sum"), op_xg_against=("xg", "sum"))
    out = table.join(f).join(a)
    out["op_xg_diff"] = out["op_xg"] - out["op_xg_against"]
    out["xg_diff_rank"] = out["op_xg_diff"].rank(ascending=False, method="min").astype(int)
    return out


def finishing_table(oof: pd.DataFrame, by: list[str], min_shots: int = 50) -> pd.DataFrame:
    """Goals vs xG per group, with how unusual the gap is.

    ``z`` = (goals - xG) / sqrt(sum p(1-p)): the gap measured in standard deviations of what an
    average finisher would score from the same shots. |z| > 2 is unlikely to be luck alone.
    """
    g = oof.assign(var=oof["xg"] * (1 - oof["xg"])).groupby(by).agg(
        shots=("xg", "size"), goals=("goal", "sum"), xg=("xg", "sum"), var=("var", "sum"))
    g = g[g["shots"] >= min_shots].copy()
    g["goals_minus_xg"] = g["goals"] - g["xg"]
    g["z"] = g["goals_minus_xg"] / np.sqrt(g["var"])
    return g.drop(columns="var").sort_values("goals_minus_xg", ascending=False)

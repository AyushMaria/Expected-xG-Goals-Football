"""Print Markdown summary tables for a shot table built by build_shots.py.

    python scripts/summarize_shots.py data/shots_all_men.parquet

Used to produce docs/DATA.md.
"""

from __future__ import annotations

import sys

import pandas as pd


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:,}" if isinstance(v, (int,)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def summarize(path: str) -> str:
    d = pd.read_parquet(path)
    d["goal"] = d["outcome"].eq("Goal")
    d["start_year"] = d["season_id"].str[:4].astype(int)
    out = []

    o = d[d["type of shot"].eq("Open Play")]
    out.append(f"**{d['game_id'].nunique():,} matches, {len(d):,} shots, of which {len(o):,} open play "
               f"({o['goal'].sum():,} goals, {100 * o['goal'].mean():.1f}% conversion).** "
               f"Open-play shots with a freeze frame: {100 * o['freeze_frame_available'].mean():.2f}%; "
               f"with the goalkeeper in the freeze frame: {100 * o['gk_in_freeze_frame'].mean():.2f}%.\n")

    g = (d.groupby("competition")
           .agg(Matches=("game_id", "nunique"), Shots=("goal", "size"),
                Open_play=("type of shot", lambda s: int(s.eq("Open Play").sum())),
                Goals=("goal", "sum"), First=("start_year", "min"), Last=("start_year", "max"),
                sb=("official xg", "sum"))
           .sort_values("Shots", ascending=False))
    g["Seasons"] = g["First"].astype(str) + "–" + g["Last"].astype(str)
    g["Conversion"] = (100 * g["Goals"] / g["Shots"]).map("{:.1f}%".format)
    g["StatsBomb xG / goals"] = (g["sb"] / g["Goals"]).map("{:.2f}".format)
    g = g.reset_index().rename(columns={"competition": "Competition", "Open_play": "Open-play shots"})
    for c in ["Matches", "Shots", "Open-play shots", "Goals"]:
        g[c] = g[c].astype(int)
    out.append("### By competition (all shot types)\n")
    out.append(_table(g[["Competition", "Seasons", "Matches", "Shots", "Open-play shots", "Goals",
                         "Conversion", "StatsBomb xG / goals"]]))

    old = o[o["start_year"] < 2000]
    out.append(f"\nPre-2000 matches: {old['game_id'].nunique()} matches, {len(old):,} open-play shots "
               f"({100 * len(old) / len(o):.1f}% of open play).\n")

    barca_games = set(o.loc[o["Team Name"].eq("Barcelona"), "game_id"])
    ll = o[o["competition"].eq("La Liga")]
    s15 = o[o["season_id"].eq("2015/2016")]
    rows = [
        ("La Liga, all matches", ll),
        ("La Liga: Barcelona's shots", ll[ll["Team Name"].eq("Barcelona")]),
        ("La Liga: opponents' shots against Barcelona", ll[ll["game_id"].isin(barca_games) & ~ll["Team Name"].eq("Barcelona")]),
        ("La Liga 2015/16, excluding Barcelona's matches", s15[s15["competition"].eq("La Liga") & ~s15["game_id"].isin(barca_games)]),
        ("Premier League 2015/16", s15[s15["competition"].eq("Premier League")]),
        ("Serie A 2015/16", s15[s15["competition"].eq("Serie A")]),
        ("Ligue 1 2015/16", s15[s15["competition"].eq("Ligue 1")]),
    ]
    t = pd.DataFrame([{
        "Group (open play)": name, "Shots": int(len(x)), "Conversion": f"{100 * x['goal'].mean():.1f}%",
        "StatsBomb xG / goals": f"{x['official xg'].sum() / x['goal'].sum():.2f}",
    } for name, x in rows])
    out.append("### League vs. Barcelona effect (open play)\n")
    out.append(_table(t))
    return "\n".join(out)


if __name__ == "__main__":
    print(summarize(sys.argv[1] if len(sys.argv) > 1 else "data/shots_all_men.parquet"))

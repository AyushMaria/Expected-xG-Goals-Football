"""Exploratory figures for the revised thesis (Methodology, Section 3.3).

Uses open-play shots from the *training* matches only (the same match-grouped split as
train_evaluate.py), so nothing seen here comes from the held-out test matches.

    python scripts/plot_eda.py --out thesis/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xg.model import modelling_set, split_by_match  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1d1d1b", "#5f5e5a", "#e6e5df"
BANDS = [0, 12, 18, 25, 200]
BAND_LABELS = ["under 12 yd", "12–18 yd", "18–25 yd", "25 yd and over"]
BAND_COLORS = [BLUE, ORANGE, AQUA, INK2]
BAND_MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def pct(s) -> float:
    return 100 * float(np.mean(s))


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)


def bar_rate(ax, labels, rates, ns, color=BLUE):
    """Bars of goal % with the value and number of shots written on each bar."""
    x = np.arange(len(labels))
    ax.bar(x, rates, color=color, width=0.65)
    top = max(rates)
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.text(i, r + top * 0.02, f"{r:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, [f"{lab}\n" + f"n={n:,}" for lab, n in zip(labels, ns)])
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel("Goals per shot (%)")
    ax.grid(axis="x", visible=False)


def bar_count(ax, labels, counts, color=BLUE):
    x = np.arange(len(labels))
    ax.bar(x, counts, color=color, width=0.65)
    top = max(counts)
    for i, c in enumerate(counts):
        ax.text(i, c + top * 0.02, f"{c:,}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel("Shots")
    ax.grid(axis="x", visible=False)


def box(ax, groups, labels, ylabel):
    bp = ax.boxplot(groups, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.55,
                    medianprops={"color": ORANGE, "linewidth": 2})
    for b in bp["boxes"]:
        b.set(facecolor="#dce9f8", edgecolor=BLUE)
    for k in ("whiskers", "caps"):
        for w in bp[k]:
            w.set(color=BLUE)
    for i, g in enumerate(groups, start=1):
        ax.text(i + 0.3, np.median(g), f"{np.median(g):.1f}", va="center", ha="left", fontsize=9, color=ORANGE)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def band_lines(ax, t, col, order, labels):
    """Goal % against a feature, one line per distance band (the confounder)."""
    band = pd.cut(t["shot distance"], BANDS, labels=BAND_LABELS)
    x = np.arange(len(order))
    for lab, c, m in zip(BAND_LABELS, BAND_COLORS, BAND_MARKERS):
        sub = t[band == lab]
        r = [pct(sub[sub[col] == v].goal) for v in order]
        ax.plot(x, r, color=c, marker=m, linewidth=2)
        ax.text(x[-1] + 0.08, r[-1], lab, color=c, va="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(order) - 1 + 0.9)
    ax.set_ylabel("Goals per shot (%)")


def band_bars(ax, t, col, order, labels, colors):
    """Goal % by distance band, one bar per category."""
    band = pd.cut(t["shot distance"], BANDS, labels=BAND_LABELS)
    w = 0.8 / len(order)
    x = np.arange(len(BAND_LABELS))
    for j, (v, lab, c) in enumerate(zip(order, labels, colors)):
        r = [pct(t[(band == b) & (t[col] == v)].goal) for b in BAND_LABELS]
        ax.bar(x + (j - (len(order) - 1) / 2) * w, r, w, color=c, label=lab)
    ax.set_xticks(x, BAND_LABELS)
    ax.set_xlabel("Shot distance")
    ax.set_ylabel("Goals per shot (%)")
    ax.legend(frameon=False, ncol=len(order), loc="upper right")
    ax.grid(axis="x", visible=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/shots_all_men.parquet")
    ap.add_argument("--out", default="thesis/images")
    ap.add_argument("--stats", default="results/eda_stats.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(a.shots)
    t, _ = split_by_match(modelling_set(raw), seed=42)
    train_ids = set(t.game_id)
    # all open-play shots of the training matches, before body parts are grouped (for counts)
    op = raw[raw["type of shot"].eq("Open Play") & (raw.season_id.str[:4].astype(int) >= 2000)
             & raw.game_id.isin(train_ids)].copy()
    op["goal"] = op.outcome.eq("Goal").astype(int)
    op["body"] = op["body part"].replace({"Right Foot": "Foot", "Left Foot": "Foot"})
    stats = {"train_shots": len(t), "train_goals": int(t.goal.sum()), "train_matches": t.game_id.nunique(),
             "goal_rate": pct(t.goal)}

    # --- shot distance
    b = (t["shot distance"] // 5).clip(upper=8).astype(int)
    labs = [f"{5*i}–{5*i+5}" for i in range(8)] + ["40+"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    g = t.groupby(b).goal
    bar_rate(ax, labs, (100 * g.mean()).tolist(), g.size().tolist())
    ax.set_xlabel("Shot distance (yards)")
    ax.set_title("The chance of scoring falls quickly with distance")
    save(fig, out, "shot distance.png")
    stats["distance_rate"] = dict(zip(labs, (100 * g.mean()).round(1).tolist()))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(0, 46, 1.5)
    for k, c, lab in [(0, BLUE, "No goal"), (1, ORANGE, "Goal")]:
        ax.hist(t[t.goal == k]["shot distance"].clip(upper=45), bins=bins, density=True,
                histtype="step", linewidth=2, color=c)
    ax.text(31, 0.031, "No goal", color=BLUE, fontsize=10)
    ax.text(17.5, 0.055, "Goal", color=ORANGE, fontsize=10)
    ax.set_xlabel("Shot distance (yards, 45+ grouped at 45)")
    ax.set_ylabel("Share of shots per yard")
    ax.set_title("Goals come from much closer than misses")
    save(fig, out, "histo.png")
    box_in = t["x location shot"].ge(102) & t["y location shot"].between(18, 62)
    stats["goals_inside_box_pct"] = pct(box_in[t.goal == 1])
    stats["shots_inside_box_pct"] = pct(box_in)

    # --- shot angle
    fig, ax = plt.subplots(figsize=(6, 4.2))
    box(ax, [t[t.goal == 0].shot_angle, t[t.goal == 1].shot_angle], ["No goal", "Goal"], "Shot angle (degrees)")
    ax.set_title("Goals are scored from wider angles")
    save(fig, out, "angle1.png")
    stats["angle_median"] = t.groupby("goal").shot_angle.median().round(1).to_dict()

    # --- body part
    vc = op["body"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bar_count(ax, ["Foot", "Head", "Other"], [int(vc.get(k, 0)) for k in ["Foot", "Head", "Other"]])
    ax.set_title("Open-play shots by body part")
    save(fig, out, "body.png")
    stats["body_counts"] = {k: int(v) for k, v in vc.items()}

    fig, ax = plt.subplots(figsize=(6, 4.2))
    g = t.groupby("body part").goal
    bar_rate(ax, ["Foot", "Head"], (100 * g.mean()).tolist(), g.size().tolist())
    ax.set_title("All distances: feet and heads convert alike")
    save(fig, out, "beforeskew.png")
    stats["body_rate_all"] = (100 * g.mean()).round(1).to_dict()

    fig, ax = plt.subplots(figsize=(6, 4.2))
    box(ax, [t[t["body part"] == k]["shot distance"] for k in ["Foot", "Head"]], ["Foot", "Head"],
        "Shot distance (yards)")
    ax.set_title("Headers are taken much closer to goal")
    save(fig, out, "bodybox.png")
    stats["body_median_distance"] = t.groupby("body part")["shot distance"].median().round(1).to_dict()
    stats["head_within_18_pct"] = pct(t[t["body part"] == "Head"]["shot distance"] <= 18)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    band_bars(ax, t, "body part", ["Foot", "Head"], ["Foot", "Head"], [BLUE, ORANGE])
    ax.set_title("At the same distance, shots with the foot convert far better")
    save(fig, out, "afterskew.png")
    w = t[t["shot distance"] <= 18]
    stats["body_rate_within_18"] = (100 * w.groupby("body part").goal.mean()).round(1).to_dict()

    # --- pressure
    pr = t["Number of opponents in 5 yards"].clip(upper=3).astype(int)
    tt = t.assign(pr=pr)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    band_lines(ax, tt, "pr", [0, 1, 2, 3], ["0", "1", "2", "3+"])
    ax.set_xlabel("Opponents within 5 yards of the shooter")
    ax.set_title("At a given distance, more pressure means fewer goals")
    save(fig, out, "5yardsbar.png")
    stats["pressure_rate_all"] = {int(k): round(v, 1) for k, v in (100 * tt.groupby("pr").goal.mean()).items()}
    stats["pressure_rate_under12"] = {int(k): round(v, 1) for k, v in
                                      (100 * tt[tt["shot distance"] < 12].groupby("pr").goal.mean()).items()}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    box(ax, [tt[tt.pr == k]["shot distance"] for k in range(4)], ["0", "1", "2", "3+"], "Shot distance (yards)")
    ax.set_xlabel("Opponents within 5 yards of the shooter")
    ax.set_title("Shots closer to goal are under more pressure")
    save(fig, out, "5yardsbox.png")
    stats["pressure_median_distance"] = {int(k): round(v, 1) for k, v in tt.groupby("pr")["shot distance"].median().items()}

    # --- players between ball and goal
    pb = t["Players between goal"].clip(upper=4).astype(int)
    g = t.groupby(pb).goal
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bar_rate(ax, ["0", "1", "2", "3", "4+"], (100 * g.mean()).tolist(), g.size().tolist())
    ax.set_xlabel("Players inside the triangle formed by the ball and the posts")
    ax.set_title("Every body in the way lowers the chance of a goal")
    save(fig, out, "opponentsbetweengoalbar.png")
    stats["between_rate"] = {int(k): round(v, 1) for k, v in (100 * g.mean()).items()}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    box(ax, [t[pb == k]["shot distance"] for k in range(5)], ["0", "1", "2", "3", "4+"], "Shot distance (yards)")
    ax.set_xlabel("Players inside the triangle formed by the ball and the posts")
    ax.set_title("Longer shots have more players in the way")
    save(fig, out, "opponentsgoalbox.png")
    stats["between_median_distance"] = {int(k): round(v, 1) for k, v in t.groupby(pb)["shot distance"].median().items()}

    # --- goalkeeper
    fig, ax = plt.subplots(figsize=(6, 4.2))
    box(ax, [t[t.goal == 0]["gk distance"], t[t.goal == 1]["gk distance"]], ["No goal", "Goal"],
        "Goalkeeper distance from goal centre (yards)")
    ax.set_title("Keepers are further off their line when goals go in")
    save(fig, out, "gkdist.png")
    stats["gk_distance_median"] = t.groupby("goal")["gk distance"].median().round(2).to_dict()
    stats["gk_distance_mean"] = t.groupby("goal")["gk distance"].mean().round(2).to_dict()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    yb = np.arange(30, 51, 1)
    ax.hist(t["y gk location"].clip(30, 50), bins=yb, color=BLUE, rwidth=0.85)
    ax.axvspan(36, 44, color=ORANGE, alpha=0.08)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.text(36.2, ax.get_ylim()[1] * 0.95, "goal mouth (posts at y = 36 and 44)", ha="left", color=ORANGE, fontsize=9)
    ax.set_xlabel("Goalkeeper y coordinate (yards; values outside 30–50 grouped at the edges)")
    ax.set_ylabel("Shots")
    ax.set_title("Keepers are usually central when the shot is taken")
    save(fig, out, "ygkloc.png")
    stats["gk_y_38_42_pct"] = pct(t["y gk location"].between(38, 42))

    fig, ax = plt.subplots(figsize=(8, 4.6))
    h = ax.hist2d(t["y gk location"], t["x gk location"], bins=[np.arange(26, 54.5, 1), np.arange(100, 120.5, 0.5)],
                  cmap="Blues", cmin=1)
    ax.plot([36, 36], [119, 120], color=INK, linewidth=3)
    ax.plot([44, 44], [119, 120], color=INK, linewidth=3)
    ax.plot([18, 62], [102, 102], color=INK2, linewidth=1)
    ax.plot([30, 30, 50, 50], [120, 114, 114, 120], color=INK2, linewidth=1)
    ax.text(27, 102.4, "penalty area", color=INK2, fontsize=8)
    ax.text(30.3, 114.4, "six-yard box", color=INK2, fontsize=8)
    ax.set_xlim(26, 54)
    ax.set_ylim(100, 120.3)
    ax.set_xlabel("y (yards)")
    ax.set_ylabel("x (yards; goal line at 120)")
    ax.grid(False)
    fig.colorbar(h[3], ax=ax, label="Shots")
    ax.set_title("Goalkeeper position at the moment of the shot")
    save(fig, out, "gkloc.png")

    # --- play pattern (raw StatsBomb categories)
    pp = op[op["body"].isin(["Foot", "Head"])].groupby("play pattern").goal.agg(["mean", "size"])
    pp = pp[pp["size"] >= 300].sort_values("mean", ascending=False)
    short = [s.replace("From ", "") for s in pp.index]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cols = [ORANGE if s == "Counter" else BLUE for s in short]
    x = np.arange(len(pp))
    ax.bar(x, 100 * pp["mean"], color=cols, width=0.65)
    for i, (r, n) in enumerate(zip(100 * pp["mean"], pp["size"])):
        ax.text(i, r + 0.3, f"{r:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x, [f"{lab}\nn={n:,}" for lab, n in zip(short, pp["size"])], fontsize=9)
    ax.set_ylabel("Goals per shot (%)")
    ax.set_ylim(0, 100 * pp["mean"].max() * 1.15)
    ax.grid(axis="x", visible=False)
    ax.set_title("Counter-attacks give the best chances")
    save(fig, out, "playpattern.png")
    stats["play_pattern_rate"] = {k: [round(100 * m, 1), int(n)] for k, m, n in zip(pp.index, pp["mean"], pp["size"])}

    vc = op[op["body"].isin(["Foot", "Head"])]["play pattern"].value_counts()
    vc = vc[vc >= 300]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bar_count(ax, [s.replace("From ", "") for s in vc.index], vc.tolist())
    ax.set_title("Open-play shots by the play pattern that led to them")
    save(fig, out, "playpatt.png")
    stats["play_pattern_grouped_rate"] = (100 * t.groupby("play pattern").goal.mean()).round(1).to_dict()

    # --- pass type
    order = ["Ground Pass", "Low Pass", "High Pass", "None"]
    labs = ["Ground", "Low", "High", "No pass"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bar_count(ax, labs, [int((t["Pass Type"] == k).sum()) for k in order])
    ax.set_title("Shots by the height of the pass before them")
    save(fig, out, "passtypeshot.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    box(ax, [t[t["Pass Type"] == k]["shot distance"] for k in order], labs, "Shot distance (yards)")
    ax.set_title("High passes lead to shots close to goal")
    save(fig, out, "passtypebox.png")
    stats["pass_median_distance"] = t.groupby("Pass Type")["shot distance"].median().round(1).to_dict()
    stats["pass_head_share"] = (100 * t.assign(h=t["body part"].eq("Head")).groupby("Pass Type").h.mean()).round(0).to_dict()

    fig, ax = plt.subplots(figsize=(9, 4.4))
    band_bars(ax, t, "Pass Type", order, labs, [BLUE, AQUA, ORANGE, INK2])
    ax.set_title("At the same distance, shots after high passes convert worst")
    save(fig, out, "passtypeperc.png")
    stats["pass_rate_all"] = (100 * t.groupby("Pass Type").goal.mean()).round(1).to_dict()
    band = pd.cut(t["shot distance"], BANDS, labels=BAND_LABELS)
    stats["pass_rate_under12"] = (100 * t[band == BAND_LABELS[0]].groupby("Pass Type").goal.mean()).round(1).to_dict()

    # --- first time
    g = t.groupby("first time").goal
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bar_rate(ax, ["Not first time", "First time"], [pct(t[t["first time"] == k].goal) for k in ["False", "True"]],
             [int((t["first time"] == k).sum()) for k in ["False", "True"]])
    ax.set_title("First-time shots convert more often")
    save(fig, out, "firsttimeperc.png")
    stats["first_time_rate"] = (100 * g.mean()).round(1).to_dict()
    stats["first_time_rate_under12"] = (100 * t[band == BAND_LABELS[0]].groupby("first time").goal.mean()).round(1).to_dict()

    # --- technique (all open-play shots, any body part)
    tq = op.groupby("technique used").goal.agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(tq))
    ax.barh(y, 100 * tq["mean"], color=[ORANGE if k == "Lob" else BLUE for k in tq.index], height=0.65)
    for i, (r, n) in enumerate(zip(100 * tq["mean"], tq["size"])):
        ax.text(r + 0.3, i, f"{r:.1f}%  ({n:,} shots, {100*n/len(op):.1f}%)", va="center", fontsize=9)
    ax.set_yticks(y, tq.index)
    ax.set_xlim(0, 100 * tq["mean"].max() * 1.6)
    ax.set_xlabel("Goals per shot (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Lobs convert best but are rare")
    save(fig, out, "tech.png")
    stats["technique"] = {k: [round(100 * m, 1), int(n), round(100 * n / len(op), 1)] for k, m, n in
                          zip(tq.index, tq["mean"], tq["size"])}

    Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(a.stats).write_text(json.dumps(stats, indent=1, default=float))
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()

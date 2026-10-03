"""Results figures for the revised thesis (Chapter 4).

Needs the outputs of train_evaluate.py and xg_tables.py in results/.

    python scripts/plot_thesis_results.py --out thesis/images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_eda import AQUA, BLUE, GRID, INK, INK2, ORANGE, save  # noqa: E402  (also sets the style)
from xg.model import FEATURES, TARGET, modelling_set, split_by_match  # noqa: E402

MODELS = {"Logistic Regression": "logistic_regression", "XGBoost": "xgboost",
          "LightGBM": "lightgbm", "CatBoost": "catboost"}
NICE = {"shot distance": "Shot distance", "shot_angle": "Shot angle", "Players between goal": "Players between ball and goal",
        "Number of opponents in 5 yards": "Opponents within 5 yards", "x location shot": "Shot x", "y location shot": "Shot y",
        "gk distance": "Goalkeeper distance", "x gk location": "Goalkeeper x", "y gk location": "Goalkeeper y",
        "body part": "Body part", "Pass Type": "Pass type", "play pattern": "Play pattern", "first time": "First time"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="data/shots_all_men.parquet")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="thesis/images")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    stats = {}

    d = modelling_set(pd.read_parquet(a.shots))
    _, te = split_by_match(d, seed=42)
    y = te[TARGET].to_numpy()
    models = {k: joblib.load(res / f"model_{v}.joblib") for k, v in MODELS.items()}
    preds = {k: m.predict_proba(te[FEATURES])[:, 1] for k, m in models.items()}

    # --- ROC curves on the held-out matches
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    styles = {"Logistic Regression": (BLUE, "-"), "XGBoost": (AQUA, "--"), "LightGBM": (INK2, ":"),
              "CatBoost": (ORANGE, "-"), "StatsBomb xG": (INK, "-.")}
    from sklearn.metrics import roc_auc_score
    allp = dict(preds, **{"StatsBomb xG": te["official xg"].to_numpy()})
    for k, p in allp.items():
        fpr, tpr, _ = roc_curve(y, p)
        c, ls = styles[k]
        ax.plot(fpr, tpr, color=c, linestyle=ls, linewidth=1.8, label=f"{k} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("ROC curves on 524 held-out matches")
    save(fig, out, "roc.png")

    # --- permutation importance (increase in test log loss when one feature is shuffled)
    rng = np.random.default_rng(0)
    imp = {}
    for k in ["Logistic Regression", "CatBoost"]:
        base = log_loss(y, preds[k])
        rows = {}
        for f in FEATURES:
            inc = []
            for _ in range(5):
                x = te[FEATURES].copy()
                x[f] = rng.permutation(x[f].to_numpy())
                inc.append(log_loss(y, models[k].predict_proba(x)[:, 1]) - base)
            rows[f] = float(np.mean(inc))
        imp[k] = rows
    imp = pd.DataFrame(imp).sort_values("CatBoost")
    stats["permutation_importance_x1000"] = (1000 * imp).round(2).to_dict()
    fig, ax = plt.subplots(figsize=(8, 5.6))
    yy = np.arange(len(imp))
    ax.barh(yy + 0.2, 1000 * imp["CatBoost"], 0.4, color=ORANGE, label="CatBoost")
    ax.barh(yy - 0.2, 1000 * imp["Logistic Regression"], 0.4, color=BLUE, label="Logistic Regression")
    ax.set_yticks(yy, [NICE[f] for f in imp.index])
    ax.set_xlabel("Increase in test log loss when the feature is shuffled (×1000)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_title("Which features the models rely on")
    save(fig, out, "importance.png")

    # --- calibration curve: best model and StatsBomb, 10 quantile bins
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for k, c, m in [("CatBoost", ORANGE, "o"), ("StatsBomb xG", INK2, "s")]:
        p = allp[k]
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
        ax.plot(g.p, g.y, marker=m, color=c, linewidth=1.8, label=k)
    ax.plot([0, 0.6], [0, 0.6], color=GRID, linewidth=1)
    ax.set_xlabel("Mean predicted xG (10 equal-size groups of shots)")
    ax.set_ylabel("Actual share of goals")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Calibration on held-out matches")
    save(fig, out, "calibration.png")

    # --- players: goals vs xG (open play, out-of-fold xG)
    oof = pd.read_parquet(res / "oof_xg.parquet")
    pl = oof.groupby("player name").agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"))
    pl = pl[pl.shots >= 150]
    stats["players_150"] = len(pl)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pl.xg, pl.goals, s=18, color=BLUE, alpha=0.6)
    lim = max(pl.xg.max(), pl.goals.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle="--")
    ax.text(60, 45, "goals = xG", color=INK2, fontsize=9)
    labels = {"Lionel Andrés Messi Cuccittini": "Messi", "Luis Alberto Suárez Díaz": "Suárez",
              "Neymar da Silva Santos Junior": "Neymar", "Andrés Iniesta Luján": "Iniesta",
              "Cristiano Ronaldo dos Santos Aveiro": "Ronaldo"}
    for full, short in labels.items():
        if full in pl.index:
            r = pl.loc[full]
            ax.scatter(r.xg, r.goals, s=40, color=ORANGE, zorder=3)
            ax.annotate(short, (r.xg, r.goals), xytext=(-38 if short == "Messi" else 6, -3),
                        textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Open-play xG (out-of-fold)")
    ax.set_ylabel("Open-play goals")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(8, lim)
    ax.set_ylim(8, lim)
    ax.set_title(f"Goals against xG, players with 150+ shots (n={len(pl)})")
    save(fig, out, "players.png")

    # --- Barcelona in La Liga, season by season
    b = oof[(oof["Team Name"] == "Barcelona") & (oof.competition == "La Liga")]
    s = b.groupby("season_id").agg(goals=("goal", "sum"), xg=("xg", "sum")).sort_index()
    lab = [f"{x[2:4]}/{x[7:9]}" if "/" in x else x for x in s.index]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    xx = np.arange(len(s))
    ax.bar(xx - 0.2, s.goals, 0.4, color=ORANGE, label="Goals")
    ax.bar(xx + 0.2, s.xg, 0.4, color=BLUE, label="xG")
    ax.set_xticks(xx, lab, fontsize=9)
    ax.set_ylabel("Open-play goals / xG")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Barcelona scored more than their xG in {(s.goals > s.xg).sum()} of {len(s)} La Liga seasons")
    save(fig, out, "barcelona_seasons.png")
    stats["barcelona_seasons_over"] = [int((s.goals > s.xg).sum()), len(s)]

    # --- Premier League 2015/16: xG difference vs points
    t = pd.read_csv(res / "table_premier_league_2015_16.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(t.op_xg_diff, t.points, s=28, color=BLUE)
    for _, r in t.iterrows():
        hi = r.team in ("Leicester City", "Arsenal")
        if hi:
            ax.scatter(r.op_xg_diff, r.points, s=50, color=ORANGE, zorder=3)
        if hi or r.position <= 4 or r.position >= 20:
            left = r.team == "Tottenham Hotspur"
            ax.annotate(r.team, (r.op_xg_diff, r.points), xytext=(-6 if left else 6, -3), textcoords="offset points",
                        ha="right" if left else "left",
                        fontsize=9, color=ORANGE if hi else INK)
    ax.set_xlabel("Open-play xG difference (xG for − xG against)")
    ax.set_ylabel("Points")
    ax.set_xlim(-32, 48)
    ax.set_title("Premier League 2015/16: points against xG difference")
    save(fig, out, "pl_2015_16.png")

    Path(res / "thesis_result_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

"""Train, tune and evaluate the xG models on StatsBomb men's open-play shots.

    python scripts/train_evaluate.py --shots data/shots_all_men.parquet --out results/

Steps:
 1. modelling set: open-play men's shots since 2000 (thesis features)
 2. hold out 20% of matches as the test set
 3. tune each model by 5-fold match-grouped CV on log loss (training matches only)
 4. score on the held-out matches: log loss, Brier, AUC, calibration (ECE, xG/goals),
    alongside StatsBomb's own xG
 5. per-competition calibration, and a Barcelona-only model (the thesis setup) for comparison
 6. out-of-fold xG for every shot (each shot predicted by a model that never saw its match)

Writes results.json, calibration.csv, oof_xg.parquet and the fitted models to --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xg.model import (FEATURES, GROUP, TARGET, calibration_table, modelling_set,  # noqa: E402
                      model_specs, scores, split_by_match, tune)

SEED = 42


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shots", default="data/shots_all_men.parquet")
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="*", default=None, help="Subset of models to run.")
    p.add_argument("--n-iter-scale", type=float, default=1.0, help="Scale the search size (e.g. 0.2 for a quick run).")
    args = p.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = modelling_set(pd.read_parquet(args.shots))
    train, test = split_by_match(d, seed=SEED)
    barca_games = set(d.loc[d["Team Name"].eq("Barcelona"), GROUP])
    res = {"data": {
        "shots": len(d), "matches": int(d[GROUP].nunique()), "goals": int(d[TARGET].sum()),
        "train_shots": len(train), "train_matches": int(train[GROUP].nunique()),
        "test_shots": len(test), "test_matches": int(test[GROUP].nunique()),
        "matches_shared_by_train_and_test": len(set(train[GROUP]) & set(test[GROUP])),
    }, "models": {}, "by_competition": {}, "barcelona_only": {}}
    log(f"modelling set {res['data']}")

    names = args.models or list(model_specs(SEED))
    fitted, preds = {}, {}
    for name in names:
        t0 = time.time()
        n_iter = max(3, int(round(model_specs(SEED)[name][2] * args.n_iter_scale)))
        search = tune(name, train, seed=SEED, n_iter=n_iter)
        p_test = search.predict_proba(test[FEATURES])[:, 1]
        fitted[name], preds[name] = search.best_estimator_, p_test
        res["models"][name] = {
            "cv_log_loss": -search.best_score_, "n_iter": n_iter,
            "best_params": {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
            "test": scores(test[TARGET], p_test), "fit_seconds": round(time.time() - t0),
        }
        joblib.dump(search.best_estimator_, out / f"model_{name.replace(' ', '_').lower()}.joblib")
        log(f"{name}: CV log loss {-search.best_score_:.4f} | test {res['models'][name]['test']}")

    res["models"]["StatsBomb xG"] = {"test": scores(test[TARGET], test["official xg"])}
    best = min(names, key=lambda n: res["models"][n]["cv_log_loss"])   # chosen on CV, not on the test set
    res["best_model"] = best
    log(f"best model by CV log loss: {best}")

    # calibration curves on the test set
    cal = []
    for name, p_ in list(preds.items()) + [("StatsBomb xG", test["official xg"].to_numpy())]:
        t = calibration_table(test[TARGET], p_).reset_index(); t["model"] = name; cal.append(t)
    pd.concat(cal).to_csv(out / "calibration.csv", index=False)

    # calibration by competition and for Barcelona (test set)
    t = test.assign(model_xg=preds[best])
    groups = {c: t[t["competition"].eq(c)] for c in t["competition"].value_counts().index}
    groups["Barcelona's shots"] = t[t["Team Name"].eq("Barcelona")]
    groups["All other shots"] = t[~t["Team Name"].eq("Barcelona")]
    for g, x in groups.items():
        if x[TARGET].sum() < 10:
            continue
        res["by_competition"][g] = {"shots": len(x), "goals": int(x[TARGET].sum()),
                                     "model_xg_over_goals": float(x["model_xg"].sum() / x[TARGET].sum()),
                                     "statsbomb_xg_over_goals": float(x["official xg"].sum() / x[TARGET].sum())}

    # the thesis setup: same model and settings, trained on Barcelona matches only
    tr_b = train[train[GROUP].isin(barca_games)]
    m_b = clone(fitted[best]).fit(tr_b[FEATURES], tr_b[TARGET])
    for label, x in [("Barcelona's shots", t[t["Team Name"].eq("Barcelona")]),
                     ("Shots in non-Barcelona matches", t[~t[GROUP].isin(barca_games)])]:
        pb = m_b.predict_proba(x[FEATURES])[:, 1]
        res["barcelona_only"][label] = {
            "shots": len(x), "goals": int(x[TARGET].sum()),
            "barcelona_only_model": scores(x[TARGET], pb),
            "pooled_model": scores(x[TARGET], x["model_xg"]),
        }
    res["barcelona_only"]["training_shots"] = len(tr_b)
    log(f"Barcelona-only comparison: {res['barcelona_only']}")

    # out-of-fold xG for every shot with the best model's settings
    oof = np.zeros(len(d))
    for k, (tr_i, te_i) in enumerate(GroupKFold(n_splits=5).split(d, groups=d[GROUP])):
        m = clone(fitted[best]).fit(d.iloc[tr_i][FEATURES], d.iloc[tr_i][TARGET])
        oof[te_i] = m.predict_proba(d.iloc[te_i][FEATURES])[:, 1]
        log(f"out-of-fold {k + 1}/5 done")
    d.assign(xg=oof)[["shot id", GROUP, "season_id", "competition", "Team Name", "player name",
                      "x location shot", "y location shot", TARGET, "official xg", "xg"]].to_parquet(out / "oof_xg.parquet", index=False)
    res["oof"] = scores(d[TARGET], oof)

    (out / "results.json").write_text(json.dumps(res, indent=2))
    log(f"wrote {out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

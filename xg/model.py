"""xG models: modelling set, features, preprocessing, models and evaluation.

The features are the ones used in the thesis (EDA+Model.ipynb, cell 97), so results stay
comparable. What changes is how models are fitted and evaluated:

* train/test split and cross-validation are grouped by match, so shots from one match
  never appear on both sides (card #13);
* hyperparameters are tuned by cross-validated log loss on the training matches only (card #15);
* evaluation reports calibration as well as discrimination (card #14).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["body part", "Pass Type", "play pattern", "first time"]
NUMERIC = [
    "shot distance", "shot_angle", "Number of opponents in 5 yards", "Players between goal",
    "x location shot", "y location shot", "gk distance", "x gk location", "y gk location",
]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "goal"
GROUP = "game_id"


# ----------------------------------------------------------------- data
def modelling_set(shots: pd.DataFrame, min_season_start: int = 2000) -> pd.DataFrame:
    """Open-play shots from matches since ``min_season_start``, with the thesis feature encoding.

    Matches the thesis preparation: body part grouped into Foot/Head (other body parts dropped),
    play pattern grouped into Regular Play / Non Regular Play, distances to the goal centre.
    """
    d = shots[shots["type of shot"].eq("Open Play")].copy()
    d = d[d["season_id"].str[:4].astype(int) >= min_season_start]
    bp = d["body part"].replace({"Right Foot": "Foot", "Left Foot": "Foot"})
    d = d[bp.isin(["Foot", "Head"])].copy()
    d["body part"] = bp[bp.isin(["Foot", "Head"])]
    d["play pattern"] = np.where(d["play pattern"].eq("Regular Play"), "Regular Play", "Non Regular Play")
    d["first time"] = d["first time"].astype(str)
    d["shot distance"] = np.hypot(d["x location shot"] - 120, d["y location shot"] - 40)
    d["gk distance"] = np.hypot(d["x gk location"] - 120, d["y gk location"] - 40)
    d[TARGET] = d["outcome"].eq("Goal").astype(int)
    return d.reset_index(drop=True)


def split_by_match(d: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Hold out a share of matches (not shots) as the test set."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(d, groups=d[GROUP]))
    return d.iloc[train_idx].reset_index(drop=True), d.iloc[test_idx].reset_index(drop=True)


# ----------------------------------------------------------------- models
def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),  # card #19
         ("num", StandardScaler(), NUMERIC)],
        verbose_feature_names_out=False,
    )


def model_specs(seed: int = 42) -> dict:
    """Estimator and search space for each model (ranges kept to valid, sensible values; card #15)."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"clf__C": loguniform(1e-3, 1e2)}, 20),
        "XGBoost": (
            XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=2, random_state=seed),
            {"clf__n_estimators": randint(100, 700), "clf__learning_rate": loguniform(0.01, 0.2),
             "clf__max_depth": randint(2, 7), "clf__min_child_weight": loguniform(1, 50),
             "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.6, 0.4),
             "clf__reg_lambda": loguniform(0.1, 20)}, 20),
        "LightGBM": (
            LGBMClassifier(n_jobs=2, random_state=seed, verbose=-1, subsample_freq=1),
            {"clf__n_estimators": randint(100, 700), "clf__learning_rate": loguniform(0.01, 0.2),
             "clf__num_leaves": randint(7, 64), "clf__min_child_samples": randint(20, 300),
             "clf__subsample": uniform(0.6, 0.4), "clf__colsample_bytree": uniform(0.6, 0.4),
             "clf__reg_lambda": loguniform(0.1, 20)}, 20),
        "CatBoost": (
            CatBoostClassifier(verbose=0, random_seed=seed, thread_count=2),
            {"clf__iterations": randint(200, 700), "clf__learning_rate": loguniform(0.02, 0.2),
             "clf__depth": randint(3, 8), "clf__l2_leaf_reg": loguniform(1, 30)}, 12),
    }


def tune(name: str, train: pd.DataFrame, seed: int = 42, n_iter: int | None = None, folds: int = 5):
    """Randomised search over the model's space, scored by log loss with match-grouped CV."""
    est, space, default_iter = model_specs(seed)[name]
    pipe = Pipeline([("pre", preprocessor()), ("clf", est)])
    search = RandomizedSearchCV(
        pipe, space, n_iter=n_iter or default_iter, scoring="neg_log_loss",
        cv=GroupKFold(n_splits=folds), random_state=seed, n_jobs=1, refit=True,
    )
    search.fit(train[FEATURES], train[TARGET], groups=train[GROUP])
    return search


# ----------------------------------------------------------------- evaluation
def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Shot-weighted mean |observed goal rate - mean xG| over equal-count bins of xG."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def scores(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "ece": expected_calibration_error(y, p),
        "xg_over_goals": float(p.sum() / np.sum(y)),
    }


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean xG vs observed goal rate in equal-count bins (for calibration curves)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(mean_xg=("p", "mean"), goal_rate=("y", "mean"), shots=("y", "size"))

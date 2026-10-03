# Does league style bias a pooled xG model?

Step 4. The question from the thesis discussion: each league has its own style, so does mixing leagues bias an xG model?

```bash
python scripts/league_comparison.py --shots data/shots_all_men.parquet --results results   # ~3 min
python scripts/plot_leagues.py
```

## Set-up
- **Data:** the four complete 2015/16 seasons in StatsBomb's open data (Premier League, La Liga, Serie A, Ligue 1): 35,645 open-play shots from 1,517 matches. Same features as the thesis.
- **Four ways of training,** each scored on the *same held-out matches* of each league (5 folds of matches):
  - **Pooled:** trained on all four leagues.
  - **Own league only:** a separate model per league.
  - **Other three only:** trained without the league being scored (leave one league out). This is the direct test of whether a model transfers between styles.
  - **Pooled + league feature:** the model is told which league each shot is from, so it can learn a per-league adjustment.
- **Models:** CatBoost (the best model in [RESULTS.md](RESULTS.md)) and logistic regression, with the settings tuned in step 3.
- **Confidence intervals:** 95% intervals come from resampling whole matches, because shots in the same match aren't independent.

## Results

### 1. Calibration: predicted ÷ actual goals (95% interval)

1.00 means the model predicted exactly the goals that were scored. CatBoost:

| League | Shots | Goals | Pooled (all four) | Own league only | Other three only | Pooled + league feature | StatsBomb xG |
|---|---|---|---|---|---|---|---|
| Premier League | 9,398 | 879 | 1.03 (0.96–1.10) | 0.98 (0.92–1.05) | 1.04 (0.97–1.11) | 1.01 (0.94–1.08) | 1.00 (0.93–1.07) |
| La Liga | 8,643 | 911 | 0.99 (0.93–1.04) | 0.98 (0.93–1.04) | 0.99 (0.93–1.04) | 0.99 (0.94–1.05) | 0.97 (0.91–1.02) |
| Serie A | 9,373 | 818 | 0.99 (0.93–1.05) | 0.98 (0.92–1.04) | 0.99 (0.93–1.05) | 0.99 (0.93–1.05) | 0.98 (0.93–1.04) |
| Ligue 1 | 8,231 | 814 | 0.98 (0.92–1.04) | 0.98 (0.92–1.05) | 0.97 (0.91–1.04) | 0.98 (0.92–1.05) | 0.96 (0.90–1.02) |

Logistic regression:

| League | Shots | Goals | Pooled (all four) | Own league only | Other three only | Pooled + league feature | StatsBomb xG |
|---|---|---|---|---|---|---|---|
| Premier League | 9,398 | 879 | 1.03 (0.97–1.11) | 1.00 (0.94–1.08) | 1.04 (0.98–1.12) | 1.00 (0.94–1.07) | 1.00 (0.93–1.07) |
| La Liga | 8,643 | 911 | 1.00 (0.94–1.05) | 1.00 (0.95–1.06) | 0.99 (0.94–1.05) | 1.00 (0.95–1.06) | 0.97 (0.91–1.02) |
| Serie A | 9,373 | 818 | 1.00 (0.94–1.06) | 1.00 (0.94–1.06) | 1.00 (0.94–1.06) | 1.00 (0.94–1.06) | 0.98 (0.93–1.04) |
| Ligue 1 | 8,231 | 814 | 0.97 (0.92–1.04) | 1.00 (0.94–1.07) | 0.97 (0.91–1.03) | 1.00 (0.94–1.07) | 0.96 (0.90–1.02) |

**Every way of training is calibrated in every league,** and every interval includes 1.00. That includes the model that never saw the league it is scoring (*Other three only*).

### 2. Accuracy: change in log loss vs. the pooled model (×1000, 95% interval)

Positive means worse than pooled; **worse**/**better** marks intervals that exclude zero. CatBoost:

![Change in log loss vs the pooled model, per league](figures/leagues.png)

| League | Pooled log loss | Own league only | Other three only | Pooled + league feature |
|---|---|---|---|---|
| Premier League | 0.2579 | +4.64 (+2.98 to +6.38) **worse** | +0.34 (-0.49 to +1.12) | +0.40 (-0.29 to +1.07) |
| La Liga | 0.2613 | +3.43 (+1.46 to +5.44) **worse** | +0.41 (-0.45 to +1.24) | +0.11 (-0.63 to +0.84) |
| Serie A | 0.2298 | +2.85 (+1.00 to +4.69) **worse** | +0.96 (+0.29 to +1.71) **worse** | -0.07 (-0.67 to +0.51) |
| Ligue 1 | 0.2561 | +4.19 (+2.13 to +6.32) **worse** | +0.64 (-0.23 to +1.49) | +0.07 (-0.67 to +0.79) |

Logistic regression:

| League | Pooled log loss | Own league only | Other three only | Pooled + league feature |
|---|---|---|---|---|
| Premier League | 0.2592 | +0.90 (-0.12 to +1.86) | +0.20 (-0.10 to +0.53) | -0.09 (-0.37 to +0.20) |
| La Liga | 0.2637 | +0.64 (-0.14 to +1.43) | +0.32 (+0.04 to +0.59) **worse** | +0.02 (-0.06 to +0.09) |
| Serie A | 0.2305 | +1.05 (+0.01 to +1.99) **worse** | +0.12 (-0.10 to +0.36) | +0.10 (-0.01 to +0.21) |
| Ligue 1 | 0.2575 | +0.37 (-1.01 to +1.70) | +0.49 (+0.16 to +0.84) **worse** | +0.15 (-0.08 to +0.38) |

## Findings

1. **Mixing the four big European leagues does not bias the model.** The pooled model is calibrated in each league (CatBoost 0.98–1.03), and every interval includes 1.00.
2. **Training on other leagues transfers.** A model that has never seen a league predicts it about as well as one that has. It is calibrated in all four, and its accuracy loss is a few ten-thousandths of log loss, only clearly different from zero in Serie A (and in La Liga and Ligue 1 for logistic regression).
3. **Splitting by league makes the model worse.** With CatBoost, own-league models are clearly worse than pooled in all four leagues: 0.003–0.005 higher log loss. That is more than the whole gap between the four algorithms in step 3 (0.0013). Fewer shots costs more than specialising gains. The simpler logistic regression loses less, but never gains.
4. **Telling the model the league adds nothing.** Pooled + league feature is indistinguishable from pooled in every league, for both models.
5. **A league of a different standard does differ.** Trained on the four European leagues and applied to the Indian Super League 2021/22, the model predicts 12% fewer goals than were scored. StatsBomb's own model, trained on far more data, shows the same gap:

| Model | ISL shots | ISL goals | Predicted ÷ actual goals | StatsBomb xG ÷ goals |
|---|---|---|---|---|
| Logistic Regression | 2,871 | 296 | 0.88 (0.80–0.98) | 0.89 (0.81–0.99) |
| CatBoost | 2,871 | 296 | 0.88 (0.80–0.98) | 0.89 (0.81–0.99) |

## What this means for the thesis
- **The concern that league style biases xG is not supported among the top European leagues.** Shot location, angle, defenders and goalkeeper position describe a chance well enough that a pooled model transfers between styles, and pooling is the more accurate choice.
- **Differences in level are a different matter.** For a league of a different standard (here the ISL), pooled xG should be recalibrated or used with care. Goals per chance differ, plausibly because of finishing and goalkeeping quality rather than style.
- **The real bias in the original thesis was one team, not one league.** Training on Barcelona's matches alone over-predicts other teams' goals by 12% ([RESULTS.md](RESULTS.md)).
- **Limits:** one season (2015/16) of four leagues from one provider, and only the thesis features. A season-by-season or team-strength analysis would need more full seasons than StatsBomb's open data has.

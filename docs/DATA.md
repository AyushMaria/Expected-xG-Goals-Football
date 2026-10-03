# Shot data: StatsBomb open data, men's football

This dataset is built with `scripts/build_shots.py` from [StatsBomb open data](https://github.com/statsbomb/open-data), and the tables below are produced by `scripts/summarize_shots.py`. Data built on 3 October 2026; StatsBomb updates the open data from time to time, so later builds can differ slightly.

```bash
python scripts/build_shots.py --gender male --out data/shots_all_men.parquet   # ~9 min fresh, ~700 MB cache
python scripts/summarize_shots.py data/shots_all_men.parquet
```

Data source: **StatsBomb**. Credit StatsBomb and use their logo in any published work based on this data.

## Summary

**2,651 matches, 66,854 shots, of which 62,279 open play (6,430 goals, 10.3% conversion).** Open-play shots with a freeze frame: 100.00%; with the goalkeeper in the freeze frame: 99.91%.

### By competition (all shot types)

| Competition | Seasons | Matches | Shots | Open-play shots | Goals | Conversion | StatsBomb xG / goals |
|---|---|---|---|---|---|---|---|
| La Liga | 1973–2020 | 867 | 21,181 | 19,612 | 2,653 | 12.5% | 0.89 |
| Premier League | 2003–2015 | 418 | 10,837 | 10,250 | 1,082 | 10.0% | 0.98 |
| Ligue 1 | 2015–2022 | 435 | 10,346 | 9,703 | 1,105 | 10.7% | 0.96 |
| Serie A | 1986–2015 | 381 | 10,033 | 9,428 | 955 | 9.5% | 0.97 |
| FIFA World Cup | 1958–2022 | 147 | 3,904 | 3,566 | 443 | 11.3% | 1.03 |
| Indian Super league | 2021–2021 | 115 | 3,095 | 2,881 | 345 | 11.1% | 0.88 |
| UEFA Euro | 2020–2024 | 102 | 2,629 | 2,450 | 281 | 10.7% | 1.08 |
| 1. Bundesliga | 2015–2023 | 68 | 1,756 | 1,658 | 200 | 11.4% | 0.95 |
| African Cup of Nations | 2023–2023 | 52 | 1,244 | 1,087 | 159 | 12.8% | 1.04 |
| Copa America | 2024–2024 | 32 | 790 | 709 | 95 | 12.0% | 1.10 |
| Champions League | 1970–2018 | 18 | 594 | 528 | 74 | 12.5% | 0.99 |
| Major League Soccer | 2023–2023 | 6 | 146 | 135 | 12 | 8.2% | 1.22 |
| UEFA Europa League | 1988–1988 | 3 | 89 | 84 | 8 | 9.0% | 0.89 |
| Copa del Rey | 1977–1983 | 3 | 74 | 65 | 8 | 10.8% | 0.82 |
| Liga Profesional | 1981–1997 | 2 | 56 | 48 | 6 | 10.7% | 1.02 |
| North American League | 1977–1977 | 1 | 50 | 48 | 3 | 6.0% | 1.34 |
| FIFA U20 World Cup | 1979–1979 | 1 | 30 | 27 | 4 | 13.3% | 0.54 |

Pre-2000 matches: 35 matches, 1,091 open-play shots (1.8% of open play).

### League vs. Barcelona effect (open play)

| Group (open play) | Shots | Conversion | StatsBomb xG / goals |
|---|---|---|---|
| La Liga, all matches | 19,612 | 12.0% | 0.89 |
| La Liga: Barcelona's shots | 7,477 | 15.8% | 0.79 |
| La Liga: opponents' shots against Barcelona | 4,359 | 8.7% | 1.04 |
| La Liga 2015/16, excluding Barcelona's matches | 7,776 | 10.2% | 0.98 |
| Premier League 2015/16 | 9,415 | 9.4% | 1.00 |
| Serie A 2015/16 | 9,396 | 8.8% | 0.98 |
| Ligue 1 2015/16 | 8,260 | 10.0% | 0.95 |

Notes on reading the tables:
- *Seasons* shows the first and last season start year in the data. Most competitions other than the four 2015/16 leagues cover only some matches of a season: one club's matches (Barcelona in La Liga, Arsenal 2003/04, Leverkusen, PSG) or finals (Champions League).
- *StatsBomb xG / goals* compares StatsBomb's own xG with actual goals: 1.00 means StatsBomb's model predicted the right number of goals; below 1 means more goals were scored than predicted.

## Findings that shape the modelling

1. **The thesis data was one team, not one league.** All 520 thesis matches involved Barcelona. In the full men's data, Barcelona's matches are 19% of open-play shots.
2. **Barcelona, not La Liga, explains the high La Liga conversion rate.** Barcelona's own shots convert at 15.8% and beat StatsBomb's xG by about 27% (xG/goals 0.79). Excluding Barcelona's matches, La Liga 2015/16 converts at 10.2%, in line with the Premier League (9.4%), Serie A (8.8%) and Ligue 1 (10.0%).
3. **A pooled model can stay calibrated across men's leagues.** StatsBomb's own model, trained across competitions, stays within 5% of actual goals in each 2015/16 league (xG/goals 0.95–1.00). This is a first indication, to be tested with this project's model (league comparison, step 4).
4. **Every open-play shot has a freeze frame**, and 99.9% include the goalkeeper, so all the thesis features can be computed across the whole dataset.

## Suggested modelling set

Open-play shots from men's matches from 2000 onwards. This excludes the 35 pre-2000 matches (1.8% of shots), which StatsBomb collected from historical footage. Penalties and direct free kicks stay out of the open-play model, as in the thesis.

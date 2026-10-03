# Team and player tables (out-of-fold xG)

Card #6. Every shot's xG here comes from a model that **never saw that shot's match**: 5-fold cross-validation grouped by match, using the best model from [RESULTS.md](RESULTS.md). These tables replace the ones at the end of `EDA+Model.ipynb`, which counted only the random 20% test sample, covered only Barcelona's matches, and used a hand-typed list of league positions.

- **Open play only:** xG and goals cover open-play shots, the shots the model covers. Penalties, direct free kicks and own goals are not included, so "goals" here is lower than a team's or player's official total.
- **Real league tables:** points and positions are calculated from StatsBomb's match results, not typed in.
- **Reading z:** *z* is the gap between goals and xG divided by how much that gap would vary by chance for an average finisher taking the same shots. Roughly, |z| above 2 is unlikely to be luck alone.

```bash
python scripts/xg_tables.py --oof results/oof_xg.parquet --out docs/TABLES.md
```

## 1. League tables 2015/16, with xG

How well each measure tracks final points (rank correlation; 1.00 would be perfect):

| League | Matches | Spearman ρ (open-play xG diff vs points) | Spearman ρ (open-play goal diff vs points) |
|---|---|---|---|
| Premier League | 380 | 0.85 | 0.93 |
| La Liga | 380 | 0.73 | 0.85 |
| Serie A | 380 | 0.83 | 0.95 |
| Ligue 1 | 377 | 0.61 | 0.65 |

Within a single season, actual goal difference tracks points more closely than xG difference. That is expected, because points are decided by actual goals. xG measures the quality of the chances a team created and allowed, and the gap between the two shows who did better or worse than their chances suggest. In 2015/16, Leicester won the Premier League while ranking 6th on open-play xG difference, and Arsenal, 1st on xG difference, finished second. Ligue 1's lower correlations partly reflect the 3 missing matches.

### Premier League 2015/16

OP = open play. Points and positions match the official final table.

| Pos | Team | Pts | Goals for (OP) | xG for | Goals against (OP) | xG against | xG diff | Rank by xG diff |
|---|---|---|---|---|---|---|---|---|
| 1 | Leicester City | 81 | 56 | 59.0 | 32 | 44.6 | 14.4 | 6 |
| 2 | Arsenal | 71 | 61 | 68.8 | 30 | 34.8 | 34.1 | 1 |
| 3 | Tottenham Hotspur | 70 | 60 | 59.7 | 31 | 35.4 | 24.2 | 2 |
| 4 | Manchester City | 66 | 62 | 58.6 | 39 | 39.4 | 19.1 | 3 |
| 5 | Manchester United | 66 | 42 | 40.7 | 29 | 38.1 | 2.6 | 8 |
| 6 | Southampton | 63 | 53 | 54.7 | 37 | 37.6 | 17.0 | 5 |
| 7 | West Ham United | 62 | 57 | 53.4 | 40 | 49.2 | 4.2 | 7 |
| 8 | Liverpool | 60 | 57 | 56.8 | 48 | 38.0 | 18.8 | 4 |
| 9 | Stoke City | 51 | 35 | 37.6 | 48 | 51.5 | -13.8 | 16 |
| 10 | Chelsea | 50 | 47 | 51.4 | 47 | 48.9 | 2.5 | 9 |
| 11 | Everton | 47 | 50 | 50.0 | 49 | 54.6 | -4.6 | 11 |
| 12 | Swansea City | 47 | 32 | 39.1 | 40 | 50.7 | -11.6 | 14 |
| 13 | Watford | 45 | 30 | 38.3 | 42 | 53.2 | -15.0 | 18 |
| 14 | West Bromwich Albion | 43 | 29 | 36.0 | 37 | 45.2 | -9.1 | 13 |
| 15 | Crystal Palace | 42 | 34 | 40.9 | 39 | 48.6 | -7.7 | 12 |
| 16 | AFC Bournemouth | 42 | 40 | 41.6 | 60 | 43.6 | -2.1 | 10 |
| 17 | Sunderland | 39 | 35 | 38.8 | 55 | 55.5 | -16.7 | 19 |
| 18 | Newcastle United | 37 | 39 | 39.5 | 57 | 53.8 | -14.3 | 17 |
| 19 | Norwich City | 34 | 36 | 40.0 | 58 | 53.8 | -13.7 | 15 |
| 20 | Aston Villa | 17 | 24 | 29.4 | 61 | 57.7 | -28.3 | 20 |
### La Liga 2015/16

OP = open play. Points and positions match the official final table.

| Pos | Team | Pts | Goals for (OP) | xG for | Goals against (OP) | xG against | xG diff | Rank by xG diff |
|---|---|---|---|---|---|---|---|---|
| 1 | Barcelona | 91 | 93 | 77.8 | 26 | 32.0 | 45.8 | 1 |
| 2 | Real Madrid | 90 | 96 | 75.6 | 36 | 43.7 | 31.9 | 2 |
| 3 | Atlético Madrid | 88 | 60 | 54.0 | 14 | 24.3 | 29.6 | 3 |
| 4 | Villarreal | 64 | 37 | 35.1 | 32 | 42.2 | -7.1 | 11 |
| 5 | Athletic Club | 62 | 53 | 48.7 | 36 | 35.2 | 13.4 | 4 |
| 6 | Celta Vigo | 60 | 49 | 46.3 | 47 | 46.7 | -0.3 | 8 |
| 7 | Sevilla | 52 | 41 | 56.0 | 41 | 43.4 | 12.6 | 5 |
| 8 | Málaga | 48 | 34 | 43.7 | 30 | 41.2 | 2.5 | 6 |
| 9 | Real Sociedad | 48 | 42 | 45.8 | 42 | 44.5 | 1.2 | 7 |
| 10 | Real Betis | 45 | 29 | 35.6 | 51 | 58.2 | -22.6 | 20 |
| 11 | Valencia | 44 | 34 | 43.9 | 43 | 53.3 | -9.4 | 15 |
| 12 | Las Palmas | 44 | 38 | 36.0 | 48 | 54.8 | -18.8 | 18 |
| 13 | Eibar | 43 | 41 | 44.2 | 55 | 47.2 | -3.0 | 9 |
| 14 | Espanyol | 43 | 38 | 44.0 | 62 | 51.6 | -7.6 | 13 |
| 15 | RC Deportivo La Coruña | 42 | 36 | 42.5 | 57 | 49.6 | -7.2 | 12 |
| 16 | Sporting Gijón | 39 | 35 | 41.2 | 55 | 53.3 | -12.0 | 16 |
| 17 | Granada | 39 | 39 | 36.3 | 60 | 55.8 | -19.5 | 19 |
| 18 | Rayo Vallecano | 38 | 52 | 50.3 | 60 | 53.5 | -3.3 | 10 |
| 19 | Getafe | 36 | 34 | 38.9 | 54 | 47.7 | -8.8 | 14 |
| 20 | Levante UD | 32 | 30 | 36.3 | 62 | 53.6 | -17.3 | 17 |
### Serie A 2015/16

OP = open play. Points and positions match the official final table.

| Pos | Team | Pts | Goals for (OP) | xG for | Goals against (OP) | xG against | xG diff | Rank by xG diff |
|---|---|---|---|---|---|---|---|---|
| 1 | Juventus | 91 | 62 | 54.6 | 13 | 21.0 | 33.6 | 2 |
| 2 | Napoli | 82 | 63 | 60.7 | 32 | 25.7 | 35.0 | 1 |
| 3 | AS Roma | 80 | 73 | 59.0 | 33 | 35.3 | 23.7 | 3 |
| 4 | Inter Milan | 67 | 43 | 48.9 | 33 | 41.9 | 7.0 | 7 |
| 5 | Fiorentina | 64 | 45 | 46.7 | 43 | 33.5 | 13.2 | 4 |
| 6 | Sassuolo | 61 | 42 | 40.6 | 36 | 39.3 | 1.3 | 10 |
| 7 | AC Milan | 57 | 41 | 47.2 | 31 | 36.1 | 11.1 | 5 |
| 8 | Lazio | 54 | 44 | 44.2 | 39 | 37.8 | 6.4 | 8 |
| 9 | Chievo | 50 | 33 | 33.7 | 42 | 46.1 | -12.4 | 15 |
| 10 | Genoa | 46 | 38 | 39.4 | 42 | 45.5 | -6.1 | 12 |
| 11 | Empoli | 46 | 38 | 36.0 | 43 | 43.7 | -7.8 | 13 |
| 12 | Torino | 45 | 40 | 43.7 | 41 | 36.3 | 7.5 | 6 |
| 13 | Atalanta | 45 | 32 | 37.4 | 40 | 39.9 | -2.4 | 11 |
| 14 | Bologna | 42 | 29 | 29.7 | 37 | 44.0 | -14.3 | 16 |
| 15 | Sampdoria | 40 | 42 | 37.8 | 51 | 54.8 | -16.9 | 18 |
| 16 | Udinese | 39 | 32 | 44.8 | 46 | 41.5 | 3.3 | 9 |
| 17 | Palermo | 39 | 36 | 34.9 | 52 | 55.5 | -20.6 | 19 |
| 18 | Carpi | 38 | 27 | 34.2 | 48 | 45.3 | -11.1 | 14 |
| 19 | Frosinone | 31 | 33 | 29.8 | 63 | 65.3 | -35.5 | 20 |
| 20 | Hellas Verona | 28 | 25 | 38.5 | 53 | 53.5 | -15.0 | 17 |
### Ligue 1 2015/16

OP = open play. StatsBomb's data is missing 3 of the 380 matches, so points and positions differ slightly from the official table (PSG finished on 96 points).

| Pos | Team | Pts | Goals for (OP) | xG for | Goals against (OP) | xG against | xG diff | Rank by xG diff |
|---|---|---|---|---|---|---|---|---|
| 1 | Paris Saint-Germain | 93 | 89 | 72.9 | 17 | 23.1 | 49.8 | 1 |
| 2 | Lyon | 65 | 59 | 59.3 | 38 | 37.6 | 21.6 | 2 |
| 3 | AS Monaco | 65 | 43 | 49.2 | 50 | 47.6 | 1.6 | 6 |
| 4 | OGC Nice | 63 | 49 | 43.1 | 31 | 42.8 | 0.3 | 7 |
| 5 | Lille | 60 | 33 | 37.4 | 24 | 31.5 | 5.9 | 4 |
| 6 | Saint-Étienne | 58 | 34 | 39.6 | 30 | 40.5 | -0.9 | 9 |
| 7 | Stade Malherbe Caen | 54 | 32 | 39.8 | 45 | 45.8 | -6.0 | 15 |
| 8 | Rennes | 52 | 42 | 38.8 | 51 | 41.5 | -2.7 | 11 |
| 9 | Angers | 50 | 33 | 30.9 | 33 | 36.9 | -6.0 | 14 |
| 10 | Bastia | 50 | 27 | 26.6 | 37 | 41.7 | -15.1 | 19 |
| 11 | Montpellier | 49 | 44 | 38.7 | 43 | 42.7 | -4.1 | 12 |
| 12 | Olympique de Marseille | 48 | 44 | 49.6 | 34 | 34.9 | 14.7 | 3 |
| 13 | Nantes | 48 | 31 | 36.9 | 34 | 39.2 | -2.2 | 10 |
| 14 | Bordeaux | 47 | 44 | 39.5 | 43 | 40.0 | -0.4 | 8 |
| 15 | Lorient | 46 | 37 | 37.5 | 50 | 50.2 | -12.7 | 18 |
| 16 | Guingamp | 44 | 41 | 38.7 | 44 | 44.6 | -5.9 | 13 |
| 17 | Toulouse | 40 | 36 | 43.5 | 49 | 39.9 | 3.7 | 5 |
| 18 | Stade de Reims | 39 | 42 | 42.1 | 49 | 49.1 | -7.1 | 16 |
| 19 | Gazélec Ajaccio | 34 | 30 | 33.9 | 47 | 41.7 | -7.8 | 17 |
| 20 | Troyes | 18 | 24 | 32.9 | 65 | 59.7 | -26.8 | 20 |

## 2. Finishing: goals vs xG over a career (open play, at least 150 shots)

Biggest over-performers:

| Player | Shots | Goals | xG | Goals − xG | z |
|---|---|---|---|---|---|
| Lionel Andrés Messi Cuccittini | 2,107 | 400 | 301.2 | +98.8 | +6.8 |
| Luis Alberto Suárez Díaz | 601 | 130 | 106.0 | +24.0 | +2.9 |
| Kylian Mbappé Lottin | 288 | 54 | 40.2 | +13.8 | +2.6 |
| Samuel Eto''o Fils | 295 | 63 | 50.4 | +12.6 | +2.2 |
| Ivan Rakitić | 221 | 27 | 17.1 | +9.9 | +2.7 |
| Thierry Henry | 314 | 50 | 40.3 | +9.7 | +1.8 |
| Zlatan Ibrahimović | 194 | 41 | 32.6 | +8.4 | +1.8 |
| Pedro Eliezer Rodríguez Ledesma | 315 | 54 | 45.7 | +8.3 | +1.5 |
| Ousmane Dembélé | 173 | 26 | 17.9 | +8.1 | +2.2 |
| Gonzalo Gerardo Higuaín | 195 | 34 | 26.3 | +7.7 | +1.8 |

Biggest under-performers:

| Player | Shots | Goals | xG | Goals − xG | z |
|---|---|---|---|---|---|
| Martin Braithwaite Christensen | 156 | 12 | 23.1 | -11.1 | -2.8 |
| Andrés Iniesta Luján | 367 | 24 | 30.2 | -6.2 | -1.3 |
| Gerard Piqué Bernabéu | 188 | 25 | 27.6 | -2.6 | -0.6 |
| Cristiano Ronaldo dos Santos Aveiro | 320 | 39 | 39.3 | -0.3 | -0.1 |
| Philippe Coutinho Correia | 245 | 20 | 19.8 | +0.2 | +0.1 |

Data coverage varies by player: Barcelona players appear in every Barcelona La Liga match from 2004/05 to 2020/21, while most other players appear only in the competitions listed in [DATA.md](DATA.md).

## 3. Barcelona in La Liga, season by season (open play)

| Season | Shots | Goals | xG | Goals − xG | Goals ÷ xG | z |
|---|---|---|---|---|---|---|
| 2004/2005 | 103 | 13 | 8.7 | 4.3 | 1.49 | +1.6 |
| 2005/2006 | 273 | 31 | 32.9 | -1.9 | 0.94 | -0.4 |
| 2006/2007 | 359 | 42 | 46.1 | -4.1 | 0.91 | -0.7 |
| 2007/2008 | 352 | 47 | 40.0 | 7.0 | 1.18 | +1.3 |
| 2008/2009 | 514 | 83 | 66.4 | 16.6 | 1.25 | +2.4 |
| 2009/2010 | 492 | 82 | 67.9 | 14.1 | 1.21 | +2.1 |
| 2010/2011 | 474 | 78 | 61.8 | 16.2 | 1.26 | +2.5 |
| 2011/2012 | 539 | 93 | 77.9 | 15.1 | 1.19 | +2.1 |
| 2012/2013 | 386 | 85 | 51.6 | 33.4 | 1.65 | +5.5 |
| 2013/2014 | 492 | 67 | 64.0 | 3.0 | 1.05 | +0.5 |
| 2014/2015 | 571 | 97 | 77.4 | 19.6 | 1.25 | +2.7 |
| 2015/2016 | 532 | 93 | 77.8 | 15.2 | 1.20 | +2.1 |
| 2016/2017 | 506 | 87 | 68.9 | 18.1 | 1.26 | +2.6 |
| 2017/2018 | 497 | 82 | 72.5 | 9.5 | 1.13 | +1.3 |
| 2018/2019 | 468 | 67 | 53.9 | 13.1 | 1.24 | +2.1 |
| 2019/2020 | 385 | 57 | 50.3 | 6.7 | 1.13 | +1.1 |
| 2020/2021 | 496 | 69 | 66.9 | 2.1 | 1.03 | +0.3 |

## 4. Players named in the thesis

| Player | Shots | Goals | xG | Goals − xG | z |
|---|---|---|---|---|---|
| Lionel Andrés Messi Cuccittini | 2,107 | 400 | 301.2 | +98.8 | +6.8 |
| Luis Alberto Suárez Díaz | 601 | 130 | 106.0 | +24.0 | +2.9 |
| Neymar da Silva Santos Junior | 407 | 74 | 67.2 | +6.8 | +1.0 |
| Andrés Iniesta Luján | 367 | 24 | 30.2 | -6.2 | -1.3 |

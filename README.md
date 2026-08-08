# World Cup Prediction 2026

Machine Learning model to predict the outcome (home win / draw / away win) and the expected scoreline of national team football matches, applied to a hypothetical Spain vs. Argentina match at the 2026 World Cup.

## Description

This project combines historical team statistics (Elo rating, recent form, attack/defense averages) to:

1. **Predict the match result** (home win, draw, away win) using a classification model.
2. **Predict the expected score** for each team using a Poisson regression model, then compute the most likely scorelines for the match.

## Methodology

- **Data cleaning:** missing values imputed with the median (chosen due to skewness and outliers present in the distributions).
- **Feature engineering:** difference variables (`elo_diff`, `attack_diff`, `defense_diff`, etc.) between home and away teams, plus context indicators (`is_neutral`, `is_world_cup`, `is_continental`).
- **Time-aware train/test split:** oldest 85% of matches for training, most recent 15% for testing — this avoids data leakage that a random split would introduce in time-series data.
- **Result model:** `RandomForestClassifier` (scikit-learn), with class balancing to account for draws being a minority class.
- **Goals model:** a separate `PoissonRegressor` trained for each team's goals, on a "symmetric" dataset (each match appears once as-is and once with teams swapped, so the model doesn't learn an ordering bias).
- **Scoreline simulation:** using each team's expected goals (λ), a joint Poisson probability matrix is built to obtain the most likely final scores.

## Repository Structure

```
world-cup-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── player_aggregates.csv        # aggregated player-level statistics
│   ├── teams_form.csv               # recent form per team
│   └── teams_match_features.csv     # main match dataset with engineered features
└── notebooks/
    └── prediccion_mundial.ipynb     # notebook with the full analysis and models
```

## How to Run It

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/world-cup-prediction.git
   cd world-cup-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook notebooks/prediccion_mundial.ipynb
   ```
   The notebook expects the CSVs inside the `data/` folder (relative path from wherever you run it).

## Data

- `teams_match_features.csv`: main dataset with match history, Elo ratings, attack/defense averages, and recent form for both teams.
- `teams_form.csv`: recent form (result streak) per team.
- `player_aggregates.csv`: aggregated player-level statistics.

## Author

Practice project on predictive modeling applied to football.

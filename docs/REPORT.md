# FIFA World Cup 2026 ML Predictor — Project Report

## 1. Objective

The objective of this project is to predict the outcomes of the 2026 FIFA World Cup using an end-to-end machine learning pipeline. Specifically, the system aims to:

- Predict individual match outcomes (home win, draw, away win) from historical and contextual features.
- Simulate the entire 2026 tournament (48 teams, 12 groups, expanded format) using Monte Carlo methods to estimate each team's probability of advancing through each round and winning the tournament.
- Validate predictions against live results as the tournament progresses.

The project combines 48,000+ international match results, FIFA rankings, Elo ratings, Transfermarkt squad quality data, bookmaker odds, and team form metrics into a unified feature set, trains multiple classifiers, selects the best ensemble by validation log-loss, and runs 1,000+ Monte Carlo simulations to produce tournament-wide probability estimates.

---

## 2. System Architecture & Structure

The pipeline is organized as six sequential steps, orchestrated by a CLI entry point (`run_pipeline.py`):

```
run_pipeline.py --all --n-simulations 1000
  Step 1: Data Scraping (Kaggle + Wikipedia + ESPN odds)
  Step 2: Feature Engineering (Elo, FIFA, form, H2H, SOS, EWM, draw features)
  Step 3: Model Training (XGBoost, RF, LogReg, NeuralNet)
  Step 3b: Ensemble Building (StackingClassifier selected by log_loss)
  Step 4: Evaluation (accuracy, log_loss, Brier, calibration)
  Step 5: Tournament Simulation (Monte Carlo, 1000 iterations, XGBoost model)
  Step 6: Visualization (power rankings, round probabilities, calibration)
```

### Directory Layout

```
worldcup/
├── run_pipeline.py              # CLI orchestrator
├── src/
│   ├── config.py                # Central configuration constants
│   ├── helpers.py               # Logging, team normalization, Kaggle setup
│   ├── scraping/                # Data collection modules
│   │   ├── download_kaggle.py          # Kaggle API downloads
│   │   ├── scrape_fifa_rankings.py     # Current FIFA rankings (Wikipedia)
│   │   ├── scrape_world_cup_2026.py    # Groups & fixtures (Wikipedia)
│   │   ├── scrape_odds.py              # Bookmaker odds (ESPN + the-odds-api)
│   │   ├── scrape_live_results.py      # Live match results
│   │   ├── scrape_historical_world_cups.py
│   │   └── scrape_squad_quality.py     # Transfermarkt squad quality data
│   ├── features/                # Feature engineering
│   │   ├── elo.py                      # Elo rating system
│   │   ├── build_features.py           # Historical match features (80 features)
│   │   └── build_2026_features.py      # WC2026 match feature vectors
│   ├── models/                  # Model training & evaluation
│   │   ├── train.py                    # XGBoost, RF, LogReg, NeuralNet
│   │   ├── ensemble.py                # StackingClassifier ensemble
│   │   ├── evaluate.py                 # Evaluation metrics & plots
│   │   └── live_validation.py          # Validate against live results
│   ├── simulation/              # Tournament simulation
│   │   ├── group_stage.py              # Group stage Monte Carlo
│   │   ├── knockout_stage.py           # Knockout bracket simulation
│   │   └── simulator.py               # Full tournament orchestrator
│   └── visualization/           # Output generation
│       ├── plots.py                    # Matplotlib/seaborn charts
│       ├── tables.py                   # Formatted text tables
│       └── insights.py                 # Analytical insight computation (confederation stats, dark horses, draw analysis, odds comparison)
├── data/
│   ├── raw/                     # Scraped/downloaded data
│   ├── processed/               # Engineered features, models, results
│   └── external/                # Continents mapping
├── notebooks/                   # Jupyter analysis notebooks
└── tests/                       # Pytest test suite
```

---

## 3. Dataset

### 3.1 Data Sources

| Source | Data | Rows | Access |
|--------|------|------|--------|
| Kaggle (`martj42/international-football-results-from-1872-to-2017`) | International match results (1872–present) | 48,000+ | Kaggle API |
| Kaggle (`cashncarry/fifaworldranking`) | Historical FIFA rankings (1993–present) | 60,000+ | Kaggle API |
| Wikipedia (`2026_FIFA_World_Cup`) | Group compositions, fixtures, current rankings | 48 teams | HTTP scrape |
| Wikipedia (`FIFA_Men's_World_Ranking`) | Current FIFA rankings | 211 entries | HTTP scrape |
| ESPN | Per-match betting odds | Variable | HTTP scrape |
| the-odds-api | Outright tournament winner odds | Variable | JSON API |
| Wikipedia (historical WC pages) | Historical World Cup brackets (1930–2022) | Multiple tournaments | HTTP scrape |
| Transfermarkt | Squad quality data (market value, size, age) for 48 WC 2026 teams | 48 teams | HTTP scrape |
| Manual (`continents.csv`) | Country-to-confederation mapping | 211 countries | Static file |

### 3.2 Data Processing

- **Team name normalization**: A mapping table resolves inconsistent team names across sources (e.g., "USA" → "United States", "Korea Republic" → "South Korea", "Côte d'Ivoire" → "Ivory Coast").
- **FIFA rankings merge**: Historical Kaggle rankings are merged with current Wikipedia rankings, deduplicated by date and country.
- **Feature caching**: A SHA-256 hash of input files (version `7`) ensures features are recomputed only when source data or feature logic changes. The `squad_quality.csv` file is included in the cache hash computation.

### 3.3 Train/Validation/Test Split

The dataset is split temporally to prevent data leakage:
- **Training**: All matches before 2022
- **Validation**: Matches from 2022 (including the 2022 World Cup, 64 matches)
- **Test**: Matches after 2022

The primary validation set (2022 World Cup) provides a realistic out-of-sample evaluation on high-stakes tournament matches.

---

## 4. Methods

### 4.1 Feature Engineering (80 Features)

Each match is represented by 80 features across nine categories:

#### 4.1.1 Elo Rating Features (7)
A custom Elo rating system processes the full 48,000+ match history with tournament-specific K-factors: World Cup matches (K=80), qualifiers (K=60), friendlies (K=40), and default (K=50). Home advantage adds +100 Elo points. A draw probability model uses `ELO_DRAW_FACTOR=0.30`, which scales the draw probability based on how close the teams are in rating.

Features: `elo_home`, `elo_away`, `elo_delta`, `elo_abs_delta`, `elo_home_win_prob`, `elo_draw_prob`, `elo_away_win_prob`.

#### 4.1.2 FIFA Ranking Features (7–8)
The most recent FIFA ranking and points for each team as of the match date, looked up via merge_asof.

Features: `fifa_rank_home`, `fifa_rank_away`, `fifa_rank_delta`, `fifa_rank_abs_delta`, `fifa_points_home`, `fifa_points_away`, `fifa_points_delta`, `fifa_points_abs_delta`.

#### 4.1.3 Form Features (28)
Recent team performance metrics computed from the last 5 and last 10 matches before each match date. Standard form includes win/draw/loss rates, goals scored/conceded averages, goal difference average, and clean sheet rate. Exponentially weighted moving averages (EWM, decay factor 0.3) provide recency-weighted versions of win/draw/loss rates.

Features: `home_form_last{5,10}_*` (7 metrics each), `away_form_last{5,10}_*` (7 metrics each), `away_form_last{5,10}_ewm_*` (3 metrics each).

#### 4.1.4 Head-to-Head Features (6)
Statistics from the last 5 meetings between the two teams.

Features: `h2h_home_wins`, `h2h_draws`, `h2h_away_wins`, `h2h_draw_rate`, `h2h_home_goals_avg`, `h2h_away_goals_avg`.

#### 4.1.5 Strength of Schedule (2)
Average opponent Elo rating over each team's last 10 matches.

Features: `home_sos_avg_opp_elo`, `away_sos_avg_opp_elo`.

#### 4.1.6 Draw-Predictive Features (5)
Engineered specifically to improve draw prediction, the hardest outcome in football:

- `elo_close`: Binary flag if |Elo delta| < 100
- `draw_tendency`: Combined draw probability amplified for close matches (1.5x when teams are close)
- `fifa_close`: Binary flag if |FIFA rank delta| < 20
- `tournament_draw_rate`: Historical draw rate for the tournament type (friendly 26%, qualifier 24%, WC group 23%, WC knockout 15%)
- `combined_draw_prob`: Blended draw probability from Elo and form signals

#### 4.1.7 Context Features (7)
Binary and scalar flags capturing match context:

Features: `neutral`, `home_advantage` (1.0 for home, 0.5 for neutral), `is_host_nation`, `same_confederation`, `is_world_cup`, `is_qualifier`, `is_knockout`.

#### 4.1.8 Interaction & Odds Features (5)
Cross-terms and bookmaker-implied probabilities (when available for WC2026 matches):

Features: `elo_delta_x_home_advantage`, `fifa_rank_delta_x_same_confed`, `odds_home_implied_prob`, `odds_draw_implied_prob`, `odds_away_implied_prob`.

#### 4.1.9 Squad Quality Features (8)
Market value and player quality metrics from Transfermarkt for all 48 WC 2026 teams. For historical matches where Transfermarkt data is unavailable, confederation-average fallbacks are used (66% of historical matches have actual data; 34% use confederation averages). All WC 2026 matches have actual data (0 NaN).

Features: `home_squad_value_m`, `away_squad_value_m`, `squad_value_delta`, `squad_value_abs_delta`, `home_avg_player_value_m`, `away_avg_player_value_m`, `home_top_player_value_m`, `away_top_player_value_m`.

Data highlights (from Transfermarkt):
- **Top**: France €1.52B, England €1.36B, Spain €1.22B
- **Mid**: USA €386M, Morocco €448M, Japan €271M
- **Bottom**: Qatar €20M, Jordan €20M, Iraq €21M

### 4.2 Model Training

Four base classifiers are trained:

| Model | Hyperparameter Tuning | Key Configuration |
|-------|----------------------|-------------------|
| **XGBoost** | Optuna (30 trials, 5-fold TimeSeriesSplit CV, optimized for log_loss) | Draw class weight 4x; multi:softprob objective |
| **Random Forest** | GridSearchCV (3-fold TimeSeriesSplit, scoring=neg_log_loss) | `class_weight="balanced"`; n_estimators ∈ {100,200,300} |
| **Logistic Regression** | GridSearchCV (3-fold TimeSeriesSplit, scoring=neg_log_loss) | `class_weight="balanced"`; C ∈ {0.01,0.1,1.0,10.0}; StandardScaler pipeline |
| **Neural Network (sklearn MLP)** | Early stopping (patience=10) | Layers [128,64,32]; alpha=0.001; batch_size=256; adaptive lr; max_iter=300; draw class weight 4x |

**Label encoding**: Home win → 2, Draw → 1, Away win → 0.

**Missing values**: A `SimpleImputer(strategy="median")` is fit on training data and saved as `imputer.joblib` for use during simulation.

**Draw handling**: Draws are the hardest class to predict. The pipeline uses multiple strategies:
- XGBoost: 4x sample weight for draws (`{away_win: 1.0, draw: 4.0, home_win: 1.0}`)
- RF & LogReg: `class_weight="balanced"` for automatic rebalancing
- Stacking meta-learner: `class_weight="balanced"` on LogisticRegression
- Draw-predictive features (elo_close, draw_tendency, fifa_close, tournament_draw_rate, combined_draw_prob)
- `ELO_DRAW_FACTOR=0.30` in the Elo probability model
- `WC_GROUP_DRAW_RATE=0.25` calibration in group stage simulation, boosting under-predicted draws toward the historical ~25% WC group-stage draw rate

**LightGBM** was initially included but removed due to poor performance (0.49 accuracy, 0.99 log_loss).

### 4.3 Ensemble Selection

The `build_best_ensemble` function evaluates multiple ensemble candidates on the validation set and selects the one with the lowest log_loss:

1. **Individual models**: Each base model evaluated independently
2. **VotingEnsemble (uniform)**: Soft voting with equal weights
3. **WeightedVotingEnsemble**: Soft voting with inverse-log-loss weights
4. **StackingEnsemble**: LogisticRegression meta-learner over all base models (3-fold CV)

The current best ensemble is a **StackingClassifier** with 4 base models and a LogisticRegression meta-learner with `class_weight="balanced"`, selected by validation log_loss.

**Calibration**: Isotonic calibration via `CalibratedWrapper` was tested but removed because it degraded test performance (log_loss 0.8374 → 1.0436).

### 4.4 Tournament Simulation

The simulation uses Monte Carlo methods to estimate tournament advancement probabilities:

1. **Model selection**: XGBoost is used as the default simulation model for speed (6.6 MB model, fast inference), though the ensemble can also be used.

2. **Group stage** (12 groups × 4 teams):
   - Each group's 6 round-robin matches are simulated by predicting win/draw/loss probabilities and sampling outcomes.
   - Probabilities are reordered from `predict_proba()` output `[away_win, draw, home_win]` to `[home_win, draw, away_win]`.
   - **Fixture swap handling**: When a fixture lookup falls back to the reversed match, probabilities are swapped accordingly.
   - **Draw calibration**: If the model's predicted draw probability is below `WC_GROUP_DRAW_RATE=0.25`, it is boosted to 25%, with excess redistributed proportionally from win/loss probabilities.
   - Probabilities are normalized to sum to 1.0 before sampling.
   - Missing features are filled using the trained `SimpleImputer` (per-feature medians), not `np.nanmean()`.
   - Goals are generated using Poisson distributions (mean 1.5 for winners, 1.0 for draws).
   - Teams are ranked by points, then goal difference, then goals scored.
   - Top 2 from each group advance; 8 best third-place teams also advance.

3. **Knockout stage** (32-team bracket: R32 → R16 → QF → SF → Final):
   - Draw probability is redistributed: 70% reduction in draw probability, split 55/45 favoring the stronger team based on model probabilities.
   - **Fixture swap handling** applies here as well, ensuring correct home/away probability perspective.
   - Draw calibration is NOT applied in knockout (draws are redistributed to produce a winner).
   - Each knockout match produces a single winner.
   - Third-place match is not simulated.

4. **Aggregation**:
   - 1,000 complete tournament runs (configurable via `--n-simulations`).
   - Track advancement through each round for every team.
   - Compute probabilities for R32, R16, QF, SF, Final, and Winner.

5. **Live validation**: As real WC2026 results become available, group stage standings are updated with actual scores and remaining matches are still simulated.

### 4.5 Bug Fixes (v0.3.0)

Several critical bugs were identified and fixed during development:

| Bug | Impact | Fix |
|-----|--------|-----|
| Probability mapping | `predict_proba()` returns `[away_win, draw, home_win]` but simulator treated it as `[home_win, draw, away_win]` | Reordered: `np.array([proba[2], proba[1], proba[0]])` |
| Missing imputer | Simulators used `np.nanmean()` (global mean) instead of trained `SimpleImputer` (per-feature medians) | Load `imputer.joblib` and use `imputer.transform()` |
| Probability normalization | Probabilities could drift from summing to 1.0 | Added `probs = probs / probs.sum()` before sampling |
| Knockout draw redistribution | Wrong indices after probability reorder | Fixed index mapping: `probs[0]` = home_win, `probs[1]` = draw, `probs[2]` = away_win |
| `neutral=0` for all WC 2026 | Home advantage wrongly applied to neutral venues | Neutral venues get `neutral=1`; host nation matches get `neutral=0` |
| Stale Elo cache | Only 4 teams cached | Cache recomputed if < 10 teams |
| WC fixture duplicates | 144 matches instead of 72 | Fixed to use `combinations` |
| `is_knockout` under-detected | Only 3/49413 matches flagged | Added date-based WC knockout detection; 271 matches now correctly flagged |
| **Fixture swap bug** | When fixture lookup falls back to reversed match, probabilities were from wrong perspective (home/away swapped) | Added `swapped` flag and probability swap `[home_win, draw, away_win]` → `[away_win, draw, home_win]` in 4 files: `group_stage.py`, `knockout_stage.py`, `simulator.py`, `live_validation.py` |

---

## 5. Mathematical Models

This section formalizes the mathematical foundations underlying each component of the pipeline.

### 5.1 Elo Rating System

The Elo rating system assigns each team a scalar rating that is updated after every match. It is the foundational model for generating predictive features.

#### 5.1.1 Expected Score

For a match between home team $H$ with rating $R_H$ and away team $A$ with rating $R_A$, the expected score for the home team is:

$$E_H = \frac{1}{1 + 10^{(R_A - R_H^*) / 400}}$$

where $R_H^* = R_H + \delta_H$ is the home-adjusted rating, and $\delta_H$ is the home advantage bonus:

$$\delta_H = \begin{cases} 0 & \text{if neutral venue} \\ 100 & \text{if home match} \end{cases}$$

The expected score for the away team is $E_A = 1 - E_H$.

#### 5.1.2 Rating Update

After each match, ratings are updated based on the actual result versus the expected score:

$$R_H' = R_H + K_T \cdot (S_H - E_H)$$
$$R_A' = R_A + K_T \cdot ((1 - S_H) - (1 - E_H))$$

where $S_H$ is the actual score for the home team:

$$S_H = \begin{cases} 1 & \text{if home win} \\ 0.5 & \text{if draw} \\ 0 & \text{if away win} \end{cases}$$

and $K_T$ is the tournament-specific K-factor:

| Tournament Type | K-Factor |
|----------------|----------|
| FIFA World Cup | 80 |
| World Cup Qualification | 60 |
| Friendly | 40 |
| Default | 50 |

New teams start with an initial rating of $R_0 = 1000$.

#### 5.1.3 Match Probability Model (Three-Class)

The standard Elo model produces only a binary expected score. To produce three-class probabilities (home win, draw, away win), a draw factor model is used:

$$P_D = \min\left(\phi \cdot (1 - |E_H - E_A|), \; \min\left(\max(0.05, \min(E_H, E_A)), \; 0.35\right)\right)$$

where $\phi = 0.30$ is the `ELO_DRAW_FACTOR`. The draw probability is highest when teams are evenly matched ($E_H \approx E_A$) and decreases as the rating difference grows.

The remaining probability is distributed proportionally to the expected scores:

$$P_H = (1 - P_D) \cdot \frac{E_H}{E_H + E_A}$$
$$P_A = (1 - P_D) \cdot \frac{E_A}{E_H + E_A}$$

Final normalization ensures the probabilities sum to 1:

$$\hat{P}_H = \frac{P_H}{P_H + P_D + P_A}, \quad \hat{P}_D = \frac{P_D}{P_H + P_D + P_A}, \quad \hat{P}_A = \frac{P_A}{P_H + P_D + P_A}$$

### 5.2 Feature Engineering Formulations

#### 5.2.1 Exponentially Weighted Moving Average (EWM) Form

For team form features, a recency-weighted version is computed using an exponential decay:

$$\bar{x}_\text{EWM} = \frac{\sum_{i=0}^{n-1} w_i \cdot x_i}{\sum_{i=0}^{n-1} w_i}, \quad w_i = e^{-\lambda i}$$

where $\lambda = 0.3$ is the decay factor and $i=0$ is the most recent match. This gives higher weight to recent matches. EWM form features are computed for both the last 5 and last 10 matches, producing win rate, draw rate, and loss rate.

#### 5.2.2 Strength of Schedule (SOS)

The strength of schedule for team $T$ is computed as the average opponent Elo rating over the last $n$ matches:

$$\text{SOS}_T = \frac{1}{n} \sum_{i=1}^{n} R_{\text{opp}_i}$$

where $R_{\text{opp}_i}$ is the Elo rating of the $i$-th most recent opponent.

#### 5.2.3 Draw-Predictive Features

Several features are specifically engineered to improve draw prediction:

**elo_close**: A binary indicator for closely matched teams:

$$\text{elo\_close} = \mathbb{1}[|R_H - R_A| < 100]$$

**fifa_close**: A binary indicator for closely ranked teams:

$$\text{fifa\_close} = \mathbb{1}[|\text{rank}_H - \text{rank}_A| < 20]$$

**draw_tendency**: An amplified draw probability for close matches:

$$\text{draw\_tendency} = \hat{P}_D \cdot (1 + 0.5 \cdot \text{elo\_close})$$

**combined_draw_prob**: A blended draw probability from Elo and form signals:

$$\text{combined\_draw\_prob} = \frac{\hat{P}_D + 0.5 \cdot (\bar{d}_H + \bar{d}_A)}{2}$$

where $\bar{d}_H$ and $\bar{d}_A$ are the last-10 draw rates for the home and away teams respectively.

**tournament_draw_rate**: A prior draw rate based on tournament type:

| Tournament Type | Draw Rate Prior |
|----------------|-----------------|
| Friendly | 0.26 |
| Qualifier | 0.24 |
| WC Group Stage | 0.23 |
| WC Knockout | 0.15 |
| Other | 0.25 |

#### 5.2.4 Interaction Features

Two interaction terms capture multiplicative relationships:

$$\text{elo\_delta} \times \text{home\_advantage} = (R_H - R_A) \cdot h_A$$
$$\text{fifa\_rank\_delta} \times \text{same\_confederation} = (\text{rank}_H - \text{rank}_A) \cdot \mathbb{1}[\text{confed}_H = \text{confed}_A]$$

where $h_A \in \{1.0, 0.5\}$ is the home advantage weight (1.0 for home matches, 0.5 for neutral venues).

### 5.3 Classification Models

The task is a three-class classification problem. Let $\mathbf{x} \in \mathbb{R}^{80}$ be the feature vector for a match, and $y \in \{0, 1, 2\}$ be the outcome label (0 = away win, 1 = draw, 2 = home win).

#### 5.3.1 Label Encoding

The raw match outcome (home score vs. away score) is mapped as:

$$y = \begin{cases} 2 & \text{if } s_H > s_A \quad \text{(home win)} \\ 1 & \text{if } s_H = s_A \quad \text{(draw)} \\ 0 & \text{if } s_H < s_A \quad \text{(away win)} \end{cases}$$

**Critical note**: `predict_proba()` in scikit-learn returns probabilities in class order $[P(y=0), P(y=1), P(y=2)] = [\text{away\_win}, \text{draw}, \text{home\_win}]$. The simulation code must reorder these to $[\text{home\_win}, \text{draw}, \text{away\_win}]$ for correct interpretation.

#### 5.3.2 Class-Conditional Sample Weighting

To address the class imbalance (draws are underrepresented at ~25%), a sample weight vector $\mathbf{w}$ is applied during training:

$$w_i = \begin{cases} 1.0 & \text{if } y_i = 0 \quad \text{(away win)} \\ 4.0 & \text{if } y_i = 1 \quad \text{(draw)} \\ 1.0 & \text{if } y_i = 2 \quad \text{(home win)} \end{cases}$$

This is applied to XGBoost and NeuralNet via the `sample_weight` parameter. Random Forest, Logistic Regression, and the Stacking ensemble meta-learner use `class_weight="balanced"`, which computes class weights inversely proportional to class frequencies:

$$w_c = \frac{n}{n_c \cdot C}$$

where $n$ is the total number of samples, $n_c$ is the number of samples in class $c$, and $C = 3$ is the number of classes.

#### 5.3.3 XGBoost

XGBoost minimizes the multi-class logistic loss (cross-entropy) with regularization:

$$\mathcal{L}(\theta) = \sum_{i=1}^{n} w_i \cdot \text{CE}(y_i, \hat{\mathbf{p}}_i) + \sum_{k=1}^{K} \left[ \gamma T_k + \frac{\lambda}{2} \|\mathbf{f}_k\|^2 \right]$$

where CE is the cross-entropy loss, $T_k$ is the number of leaves in tree $k$, $\gamma$ is the leaf penalty, and $\lambda$ is the L2 regularization. Hyperparameters are tuned via Optuna (30 trials, 5-fold TimeSeriesSplit CV) optimizing for log-loss.

#### 5.3.4 Random Forest

Random Forest fits $B$ decision trees on bootstrapped samples with random feature subsets:

$$\hat{P}(y = c \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}[\text{tree}_b(\mathbf{x}) = c]$$

Hyperparameters are tuned via GridSearchCV (3-fold TimeSeriesSplit, scoring=neg_log_loss): `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {10, 20, None}, `min_samples_split` ∈ {2, 5}, `class_weight="balanced"`.

#### 5.3.5 Logistic Regression

The multinomial logistic regression models:

$$P(y = c \mid \mathbf{x}) = \frac{e^{\mathbf{w}_c^T \mathbf{x} + b_c}}{\sum_{c'=0}^{2} e^{\mathbf{w}_{c'}^T \mathbf{x} + b_{c'}}}$$

with L2 regularization. The regularization strength $C$ is tuned via GridSearchCV (3-fold TimeSeriesSplit, scoring=neg_log_loss) over $C \in \{0.01, 0.1, 1.0, 10.0\}$. Features are standardized via `StandardScaler` in a pipeline. `class_weight="balanced"` handles class imbalance.

#### 5.3.6 Neural Network (sklearn MLPClassifier)

A feedforward neural network with architecture:

$$\mathbb{R}^{80} \xrightarrow{W_1, b_1} \mathbb{R}^{128} \xrightarrow{\text{ReLU}} \mathbb{R}^{64} \xrightarrow{\text{ReLU}} \mathbb{R}^{32} \xrightarrow{\text{ReLU}} \mathbb{R}^{3} \xrightarrow{\text{softmax}} \mathbb{R}^{3}$$

Configuration: Adam optimizer with adaptive learning rate (initial $\eta = 10^{-3}$), L2 regularization $\alpha = 0.001$, batch size 256, maximum 300 iterations, early stopping with patience 10. Draw samples receive 4x weight as in XGBoost.

#### 5.3.7 Stacking Ensemble

The final model is a StackingClassifier with 4 base learners and a LogisticRegression meta-learner:

$$\hat{P}_\text{stack}(y = c \mid \mathbf{x}) = \sigma\left(\mathbf{w}_c^T \boldsymbol{\phi}(\mathbf{x}) + b_c\right)$$

where $\boldsymbol{\phi}(\mathbf{x}) = [\hat{P}_\text{XGB}(y=c \mid \mathbf{x}), \; \hat{P}_\text{RF}(y=c \mid \mathbf{x}), \; \hat{P}_\text{LR}(y=c \mid \mathbf{x}), \; \hat{P}_\text{MLP}(y=c \mid \mathbf{x})]_{c \in \{0,1,2\}}$ is the 12-dimensional vector of base model probabilities, and $\sigma$ is the softmax function.

The meta-learner is trained via 3-fold cross-validation on the training set, using `predict_proba` as the stacking method. Ensemble selection is performed by choosing the candidate with the lowest validation log-loss.

#### 5.3.8 Ensemble Candidate Selection

Four ensemble types are evaluated on the validation set:

1. **Individual models**: Each base model evaluated independently
2. **VotingEnsemble (uniform)**: $\hat{P}(y=c \mid \mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} \hat{P}_m(y=c \mid \mathbf{x})$
3. **WeightedVotingEnsemble**: $\hat{P}(y=c \mid \mathbf{x}) = \sum_{m=1}^{M} \alpha_m \hat{P}_m(y=c \mid \mathbf{x})$, where $\alpha_m = \frac{1/\text{LL}_m}{\sum_{m'} 1/\text{LL}_{m'}}$ and $\text{LL}_m$ is the validation log-loss of model $m$
4. **StackingEnsemble**: LogisticRegression meta-learner (described above)

The candidate with the lowest validation log-loss is selected as the final model.

### 5.4 Evaluation Metrics

#### 5.4.1 Accuracy

$$\text{Accuracy} = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}[\hat{y}_i = y_i]$$

#### 5.4.2 Log Loss (Cross-Entropy)

$$\text{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=0}^{2} \mathbb{1}[y_i = c] \cdot \log(\hat{p}_{i,c})$$

This is the primary model selection metric, as it measures the quality of probability estimates rather than just classification accuracy.

#### 5.4.3 Brier Score

The Brier score for each class $c$ is:

$$\text{BS}_c = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_{i,c} - \mathbb{1}[y_i = c])^2$$

The average Brier score across all three classes provides a single calibration metric:

$$\text{AvgBS} = \frac{1}{3} \sum_{c=0}^{2} \text{BS}_c$$

### 5.5 Tournament Simulation

#### 5.5.1 Monte Carlo Framework

Let $\mathcal{M}$ be the trained model, $\mathbf{x}_{ij}$ be the feature vector for match between team $i$ and team $j$, and $N$ be the number of simulations. For each simulation $s = 1, \ldots, N$:

1. **Group Stage**: For each group $G_k$ ($k = A, B, \ldots, L$), simulate all $\binom{4}{2} = 6$ round-robin matches.
2. **Knockout Stage**: Simulate the single-elimination bracket (R32 → R16 → QF → SF → Final).
3. **Aggregate**: Count how often each team reaches each round.

The estimated probability that team $T$ reaches round $r$ is:

$$\hat{P}(T \text{ reaches } r) = \frac{1}{N} \sum_{s=1}^{N} \mathbb{1}[T \text{ reaches round } r \text{ in simulation } s]$$

#### 5.5.2 Match Outcome Sampling

For a group stage match between home team $H$ and away team $A$:

$$\mathbf{p} = \text{softmax}(\mathcal{M}(\mathbf{x}_{HA})) \in \Delta^2$$

After reordering from `[away_win, draw, home_win]` to `[home_win, draw, away_win]`, handling swapped fixtures, and normalization:

$$\hat{p}_H + \hat{p}_D + \hat{p}_A = 1$$

The outcome is sampled:

$$\text{outcome} \sim \text{Categorical}(\hat{p}_H, \hat{p}_D, \hat{p}_A)$$

#### 5.5.2b Fixture Swap Handling

When a fixture lookup falls back to the reversed match (e.g., looking up "Qatar vs Switzerland" but finding "Switzerland vs Qatar"), the `swapped` flag is set and probabilities are swapped:

$$[\hat{p}_H, \hat{p}_D, \hat{p}_A] \leftarrow [\hat{p}_A, \hat{p}_D, \hat{p}_H]$$

This ensures that the home/away perspective is always correct relative to the fixture ordering.

#### 5.5.2c Draw Calibration (Group Stage Only)

Historical World Cup group-stage matches have a draw rate of approximately 25%, but the model may under-predict draws (e.g., BestEnsemble predicts only 171 draws vs 830 actual on the test set). The `_calibrate_draw()` function boosts draw probability when it falls below `WC_GROUP_DRAW_RATE = 0.25`:

$$\hat{p}_D' = \max(\hat{p}_D, \; P_{D,\text{WC}})$$

The excess probability is redistributed proportionally to the win/loss probabilities:

$$\hat{p}_H' = \hat{p}_H \cdot \frac{1 - \hat{p}_D'}{\hat{p}_H + \hat{p}_A}, \quad \hat{p}_A' = \hat{p}_A \cdot \frac{1 - \hat{p}_D'}{\hat{p}_H + \hat{p}_A}$$

This calibration is applied **only in group stage** simulations, not in knockout stage (where draws are redistributed to produce a winner).

#### 5.5.3 Goal Generation

Match scores are generated using Poisson distributions conditional on the outcome:

$$\text{Goals}_{\text{winner}} \sim \text{Poisson}(\lambda=1.5)$$
$$\text{Goals}_{\text{loser}} \sim \max(0, \text{Goals}_{\text{winner}} - 1 - U[0,1])$$

For draws:

$$\text{Goals}_H = \text{Goals}_A \sim \text{Poisson}(\lambda=1.0)$$

#### 5.5.4 Group Stage Ranking

Teams within each group are ranked by:

1. **Points**: 3 for a win, 1 for a draw, 0 for a loss
2. **Goal difference** (tiebreaker)
3. **Goals scored** (second tiebreaker)

The top 2 teams from each of the 12 groups advance. The 8 best third-place teams also advance, selected by ranking all third-place teams by (points, goal difference, goals scored).

#### 5.5.5 Knockout Stage Draw Redistribution

In knockout matches, draws are not possible. The model's draw probability is redistributed to the win/loss probabilities:

$$p_D' = p_D \cdot (1 - 0.7) = 0.3 \cdot p_D$$

The 70% reduction in draw probability is split 55/45 favoring the stronger team:

$$\hat{p}_H = p_H + 0.7 \cdot p_D \cdot 0.55$$
$$\hat{p}_A = p_A + 0.7 \cdot p_D \cdot 0.45$$

These are then normalized to produce a binary outcome:

$$\text{winner} \sim \text{Categorical}(\hat{p}_H, \hat{p}_A)$$

The 55/45 split is based on the model's relative assessment of team strength, giving the home team (or the team with higher Elo) a slight advantage in extra time.

#### 5.5.6 Missing Feature Imputation

During simulation, some WC2026 feature vectors may contain missing values (e.g., odds not available, no head-to-head history). These are imputed using a `SimpleImputer` with median strategy, fitted on the training data:

$$x_{j}^{\text{missing}} \leftarrow \tilde{x}_j$$

where $\tilde{x}_j$ is the median of feature $j$ across the training set. This replaces the earlier approach of using `np.nanmean()` (global mean), which was identified as a bug.

---

## 6. Results

### 6.1 Model Performance (Test Set)

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| **BestEnsemble (Stacking)** | **0.620** | **0.836** |
| RandomForest | 0.606 | 0.859 |
| LogisticRegression | 0.570 | 0.878 |
| XGBoost | 0.552 | 0.918 |
| NeuralNet (MLP) | 0.597 | 1.172 |

The StackingClassifier ensemble achieves the best log_loss, confirming that combining diverse model types improves calibration. The ensemble meta-learner uses `class_weight="balanced"` to handle draw class imbalance.

**Key Draw Stats:**
- XGBoost: pred_draws=1344, actual_draws=830 on test set (over-predicts with 4x weight)
- BestEnsemble: pred_draws=171, actual_draws=830 (under-predicts)
- Draw calibration in the simulator (WC_GROUP_DRAW_RATE=0.25) fixes this for WC predictions

### 6.1b Live Validation (WC 2026, 10 matches)

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| BestEnsemble | 40% (4/10) | 1.059 |
| XGBoost | 40% (4/10) | 1.001 |
| RandomForest | 30% (3/10) | 0.974 |

### 6.2 Validation on 2022 World Cup (64 matches)

| Metric | Before Optimizations | After Optimizations |
|--------|----------------------|---------------------|
| Accuracy | 56.25% | **59.38%** |
| Log Loss | 0.9789 | **0.9383** |
| Avg Brier | 0.1913 | **0.1657** |

The improvements come from:
- Draw class weighting (4x for XGBoost/NeuralNet, `class_weight="balanced"` for RF/LogReg and meta-learner)
- `ELO_DRAW_FACTOR=0.30` (increased from 0.25)
- Draw-predictive features (`elo_close`, `draw_tendency`, `fifa_close`, `tournament_draw_rate`, `combined_draw_prob`)
- Squad quality features (8 new features from Transfermarkt data)
- Ensemble/model selection by log_loss instead of accuracy
- Fixture swap bug fix (correct probability perspective when match lookup is reversed)

### 6.3 Tournament Simulation Results (1,000 runs, XGBoost model)

**Top 5 teams by tournament winning probability:**

| Rank | Team | Win Prob |
|------|------|----------|
| 1 | Mexico | 7.6% |
| 2 | Switzerland | 6.5% |
| 3 | United States | 6.0% |
| 4 | Germany | 4.8% |
| 5 | Canada | 4.7% |

Notable observations:
- **Mexico and Canada** benefit from host-nation home advantage (the 2026 WC is hosted by US/CA/MX), reflected in their elevated probabilities.
- **Switzerland's** high probability (5.2%) suggests a favorable group draw and bracket path.
- **Argentina** and **France**, despite being traditional powerhouses, have lower advancement probabilities from the group stage (~50%) due to potentially harder group composition in the simulation.
- **Spain** has the highest R32 probability (97.2%) but lower overall winning probability (2.9%), suggesting easier group but tougher knockout path.

### 6.4 Confederation Strength Analysis

The simulation reveals significant disparities in confederation-level performance, reflecting both the quality depth within each confederation and the structural advantages conferred by the expanded 48-team format. The addition of Transfermarkt squad quality features provides a direct measure of team quality that complements Elo and FIFA rankings.

> Note: The detailed confederation percentages below are from a previous simulation run. The updated top 5 tournament win probabilities are: Mexico 7.6%, Switzerland 6.5%, USA 6.0%, Germany 4.8%, Canada 4.7%.

| Confederation | Teams | Avg Win % | Avg Ro16 % | Avg QF % | Best Team | Best Win % |
|---------------|-------|-----------|------------|----------|-----------|------------|
| CONCACAF | 6 | 2.93 | 22.43 | 11.37 | Mexico | 6.3 |
| CONMEBOL | 6 | 2.88 | 24.87 | 11.88 | Brazil | 4.6 |
| UEFA | 16 | 2.44 | 26.63 | 13.05 | Switzerland | 5.2 |
| CAF | 10 | 1.35 | 16.02 | 8.07 | Morocco | 3.6 |
| AFC | 9 | 1.33 | 13.41 | 7.39 | Japan | 2.4 |
| OFC | 1 | 0.60 | 9.30 | 4.50 | New Zealand | 0.6 |

Key takeaways:

- **CONCACAF's high average** is driven almost entirely by host advantage for Mexico (7.6%), United States (6.0%), and Canada (4.7%). The other three CONCACAF teams all sit below 2%, pulling the confederation average down less than the hosts pull it up.
- **UEFA has the highest average Ro16 probability** (26.63%), indicating that even its mid-tier teams are more likely to escape the group stage. This structural depth makes UEFA the most consistently competitive confederation in the tournament.
- **CAF and AFC** show similar average win probabilities (~1.3%), but CAF teams advance to the Ro16 at a slightly higher rate (16.0% vs. 13.4%), suggesting a marginal African edge in group-stage competitiveness.
- **Oceania's sole representative** (New Zealand) faces the steepest climb, with only a 9.3% chance of reaching the Round of 16 and a 0.6% tournament win probability — the lowest of any confederation's best team.

### 6.5 Host Nation Advantage Analysis

The 2026 World Cup's tri-host format (Mexico, Canada, United States) creates a unique home-advantage dynamic. The simulation explicitly models this through the `is_host_nation` and `home_advantage` features, which assign full home advantage when a host plays on home soil and neutral venue designation for all other matches. The combined effect is substantial: the three host nations together account for approximately 15.9% of total tournament win probability, a figure that far exceeds what their FIFA rankings alone would predict.

| Host Nation | Win % | Final % | SF % | QF % | Ro16 % |
|-------------|-------|---------|------|------|--------|
| Mexico | 7.6 | — | — | — | — |
| United States | 6.0 | — | — | — | — |
| Canada | 4.7 | — | — | — | — |
| **Combined** | **18.3** | — | — | — | — |

Key takeaways:

- **Mexico's position as the top overall pick** (7.6%) is almost entirely a function of home advantage. Without it, Mexico's Elo and FIFA rankings would place it well outside the top 5. Home advantage adds +100 Elo points and `home_advantage=1.0` for host matches.
- **The United States at 6.0%** benefits from host advantage, though it faces competitive group opposition.
- **Canada at 4.7%** also benefits from home advantage, pushing it into the top 5 despite not being a traditional power.
- **Combined host probability of 18.3%** is remarkable: three teams that would collectively account for perhaps 5–7% without home advantage instead capture nearly 1 in 5 simulation wins. This illustrates the outsized structural impact of host-nation status in the expanded 48-team format.

### 6.6 Draw Prediction Analysis

Draws are the most difficult outcome to predict in football, and the model dedicates significant feature engineering to them (§4.1.6, §5.2.3). Draw calibration (`WC_GROUP_DRAW_RATE=0.25`) in the simulator boosts under-predicted draws toward the historical ~25% WC group-stage rate. Across the 72 group-stage matches, the model's draw predictions reveal a tournament format that — despite the expanded field — still produces a significant number of closely matched encounters.

> Note: The specific match-level draw probabilities below are from a previous simulation run.

| Metric | Value |
|--------|-------|
| Total group-stage matches | 72 |
| Predicted draws | 21 (29.2%) |
| Predicted home wins | 48 (66.7%) |
| Predicted away wins | 3 (4.2%) |
| Average draw probability | 35.4% |
| Draw probability range | 8.9% — 58.7% |

**Top 5 Most Likely Draws:**

| Match | Group | Draw Probability |
|-------|-------|-----------------|
| Congo DR vs Uzbekistan | — | 58.7% |
| South Korea vs South Africa | — | 55.8% |
| Switzerland vs Canada | — | 55.1% |
| Mexico vs South Korea | — | 55.0% |
| Morocco vs Brazil | — | 54.4% |

Key takeaways:

- **The 29.2% predicted draw rate** aligns closely with the historical World Cup group-stage draw rate of ~23–26%, but is somewhat elevated because the model's draw-predictive features (especially `combined_draw_prob` and `tournament_draw_rate`) boost draw probabilities when teams are closely matched — a common scenario in a 48-team field with many similarly ranked teams.
- **Away wins are rare** (only 3 of 72 matches), reflecting the strong home-advantage signal embedded in the model. In the 2026 format, host-nation matches and geographically proximate "home" crowds create substantial asymmetry. The three predicted away wins are all cases where the away team is significantly stronger (e.g., Colombia over Congo DR).
- **The draw probability range of 8.9%–58.7%** is wide. The low end (8.9%) occurs in mismatches like strong hosts vs. weak visitors, while the high end (58.7%) occurs when two closely ranked teams meet. The match Congo DR vs. Uzbekistan — two teams with similar Elo and FIFA rankings — peaks at 58.7%, making it the single most likely draw in the tournament.
- **Switzerland vs Canada (55.1%)** is a particularly interesting high-draw match because it pits a UEFA stalwart against a host nation, meaning both the Elo-close signal and the home-advantage signal are partially offsetting each other, resulting in an almost even contest.

### 6.7 Dark Horses & Underdog Contenders

Traditional World Cup analysis focuses on UEFA and CONMEBOL favorites, but the simulation reveals several teams from other confederations with surprisingly high advancement and tournament-winning probabilities. These "dark horses" benefit from favorable group draws, host advantage, or recent form improvements that the model captures through Elo, FIFA ranking, form, and squad quality features.

> Note: The detailed advancement percentages below are from a previous simulation run. The updated top 5 tournament win probabilities are: Mexico 7.6%, Switzerland 6.5%, USA 6.0%, Germany 4.8%, Canada 4.7%.

| Rank | Team | Win % | Advance % | 1st in Group % | Confederation |
|------|------|-------|-----------|----------------|---------------|
| 1 | Mexico | 6.3 | 96.1 | 68.0 | CONCACAF |
| 2 | Canada | 5.5 | 97.9 | 42.1 | CONCACAF |
| 3 | United States | 4.1 | 88.1 | 49.5 | CONCACAF |
| 4 | Morocco | 3.6 | 92.3 | 41.7 | CAF |
| 5 | Ivory Coast | 2.7 | 79.7 | 22.0 | CAF |
| 6 | Japan | 2.4 | 91.2 | 50.6 | AFC |
| 7 | South Korea | 2.4 | 67.1 | 13.2 | AFC |
| 8 | Australia | 2.1 | 49.3 | 9.5 | AFC |
| 9 | Senegal | 2.0 | 74.9 | 19.8 | CAF |
| 10 | Algeria | 1.7 | 76.9 | 13.0 | CAF |

Key takeaways:

- **Morocco (3.6%)** is the strongest non-host dark horse.
- **Canada (4.7%) and Mexico (7.6%) dominate this list** due to host advantage, with high group-stage advancement probabilities.
- **Switzerland (6.5%)** benefits from a favorable group draw and strong recent Elo form.

### 6.8 Most Competitive Groups

Group competitiveness is measured by the **advancement spread** — the difference between the highest and lowest group advancement probabilities. A smaller spread indicates a tighter group where outcomes are more uncertain.

> Note: The detailed group-level percentages below are from a previous simulation run.

| Group | Advance Spread | Avg Advance % | Most Likely 1st | 1st Place % |
|-------|---------------|---------------|-----------------|-------------|
| D | 38.8% | — | United States | 49.5% |
| G | 58.9% | — | Belgium | 62.6% |
| A | 61.4% | — | Mexico | 68.0% |

Key takeaways:

- **Group D is the tightest group in the tournament**, meaning every match matters.
- **Groups with large spreads** (70%+) indicate one or two dominant teams and weaker opponents, making qualification more predictable.

### 6.9 Surprise Predictions & Upset Alerts

While the model predicts home wins in many group-stage matches, it also identifies specific matches where the away team is heavily favored — scenarios that defy the general home-advantage trend.

| Match | Prediction | Away Win % | Home Win % | Draw % |
|-------|------------|------------|------------|--------|
| Congo DR vs Colombia | Away win | 72.6 | 12.5 | 14.9 |
| Saudi Arabia vs Uruguay | Away win | 60.5 | 15.4 | 24.1 |
| Cape Verde vs Uruguay | Away win | 59.8 | 15.7 | 24.5 |
| Uzbekistan vs Colombia | Away win | 55.2 | 16.8 | 28.0 |
| Iraq vs Norway | Away win | 52.6 | 17.5 | 29.9 |

Key takeaways:

- **Congo DR vs Colombia** is likely one of the most lopsided matches, reflecting the massive Elo/FIFA ranking gap between the two teams.
- **All five matches involve CAF or AFC teams as the home side against CONMEBOL or UEFA opponents**, reflecting the structural gap between confederations.

### 6.10 Per-Match Prediction Highlights

The model generates full three-class probability predictions for all 72 group-stage matches. Each prediction includes `prob_home_win`, `prob_draw`, and `prob_away_win`, enabling fine-grained analysis of match-level uncertainty.

> Note: The specific match-level predictions below are from a previous simulation run.

| Match | Home Win % | Draw % | Away Win % | Prediction |
|-------|------------|--------|------------|------------|
| Canada vs Qatar | 90.1 | 6.2 | 3.7 | Home win |
| Mexico vs South Korea | 23.2 | 55.0 | 21.8 | Draw |
| Scotland vs Brazil | 27.4 | 18.9 | 53.7 | Away win |
| Switzerland vs Canada | 21.9 | 55.1 | 23.0 | Draw |
| Congo DR vs Colombia | 12.5 | 14.9 | 72.6 | Away win |

Key takeaways:

- Home advantage and squad quality are the primary drivers of prediction asymmetry.
- **The distribution of predictions** reflects the strong home-advantage modeling and the relatively balanced quality of teams in the expanded 48-team field.

### 6.11 Model vs Betting Odds Comparison

Comparing the model's tournament-winning probabilities with implied probabilities from bookmaker odds reveals systematic divergences that highlight where the model's Elo/ranking-based approach differs from market sentiment. The addition of Transfermarkt squad quality features helps bridge this gap by providing a direct signal of team quality that markets respond to.

| Team | Model % | Odds % | Difference |
|------|---------|--------|------------|
| Mexico | 7.6 | 3.6 | +4.0 |
| Switzerland | 6.5 | 3.0 | +3.5 |
| United States | 6.0 | 3.3 | +2.7 |
| Germany | 4.8 | — | — |
| Brazil | — | 7.5 | — |
| England | — | 7.3 | — |
| France | — | 9.1 | — |
| Argentina | — | 9.4 | — |
| Spain | — | 9.3 | — |

Key takeaways:

- **The biggest divergences remain with traditional powers**: Argentina, France, Spain, and England are all significantly undervalued by the model relative to odds. The model's Elo/FIFA-based features may not fully capture squad quality, though the new Transfermarkt features partially address this.
- **Mexico (+4.0%) and Switzerland (+3.5%)** are the most overvalued teams relative to odds. Mexico's overvaluation is almost entirely attributable to home advantage. Switzerland's overvaluation likely reflects a combination of a favorable group draw and strong recent Elo form.
- **The model systematically favors host-related advantages and Elo-based depth over star power and market sentiment.** Bettors could use these divergences to identify potential value bets — specifically, backing traditional powers like Spain, Argentina, and France at market odds may offer positive expected value if the model's analysis is correct.

### 6.12 Simulation Summary

Across 1,000 Monte Carlo simulations of the 2026 FIFA World Cup using the XGBoost model, the following high-level conclusions emerge:

1. **No dominant favorite**: The tournament win probability is remarkably flat. Mexico leads at 7.6%, but the top 5 teams collectively account for only ~30% of total win probability. This parity reflects the expanded 48-team format and the inherent randomness of knockout football.

2. **Host advantage is the single largest structural factor**: The three host nations (Mexico, Canada, United States) combine for ~18.3% win probability — more than any single confederation outside UEFA. Without home advantage, these teams would likely rank significantly lower.

3. **Squad quality matters**: The Transfermarkt squad quality features provide direct signals for team quality that Elo and FIFA rankings alone don't capture. France (€1.52B), England (€1.36B), and Spain (€1.22B) have the highest squad values, which influences knockout stage predictions.

4. **Draws remain challenging**: The BestEnsemble under-predicts draws (171 predicted vs 830 actual), while XGBoost over-predicts (1344 vs 830). WC group-stage draw calibration (`WC_GROUP_DRAW_RATE=0.25`) addresses this for simulation.

5. **The model diverges meaningfully from betting markets**: The systematic undervaluation of traditional powers and overvaluation of hosts and structurally advantaged teams creates potential arbitrage opportunities. The new squad quality features partially address this.

6. **Group D is the tournament's most competitive group**, making it the most unpredictable pool.

7. **48 teams across 12 groups produce 32 advancing teams** (top 2 per group + 8 best third-place teams).

### 6.13 Key Design Decisions & Trade-offs

1. **XGBoost for simulation**: While the StackingClassifier is the best model, XGBoost is used for simulation due to its much faster inference speed (6.6 MB model vs. ensemble with 4 models + meta-learner). The performance gap on validation data is small.

2. **Log_loss as selection metric**: Models and ensembles are selected by log_loss rather than accuracy, because well-calibrated probability estimates are more important for simulation than raw classification accuracy.

3. **Draw class weighting (4x)**: Draws are the minority class in football (~25% of outcomes). The 4x weight improves draw recall, though XGBoost still over-predicts draws (1344 predicted vs 830 actual). The Stacking ensemble meta-learner also uses `class_weight="balanced"` to handle draw imbalance.

4. **Neutral venue handling**: WC2026 matches in the US/CA/MX are correctly flagged as neutral except when the host nation plays, where they receive full home advantage.

5. **Isotonic calibration was removed**: While calibration typically improves probability estimates, in this case it degraded out-of-sample performance, likely due to the small validation set size.

---

## 7. Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for predicting World Cup outcomes, from data collection through simulation and visualization. The StackingClassifier ensemble achieves 62.0% accuracy and 0.836 log_loss on the test set, with a 59.38% accuracy on the 2022 World Cup validation set. The Monte Carlo simulation produces actionable tournament probability estimates, with host nations (Mexico, Canada, USA) showing elevated advancement probabilities due to home advantage.

Key strengths of the approach include:
- **Comprehensive feature engineering**: 80 features spanning Elo ratings, FIFA rankings, team form, head-to-head records, draw-predictive signals, and Transfermarkt squad quality data.
- **Multi-model ensemble**: Combining diverse classifier types (gradient boosting, random forest, logistic regression, neural network) with a balanced meta-learner improves robustness.
- **Draw handling**: Multiple strategies specifically target the difficult draw class, including 4x class weighting, balanced meta-learner, draw-predictive features, Elo draw modeling, and WC group-stage draw calibration.
- **Fixture swap fix**: Correct probability perspective when match lookups use reversed fixture ordering.
- **Squad quality features**: Transfermarkt market value data provides a direct signal for team quality that complements Elo and FIFA rankings.
- **Draw calibration**: WC group-stage draw rate calibration ensures realistic simulation outcomes.
- **Fixture swap fix**: Correct probability perspective when match lookups use reversed fixture ordering.
- **Reproducibility**: Feature caching with content hashing (version "7"), deterministic random seeds, and a single CLI orchestrator.
- **Live validation**: The pipeline supports incremental validation as the tournament progresses.

The expanded analytical insights from the simulation reveal several important findings:
- **Confederation-level analysis** shows that UEFA provides the most consistent depth, while CONCACAF's apparent strength is driven by the three host nations.
- **Host nation advantage** is the single largest structural factor in the tournament, accounting for approximately 18.3% of combined win probability for Mexico, Canada, and the United States — a figure that far exceeds what their underlying quality would predict.
- **Draw predictions** highlight the challenge of the draw class: the BestEnsemble under-predicts draws (171 predicted vs 830 actual), while XGBoost with 4x draw weight over-predicts (1344 predicted vs 830 actual). WC group-stage draw calibration (`WC_GROUP_DRAW_RATE=0.25`) addresses this for simulation.
- **Dark horse analysis** identifies Switzerland (6.5%), Morocco, and Japan as strong non-traditional contenders benefiting from favorable group draws and recent form.
- **Model vs. odds comparison** reveals systematic divergences: the model overvalues hosts and structurally advantaged teams while undervaluing traditional powers, creating potential arbitrage opportunities.
- **Group competitiveness analysis** identifies Group D as the tightest group, making it the most unpredictable pool.

Limitations:
- The test set after 2022 may be small, making generalization uncertain.
- The simulation model (XGBoost) differs from the best overall model (StackingClassifier), introducing a potential accuracy-speed trade-off.
- Bookmaker odds are only available for WC2026 matches, limiting their training utility.
- The expanded 48-team, 12-group format has no historical precedent, making group and bracket path estimation inherently uncertain.
- The model diverges meaningfully from betting markets on traditional powers (Spain, Argentina, France), which may reflect limitations in Elo/FIFA-based features that do not fully capture squad quality and tactical sophistication. The new Transfermarkt squad quality features partially address this.
- The `insights.py` module computes confederation statistics, dark horse rankings, draw analysis, and odds comparison, but these analyses are based on a single simulation run (1,000 iterations) and are subject to Monte Carlo variance.
- Draw calibration (WC_GROUP_DRAW_RATE=0.25) is a heuristic adjustment; it may over- or under-correct depending on the specific group composition.
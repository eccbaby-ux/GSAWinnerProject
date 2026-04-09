# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bat
Run_GSA.bat          # Generate today's predictions (fetch odds → predict → risk filter)
Train_GSA.bat        # Train/learn from completed matches (results → ELO → weights → Shishka → Ticha)
Run_Dashboard.bat    # Launch Streamlit dashboard (streamlit run "deshbord giboi.py")
```

Run individual steps manually:
```bat
python -Xutf8 winner_auto_fetcher.py   # Scrape Winner odds → winner_odds_cache.json
python -Xutf8 v76_Master_Nachshon.py   # Run prediction engine → analysis_results_v76.json
python -Xutf8 dadima_correction.py     # Apply risk filters to analysis_results_v76.json
python -Xutf8 result_updater.py        # Fetch final scores from API-Sports
python -Xutf8 elo_updater.py           # Recompute ELO ratings from all closed matches
python -Xutf8 v79_Auto_Learner.py      # Optimize model/market weights by ROI
python -Xutf8 shishka_run_and_save.py  # Run data-quality validation gate
python -Xutf8 ticha_system.py          # Backward-learning weight refinement
```

## Architecture

### Two Pipelines

**Run_GSA.bat** (daily, before matches):
```
winner_auto_fetcher.py  →  v76_Master_Nachshon.py  →  dadima_correction.py
(scrape odds)              (predict + classify)         (risk filter)
→ winner_odds_cache.json   → analysis_results_v76.json  → DB updated
```

**Train_GSA.bat** (after match results are in, weekly):
```
result_updater → elo_updater → v79_Auto_Learner → shishka_run_and_save → ticha_system
(fetch results)  (ELO ratings) (ROI-optimize weights) (validate quality)  (refine weights)
```

### Core Prediction: v76_Master_Nachshon.py

For each match:
1. **Team resolution** via `mapper.py` (Hebrew↔English fuzzy matching → API team ID)
2. **Lambda estimation**: xG from API-Sports last 10 games → weighted average → ELO fallback if API has no data
3. **Dixon-Coles Poisson** (`rho=0.12`, `HOME_ADVANTAGE_FACTOR=1.12`) → model_prob_1/x/2
4. **Market odds** from `winner_odds_cache.json` → Shin's vig removal → market_prob_1/x/2
5. **Blend**: `w_model * model_probs + w_market * market_probs` (weights from DB `weights` table)
6. **EV classification** → tier (gold/value/sting) → save to `matches` table + `analysis_results_v76.json`

### Learning Layer

Three systems update the blend weights in `weights` table. **Ticha runs last and overwrites all others:**

| System | File | Optimizes | Runs |
|--------|------|-----------|------|
| Auto-Learner | `v79_Auto_Learner.py` | ROI (walk-forward) | Step 3 |
| Ticha | `ticha_system.py` | ROI (EV-filtered, per-bet) | Step 5 — **final** |

**EV filter** (applied in both systems): only count a bet when `combined_prob × odds > 1`. ROI is computed per bet placed, not per game. Minimum 5 bets required for meaningful ROI.

**Shishka** (`shishka_check.py`) is a validation gate — it does not modify weights; it logs pass/fail + Brier score + drift detection to `learning_log`.

### Database: gsa_history.db

Key tables:
- `matches` — all predictions + outcomes. `actual_result` is NULL until `result_updater` fills it.
- `weights` (id=1) — active `w_model`, `w_market`, `w_ticha` used by v76
- `elo_ratings` — per-team ELO (team_name TEXT PRIMARY KEY, elo REAL)
- `learning_log` — timestamped entries for each train run (event_type: auto_learner / ticha / shishka)
- `bankroll_state` (id=1) — current balance
- `bet_slips` — placed bets with stake, outcome, ROI

### Risk Management: dadima_correction.py + Dashboard

Applied in two layers:
1. **dadima_correction.py**: Bayesian shrinkage (60% system / 40% market), EV ≥ 2%, ¼ Kelly sizing, 2% bankroll cap
2. **Dashboard** (`deshbord giboi.py`): `MIN_EV_THRESHOLD=0.04` (4%), `MIN_ODDS=1.50`, max 4 bets/day, 10% daily stop-loss

### Key Config (v76_Master_Nachshon.py)

```python
FOOTBALL_API_KEY = "e49cdc2ba079c654d1dbc88fb16bfa75"  # API-Sports key
HOME_ADVANTAGE_FACTOR = 1.12
AWAY_DISADVANTAGE_FACTOR = 0.88
DIXON_COLES_RHO = 0.12
```

### Files & Their Roles

| File | Role |
|------|------|
| `matches.txt` | Upcoming match list consumed by v76 |
| `winner_odds_cache.json` | Current Winner betting odds (scraped) |
| `analysis_results_v76.json` | Today's predictions output (JSON array) |
| `hebrew_to_id_mapping_new.csv` | Team name → API-Sports ID mapping |
| `translation_cache.json` | Cached Hebrew→English translations |
| `ticha_params.json` | Ticha neural net weights (W, b) backup |
| `calibration_params.json` | Platt Scaling calibration params backup |

### Known Limitations

- **ELO spread is small** (~64 pts currently, ~1.1 games/team). Impact grows as data accumulates. The `Global_Avg` fallback (flat `league_anchor=1.35` for teams with no API history) affects ~44% of stored matches.
- **w_model is 0%** in current weights — the Poisson model's probabilities are near-uniform (Brier ≈ 0.663) because many matches used the Global_Avg fallback. ELO will improve this over time.
- Ticha and Auto-Learner both write to `weights` table. Ticha runs last → its values are authoritative.

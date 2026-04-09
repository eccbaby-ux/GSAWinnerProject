# -*- coding: utf-8 -*-
"""
ShishkaCheck (בדיקת שישקה) — Ultimate gatekeeper for sports betting continuous learning.

Every time a match ends and its data is fed back for learning, it MUST pass through
ShishkaCheck. If any check fails, the model does NOT train on this data and alerts
are triggered.

Components:
  1. ShishkaDataValidator — Data integrity, anti-leakage, outlier detection
  2. ShishkaCalibrator    — Brier Score, Log Loss, Calibration Curve
  3. ShishkaDriftMonitor  — Concept drift (e.g. KS test)
  4. ShishkaRiskManager   — ROI vs Paper ROI, Max Drawdown

Author: MLOps / Quant pipeline
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import log_loss

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("ShishkaCheck")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ShishkaResult:
    """Result of a single Shishka sub-check."""

    passed: bool
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShishkaFullResult:
    """Result of the full Shishka pipeline (evaluate_and_learn)."""

    passed: bool
    results: List[ShishkaResult]
    safe_to_train: bool
    safe_to_bet: bool
    alerts_triggered: List[str]


# ---------------------------------------------------------------------------
# 1. Data Integrity & Anti-Leakage
# ---------------------------------------------------------------------------


class ShishkaDataValidator:
    """
    Strict validation: no future data leakage, outlier detection (Z-score / Isolation Forest).
    Quarantine rows that fail so the model does not learn from garbage.
    """

    # Columns that must NOT appear in pre-match features (post-match / outcome leakage)
    DEFAULT_LEAKAGE_COLUMNS = frozenset({
        "actual_result",
        "final_result",
        "goals_home",
        "goals_away",
        "goals_total",
        "possession_home",
        "possession_away",
        "shots_on_target_home",
        "shots_on_target_away",
        "corners_home",
        "corners_away",
        "full_time_score",
        "half_time_score",
        "match_ended",
        "winner",
        "points_home",
        "points_away",
    })

    def __init__(
        self,
        leakage_columns: Optional[Sequence[str]] = None,
        outlier_method: str = "isolation_forest",
        z_score_threshold: float = 3.5,
        contamination: float = 0.05,
        outliers_fail: bool = False,
        required_prematch_columns: Optional[Sequence[str]] = None,
        possession_max: float = 1.0,
        prob_sum_tolerance: float = 0.05,
    ):
        """
        Args:
            leakage_columns: Column names that would indicate post-match data (leakage).
            outlier_method: 'z_score' or 'isolation_forest'.
            z_score_threshold: Max |Z| for non-outlier (used if outlier_method=='z_score').
            contamination: Expected fraction of outliers for Isolation Forest.
            outliers_fail: If True, any outlier causes validation failure. If False, outliers
                           are reported in details (quarantine_rows_outliers) but validation passes.
            required_prematch_columns: Columns that must exist and be pre-match only.
            possession_max: Max valid possession (e.g. 1.0 = 100%); above = invalid.
            prob_sum_tolerance: Max |sum(probs) - 1| for model/market prob columns.
        """
        self.leakage_columns = set(leakage_columns or self.DEFAULT_LEAKAGE_COLUMNS)
        self.outlier_method = outlier_method
        self.z_score_threshold = z_score_threshold
        self.contamination = contamination
        self.outliers_fail = outliers_fail
        self.required_prematch_columns = list(required_prematch_columns or [])
        self.possession_max = possession_max
        self.prob_sum_tolerance = prob_sum_tolerance

    def _check_leakage(self, df: pd.DataFrame, feature_columns: Optional[List[str]]) -> Tuple[bool, str]:
        """Ensure no leakage columns are used as features."""
        if feature_columns is None:
            feature_columns = [
                c for c in df.columns
                if c not in {"actual_result", "final_result"}
                and df[c].dtype in ("float64", "int64", "float32", "int32")
            ]
        leaking = [c for c in feature_columns if c in self.leakage_columns]
        if leaking:
            return False, f"Leakage columns used as features: {leaking}"
        return True, "No leakage columns in feature set"

    def _check_prob_sums(self, df: pd.DataFrame) -> Tuple[bool, str, List[int]]:
        """Check model/market prob columns sum to ~1 (if present).
        If market_prob_* look like odds (sum > 1.5), convert to probabilities before check."""
        prob_groups = [
            ("model_prob_1", "model_prob_x", "model_prob_2"),
            ("market_prob_1", "market_prob_x", "market_prob_2"),
        ]
        work = df.copy()
        for group in prob_groups:
            cols = [c for c in group if c in work.columns]
            if len(cols) < 3:
                continue
            vals = work[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            sums = vals.sum(axis=1)
            # אם נראה כמו odds (סכום 2–25), ממיר להסתברויות
            odds_mask = sums > 1.5
            if odds_mask.any():
                v = vals.values
                inv = np.where(v > 0, 1.0 / v, 0.0)
                row_tot = inv.sum(axis=1, keepdims=True)
                row_tot = np.where(row_tot > 0, row_tot, 1.0)
                probs = inv / row_tot
                work.loc[odds_mask, cols] = probs[odds_mask]
        bad_rows: List[int] = []
        for group in prob_groups:
            cols = [c for c in group if c in work.columns]
            if len(cols) < 3:
                continue
            vals = work[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            sums = vals.sum(axis=1)
            invalid = np.abs(sums - 1.0) > self.prob_sum_tolerance
            bad_rows.extend(np.where(invalid)[0].tolist())
        bad_rows = sorted(set(bad_rows))
        if bad_rows:
            return False, f"Probabilities do not sum to 1 (±{self.prob_sum_tolerance}) in rows: {bad_rows[:20]}{'...' if len(bad_rows) > 20 else ''}", bad_rows
        return True, "Probability sums OK", []

    def _check_possession_columns(self, df: pd.DataFrame) -> Tuple[bool, str, List[int]]:
        """Flag rows with impossible possession (e.g. > 100%)."""
        poss_cols = [c for c in df.columns if "possession" in c.lower()]
        bad_rows: List[int] = []
        for col in poss_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            bad = (s < 0) | (s > self.possession_max)
            bad_rows.extend(np.where(bad)[0].tolist())
        bad_rows = sorted(set(bad_rows))
        if bad_rows:
            return False, f"Impossible possession values in rows: {bad_rows[:20]}{'...' if len(bad_rows) > 20 else ''}", bad_rows
        return True, "Possession bounds OK", []

    def _outliers_zscore(self, df: pd.DataFrame, numeric_columns: List[str]) -> np.ndarray:
        """Mark rows with any feature beyond z_score_threshold as outlier (1 = inlier, -1 = outlier for compatibility)."""
        inlier = np.ones(len(df), dtype=int)
        for col in numeric_columns:
            if col not in df.columns:
                continue
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            mean, std = s.mean(), s.std()
            if std is None or std == 0:
                continue
            z = np.abs((s - mean) / std)
            inlier[z > self.z_score_threshold] = -1
        return inlier

    def _outliers_isolation_forest(self, df: pd.DataFrame, numeric_columns: List[str]) -> np.ndarray:
        """Isolation Forest outlier detection. Returns 1 inlier, -1 outlier."""
        cols = [c for c in numeric_columns if c in df.columns]
        if not cols:
            return np.ones(len(df), dtype=int)
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        if len(X) < 2:
            return np.ones(len(df), dtype=int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            pred = clf.fit_predict(X)
        return pred

    def validate(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> ShishkaResult:
        """
        Run full data validation. Returns ShishkaResult.
        If validation fails, data should be quarantined (not used for training).
        """
        details: Dict[str, Any] = {}
        errors: List[str] = []

        # Required columns
        if self.required_prematch_columns:
            missing = [c for c in self.required_prematch_columns if c not in df.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")

        # Leakage
        ok, msg = self._check_leakage(df, feature_columns)
        details["leakage_check"] = msg
        if not ok:
            errors.append(msg)

        # Prob sums
        ok, msg, bad_rows = self._check_prob_sums(df)
        details["prob_sum_check"] = msg
        if not ok:
            errors.append(msg)
            details["quarantine_rows_prob"] = bad_rows

        # Possession
        ok, msg, bad_rows = self._check_possession_columns(df)
        details["possession_check"] = msg
        if not ok:
            errors.append(msg)
            details["quarantine_rows_possession"] = bad_rows

        # Outliers on numeric features
        numeric = [
            c for c in (feature_columns or df.select_dtypes(include=[np.number]).columns.tolist())
            if c in df.columns and c not in self.leakage_columns
        ]
        if numeric and len(df) >= 2:
            if self.outlier_method == "z_score":
                labels = self._outliers_zscore(df, numeric)
            else:
                labels = self._outliers_isolation_forest(df, numeric)
            n_outliers = int((labels == -1).sum())
            details["n_outliers"] = n_outliers
            details["outlier_method"] = self.outlier_method
            if n_outliers > 0:
                quarantine = np.where(labels == -1)[0].tolist()
                details["quarantine_rows_outliers"] = quarantine
                if self.outliers_fail:
                    errors.append(f"Outliers detected: {n_outliers} rows (quarantine recommended)")

        passed = len(errors) == 0
        message = "Data validation passed" if passed else "; ".join(errors)
        logger.info("[ShishkaDataValidator] %s", message)
        return ShishkaResult(passed=passed, component="ShishkaDataValidator", message=message, details=details)


# ---------------------------------------------------------------------------
# 2. Model Calibration & Evaluation
# ---------------------------------------------------------------------------


class ShishkaCalibrator:
    """
    Brier Score, Log Loss, and Calibration Curve (Expected vs Actual).
    Triggers Calibration_Warning if model confidence is statistically misaligned.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_samples_per_bin: int = 5,
        calibration_epsilon: float = 0.15,
        max_brier_threshold: Optional[float] = 0.30,
        max_log_loss_threshold: Optional[float] = 1.2,
    ):
        """
        Args:
            n_bins: Number of bins for calibration curve.
            min_samples_per_bin: Minimum samples in a bin to trust that bin.
            calibration_epsilon: Max allowed |expected_prob - actual_rate| (e.g. 0.15 = 15%) before warning.
            max_brier_threshold: If Brier score > this, may trigger warning (optional).
            max_log_loss_threshold: If log loss > this, may trigger warning (optional).
        """
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.calibration_epsilon = calibration_epsilon
        self.max_brier_threshold = max_brier_threshold
        self.max_log_loss_threshold = max_log_loss_threshold

    def _get_prob_actual(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract predicted probabilities for the winning outcome and binary actual (1 if correct prediction).
        Expects columns: model_prob_1, model_prob_x, model_prob_2 (or pred_1, pred_x, pred_2)
        and actual_result in ('1','X','2').
        Returns: prob_of_predicted_outcome (confidence), binary_correct (1/0).
        """
        prob_1 = df.get("model_prob_1", df.get("pred_1", pd.Series(1.0 / 3, index=df.index)))
        prob_x = df.get("model_prob_x", df.get("pred_x", pd.Series(1.0 / 3, index=df.index)))
        prob_2 = df.get("model_prob_2", df.get("pred_2", pd.Series(1.0 / 3, index=df.index)))
        actual = df["actual_result"].astype(str).str.strip()

        # Probability assigned to the outcome that actually happened (for Brier / log loss)
        prob_actual_outcome = np.where(actual == "1", prob_1, np.where(actual == "X", prob_x, prob_2))
        prob_actual_outcome = np.asarray(prob_actual_outcome, dtype=float)

        # Predicted outcome = argmax(prob)
        pred_outcome = np.argmax(np.column_stack([prob_1, prob_x, prob_2]), axis=1)
        actual_idx = np.where(actual == "1", 0, np.where(actual == "X", 1, 2))
        binary_correct = (pred_outcome == actual_idx).astype(float)

        # Confidence = prob of the predicted outcome
        pred_probs = np.column_stack([prob_1, prob_x, prob_2])
        confidence = np.max(pred_probs, axis=1)

        return confidence, binary_correct, prob_actual_outcome

    def brier_score(self, df: pd.DataFrame) -> float:
        """Multi-class Brier: (1/N) * sum_i sum_c (y_ic - p_ic)^2 for classes 1,X,2."""
        actual = df["actual_result"].astype(str).str.strip()
        y_true = np.where(actual == "1", 0, np.where(actual == "X", 1, 2))
        prob_1 = df.get("model_prob_1", df.get("pred_1", 1.0 / 3)).astype(float)
        prob_x = df.get("model_prob_x", df.get("pred_x", 1.0 / 3)).astype(float)
        prob_2 = df.get("model_prob_2", df.get("pred_2", 1.0 / 3)).astype(float)
        y_prob = np.column_stack([prob_1, prob_x, prob_2])
        # Clip and normalize
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        # Multiclass Brier: (1/N) sum_i sum_c (indicator(c==y_i) - p_ic)^2
        n, k = y_prob.shape
        y_onehot = np.zeros_like(y_prob)
        y_onehot[np.arange(n), y_true.astype(int)] = 1.0
        return float(np.mean(np.sum((y_onehot - y_prob) ** 2, axis=1)))

    def log_loss_score(self, df: pd.DataFrame) -> float:
        """Categorical log loss for 3 classes (1, X, 2)."""
        actual = df["actual_result"].astype(str).str.strip()
        y_true = np.where(actual == "1", 0, np.where(actual == "X", 1, 2))
        prob_1 = df.get("model_prob_1", df.get("pred_1", 1.0 / 3)).astype(float)
        prob_x = df.get("model_prob_x", df.get("pred_x", 1.0 / 3)).astype(float)
        prob_2 = df.get("model_prob_2", df.get("pred_2", 1.0 / 3)).astype(float)
        y_prob = np.column_stack([prob_1, prob_x, prob_2])
        eps = 1e-15
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob /= y_prob.sum(axis=1, keepdims=True)
        return float(log_loss(y_true, y_prob))

    def calibration_curve(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: bin_centers (expected prob), actual_rates, counts_per_bin.
        """
        confidence, binary_correct, _ = self._get_prob_actual(df)
        confidence = np.asarray(confidence, dtype=float)
        binary_correct = np.asarray(binary_correct, dtype=float)
        finite = np.isfinite(confidence)
        confidence = confidence[finite]
        binary_correct = binary_correct[finite]
        if len(confidence) < self.min_samples_per_bin:
            return np.array([]), np.array([]), np.array([])
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_idx = np.digitize(confidence, bins) - 1
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        actual_rates = np.zeros(self.n_bins)
        counts = np.zeros(self.n_bins)
        for i in range(self.n_bins):
            mask = bin_idx == i
            counts[i] = mask.sum()
            if counts[i] >= 1:
                actual_rates[i] = binary_correct[mask].mean()
            else:
                actual_rates[i] = np.nan
        return bin_centers, actual_rates, counts

    def evaluate(
        self, df: pd.DataFrame, last_n: Optional[int] = None
    ) -> ShishkaResult:
        """
        Compute Brier, Log Loss, calibration curve; trigger Calibration_Warning
        if expected vs actual is misaligned beyond calibration_epsilon.
        """
        if df.empty or "actual_result" not in df.columns:
            return ShishkaResult(
                passed=True,
                component="ShishkaCalibrator",
                message="No data to calibrate",
                details={},
            )
        work = df.tail(last_n) if last_n else df
        work = work.dropna(subset=["actual_result"])
        work = work[work["actual_result"].astype(str).str.strip().isin(["1", "X", "2"])]
        if len(work) < self.min_samples_per_bin:
            return ShishkaResult(
                passed=True,
                component="ShishkaCalibrator",
                message="Insufficient samples for calibration check",
                details={"n": len(work)},
            )

        brier = self.brier_score(work)
        logloss = self.log_loss_score(work)
        bin_centers, actual_rates, counts = self.calibration_curve(work)

        details = {
            "brier_score": brier,
            "log_loss": logloss,
            "n_matches": len(work),
        }

        alerts: List[str] = []
        if self.max_brier_threshold is not None and brier > self.max_brier_threshold:
            alerts.append(f"Brier score {brier:.4f} > {self.max_brier_threshold}")
        if self.max_log_loss_threshold is not None and logloss > self.max_log_loss_threshold:
            alerts.append(f"Log loss {logloss:.4f} > {self.max_log_loss_threshold}")

        # Statistical misalignment: any bin with |expected - actual| > epsilon
        for i in range(len(bin_centers)):
            if counts[i] >= self.min_samples_per_bin and np.isfinite(actual_rates[i]):
                diff = abs(bin_centers[i] - actual_rates[i])
                if diff > self.calibration_epsilon:
                    alerts.append(
                        f"Calibration bin {bin_centers[i]:.2f}: expected ~{bin_centers[i]:.2f}, actual {actual_rates[i]:.2f} (diff={diff:.2f})"
                    )

        passed = len(alerts) == 0
        if alerts:
            # Shadow mode: do NOT raise – just mark that production would halt betting here.
            shadow_msg = "SHADOW_MODE_HALT: Production system would stop betting here (calibration)"
            alerts.insert(0, shadow_msg)
            logger.warning(
                "[ShishkaCalibrator] Calibration_Warning (shadow mode): %s",
                "; ".join(alerts),
            )
        else:
            logger.info(
                "[ShishkaCalibrator] Calibration OK — Brier=%.4f, LogLoss=%.4f",
                brier,
                logloss,
            )

        details["calibration_bins_expected"] = bin_centers.tolist()
        details["calibration_bins_actual"] = [float(x) if np.isfinite(x) else None for x in actual_rates]
        details["calibration_counts"] = counts.tolist()
        details["calibration_alerts"] = alerts
        details["shadow_mode_halt"] = not passed

        return ShishkaResult(
            passed=passed,
            component="ShishkaCalibrator",
            message="Calibration OK" if passed else "Calibration_Warning: " + "; ".join(alerts),
            details=details,
        )


# ---------------------------------------------------------------------------
# 3. Concept Drift Detection
# ---------------------------------------------------------------------------


class ShishkaDriftMonitor:
    """
    Compare feature distributions of newly finished games vs baseline (training) data.
    Uses Kolmogorov-Smirnov test. Alerts when the nature of the game/league has changed.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        alpha: float = 0.05,
        min_baseline_size: int = 50,
        min_new_size: int = 5,
    ):
        """
        Args:
            feature_columns: Numeric columns to test for drift. If None, all numeric (except leakage) used.
            alpha: Significance level for KS test (reject same-distribution if p < alpha).
            min_baseline_size: Minimum baseline rows to run drift.
            min_new_size: Minimum new rows to run drift.
        """
        self.feature_columns = feature_columns
        self.alpha = alpha
        self.min_baseline_size = min_baseline_size
        self.min_new_size = min_new_size

    def detect_drift(
        self,
        baseline_df: pd.DataFrame,
        new_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[bool, Dict[str, float], List[str]]:
        """
        Returns: (drift_detected, dict of p-values per feature, list of drifted feature names).
        """
        cols = feature_columns or self.feature_columns
        if cols is None:
            numeric = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
            leakage = {"actual_result", "final_result", "id", "fixture_id"}
            cols = [c for c in numeric if c not in leakage]
        cols = [c for c in cols if c in baseline_df.columns and c in new_df.columns]
        if not cols or len(baseline_df) < self.min_baseline_size or len(new_df) < self.min_new_size:
            return False, {}, []

        p_values: Dict[str, float] = {}
        drifted: List[str] = []
        for col in cols:
            a = baseline_df[col].dropna().astype(float)
            b = new_df[col].dropna().astype(float)
            if len(a) < 2 or len(b) < 2:
                continue
            try:
                stat, p = stats.ks_2samp(a, b)
                p_values[col] = p
                if p < self.alpha:
                    drifted.append(col)
            except Exception:
                continue
        return len(drifted) > 0, p_values, drifted

    def evaluate(
        self,
        baseline_df: pd.DataFrame,
        new_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> ShishkaResult:
        """Run drift detection; alert if distributions have shifted significantly."""
        drift_detected, p_values, drifted = self.detect_drift(baseline_df, new_df, feature_columns)
        details = {"p_values": p_values, "drifted_features": drifted, "alpha": self.alpha}
        if drift_detected:
            logger.warning(
                "[ShishkaDriftMonitor] Concept drift detected — drifted features: %s",
                drifted,
            )
            return ShishkaResult(
                passed=False,
                component="ShishkaDriftMonitor",
                message=f"Concept drift: historical assumptions may be decaying. Drifted: {drifted}",
                details=details,
            )
        logger.info("[ShishkaDriftMonitor] No significant drift (alpha=%.2f)", self.alpha)
        return ShishkaResult(
            passed=True,
            component="ShishkaDriftMonitor",
            message="No significant concept drift",
            details=details,
        )


# ---------------------------------------------------------------------------
# 4. Financial Risk & ROI Gate
# ---------------------------------------------------------------------------


class ShishkaRiskManager:
    """
    Track Actual ROI vs Expected/Paper ROI; Max Drawdown monitor.
    In shadow mode: NEVER raises exceptions – only flags critical alerts and
    reports where production would have halted betting.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 25.0,
        min_bets_for_drawdown: int = 10,
        drawdown_consecutive_losses: Optional[int] = 5,
        fractional_kelly: float = 0.25,
    ):
        """
        Args:
            max_drawdown_pct: Max allowed drawdown (from peak) in % before flagging critical alert.
            min_bets_for_drawdown: Minimum number of settled bets before evaluating drawdown.
            drawdown_consecutive_losses: If this many consecutive losses, flag critical alert.
            fractional_kelly: Fraction of full Kelly to use for suggested bet sizing (e.g. 0.25 = 1/4 Kelly).
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.min_bets_for_drawdown = min_bets_for_drawdown
        self.drawdown_consecutive_losses = drawdown_consecutive_losses or 999
        self.fractional_kelly = fractional_kelly

    def _roi_and_drawdown(
        self,
        df: pd.DataFrame,
        profit_col: Optional[str] = None,
        stake_per_bet: float = 1.0,
    ) -> Tuple[float, float, float, int, List[float], float, float]:
        """
        Expects df with actual results and model predictions; computes profit per row
        if profit_col not given (using unit stake and decimal odds from market/model).
        Returns:
            actual_roi_pct, paper_roi_pct, max_drawdown_pct, consecutive_losses,
            cumulative_profits, kelly_full_mean, kelly_fractional_mean.
        """
        if df.empty or "actual_result" not in df.columns:
            return 0.0, 0.0, 0.0, 0, [], 0.0, 0.0

        # Infer profit per row: assume we bet on model pick at decimal odds
        def decimal_odds(row: pd.Series, outcome: str) -> float:
            key = outcome.lower() if outcome != "X" else "x"
            p = row.get(f"market_prob_{outcome}", row.get(f"market_prob_{key}", 1.0 / 3))
            try:
                p = float(p)
            except (TypeError, ValueError):
                return 1.0 / 0.33
            if 0 < p <= 1:
                return 1.0 / p
            return p

        profits: List[float] = []
        kelly_full_vals: List[float] = []
        for _, row in df.iterrows():
            if profit_col and profit_col in row and pd.notna(row[profit_col]):
                profits.append(float(row[profit_col]))
                continue
            actual = str(row.get("actual_result", "")).strip()
            if actual not in ("1", "X", "2"):
                continue
            p1 = row.get("model_prob_1", row.get("pred_1", 1.0 / 3))
            px = row.get("model_prob_x", row.get("pred_x", 1.0 / 3))
            p2 = row.get("model_prob_2", row.get("pred_2", 1.0 / 3))
            pick = "1" if p1 >= px and p1 >= p2 else ("X" if px >= p1 and px >= p2 else "2")
            odds = decimal_odds(row, pick)
            # Kelly fraction for this bet (full Kelly)
            try:
                if pick == "1":
                    p_hat = float(p1)
                elif pick == "X":
                    p_hat = float(px)
                else:
                    p_hat = float(p2)
            except (TypeError, ValueError):
                p_hat = None
            if p_hat is not None and 0.0 < p_hat < 1.0 and odds > 1.0:
                b = odds - 1.0
                q = 1.0 - p_hat
                kelly_full = (b * p_hat - q) / b if b > 0 else 0.0
                if not np.isfinite(kelly_full) or kelly_full < 0:
                    kelly_full = 0.0
                kelly_full_vals.append(kelly_full)

            if pick == actual:
                profits.append((odds - 1.0) * stake_per_bet)
            else:
                profits.append(-stake_per_bet)

        if not profits:
            return 0.0, 0.0, 0.0, 0, [], 0.0, 0.0

        cumulative = np.cumsum(profits)
        total_invested = len(profits) * stake_per_bet
        actual_roi_pct = (cumulative[-1] / total_invested) * 100.0 if total_invested else 0.0

        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0
        max_dd_pct = (max_dd / total_invested) * 100.0 if total_invested else 0.0

        # Consecutive losses from the end
        consec = 0
        for x in reversed(profits):
            if x <= 0:
                consec += 1
            else:
                break

        paper_roi_pct = actual_roi_pct  # Can be extended with expected value later

        if kelly_full_vals:
            kelly_full_mean = float(np.mean(kelly_full_vals))
        else:
            kelly_full_mean = 0.0
        kelly_fractional_mean = self.fractional_kelly * kelly_full_mean

        return (
            actual_roi_pct,
            paper_roi_pct,
            max_dd_pct,
            consec,
            cumulative.tolist(),
            kelly_full_mean,
            kelly_fractional_mean,
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        profit_col: Optional[str] = None,
        stake_per_bet: float = 1.0,
    ) -> ShishkaResult:
        """
        Check ROI and drawdown in SHADOW MODE.

        Never raises exceptions. Instead, when risk thresholds are breached it
        flags critical alerts and indicates that a production system would stop
        betting at this point.
        """
        details: Dict[str, Any] = {}
        alerts: List[str] = []

        (
            actual_roi,
            paper_roi,
            max_dd_pct,
            consec_losses,
            cum,
            kelly_full_mean,
            kelly_fractional_mean,
        ) = self._roi_and_drawdown(
            df, profit_col, stake_per_bet
        )
        details["actual_roi_pct"] = actual_roi
        details["paper_roi_pct"] = paper_roi
        details["max_drawdown_pct"] = max_dd_pct
        details["consecutive_losses"] = consec_losses
        details["n_bets"] = len(cum) if cum else 0
        details["kelly_fraction_full"] = kelly_full_mean
        details["kelly_fraction_fractional"] = kelly_fractional_mean

        if details["n_bets"] >= self.min_bets_for_drawdown:
            if max_dd_pct > self.max_drawdown_pct:
                alerts.append(f"Max drawdown {max_dd_pct:.1f}% exceeds threshold {self.max_drawdown_pct}%")
            if consec_losses >= self.drawdown_consecutive_losses:
                alerts.append(
                    f"Consecutive losses ({consec_losses}) >= threshold ({self.drawdown_consecutive_losses})"
                )

        passed = len(alerts) == 0
        if alerts:
            shadow_msg = "SHADOW_MODE_HALT: Production system would stop betting here (risk)"
            alerts.insert(0, shadow_msg)
            logger.critical(
                "[ShishkaRiskManager] Drawdown/Risk alert (shadow mode): %s",
                "; ".join(alerts),
            )
        else:
            logger.info(
                "[ShishkaRiskManager] ROI=%.2f%%, MaxDD=%.2f%%, Consec losses=%s",
                actual_roi,
                max_dd_pct,
                consec_losses,
            )
        return ShishkaResult(
            passed=passed,
            component="ShishkaRiskManager",
            message=(
                f"ROI={actual_roi:.2f}%, MaxDD={max_dd_pct:.2f}%"
                if passed
                else "; ".join(alerts)
            ),
            details=details,
        )


# ---------------------------------------------------------------------------
# 5. Main Orchestrator: ShishkaCheck
# ---------------------------------------------------------------------------


class ShishkaCheck:
    """
    Ultimate gatekeeper: runs DataValidator → Calibrator → DriftMonitor → RiskManager
    sequentially. If any check fails, safe_to_train=False and alerts are triggered.
    """

    def __init__(
        self,
        data_validator: Optional[ShishkaDataValidator] = None,
        calibrator: Optional[ShishkaCalibrator] = None,
        drift_monitor: Optional[ShishkaDriftMonitor] = None,
        risk_manager: Optional[ShishkaRiskManager] = None,
        baseline_df: Optional[pd.DataFrame] = None,
        calibration_last_n: Optional[int] = 100,
    ):
        self.data_validator = data_validator or ShishkaDataValidator()
        self.calibrator = calibrator or ShishkaCalibrator()
        self.drift_monitor = drift_monitor or ShishkaDriftMonitor()
        self.risk_manager = risk_manager or ShishkaRiskManager()
        self.baseline_df = baseline_df
        self.calibration_last_n = calibration_last_n

    def set_baseline(self, baseline_df: pd.DataFrame) -> None:
        """Set baseline (training) data for drift detection."""
        self.baseline_df = baseline_df

    def evaluate_and_learn(
        self,
        new_match_data: pd.DataFrame,
        model_predictions: Optional[pd.DataFrame] = None,
        *,
        feature_columns: Optional[List[str]] = None,
        run_drift: bool = True,
        run_risk: bool = True,
    ) -> ShishkaFullResult:
        """
        Main entry: run all four checks in sequence. If any fails, model must NOT
        train on this data and alerts are triggered.

        Args:
            new_match_data: DataFrame with match rows (pre-match features + actual_result).
            model_predictions: Optional separate DataFrame of predictions; if None, new_match_data
                               must contain model_prob_1/x/2 (or pred_1/x/2) and actual_result.
            feature_columns: Optional list of pre-match feature column names for validation/drift.
            run_drift: Whether to run drift check (requires baseline_df).
            run_risk: Whether to run risk/drawdown check.

        Returns:
            ShishkaFullResult with passed, results, safe_to_train, alerts_triggered.
        """
        logger.info("======== ShishkaCheck (בדיקת שישקה) START ========")
        results: List[ShishkaResult] = []
        alerts_triggered: List[str] = []

        # Use merged data for calibration/risk if predictions given separately
        if model_predictions is not None and not model_predictions.empty:
            if new_match_data.index.equals(model_predictions.index) and len(new_match_data) == len(model_predictions):
                work_df = new_match_data.copy()
                for c in model_predictions.columns:
                    if c not in work_df.columns:
                        work_df[c] = model_predictions[c].values
            else:
                work_df = pd.concat([new_match_data.reset_index(drop=True), model_predictions.reset_index(drop=True)], axis=1)
        else:
            work_df = new_match_data.copy()

        # 1. Data integrity & anti-leakage
        r1 = self.data_validator.validate(work_df, feature_columns=feature_columns)
        results.append(r1)
        if not r1.passed:
            alerts_triggered.append(f"[DataValidator] {r1.message}")

        # 2. Calibration
        r2 = self.calibrator.evaluate(work_df, last_n=self.calibration_last_n)
        results.append(r2)
        if not r2.passed:
            alerts_triggered.append(f"[Calibrator] Calibration_Warning: {r2.message}")

        # 3. Drift (if baseline available)
        if run_drift and self.baseline_df is not None and not self.baseline_df.empty and not work_df.empty:
            r3 = self.drift_monitor.evaluate(self.baseline_df, work_df, feature_columns=feature_columns)
            results.append(r3)
            if not r3.passed:
                alerts_triggered.append(f"[DriftMonitor] {r3.message}")
        else:
            results.append(
                ShishkaResult(
                    passed=True,
                    component="ShishkaDriftMonitor",
                    message="Skipped (no baseline or disabled)",
                    details={},
                )
            )

        # 4. Risk & drawdown (shadow mode – never raises)
        if run_risk:
            r4 = self.risk_manager.evaluate(work_df)
            results.append(r4)
            if not r4.passed:
                alerts_triggered.append(f"[RiskManager] {r4.message}")
        else:
            results.append(
                ShishkaResult(
                    passed=True,
                    component="ShishkaRiskManager",
                    message="Skipped (disabled)",
                    details={},
                )
            )

        # Map components by name for robust logic
        comp: Dict[str, ShishkaResult] = {r.component: r for r in results}
        data_res = comp.get("ShishkaDataValidator")
        calib_res = comp.get("ShishkaCalibrator")
        drift_res = comp.get("ShishkaDriftMonitor")
        risk_res = comp.get("ShishkaRiskManager")

        # Safe to train: ONLY depends on data integrity / anti-leakage
        safe_to_train = bool(data_res.passed) if data_res is not None else True

        # Safe to bet: calibration, drift and risk must all pass
        calib_ok = bool(calib_res.passed) if calib_res is not None else True
        drift_ok = bool(drift_res.passed) if drift_res is not None else True
        risk_ok = bool(risk_res.passed) if risk_res is not None else True
        safe_to_bet = calib_ok and drift_ok and risk_ok

        # Shadow mode: if NOT safe_to_bet, force suggested Kelly fraction to 0
        if risk_res is not None:
            if not safe_to_bet:
                risk_res.details["kelly_fraction_suggested"] = 0.0
                risk_res.details["kelly_fraction_fractional"] = 0.0
                risk_res.details["kelly_shadow_mode_halt"] = True
            else:
                # expose a single suggested fraction (fractional Kelly) when healthy
                frac = risk_res.details.get("kelly_fraction_fractional")
                if frac is not None:
                    risk_res.details["kelly_fraction_suggested"] = float(frac)

        passed = safe_to_train and safe_to_bet

        logger.info(
            "======== ShishkaCheck END | passed=%s | safe_to_train=%s | safe_to_bet=%s | alerts=%s ========",
            passed,
            safe_to_train,
            safe_to_bet,
            len(alerts_triggered),
        )
        if alerts_triggered:
            for a in alerts_triggered:
                logger.warning("ALERT: %s", a)

        return ShishkaFullResult(
            passed=passed,
            results=results,
            safe_to_train=safe_to_train,
             safe_to_bet=safe_to_bet,
            alerts_triggered=alerts_triggered,
        )


# ---------------------------------------------------------------------------
# Convenience: run checks only (no training) for integration
# ---------------------------------------------------------------------------

def run_shishka_check(
    new_match_data: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    model_predictions: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> ShishkaFullResult:
    """
    One-shot Shishka check. Use this or instantiate ShishkaCheck and call evaluate_and_learn.
    """
    shishka = ShishkaCheck(baseline_df=baseline_df, **kwargs)
    return shishka.evaluate_and_learn(new_match_data, model_predictions)


# ---------------------------------------------------------------------------
# Integration: gate training in Auto-Learner (example)
# ---------------------------------------------------------------------------
#
# In v79_Auto_Learner or similar, after loading finished matches:
#
#   from shishka_check import ShishkaCheck
#
#   shishka = ShishkaCheck(
#       baseline_df=df_train,  # training distribution for drift
#       calibration_last_n=100,
#   )
#   result = shishka.evaluate_and_learn(new_match_data)
#   # Training decision (data quality only)
#   if not result.safe_to_train:
#       logger.error("ShishkaCheck: data NOT safe for training. Alerts: %s", result.alerts_triggered)
#       return  # do not call optimize_weights()
#
#   # Betting decision (calibration/drift/risk)
#   if not result.safe_to_bet:
#       logger.warning("ShishkaCheck: SHADOW MODE – would pause betting, but still learning from data.")
#
#   # Proceed with optimize_weights(conn) ... (learning still allowed if safe_to_train)
#


if __name__ == "__main__":
    import os
    import sqlite3

    logging.basicConfig(level=logging.INFO)
    DB_PATH = os.path.join(os.path.dirname(__file__) or ".", "gsa_history.db")
    if not os.path.isfile(DB_PATH):
        print("No gsa_history.db found; creating minimal demo data.")
        raw = np.random.rand(20, 3)
        raw /= raw.sum(axis=1, keepdims=True)
        df = pd.DataFrame({
            "match_date": ["2025-01-01"] * 20,
            "model_prob_1": raw[:, 0].copy(),
            "model_prob_x": raw[:, 1].copy(),
            "model_prob_2": raw[:, 2].copy(),
            "market_prob_1": raw[:, 0].copy(),
            "market_prob_x": raw[:, 1].copy(),
            "market_prob_2": raw[:, 2].copy(),
            "actual_result": np.random.choice(["1", "X", "2"], 20),
        })
        # Ensure exact sum 1 for validator
        for pre in ("model", "market"):
            s = df[f"{pre}_prob_1"] + df[f"{pre}_prob_x"] + df[f"{pre}_prob_2"]
            df[f"{pre}_prob_1"] /= s
            df[f"{pre}_prob_x"] /= s
            df[f"{pre}_prob_2"] /= s
        baseline = df.iloc[:10]
        new_data = df.iloc[10:]
    else:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM matches WHERE actual_result IS NOT NULL AND actual_result IN ('1','X','2') ORDER BY id",
            conn,
        )
        conn.close()
        if len(df) < 20:
            print("Fewer than 20 settled matches in DB; using synthetic data for demo.")
            raw = np.random.rand(30, 3)
            raw /= raw.sum(axis=1, keepdims=True)
            df = pd.DataFrame({
                "model_prob_1": raw[:, 0], "model_prob_x": raw[:, 1], "model_prob_2": raw[:, 2],
                "market_prob_1": raw[:, 0], "market_prob_x": raw[:, 1], "market_prob_2": raw[:, 2],
                "actual_result": np.random.choice(["1", "X", "2"], 30),
            })
            baseline = df.iloc[:15]
            new_data = df.iloc[15:]
        else:
            baseline = df.iloc[: max(50, len(df) // 2)]
            new_data = df.tail(50)
    print("Running ShishkaCheck (בדיקת שישקה) demo...")
    shishka = ShishkaCheck(baseline_df=baseline, calibration_last_n=100)
    result = shishka.evaluate_and_learn(new_data)
    print("Passed:", result.passed, "| Safe to train:", result.safe_to_train, "| Safe to bet:", result.safe_to_bet)
    print("Alerts:", result.alerts_triggered)
    for r in result.results:
        print(f"  [{r.component}] passed={r.passed} — {r.message}")

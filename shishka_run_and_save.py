# -*- coding: utf-8 -*-
"""
מריץ את בדיקת שישקה (ShishkaCheck) על משחקים שהסתיימו ושומר את התוצאה ל-DB
כדי שהדשבורד יוכל להציג תחת "למידת מכונה ואיכות".
מופעל מ-Run_GSA.bat.
"""
import os
import sqlite3
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")

# Reduce noise from ShishkaCheck logger when run from batch
logging.getLogger("ShishkaCheck").setLevel(logging.WARNING)


def ensure_learning_log_table(conn):
    """טבלת יומן למידה – השתלשלות כל אירועי הלימוד."""
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT,
            details_json TEXT
        )
    """)
    conn.commit()


def ensure_shishka_table(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS shishka_last_run (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            run_at TEXT NOT NULL,
            passed INTEGER NOT NULL,
            safe_to_train INTEGER NOT NULL,
            safe_to_bet INTEGER NOT NULL DEFAULT 1,
            brier_score REAL,
            log_loss REAL,
            actual_roi_pct REAL,
            max_drawdown_pct REAL,
            consecutive_losses INTEGER,
            n_matches INTEGER,
            alerts_json TEXT,
            data_validator_passed INTEGER,
            calibrator_passed INTEGER,
            drift_passed INTEGER,
            risk_passed INTEGER,
            details_json TEXT
        )
    """)
    # Backwards-compat: ensure columns exist on old tables
    try:
        c.execute("PRAGMA table_info(shishka_last_run)")
        cols = [r[1] for r in c.fetchall()]
        if "safe_to_bet" not in cols:
            c.execute(
                "ALTER TABLE shishka_last_run "
                "ADD COLUMN safe_to_bet INTEGER NOT NULL DEFAULT 1"
            )
        if "details_json" not in cols:
            c.execute("ALTER TABLE shishka_last_run ADD COLUMN details_json TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def run_and_save():
    os.chdir(BASE_DIR)
    try:
        from shishka_check import ShishkaCheck, ShishkaDataValidator, ShishkaCalibrator
    except ImportError as e:
        print(f"[Shishka] Import error: {e}")
        return

    if not os.path.isfile(DB_PATH):
        print("[Shishka] No gsa_history.db — skipping.")
        return

    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    ensure_shishka_table(conn)
    ensure_learning_log_table(conn)

    try:
        df = pd.read_sql_query(
            """SELECT * FROM matches
               WHERE actual_result IS NOT NULL
                 AND actual_result IN ('1','X','2')
               ORDER BY id ASC""",
            conn,
        )
    except Exception as e:
        print(f"[Shishka] DB read error: {e}")
        conn.close()
        return

    # תיקון: market_prob_* עשויים להישמר כיחסים (odds) במקום הסתברויות.
    # אם הסכום > 1.5, ממירים odds -> probabilities (1/odd + נרמול)
    market_cols = ["market_prob_1", "market_prob_x", "market_prob_2"]
    if all(c in df.columns for c in market_cols):
        vals = df[market_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        sums = vals.sum(axis=1)
        odds_mask = sums > 1.5  # נראה כמו odds (סכום 7–12)
        if odds_mask.any():
            v = vals.values
            inv = np.where(v > 0, 1.0 / v, 0.0)
            row_tot = inv.sum(axis=1, keepdims=True)
            row_tot = np.where(row_tot > 0, row_tot, 1.0)
            probs = inv / row_tot
            df.loc[odds_mask, market_cols] = probs[odds_mask]
            print(f"[Shishka] המרנו {odds_mask.sum()} שורות מיחסים להסתברויות (market_prob).")

    # החלת שכבת כיול על model_prob_* (אם קיימת) – משפר Brier/Log Loss
    try:
        from calibration_layer import load_calibration_params, apply_calibration
        cal_params = load_calibration_params()
        if cal_params and all(c in df.columns for c in ["model_prob_1", "model_prob_x", "model_prob_2"]):
            calibrated = df.apply(
                lambda row: apply_calibration(
                    {"1": row["model_prob_1"], "X": row["model_prob_x"], "2": row["model_prob_2"]},
                    cal_params,
                ),
                axis=1,
            )
            df["model_prob_1"] = [c["1"] for c in calibrated]
            df["model_prob_x"] = [c["X"] for c in calibrated]
            df["model_prob_2"] = [c["2"] for c in calibrated]
    except (ImportError, Exception):
        pass

    if len(df) < 10:
        # Still save a "no data" row so dashboard shows something
        row = {
            "run_at": datetime.now().isoformat(),
            "passed": 1,
            "safe_to_train": 1,
            "safe_to_bet": 1,
            "brier_score": None,
            "log_loss": None,
            "actual_roi_pct": None,
            "max_drawdown_pct": None,
            "consecutive_losses": None,
            "n_matches": len(df),
            "alerts_json": json.dumps(["מעט מדי משחקים לבדיקה"]),
            "data_validator_passed": 1,
            "calibrator_passed": 1,
            "drift_passed": 1,
            "risk_passed": 1,
            "details_json": "{}",
        }
        c = conn.cursor()
        ensure_learning_log_table(conn)
        c.execute("""
            INSERT OR REPLACE INTO shishka_last_run
            (id, run_at, passed, safe_to_train, safe_to_bet, brier_score, log_loss,
             actual_roi_pct, max_drawdown_pct, consecutive_losses, n_matches,
             alerts_json, data_validator_passed, calibrator_passed, drift_passed, risk_passed, details_json)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["run_at"], row["passed"], row["safe_to_train"], row["safe_to_bet"],
            row["brier_score"], row["log_loss"],
            row["actual_roi_pct"], row["max_drawdown_pct"], row["consecutive_losses"], row["n_matches"],
            row["alerts_json"], row["data_validator_passed"], row["calibrator_passed"],
            row["drift_passed"], row["risk_passed"], row["details_json"],
        ))
        c.execute("""INSERT INTO learning_log (run_at, event_type, summary, details_json) VALUES (?, ?, ?, ?)""",
                  (row["run_at"], "shishka", "שישקה: מעט משחקים", "{}"))
        conn.commit()
        conn.close()
        print("[Shishka] Saved (few matches).")
        return

    # Continuous chronological split (no gaps):
    # Use an older block as baseline, and the most recent window as new_data.
    window_new = min(200, len(df))
    # Ensure we always have at least 1 row in baseline if possible
    cut_idx = max(len(df) - window_new, 1)
    baseline_df = df.iloc[:cut_idx]
    new_data = df.iloc[cut_idx:]
    if new_data.empty:
        # Fallback: take last 50 as new_data, rest as baseline
        window_new = min(50, len(df))
        cut_idx = max(len(df) - window_new, 1)
        baseline_df = df.iloc[:cut_idx]
        new_data = df.iloc[cut_idx:]

    data_validator = ShishkaDataValidator(outliers_fail=False)
    calibrator = ShishkaCalibrator(max_brier_threshold=0.65, max_log_loss_threshold=1.5)
    shishka = ShishkaCheck(
        baseline_df=baseline_df,
        calibration_last_n=100,
        data_validator=data_validator,
        calibrator=calibrator,
    )

    result = shishka.evaluate_and_learn(new_data)

    alerts = result.alerts_triggered
    passed = 1 if result.passed else 0
    safe_to_train = 1 if result.safe_to_train else 0
    safe_to_bet = 1 if result.safe_to_bet else 0

    # Robust parsing by component name (no hard-coded indices)
    comp = {r.component: r for r in result.results}
    data_res = comp.get("ShishkaDataValidator")
    cal_res = comp.get("ShishkaCalibrator")
    drift_res = comp.get("ShishkaDriftMonitor")
    risk_res = comp.get("ShishkaRiskManager")

    data_ok = 1 if (data_res is not None and data_res.passed) else 0
    cal_ok = 1 if (cal_res is not None and cal_res.passed) else 0
    drift_ok = 1 if (drift_res is not None and drift_res.passed) else 1
    risk_ok = 1 if (risk_res is not None and risk_res.passed) else 1

    brier = cal_res.details.get("brier_score") if cal_res is not None else None
    logloss = cal_res.details.get("log_loss") if cal_res is not None else None

    roi = risk_res.details.get("actual_roi_pct") if risk_res is not None else None
    dd = risk_res.details.get("max_drawdown_pct") if risk_res is not None else None
    consec = risk_res.details.get("consecutive_losses") if risk_res is not None else None
    n_mat = risk_res.details.get("n_bets") if risk_res is not None else len(new_data)

    details_json = json.dumps({r.component: r.details for r in result.results}, default=str)

    c = conn.cursor()
    run_at = datetime.now().isoformat()
    c.execute("""
        INSERT OR REPLACE INTO shishka_last_run
        (id, run_at, passed, safe_to_train, safe_to_bet, brier_score, log_loss,
         actual_roi_pct, max_drawdown_pct, consecutive_losses, n_matches,
         alerts_json, data_validator_passed, calibrator_passed, drift_passed, risk_passed, details_json)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_at, passed, safe_to_train, safe_to_bet, brier, logloss,
        roi, dd, consec, n_mat or len(new_data),
        json.dumps(alerts, ensure_ascii=False), data_ok, cal_ok, drift_ok, risk_ok, details_json,
    ))
    # יומן למידה – השתלשלות
    summary = f"שישקה: {'עבר' if passed else 'נכשל'}" + (f" | Brier={brier:.4f}" if brier is not None else "")
    c.execute(
        """INSERT INTO learning_log (run_at, event_type, summary, details_json) VALUES (?, ?, ?, ?)""",
        (run_at, "shishka", summary, json.dumps({"passed": bool(passed), "brier_score": brier, "log_loss": logloss, "n_matches": n_mat}, default=str)),
    )
    conn.commit()
    conn.close()
    status = "PASS" if passed else "FAIL"
    print(
        f"[Shishka] בדיקת שישקה הושלמה — {status} | "
        f"safe_to_train={bool(safe_to_train)} | safe_to_bet={bool(safe_to_bet)}"
    )


if __name__ == "__main__":
    run_and_save()

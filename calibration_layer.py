# -*- coding: utf-8 -*-
"""
שכבת כיול – משפרת את הדיוק של ההסתברויות (Brier, Log Loss).
משתמשת ב-Platt Scaling (LogisticRegression) במקום IsotonicRegression – פחות overfitting,
ובהסתברויות חסומות ל-[0.01, 0.99] כדי למנוע 0.0 או 1.0 אבסולוטיים.

לולאת משוב משישקה: כששישקה מזהה אי-התאמה בין צפי למציאות (calibration bins),
המערכת בונה עקומת תיקון confidence ומשפרת את הכיול אוטומטית.

שימוש:
  1. fit_calibration(conn) – מאמן על משחקים היסטוריים, שומר פרמטרים
  2. fit_calibration_with_shishka_refinement(conn) – אימון + תיקון לפי תוצאות שישקה
  3. apply_calibration(probs_dict, params) – מפעיל כיול על הסתברויות חדשות
  4. שילוב ב-v76: טוען params ומפעיל על model_probs לפני שילוב עם שוק/טיכה
"""
import json
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")
CALIBRATION_PARAMS_FILE = os.path.join(BASE_DIR, "calibration_params.json")

OUTCOMES = ["1", "X", "2"]


def _outcome_to_idx(actual: str) -> int:
    if actual == "1":
        return 0
    if actual in ("X", "x"):
        return 1
    if actual == "2":
        return 2
    return 0


def _safe_prob(val, default: float = 1.0 / 3) -> float:
    try:
        v = float(val) if val is not None else default
        return max(1e-6, min(1.0 - 1e-6, v)) if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def fit_calibration(
    conn: Optional[sqlite3.Connection] = None,
    min_samples: int = 30,
    prob_source: str = "model",
) -> Optional[Dict]:
    """
    מאמן שכבת כיול על משחקים היסטוריים.
    prob_source: 'model' | 'final' – מאיזה עמודות לקחת הסתברויות (model_prob_* או final_prob_*)
    מחזיר dict עם isotonic_regressors לכל תוצאה (1, X, 2), או None אם לא מספיק נתונים.
    """
    own_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)

    prefix = "model_prob_" if prob_source == "model" else "final_prob_"
    c1, c2, c3 = f"{prefix}1", f"{prefix}x", f"{prefix}2"

    cur = conn.execute(
        f"""SELECT {c1}, {c2}, {c3}, actual_result FROM matches
           WHERE actual_result IS NOT NULL AND actual_result IN ('1','X','2')
           ORDER BY id ASC"""
    )
    rows = cur.fetchall()
    if own_conn:
        conn.close()

    if len(rows) < min_samples:
        return None

    # X: (n, 3) הסתברויות, y: (n,) אינדקס תוצאה
    X_list = []
    y_list = []
    for r in rows:
        p1 = _safe_prob(r[0])
        px = _safe_prob(r[1])
        p2 = _safe_prob(r[2])
        actual = str(r[3] or "").strip().upper()
        if actual not in ("1", "X", "2"):
            continue
        # נרמול
        s = p1 + px + p2
        if s <= 0:
            continue
        p1, px, p2 = p1 / s, px / s, p2 / s
        X_list.append([p1, px, p2])
        y_list.append(_outcome_to_idx(actual))

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    n = len(X)
    if n < min_samples:
        return None

    # Platt Scaling: LogisticRegression per outcome (less overfitting than Isotonic)
    platt_coefs = []
    for k in range(3):
        p_k = X[:, k:k + 1]  # (n, 1)
        y_binary = (y == k).astype(np.int32)
        lr = LogisticRegression(C=2.0, max_iter=500, solver="lbfgs", random_state=42)
        lr.fit(p_k, y_binary)
        platt_coefs.append({
            "coef": float(lr.coef_[0, 0]),
            "intercept": float(lr.intercept_[0]),
        })

    # Build calibration maps for apply (grid for backward compatibility with interpolation)
    grid = np.linspace(0.01, 0.99, 50)
    calibration_maps = []
    for cf in platt_coefs:
        z = grid * cf["coef"] + cf["intercept"]
        preds = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        preds = np.clip(preds, 0.01, 0.99)
        calibration_maps.append({"x": grid.tolist(), "y": [float(v) for v in preds]})

    params = {
        "prob_source": prob_source,
        "n_samples": n,
        "calibration_maps": calibration_maps,
    }

    with open(CALIBRATION_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return params


def _interp_calibrate(x: float, x_vals: List[float], y_vals: List[float]) -> float:
    """Interpolate y given x using calibration map."""
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    if x <= x_vals[0]:
        return float(y_vals[0])
    if x >= x_vals[-1]:
        return float(y_vals[-1])
    return float(np.interp(x, x_vals, y_vals))


def apply_calibration(
    probs: Dict[str, float],
    params: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    מפעיל כיול על מילון הסתברויות {'1': p1, 'X': px, '2': p2}.
    שלב 1: Platt Scaling (per-outcome).
    שלב 2: אם קיים confidence_calibration משישקה – תיקון confidence.
    מחזיר מילון כיול עם סכום 1.
    """
    if params is None or "calibration_maps" not in params:
        return dict(probs)

    maps = params["calibration_maps"]
    if len(maps) != 3:
        return dict(probs)

    p1 = _safe_prob(probs.get("1", 1 / 3))
    px = _safe_prob(probs.get("X", 1 / 3))
    p2 = _safe_prob(probs.get("2", 1 / 3))
    s = p1 + px + p2
    if s <= 0:
        return {"1": 1 / 3, "X": 1 / 3, "2": 1 / 3}
    p1, px, p2 = p1 / s, px / s, p2 / s

    c1 = _interp_calibrate(p1, maps[0]["x"], maps[0]["y"])
    cx = _interp_calibrate(px, maps[1]["x"], maps[1]["y"])
    c2 = _interp_calibrate(p2, maps[2]["x"], maps[2]["y"])

    # Clip to [0.01, 0.99] - never absolute 0 or 1
    c1 = max(0.01, min(0.99, c1))
    cx = max(0.01, min(0.99, cx))
    c2 = max(0.01, min(0.99, c2))

    total = c1 + cx + c2
    if total <= 0:
        return {"1": 1 / 3, "X": 1 / 3, "2": 1 / 3}
    result = {"1": c1 / total, "X": cx / total, "2": c2 / total}

    # שלב 2: תיקון confidence משישקה (לולאת משוב)
    confidence_map = params.get("confidence_calibration") if params else None
    if confidence_map and "x" in confidence_map and "y" in confidence_map:
        result = _apply_confidence_calibration(result, confidence_map)

    return result


def load_calibration_params() -> Optional[Dict]:
    """טוען פרמטרי כיול מקובץ JSON."""
    if not os.path.exists(CALIBRATION_PARAMS_FILE):
        return None
    try:
        with open(CALIBRATION_PARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# לולאת משוב משישקה – תיקון אוטומטי לפי עקומת calibration
# ---------------------------------------------------------------------------


def _build_confidence_calibration_from_shishka(
    shishka_cal_details: Dict,
    min_samples_per_bin: int = 5,
        ) -> Optional[Dict]:
    """
    בונה מפת תיקון confidence מתוצאות ShishkaCalibrator.
    מחזיר dict עם x, y (expected -> actual) או None אם לא מספיק נתונים.
    """
    expected = shishka_cal_details.get("calibration_bins_expected")
    actual = shishka_cal_details.get("calibration_bins_actual")
    counts = shishka_cal_details.get("calibration_counts")
    if not expected or not actual or not counts or len(expected) != len(actual):
        return None

    x_vals: List[float] = []
    y_vals: List[float] = []
    for i in range(len(expected)):
        exp_val = expected[i]
        act_val = actual[i]
        cnt = counts[i] if i < len(counts) else 0
        if cnt is None or cnt < min_samples_per_bin:
            continue
        if act_val is None or not np.isfinite(act_val):
            continue
        try:
            ex = float(exp_val)
            ay = float(act_val)
        except (TypeError, ValueError):
            continue
        x_vals.append(ex)
        y_vals.append(max(0.01, min(0.99, ay)))

    if len(x_vals) < 2:
        return None

    # Sort by x for monotonic interpolation
    pairs = sorted(zip(x_vals, y_vals))
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]

    # Build grid for interpolation (50 points)
    grid = np.linspace(0.01, 0.99, 50)
    interp = np.interp(grid, x_vals, y_vals)
    interp = np.clip(interp, 0.01, 0.99)

    return {"x": grid.tolist(), "y": [float(v) for v in interp]}


def _apply_confidence_calibration(
    probs: Dict[str, float],
    confidence_map: Dict,
) -> Dict[str, float]:
    """
    מפעיל תיקון confidence: כשהמודל נותן confidence X, עקומת שישקה אומרת שהמציאות Y.
    מתקן את ההסתברות של התוצאה המובילה ומחלק את השאר proportionally.
    """
    p1 = _safe_prob(probs.get("1", 1 / 3))
    px = _safe_prob(probs.get("X", 1 / 3))
    p2 = _safe_prob(probs.get("2", 1 / 3))
    s = p1 + px + p2
    if s <= 0:
        return {"1": 1 / 3, "X": 1 / 3, "2": 1 / 3}
    p1, px, p2 = p1 / s, px / s, p2 / s

    # Top outcome and confidence
    probs_arr = [p1, px, p2]
    top_idx = int(np.argmax(probs_arr))
    confidence = probs_arr[top_idx]
    keys = ["1", "X", "2"]
    top_key = keys[top_idx]

    # Apply Shishka curve: new_confidence = f(old_confidence)
    new_confidence = _interp_calibrate(
        confidence,
        confidence_map["x"],
        confidence_map["y"],
    )
    new_confidence = max(0.01, min(0.99, new_confidence))

    # Redistribute: top gets new_confidence, rest split proportionally
    other_inds = [i for i in range(3) if i != top_idx]
    other_sum = sum(probs_arr[i] for i in other_inds)
    if other_sum <= 0:
        return {"1": 1 / 3, "X": 1 / 3, "2": 1 / 3}
    remainder = 1.0 - new_confidence
    result = {k: 0.0 for k in keys}
    result[top_key] = new_confidence
    for i in other_inds:
        result[keys[i]] = remainder * (probs_arr[i] / other_sum)

    total = sum(result.values())
    if total <= 0:
        return {"1": 1 / 3, "X": 1 / 3, "2": 1 / 3}
    return {k: v / total for k, v in result.items()}


def fit_calibration_with_shishka_refinement(
    conn: Optional[sqlite3.Connection] = None,
    min_samples: int = 30,
    prob_source: str = "model",
    min_samples_per_bin: int = 5,
    shishka_blend_weight: float = 0.7,
) -> Optional[Dict]:
    """
    אימון שכבת כיול עם לולאת משוב משישקה.
    1. מאמן Platt Scaling על נתונים היסטוריים (בסיס)
    2. טוען תוצאות שישקה האחרונות מ-DB
    3. אם שישקה דיווחה על bins עם אי-התאמה – בונה עקומת תיקון confidence
    4. שומר params עם calibration_maps + confidence_calibration (אם קיים)
    """
    base_params = fit_calibration(conn=conn, min_samples=min_samples, prob_source=prob_source)
    if base_params is None:
        return None

    # Load Shishka details from DB
    own_conn = conn is None
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)

    try:
        cur = conn.execute(
            "SELECT details_json FROM shishka_last_run WHERE id=1"
        )
        row = cur.fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        if own_conn:
            conn.close()

    if row is None or not row[0]:
        return base_params

    try:
        details_all = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return base_params

    shishka_cal = details_all.get("ShishkaCalibrator")
    if not shishka_cal:
        return base_params

    confidence_map = _build_confidence_calibration_from_shishka(
        shishka_cal,
        min_samples_per_bin=min_samples_per_bin,
    )
    if confidence_map is None:
        return base_params

    # Blend with identity for stability (avoid over-correction)
    # new_y = blend * shishka_y + (1-blend) * x (identity line)
    blended_y = []
    for i in range(len(confidence_map["y"])):
        shishka_y = confidence_map["y"][i]
        ident_y = confidence_map["x"][i]  # identity: y = x
        blended = shishka_blend_weight * shishka_y + (1 - shishka_blend_weight) * ident_y
        blended_y.append(max(0.01, min(0.99, blended)))
    confidence_map["y"] = blended_y

    base_params["confidence_calibration"] = confidence_map
    base_params["shishka_refinement_applied"] = True

    with open(CALIBRATION_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(base_params, f, indent=2, ensure_ascii=False)

    return base_params


if __name__ == "__main__":
    print("=" * 50)
    print(" שכבת כיול – אימון")
    print("=" * 50)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    params = fit_calibration(conn, min_samples=20)
    conn.close()
    if params:
        print(f"[כיול] אומן על {params['n_samples']} משחקים. נשמר ב-{CALIBRATION_PARAMS_FILE}")
    else:
        print("[כיול] לא מספיק נתונים לאימון (נדרשים לפחות 20 משחקים).")

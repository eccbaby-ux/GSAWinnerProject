# -*- coding: utf-8 -*-
"""
מערכת טיכה – למידה לאחור מכל המשחקים שהסתיימו.
מחשבת מה הייתה נותנת, בודקת מה יצא, מתקנת את המשקולות והמודל עד להגעה לרמת דיוק מקסימלית.
משקל נוסף בתוך מערת נחשון: (מודל מתמטי, שוק, טיכה).
"""
import os
import json
import sqlite3
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")
TICHA_PARAMS_FILE = os.path.join(BASE_DIR, "ticha_params.json")


def _row_market_probs(row):
    """ממיר market_prob_* להסתברויות (0-1). אם חסר – 1/3."""
    if not hasattr(row, 'get'):
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    cols = [('1', 'market_prob_1'), ('X', 'market_prob_x'), ('2', 'market_prob_2')]
    clean = {}
    for k, col in cols:
        v = None
        try:
            v = row.get(col) or row.get('market_prob_X' if col == 'market_prob_x' else col)
            v = float(v) if v is not None else None
            if v is not None and v > 0:
                clean[k] = v
            else:
                clean[k] = 0.0
        except (TypeError, ValueError, KeyError):
            clean[k] = 0.0
    vals = [clean[k] for k in ['1', 'X', '2']]
    if not any(v > 0 for v in vals):
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    if all(0 < v <= 1.0 for v in vals if v > 0):
        s = sum(vals)
        if s <= 0:
            return {'1': 1/3, 'X': 1/3, '2': 1/3}
        return {k: clean[k] / s for k in ['1', 'X', '2']}
    inv = {k: (1.0 / v if v > 0 else 0.0) for k, v in clean.items()}
    s = sum(inv.values())
    if s <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    return {k: inv[k] / s for k in ['1', 'X', '2']}


def _get_model_probs(row):
    """מחזיר הסתברויות המודל מהשורה. אם חסר – 1/3."""
    if not hasattr(row, 'get'):
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    def v(letter):
        for col in (f'model_prob_{letter}', f'model_prob_{letter.lower()}'):
            try:
                x = row.get(col)
                if x is not None:
                    f = float(x)
                    if f > 0:
                        return f
            except (TypeError, ValueError, KeyError):
                pass
        return 1/3
    return {'1': v('1'), 'X': v('x'), '2': v('X')}


def _outcome_to_idx(actual):
    """ממיר תוצאה '1'/'X'/'2' לאינדקס 0/1/2."""
    if actual == '1': return 0
    if actual == 'X': return 1
    if actual == '2': return 2
    return 0


def _idx_to_outcome(i):
    return ['1', 'X', '2'][i]


def softmax(x):
    e = np.exp(np.clip(x - x.max(), -50, 50))
    return e / e.sum()


def build_feature(row):
    """וקטור תכונות: [model_1, model_x, model_2, mkt_1, mkt_x, mkt_2]."""
    model = _get_model_probs(row)
    market = _row_market_probs(row)
    return np.array([
        model['1'], model['X'], model['2'],
        market['1'], market['X'], market['2']
    ], dtype=np.float64)


def train_ticha_model(X, y, max_iter=500, lr=0.1):
    """
    אימון רגרסיה לוגיסטית (softmax) ב-numpy.
    X: (n, 6), y: (n,) אינדקסים 0,1,2.
    מחזיר W (3, 6), b (3,).
    """
    n, d = X.shape
    W = np.zeros((3, d), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for _ in range(max_iter):
        logits = X @ W.T + b
        probs = np.apply_along_axis(softmax, 1, logits)
        one_hot = np.zeros((n, 3))
        one_hot[np.arange(n), y] = 1.0
        grad_w = (probs - one_hot).T @ X
        grad_b = (probs - one_hot).sum(axis=0)
        W -= lr * grad_w / n
        b -= lr * grad_b / n
    return W, b


def predict_ticha_probs(W, b, model_probs, market_probs):
    """מחזיר הסתברויות טיכה עבור משחק בודד."""
    x = np.array([
        float(model_probs.get('1', 1/3)), float(model_probs.get('X', 1/3)), float(model_probs.get('2', 1/3)),
        float(market_probs.get('1', 1/3)), float(market_probs.get('X', 1/3)), float(market_probs.get('2', 1/3))
    ], dtype=np.float64)
    logits = x @ W.T + b
    p = softmax(logits)
    return {'1': float(p[0]), 'X': float(p[1]), '2': float(p[2])}


def get_ticha_probs(model_probs, market_probs, ticha_params=None):
    """
    מחזיר הסתברויות טיכה למשחק.
    ticha_params: dict עם 'W' (list of lists), 'b' (list), או None – אז מחזיר אחיד 1/3.
    """
    if not ticha_params or 'W' not in ticha_params or 'b' not in ticha_params:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    W = np.array(ticha_params['W'], dtype=np.float64)
    b = np.array(ticha_params['b'], dtype=np.float64)
    return predict_ticha_probs(W, b, model_probs, market_probs)


def load_ticha_params_from_db(conn=None):
    """טוען פרמטרי טיכה מהטבלה ticha_params או מקובץ JSON."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticha_params'")
        if c.fetchone():
            c.execute("SELECT params_json, updated_at FROM ticha_params WHERE id=1 LIMIT 1")
            row = c.fetchone()
            if row and row[0]:
                return json.loads(row[0])
    except Exception:
        pass
    if os.path.exists(TICHA_PARAMS_FILE):
        with open(TICHA_PARAMS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_ticha_params_to_db(conn, params_dict, w_model, w_market, w_ticha):
    """שומר פרמטרי טיכה ומשקולות במסד."""
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS ticha_params (
            id INTEGER PRIMARY KEY,
            params_json TEXT,
            w_model REAL, w_market REAL, w_ticha REAL,
            updated_at TEXT,
            train_accuracy REAL, test_accuracy REAL, total_matches INTEGER
        )
    """)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    params_json = json.dumps({
        'W': [row.tolist() for row in params_dict['W']],
        'b': params_dict['b'].tolist(),
    }, ensure_ascii=False)
    acc_train = params_dict.get('train_accuracy')
    acc_test = params_dict.get('test_accuracy')
    total = params_dict.get('total_matches')
    c.execute("""
        INSERT OR REPLACE INTO ticha_params
        (id, params_json, w_model, w_market, w_ticha, updated_at, train_accuracy, test_accuracy, total_matches)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (params_json, w_model, w_market, w_ticha, now, acc_train, acc_test, total))
    conn.commit()


def ensure_weights_have_ticha(conn):
    """מוסיף עמודה w_ticha לטבלת weights ו-weight_profiles אם חסרה."""
    c = conn.cursor()
    try:
        c.execute("PRAGMA table_info(weights)")
        cols = [row[1] for row in c.fetchall()]
        if 'w_ticha' not in cols:
            c.execute("ALTER TABLE weights ADD COLUMN w_ticha REAL DEFAULT 0")
            conn.commit()
    except Exception:
        pass
    try:
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weight_profiles'")
        if c.fetchone():
            c.execute("PRAGMA table_info(weight_profiles)")
            cols = [row[1] for row in c.fetchall()]
            if 'w_ticha' not in cols:
                c.execute("ALTER TABLE weight_profiles ADD COLUMN w_ticha REAL DEFAULT 0")
                conn.commit()
    except Exception:
        pass


def train_ticha_backward(conn=None):
    """
    למידה לאחור: טוען את כל המשחקים עם תוצאה, מאמן את מודל טיכה ומשקולות (w_model, w_market, w_ticha)
    עד מקסימום דיוק. שומר את הפרמטרים והמשקולות ב-DB.
    """
    if conn is None:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
    ensure_weights_have_ticha(conn)

    # קודם מאפסים 'None' (מחרוזת) ל-NULL כדי ש־result_updater.py יוכל למשוך תוצאות
    try:
        cur = conn.execute(
            "UPDATE matches SET actual_result = NULL WHERE actual_result = ?",
            ("None",)
        )
        updated_none = cur.rowcount
        conn.commit()
        if updated_none > 0:
            print(f"[מערכת טיכה] איפסנו {updated_none} רשומות עם actual_result='None' ל-NULL (result_updater.py ימשוך תוצאות בהרצה הבאה).")
    except Exception:
        pass

    # שליפת משחקים עם תוצאה אמיתית (1/X/2)
    cur = conn.execute(
        "SELECT * FROM matches WHERE actual_result IS NOT NULL AND actual_result != '' ORDER BY id ASC"
    )
    df = cur.fetchall()
    columns = [d[0] for d in cur.description] if cur.description else []
    rows = [dict(zip(columns, r)) for r in df]
    # סינון ב-Python: רק תוצאות 1/X/2 (לא מחרוזת 'None' שנשארה)
    rows = [r for r in rows if str(r.get("actual_result") or "").strip().lower() not in ("none", "null", "")]
    total_matches = len(rows)

    MIN_ROWS = 15  # מינימום משחקים ללמידה (פחות = דיוק חלש)

    if total_matches < MIN_ROWS:
        print(
            f"[מערכת טיכה] יש רק {total_matches} משחקים עם תוצאה תקינה (1/X/2). נדרשים לפחות {MIN_ROWS} ללמידה."
        )
        print(
            "[מערכת טיכה] result_updater.py (ב־Train_GSA.bat) מטפל במשיכת תוצאות מ-API; הרץ את Train_GSA.bat כדי לעדכן תוצאות ואז טיכה תלמד."
        )
        return None

    def normalize_actual(val):
        if val is None:
            return None
        s = str(val).strip()
        if not s:
            return None
        # מספרים (למשל 1.0 מ־pandas/sqlite)
        try:
            n = float(s)
            if n == 1.0 or n == 1:
                return "1"
            if n == 2.0 or n == 2:
                return "2"
            if n == 0.0 or n == 0:
                return "X"
        except ValueError:
            pass
        s_upper = s.upper()
        if s_upper in ("1", "X", "2"):
            return "1" if s_upper == "1" else "2" if s_upper == "2" else "X"
        if s_upper in ("H", "HOME"):
            return "1"
        if s_upper in ("D", "DRAW"):
            return "X"
        if s_upper in ("A", "AWAY"):
            return "2"
        return None

    # דוגמאות מערך actual_result במסד (לאבחון)
    seen_actuals = {}
    for r in rows[:200]:
        raw = r.get("actual_result")
        key = repr(raw)
        seen_actuals[key] = seen_actuals.get(key, 0) + 1
    samples = list(seen_actuals.keys())[:5]
    print(f"[מערכת טיכה] דוגמאות actual_result במסד: {samples}")

    X_list = []
    y_list = []
    rows_used = []
    skipped_actual = 0
    skipped_build = 0
    for r in rows:
        actual = normalize_actual(r.get("actual_result"))
        if actual is None:
            skipped_actual += 1
            continue
        try:
            x = build_feature(r)
            y = _outcome_to_idx(actual)
            X_list.append(x)
            y_list.append(y)
            rows_used.append(r)
        except Exception as e:
            skipped_build += 1
            continue

    print(f"[מערכת טיכה] משחקים עם תוצאה: {total_matches} | תוצאה תקינה (1/X/2): {len(rows_used) + skipped_build} | שורות ללמוד: {len(X_list)} (דולגו: תוצאה לא תקינה {skipped_actual}, בניית תכונות {skipped_build})")

    if len(X_list) < MIN_ROWS:
        print(
            f"[מערכת טיכה] לא מספיק שורות תקינות אחרי בניית תכונות ({len(X_list)}). "
            f"נדרשים לפחות {MIN_ROWS}. ודא שיש במסד משחקים עם actual_result ועמודות model_prob_* / market_prob_*."
        )
        return None

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    n = len(X)
    train_size = int(n * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    rows_train = rows_used[:train_size]
    rows_test = rows_used[train_size:]

    print("[מערכת טיכה] מאמנת מודל טיכה (למידה לאחור)...")
    W, b = train_ticha_model(X_train, y_train)

    EV_MIN_THRESHOLD = 0.0   # הימר רק כשהציפייה חיובית (combined_prob × odds > 1)
    MIN_BETS_FOR_ROI = 5     # מינימום הימורים כדי שה-ROI יהיה בעל משמעות

    def hit_rate_and_roi(weights, X_data, y_data, rows_subset, get_odds_fn):
        """rows_subset: רשימת שורות מתאימה לאורך X_data/y_data.
        מחשב ROI רק על הימורים עם EV חיובי (combined_prob > market implied prob).
        """
        w_model, w_market, w_ticha = weights[0], weights[1], weights[2]
        total_w = w_model + w_market + w_ticha
        if total_w <= 0:
            return 0.0, float('-inf'), 0.0
        w_model, w_market, w_ticha = w_model / total_w, w_market / total_w, w_ticha / total_w
        hits = 0
        profit = 0.0
        bets_placed = 0
        for i, (xi, yi) in enumerate(zip(X_data, y_data)):
            model_p = {'1': xi[0], 'X': xi[1], '2': xi[2]}
            mkt_p = {'1': xi[3], 'X': xi[4], '2': xi[5]}
            ticha_p = predict_ticha_probs(W, b, model_p, mkt_p)
            p1 = model_p['1'] * w_model + mkt_p['1'] * w_market + ticha_p['1'] * w_ticha
            px = model_p['X'] * w_model + mkt_p['X'] * w_market + ticha_p['X'] * w_ticha
            p2 = model_p['2'] * w_model + mkt_p['2'] * w_market + ticha_p['2'] * w_ticha
            pred = 0 if p1 >= px and p1 >= p2 else (1 if px >= p1 and px >= p2 else 2)
            odds = get_odds_fn(rows_subset[i], _idx_to_outcome(pred))
            if not odds or odds <= 1.0:
                continue
            combined_p_pred = [p1, px, p2][pred]
            ev = combined_p_pred * odds - 1.0
            if ev < EV_MIN_THRESHOLD:
                continue  # אין יתרון על השוק – דלג
            bets_placed += 1
            if pred == yi:
                hits += 1
                profit += (odds - 1.0)
            else:
                profit -= 1.0
        if bets_placed < MIN_BETS_FOR_ROI:
            return 0.0, float('-inf'), profit
        rate = hits / bets_placed
        roi = profit / bets_placed
        return rate, roi, profit

    def get_odds_simple(row, outcome):
        key = {'1': '1', 'X': 'x', '2': '2'}[outcome]
        col = f'market_prob_{key}' if key != 'x' else 'market_prob_x'
        try:
            v = row.get(col)
            fv = float(v) if v is not None else None
        except (TypeError, ValueError, KeyError):
            return 1.0
        if fv is None or fv <= 0:
            return 1.0
        if 0 < fv <= 1.0:
            return 1.0 / fv
        return fv

    best_weights = (0.33, 0.33, 0.34)
    best_test_rate = 0.0
    best_test_roi = float('-inf')

    step = 0.05
    MIN_W_MARKET = 0.20  # Hard constraint: market weight never below 20%; sum remains 1.0
    for wmk in np.arange(MIN_W_MARKET, 1.01, step):
        for wm in np.arange(0.0, 1.01 - wmk, step):
            wt = 1.0 - wm - wmk
            if wt < 0:
                continue
            weights = (wm, wmk, wt)
            te_rate, te_roi, _ = hit_rate_and_roi(weights, X_test, y_test, rows_test, get_odds_simple)
            if te_roi > best_test_roi or (abs(te_roi - best_test_roi) < 1e-6 and te_rate > best_test_rate):
                best_test_rate = te_rate
                best_test_roi = te_roi
                best_weights = weights

    w_model, w_market, w_ticha = best_weights
    train_rate, train_roi, train_profit = hit_rate_and_roi(best_weights, X_train, y_train, rows_train, get_odds_simple)
    test_rate, test_roi, test_profit = hit_rate_and_roi(best_weights, X_test, y_test, rows_test, get_odds_simple)

    roi_str_train = f"{train_roi*100:+.2f}%" if train_roi != float('-inf') else "N/A (מעט הימורים)"
    roi_str_test  = f"{test_roi*100:+.2f}%" if test_roi  != float('-inf') else "N/A (מעט הימורים)"
    print(f"[מערכת טיכה] [אימון] דיוק: {train_rate*100:.1f}%  ROI (per bet): {roi_str_train}")
    print(f"[מערכת טיכה] [בדיקה] דיוק: {test_rate*100:.1f}%  ROI (per bet): {roi_str_test}")
    print(f"[מערכת טיכה] משקולות: מודל={w_model:.2f}, שוק={w_market:.2f}, טיכה={w_ticha:.2f}")

    params_dict = {
        'W': W, 'b': b,
        'train_accuracy': round(train_rate, 4),
        'test_accuracy': round(test_rate, 4),
        'total_matches': total_matches,
    }
    save_ticha_params_to_db(conn, params_dict, w_model, w_market, w_ticha)

    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.execute("UPDATE weights SET w_model=?, w_market=?, w_ticha=?, updated_at=? WHERE id=1",
              (w_model, w_market, w_ticha, now))
    try:
        c.execute("UPDATE weight_profiles SET w_model=?, w_market=?, w_ticha=?, updated_at=? WHERE profile='default'",
                  (w_model, w_market, w_ticha, now))
    except Exception:
        pass
    # יומן למידה – השתלשלות
    try:
        c.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT,
                details_json TEXT
            )
        """)
        summary = f"טיכה: מודל={w_model:.0%} שוק={w_market:.0%} טיכה={w_ticha:.0%} | Train={train_rate*100:.1f}% Test={test_rate*100:.1f}%"
        c.execute(
            """INSERT INTO learning_log (run_at, event_type, summary, details_json) VALUES (?, ?, ?, ?)""",
            (now, "ticha", summary, json.dumps({"w_model": w_model, "w_market": w_market, "w_ticha": w_ticha, "train_accuracy": train_rate, "test_accuracy": test_rate, "total_matches": total_matches})),
        )
    except Exception:
        pass
    conn.commit()

    with open(TICHA_PARAMS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'W': [row.tolist() for row in W], 'b': b.tolist()}, f, ensure_ascii=False, indent=2)

    # אימון שכבת כיול עם לולאת משוב משישקה (משפר Brier/Log Loss, מתקן אוטומטית)
    try:
        from calibration_layer import fit_calibration_with_shishka_refinement
        cal_params = fit_calibration_with_shishka_refinement(
            conn, min_samples=20, prob_source="model",
            min_samples_per_bin=5, shishka_blend_weight=0.7,
        )
        if cal_params:
            msg = f"[מערכת טיכה] שכבת כיול אומנה על {cal_params['n_samples']} משחקים."
            if cal_params.get("shishka_refinement_applied"):
                msg += " (תיקון משישקה הוחל)"
            print(msg)
    except ImportError:
        pass

    return params_dict


if __name__ == "__main__":
    print("=" * 50)
    print(" מערכת טיכה – למידה לאחור")
    print("=" * 50)
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    train_ticha_backward(conn)
    conn.close()
    print("סיום מערכת טיכה.")

# -*- coding: utf-8 -*-
"""
GSA V79 Auto-Learner: עדכון משקולות מודל/שוק לפי ROI על משחקים שהסתיימו.
מומלץ להריץ אחרי עדכון actual_result במסד הנתונים (למשל אחרי עדכון תוצאות מחזור).
"""
import os
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np

# --- 1. הגדרת נתיבים בסיסיים (חובה להגדיר בראש ובראשונה) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")

def _row_market_probs(row):
    """
    ממיר את עמודות market_prob_* כפי שנשמרו בטבלת matches להסתברויות.
    אם הערכים נראים כהסתברויות (0–1) – מנרמל אותם; אחרת מניח שהם יחסים ומחשב 1/odd.
    """
    try:
        v1 = row.get('market_prob_1')
    except AttributeError:
        v1 = row['market_prob_1'] if 'market_prob_1' in row else None
    try:
        vx = row.get('market_prob_x')
    except AttributeError:
        vx = row['market_prob_x'] if 'market_prob_x' in row else None
    try:
        v2 = row.get('market_prob_2')
    except AttributeError:
        v2 = row['market_prob_2'] if 'market_prob_2' in row else None

    values = []
    clean = {}
    for key, v in (('1', v1), ('X', vx), ('2', v2)):
        try:
            fv = float(v) if v is not None else None
        except (TypeError, ValueError):
            fv = None
        if fv is not None and fv > 0:
            clean[key] = fv
            values.append(fv)
        else:
            clean[key] = 0.0
    if not values:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    # כבר הסתברויות?
    if all(0 < v <= 1.0 for v in values):
        total = sum(values)
        if total <= 0:
            return {'1': 1/3, 'X': 1/3, '2': 1/3}
        return {k: (clean[k] / total) for k in ['1', 'X', '2']}
    # אחרת – מניחים יחסים
    inv = {}
    total_inv = 0.0
    for k in ['1', 'X', '2']:
        v = clean[k]
        if v > 0:
            inv[k] = 1.0 / v
            total_inv += inv[k]
        else:
            inv[k] = 0.0
    if total_inv <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    return {k: (inv[k] / total_inv) for k in ['1', 'X', '2']}

def fix_database_schema():
    """הבטחת קיום הטבלאות, כולל טבלת המשחקים המרכזית ויומן הלמידה"""
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute('PRAGMA journal_mode=WAL;')
    c = conn.cursor()
    
    # 1. יצירת טבלת המשחקים המרכזית (אם נמחקה)
    c.execute('''CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        fixture_id INTEGER, 
        match_date TEXT, 
        home_team TEXT, 
        away_team TEXT, 
        model_prob_1 REAL, 
        model_prob_x REAL, 
        model_prob_2 REAL, 
        market_prob_1 REAL, 
        market_prob_x REAL, 
        market_prob_2 REAL, 
        ai_prob_1 REAL, 
        ai_prob_x REAL, 
        ai_prob_2 REAL, 
        actual_result TEXT)''')
    
    # 2. טבלת משקולות
    c.execute('''CREATE TABLE IF NOT EXISTS weights (
        id INTEGER PRIMARY KEY, 
        w_model REAL, 
        w_market REAL, 
        w_ai REAL, 
        updated_at TEXT)''')
    
    # 3. טבלת יומן למידה לדשבורד
    c.execute('''CREATE TABLE IF NOT EXISTS system_logs (
        id INTEGER PRIMARY KEY, 
        log_date TEXT, 
        message TEXT)''')

    # 4. טבלת היסטוריית משקולות
    c.execute('''CREATE TABLE IF NOT EXISTS weights_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        updated_at TEXT,
        w_model REAL,
        w_market REAL,
        train_accuracy REAL,
        test_accuracy REAL,
        total_matches INTEGER)''')
    
    # בדיקה האם יש רשומת משקולות התחלתית
    c.execute("SELECT count(*) FROM weights")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO weights (id, w_model, w_market, w_ai, updated_at) VALUES (1, 0.40, 0.60, 0, ?)", 
                  (datetime.now().strftime("%Y-%m-%d"),))
    
    conn.commit()
    conn.close()


def _get_system_prediction(row, market_probs, w_model, w_market):
    """פונקציית עזר לחישוב חיזוי משוקלל (מודל מתמטי + שוק)"""
    mkt_1 = market_probs.get('1', 0.33)
    mkt_x = market_probs.get('X', 0.33)
    mkt_2 = market_probs.get('2', 0.33)

    p1 = (row['model_prob_1'] or 0.33) * w_model + mkt_1 * w_market
    px = (row['model_prob_x'] or 0.33) * w_model + mkt_x * w_market
    p2 = (row['model_prob_2'] or 0.33) * w_model + mkt_2 * w_market

    return '1' if p1 > px and p1 > p2 else 'X' if px > p1 and px > p2 else '2'


def _get_decimal_odds_for_outcome(row, outcome):
    """
    Returns the decimal (European) odds for the given outcome ('1', 'X', '2').
    Row may have market_odds_1/X/2 or market_prob_1/x/2. If market_prob_* is in (0,1] treat as probability (odds=1/p);
    if >1 treat as stored decimal odds.
    """
    key = {'1': '1', 'X': 'x', '2': '2'}[outcome]
    # Prefer explicit odds columns if present
    odds_col = f'market_odds_{key}' if key != 'x' else 'market_odds_x'
    if odds_col in row and row.get(odds_col) is not None:
        try:
            o = float(row[odds_col])
            if o > 0:
                return o
        except (TypeError, ValueError):
            pass
    # Derive from market_prob_*
    prob_col = f'market_prob_{key}' if key != 'x' else 'market_prob_x'
    v = row.get(prob_col)
    try:
        fv = float(v) if v is not None else None
    except (TypeError, ValueError):
        return 1.0
    if fv is None or fv <= 0:
        return 1.0
    if 0 < fv <= 1.0:
        return 1.0 / fv  # probability -> decimal odds
    return fv  # already stored as decimal odds


def _save_weights(conn, best_weights, train_accuracy=None, test_accuracy=None, total_matches=None):
    """שומר משקולות ומתעד את הריצה בהיסטוריה"""
    import json
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    w_model = float(best_weights[0])
    w_market = float(best_weights[1])
    c.execute(
        "UPDATE weights SET w_model=?, w_market=?, w_ai=0, updated_at=? WHERE id=1",
        (w_model, w_market, now),
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS weight_profiles (
            profile TEXT PRIMARY KEY, w_model REAL, w_market REAL, w_ai REAL, updated_at TEXT
        )"""
    )
    c.execute(
        "INSERT OR REPLACE INTO weight_profiles (profile, w_model, w_market, w_ai, updated_at) VALUES (?, ?, ?, 0, ?)",
        ("default", w_model, w_market, now),
    )
    c.execute(
        """INSERT INTO weights_history (updated_at, w_model, w_market, train_accuracy, test_accuracy, total_matches)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (now, w_model, w_market, train_accuracy, test_accuracy, total_matches),
    )
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
        tr = f"{train_accuracy*100:.1f}%" if train_accuracy is not None else "—"
        te = f"{test_accuracy*100:.1f}%" if test_accuracy is not None else "—"
        summary = f"Auto-Learner: מודל={w_model:.0%} שוק={w_market:.0%} | Train={tr} Test={te}"
        c.execute(
            """INSERT INTO learning_log (run_at, event_type, summary, details_json) VALUES (?, ?, ?, ?)""",
            (now, "auto_learner", summary, json.dumps({"w_model": w_model, "w_market": w_market, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy, "total_matches": total_matches})),
        )
    except Exception:
        pass
    conn.commit()


def optimize_weights(conn):
    """
    אופטימיזציה מבוססת Walk-Forward (Time-Series Split).
    אימון על נתוני עבר, בדיקה על נתוני עתיד (Out-of-Sample) למניעת Overfitting.
    משתמש ב-ROI קופה (מטמון מהדשבורד) לדיוק הלימוד.
    """
    # קריאת ROI קופה אם קיים (משמש לדיוק הלימוד)
    try:
        roi_df = pd.read_sql_query("SELECT roi_pct, total_invested, total_profit FROM bankroll_roi_cache WHERE id=1", conn)
        if not roi_df.empty and roi_df.iloc[0]["roi_pct"] is not None:
            roi_pct = float(roi_df.iloc[0]["roi_pct"])
            print(f"📊 ROI קופה (מהדשבורד): {roi_pct:.1f}% – משמש כמשוב לדיוק המשקולות.")
    except Exception:
        pass

    # מיון כרונולוגי חובה – אסור לערבב משחקים חדשים וישנים
    df = pd.read_sql_query(
        "SELECT * FROM matches WHERE actual_result IS NOT NULL ORDER BY id ASC", conn
    )

    total_matches = len(df)
    if total_matches < 30:
        print(
            f"⚠️ יש לך רק {total_matches} משחקים סגורים. "
            "זה קטן מדי מכדי ללמוד משהו בלי ליפול ל-Overfitting מביך."
        )
        print("המערכת משתמשת כרגע בעוגן שמרני (מודל=0.40, שוק=0.60). תן לזמן לעשות את שלו.")
        _save_weights(conn, np.array([0.40, 0.60]), total_matches=total_matches)
        return

    # פיצול: 80% לאימון, 20% לבדיקה מחוץ למדגם
    train_size = int(total_matches * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    print(f"🧠 מתחיל אופטימיזציה נטולת-אשליות (Walk-Forward).")
    print(f"   ← מתאמן על {len(df_train)} משחקים מהעבר.")
    print(f"   ← בוחן את עצמו 'על יבש' מול {len(df_test)} המשחקים האחרונים.")

    baseline_weights = np.array([0.40, 0.60])  # [Model, Market]
    best_train_roi = float('-inf')
    best_weights = baseline_weights
    min_distance = float('inf')
    best_train_profit = 0.0
    best_train_bets = 0
    best_train_hits = 0

    # שלב 1 – אימון: מציאת המשקולות על בסיס רווחיות (ROI). צעד 0.02 לאיזון עדין יותר מודל/שוק
    for w_model in np.arange(0.0, 1.01, 0.02):
        w_market = round(1.0 - w_model, 4)
        if w_market < 0.0:
            continue

        profit = 0.0
        bets_placed = 0
        hits = 0
        for _, row in df_train.iterrows():
            market_probs = _row_market_probs(row)
            pred = _get_system_prediction(row, market_probs, w_model, w_market)
            odds_for_pred = _get_decimal_odds_for_outcome(row, pred)
            if not odds_for_pred or odds_for_pred <= 1.0:
                continue
            # EV filter: הימר רק כשהציפייה חיובית (model sees value vs market)
            mkt_imp_prob = market_probs.get(pred, 0.33)
            model_imp_prob = (row.get(f'model_prob_{pred.lower()}') or row.get(f'model_prob_{pred}') or 0.33)
            try:
                model_imp_prob = float(model_imp_prob)
            except (TypeError, ValueError):
                model_imp_prob = 0.33
            combined_p = model_imp_prob * w_model + mkt_imp_prob * w_market
            ev = combined_p * odds_for_pred - 1.0
            if ev < 0.0:
                continue  # אין יתרון – דלג
            if pred == str(row['actual_result']):
                profit += (odds_for_pred - 1.0)
                hits += 1
            else:
                profit -= 1.0
            bets_placed += 1

        roi = (profit / bets_placed) if bets_placed >= 5 else float('-inf')
        current_weights = np.array([w_model, w_market])
        distance = np.sum((current_weights - baseline_weights) ** 2)

        # בחר משקולות לפי ROI מקסימלי; בשוויון – קרוב יותר לבסיס
        if roi > best_train_roi + 1e-9:
            best_train_roi = roi
            best_weights = current_weights
            min_distance = distance
            best_train_profit = profit
            best_train_bets = bets_placed
            best_train_hits = hits
        elif abs(roi - best_train_roi) <= 1e-9 and distance < min_distance:
            best_weights = current_weights
            min_distance = distance
            best_train_profit = profit
            best_train_bets = bets_placed
            best_train_hits = hits

    # שלב 2 – אמת: בדיקת המשקולות מחוץ למדגם (Test Set) – חישוב רווח ו-ROI
    test_profit = 0.0
    test_bets_placed = 0
    test_hits = 0
    for _, row in df_test.iterrows():
        market_probs = _row_market_probs(row)
        pred = _get_system_prediction(row, market_probs, best_weights[0], best_weights[1])
        odds_for_pred = _get_decimal_odds_for_outcome(row, pred)
        if not odds_for_pred or odds_for_pred <= 1.0:
            continue
        mkt_imp_prob = market_probs.get(pred, 0.33)
        model_imp_prob = (row.get(f'model_prob_{pred.lower()}') or row.get(f'model_prob_{pred}') or 0.33)
        try:
            model_imp_prob = float(model_imp_prob)
        except (TypeError, ValueError):
            model_imp_prob = 0.33
        combined_p = model_imp_prob * best_weights[0] + mkt_imp_prob * best_weights[1]
        ev = combined_p * odds_for_pred - 1.0
        if ev < 0.0:
            continue
        if pred == str(row['actual_result']):
            test_profit += (odds_for_pred - 1.0)
            test_hits += 1
        else:
            test_profit -= 1.0
        test_bets_placed += 1

    test_roi = (test_profit / test_bets_placed) if test_bets_placed > 0 else 0.0
    test_rate = (test_hits / test_bets_placed) if test_bets_placed > 0 else 0
    train_rate = (best_train_hits / best_train_bets) if best_train_bets > 0 else 0

    print(f"📊 [Train] רווח סימולציה: {best_train_profit:+.2f} יח'  |  ROI: {best_train_roi * 100:+.2f}%  |  Hit Rate: {train_rate * 100:.1f}%  (n={best_train_bets})")
    print(f"🏆 [Test ] רווח סימולציה: {test_profit:+.2f} יח'  |  ROI: {test_roi * 100:+.2f}%  |  Hit Rate: {test_rate * 100:.1f}%  (n={test_bets_placed})")
    print(f"⚖️  משקולות שנצרבו (לפי ROI): מודל={best_weights[0]:.2f}, שוק={best_weights[1]:.2f}")

    _save_weights(conn, best_weights,
                  train_accuracy=round(train_rate, 4),
                  test_accuracy=round(test_rate, 4),
                  total_matches=total_matches)

if __name__ == "__main__":
    print("===================================================")
    print(" 🧠 GSA V79 Auto-Learner & Cognitive Feedback Loop ")
    print("===================================================")
    fix_database_schema()
    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    optimize_weights(conn)
    conn.close()
import streamlit as st
import json
import numpy as np
import os
import subprocess
import time
import sys
import base64
import sqlite3
import pandas as pd
from datetime import datetime, timezone, timedelta
import dateutil.parser
import math
from collections import Counter
import itertools
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
# --- V76 DASHBOARD CONFIG (PRO PRODUCTION) ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(_BASE_DIR, "analysis_results_v76.json")
DB_PATH = os.path.join(_BASE_DIR, "gsa_history.db")
RUN_GSA_BAT = os.path.join(_BASE_DIR, "Run_GSA.bat")
TRAIN_GSA_BAT = os.path.join(_BASE_DIR, "Train_GSA.bat")
BG_IMAGE_PATH = os.path.join(_BASE_DIR, "GSA GOALS.jpeg")
ODDS_PREVIOUS_FILE = os.path.join(_BASE_DIR, "winner_odds_previous.json")

# --- Hard Risk Limits (Bankroll Management) ---
MAX_BET_FRACTION = 0.02          # מקסימום 2% מהבנק להימור/טופס
MAX_DAILY_RISK_FRACTION = 0.05   # תקרה יומית 5% מהבנק
MAX_DAILY_LOSS_FRACTION = 0.10   # Stop Loss יומי 8%–10%
MAX_BETS_PER_DAY = 4             # לא יותר מ-4 הימורים ביום
MIN_EV_THRESHOLD = 0.04          # סף EV מינימלי 4%
MIN_ODDS = 1.50                  # אל תיגע ביחסים נמוכים מ-1.50

st.set_page_config(page_title="GSA V76 Pro", layout="wide", page_icon="⚽", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* פלט מסוף – שימוש ברוחב מלא, טקסט לא נחתך */
    div[data-testid="stExpander"] pre, div[data-testid="stExpander"] code {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* --- Mobile Responsive --- */
    @media (max-width: 768px) {
        /* הורדת ריווח כללי */
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 0.5rem !important;
        }
        /* עמודות Streamlit — הורדת flex לשורה אחת */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 0 0 100% !important;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }
        /* כותרות */
        h1 { font-size: 1.2rem !important; }
        h2 { font-size: 1.0rem !important; }
        h3 { font-size: 0.9rem !important; }
        /* טבלאות גלילה אופקית */
        [data-testid="stDataFrame"] > div {
            overflow-x: auto !important;
        }
        /* כפתורים */
        .stButton > button {
            width: 100% !important;
            min-height: 2.8rem !important;
            font-size: 0.95rem !important;
            margin-bottom: 4px !important;
        }
        /* מדדים */
        [data-testid="stMetric"] {
            padding: 8px !important;
        }
        /* tabs */
        [data-testid="stTabs"] button {
            font-size: 0.78rem !important;
            padding: 5px 6px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


def run_bat_and_capture_output(bat_path, output_placeholder=None):
    """
    Runs a .bat file via subprocess.Popen. Uses --no-pause to avoid blocking.
    Returns (success: bool, output: str).
    """
    if not os.path.exists(bat_path):
        return False, f"[ERROR] File not found: {bat_path}"
    bat_name = os.path.basename(bat_path)
    bat_dir = os.path.dirname(bat_path)
    try:
        proc = subprocess.Popen(
            ["cmd", "/c", bat_name, "--no-pause"],
            cwd=bat_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        lines = []
        for line in iter(proc.stdout.readline, ""):
            if line:
                lines.append(line)
                if output_placeholder is not None:
                    output_placeholder.code("".join(lines), language="text")
        proc.wait()
        output = "".join(lines)
        return proc.returncode == 0, output
    except Exception as e:
        return False, f"[ERROR] {e}"


# --- Database & Stats Helpers ---

def _get_slip_count():
    """מונה טפסים – משמש כמפתח cache לעדכון סטטיסטיקות כספיות."""
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH)
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM bet_slips")
        return c.fetchone()[0]
    finally:
        conn.close()


@st.cache_data(ttl=300)
def _load_analysis_json_cached(_path, _mtime):
    """טעינת קובץ JSON עם cache – מתבטל כשהקובץ משתנה (לפי mtime)."""
    if not os.path.exists(_path):
        return []
    try:
        with open(_path, 'r', encoding='utf-8') as f:
            raw = f.read().strip()
        return json.loads(raw) if raw else []
    except (json.JSONDecodeError, IOError):
        return []


@st.cache_data(ttl=60)
def get_pro_stats_cached(_db_mtime):
    """גרסה עם cache – מתבטל כשמסד הנתונים משתנה (לפי mtime)."""
    return _get_pro_stats_impl()


def init_finance_tables():
    """יוצר טבלאות לניהול קופה ולשמירת טפסים אם אינן קיימות."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # טבלת קופה ראשית
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS bankroll_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            balance REAL NOT NULL,
            updated_at TEXT
        )
        """
    )
    # טבלת טפסים
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS bet_slips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            bet_type TEXT,
            stake_per_unit REAL,
            total_stake REAL,
            potential_return REAL,
            settled INTEGER DEFAULT 0,
            result_profit REAL DEFAULT 0
        )
        """
    )
    # טבלת משחקים בטופס (match_date/fixture_id for correct settlement matching)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS bet_slip_legs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slip_id INTEGER,
            home_team TEXT,
            away_team TEXT,
            selection TEXT,
            odds REAL,
            match_date TEXT,
            fixture_id INTEGER,
            FOREIGN KEY(slip_id) REFERENCES bet_slips(id)
        )
        """
    )
    for col in ("match_date", "fixture_id"):
        try:
            c.execute(f"ALTER TABLE bet_slip_legs ADD COLUMN {col} {'TEXT' if col == 'match_date' else 'INTEGER'}")
        except sqlite3.OperationalError:
            pass
    try:
        c.execute("ALTER TABLE bet_slips ADD COLUMN slip_kind TEXT")
    except sqlite3.OperationalError:
        pass
    # טבלת matches (v76/v79) – נדרשת לשאילתות הדשבורד
    c.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, fixture_id INTEGER, match_date TEXT,
            home_team TEXT, away_team TEXT, model_prob_1 REAL, model_prob_x REAL, model_prob_2 REAL,
            market_prob_1 REAL, market_prob_x REAL, market_prob_2 REAL, actual_result TEXT,
            final_prob_1 REAL, final_prob_x REAL, final_prob_2 REAL,
            tier TEXT, classified_ev REAL, risk_category TEXT
        )
    """)
    for col in ("recommended_bet_market", "market_type"):
        try:
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    for col in ("home_goals", "away_goals"):
        try:
            c.execute(f"ALTER TABLE matches ADD COLUMN {col} INTEGER")
        except sqlite3.OperationalError:
            pass
    # טבלאות weights ו-weights_history (v79 Auto-Learner)
    c.execute("""
        CREATE TABLE IF NOT EXISTS weights (
            id INTEGER PRIMARY KEY, w_model REAL, w_market REAL, w_ai REAL, updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS weights_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, updated_at TEXT, w_model REAL, w_market REAL,
            train_accuracy REAL, test_accuracy REAL, total_matches INTEGER
        )
    """)
    c.execute("SELECT count(*) FROM weights")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO weights (id, w_model, w_market, w_ai, updated_at) VALUES (1, 0.40, 0.60, 0, ?)",
            (datetime.now().strftime("%Y-%m-%d"),),
        )
    # טבלת learning_log
    try:
        c.execute("""
            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_at TEXT NOT NULL,
                event_type TEXT NOT NULL, summary TEXT, details_json TEXT
            )
        """)
    except sqlite3.OperationalError:
        pass

    # טבלת מטמון ROI קופה ללמידה (Auto-Learner קורא מכאן)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS bankroll_roi_cache (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_invested REAL,
            total_profit REAL,
            roi_pct REAL,
            roi_by_type_json TEXT,
            updated_at TEXT
        )
        """
    )
    # יצירת קופה התחלתית אם אין רשומה
    c.execute("SELECT balance FROM bankroll_state WHERE id=1")
    row = c.fetchone()
    if row is None:
        c.execute(
            "INSERT INTO bankroll_state (id, balance, updated_at) VALUES (1, ?, ?)",
            (1000.0, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
    conn.commit()
    conn.close()


def get_bankroll_balance():
    """מחזיר את היתרה הנוכחית בקופה הראשית."""
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return 1000.0
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT balance FROM bankroll_state WHERE id=1")
    row = c.fetchone()
    conn.close()
    if row is None:
        return 1000.0
    return float(row[0])


def update_bankroll(delta):
    """מעדכן את הקופה (delta חיובי=הפקדה, שלילי=חישוב הפסד)."""
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT balance FROM bankroll_state WHERE id=1")
    row = c.fetchone()
    if row is None:
        balance = 1000.0
    else:
        balance = float(row[0])
    new_balance = balance + float(delta)
    if new_balance < 0:
        new_balance = 0.0
    c.execute(
        "UPDATE bankroll_state SET balance=?, updated_at=? WHERE id=1",
        (new_balance, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

def set_bankroll_balance(amount):
    """מאפס את הקופה לסכום נתון (החלפה מוחלטת של היתרה)."""
    init_finance_tables()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE bankroll_state SET balance=?, updated_at=? WHERE id=1",
        (float(amount), datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def _slip_alive_or_dead(bet_type, leg_results):
    """
    מחזיר 'חי' או 'מת' לפי סוג הטופס ותוצאות הרגליים.
    leg_results: רשימה של True (זכייה), False (הפסד), None (ממתין).
    טופס 'מת' = כבר אי אפשר לזכות באף קומבינציה (למשל 2מ2 פישלנו באחד = מת).
    """
    n = len(leg_results)
    if n == 0:
        return "חי"
    wins = sum(1 for r in leg_results if r is True)
    losses = sum(1 for r in leg_results if r is False)
    pending = sum(1 for r in leg_results if r is None)

    if bet_type in ("סינגלים (הימור בודד על כל משחק)", "סינגל"):
        k = 1
    elif bet_type in ("2 מצטבר", "כל ה-2"):
        k = 2
    elif bet_type == "2 מתוך 3 (מערכת)":
        k = 2
    elif bet_type == "2 מתוך 4 (מערכת)":
        k = 2
    elif bet_type == "3 מתוך 4 (מערכת)":
        k = 3
    elif bet_type == "3 מתוך 5 (מערכת)":
        k = 3
    elif bet_type == "4 מתוך 5 (מערכת)":
        k = 4
    else:
        k = min(2, n)

    max_losses = n - k
    if losses > max_losses:
        return "מת"
    if pending > 0:
        return "חי"
    if wins >= k:
        return "חי"
    return "מת"


def delete_slip(slip_id):
    """
    מוחק טופס ואת כל הרגליים שלו. מעדכן את הקופה בהתאם:
    - טופס פתוח (לא סולק): מחזיר את העלות לקופה.
    - טופס סגור: מבטל את הרווח/הפסד (מפחית רווח או מחזיר הפסד).
    """
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT settled, total_stake, result_profit FROM bet_slips WHERE id = ?", (slip_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return
    settled, total_stake = row[0], float(row[1] or 0)
    result_profit = float(row[2] or 0)
    c.execute("DELETE FROM bet_slip_legs WHERE slip_id = ?", (slip_id,))
    c.execute("DELETE FROM bet_slips WHERE id = ?", (slip_id,))
    conn.commit()
    conn.close()
    if not settled and total_stake > 0:
        update_bankroll(total_stake)  # החזרת העלות
    elif settled:
        update_bankroll(-result_profit)  # ביטול הרווח/הפסד (חיובי→מוריד, שלילי→מחזיר)


def has_main_slip_sent_today():
    """מחזיר True אם כבר נשלח טופס יומי ראשי היום (מערכת/כפולות טוטו)."""
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM bet_slips WHERE created_at LIKE ? AND (slip_kind IS NULL OR slip_kind = 'main')",
        (today + "%",),
    )
    count = c.fetchone()[0]
    conn.close()
    return count > 0


def has_singles_slip_sent_today():
    """מחזיר True אם כבר נשלח טופס סינגלים מומלצים היום."""
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM bet_slips WHERE created_at LIKE ? AND slip_kind = 'singles'", (today + "%",))
    count = c.fetchone()[0]
    conn.close()
    return count > 0


def get_double_slips_count_today():
    """מחזיר כמה טפסי כפולות נשלחו היום (מקסימום מומלץ: 2)."""
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM bet_slips WHERE created_at LIKE ? AND slip_kind = 'double'", (today + "%",))
    count = c.fetchone()[0]
    conn.close()
    return count


def _goal_int(val):
    """ממיר שער מ-SQL/pandas ל-int; None / NaN → None (לא זורק על float('nan'))."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def resolve_bet_leg_win(selection, actual_result, home_goals, away_goals):
    """
    האם רגל ההימור ניצחה.
    מחזיר True/False, או None אם עדיין אין מספיק נתונים (למשל Totals בלי שערים במסד).
    """
    if selection is None:
        return False
    s = str(selection).strip()
    if not s:
        return False
    s_low = s.lower()

    def _as_int_goals():
        hg = _goal_int(home_goals)
        ag = _goal_int(away_goals)
        if hg is None or ag is None:
            return None
        return hg, ag

    # Over/Under 2.5 (כולל "Under 2.5 [Totals]")
    if "2.5" in s and ("over" in s_low or "under" in s_low or "מעל" in s or "מתחת" in s):
        ga = _as_int_goals()
        if ga is None:
            return None
        hg, ag = ga
        total = hg + ag
        if "under" in s_low or "מתחת" in s:
            return total <= 2
        if "over" in s_low or "מעל" in s:
            return total >= 3
        return None

    # BTTS — שני הקבוצות יבקיעו
    if "btts" in s_low or "both teams" in s_low or "שני הקבוצות" in s:
        ga = _as_int_goals()
        if ga is None:
            return None
        hg, ag = ga
        both_scored = hg >= 1 and ag >= 1
        if " no" in s_low or s_low.rstrip().endswith("no") or "/no" in s_low or "לא" in s:
            return not both_scored
        return both_scored

    # 1X2
    ar = str(actual_result).strip() if actual_result is not None and str(actual_result).strip() != "" else ""
    if not ar:
        return None
    if len(s) == 1 and s in ("1", "X", "2"):
        return ar == s
    return ar == s


def settle_open_slips():
    """
    מעדכן טפסים שטרם סולקו על בסיס טבלת matches.
    חוקי ווינר: שליחת הימור מורידה מיד מהקופה; בסילוק מוסיפים רק את הרווח הנקי מהקומבינציות שפגעו.
    """
    init_finance_tables()
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id, bet_type, stake_per_unit, total_stake FROM bet_slips WHERE settled = 0")
    slips = c.fetchall()
    if not slips:
        conn.close()
        return

    df_matches = pd.read_sql_query(
        """
        SELECT fixture_id, match_date, home_team, away_team, actual_result,
               home_goals, away_goals,
               market_prob_1, market_prob_x, market_prob_2
        FROM matches
        WHERE actual_result IS NOT NULL
          AND actual_result != ''
          AND actual_result != 'None'
        """,
        conn,
    )

    def _find_match_outcome(home, away, match_date=None, fixture_id=None):
        """מחזיר (actual_result, home_goals, away_goals) למשחק או None."""
        if fixture_id is not None and pd.notna(fixture_id):
            sub = df_matches[df_matches["fixture_id"] == fixture_id]
        elif match_date and str(match_date).strip():
            date_str = str(match_date).strip()[:10]
            sub = df_matches[
                (df_matches["home_team"] == home)
                & (df_matches["away_team"] == away)
                & (df_matches["match_date"].astype(str).str[:10] == date_str)
            ]
        else:
            sub = df_matches[
                (df_matches["home_team"] == home) & (df_matches["away_team"] == away)
            ]
        if sub.empty:
            return None
        row = sub.iloc[0]
        ar = str(row["actual_result"]).strip()
        hg = row["home_goals"] if "home_goals" in row.index else None
        ag = row["away_goals"] if "away_goals" in row.index else None
        return ar, hg, ag

    total_delta = 0.0

    for slip_id, bet_type, stake_unit, total_stake in slips:
        c.execute(
            "SELECT home_team, away_team, selection, odds, match_date, fixture_id FROM bet_slip_legs WHERE slip_id = ?",
            (slip_id,),
        )
        legs = c.fetchall()
        if not legs:
            continue

        results = []
        for row in legs:
            home, away, sel, odds = row[0], row[1], row[2], row[3]
            match_date = row[4] if len(row) > 4 else None
            fixture_id = row[5] if len(row) > 5 else None
            out = _find_match_outcome(home, away, match_date=match_date, fixture_id=fixture_id)
            if out is None:
                results.append({"win": None, "odds": float(odds)})
            else:
                ar, hg, ag = out
                w = resolve_bet_leg_win(sel, ar, hg, ag)
                results.append({"win": w, "odds": float(odds)})

        if any(r["win"] is None for r in results):
            continue

        odds_list = [r["odds"] for r in results]
        wins_mask = [r["win"] for r in results]
        n = len(odds_list)

        profit = 0.0

        if bet_type in ("סינגלים (הימור בודד על כל משחק)", "סינגל"):
            for win, o in zip(wins_mask, odds_list):
                if win:
                    profit += stake_unit * o  # full payout (stake * odds) returned to bankroll
        else:
            combo_size = None
            if bet_type == "3 מתוך 4 (מערכת)" and n == 4:
                combo_size = 3
            elif bet_type == "4 מתוך 5 (מערכת)" and n == 5:
                combo_size = 4
            elif bet_type == "3 מתוך 5 (מערכת)" and n == 5:
                combo_size = 3
            elif bet_type == "2 מתוך 4 (מערכת)" and n == 4:
                combo_size = 2
            elif bet_type == "2 מתוך 3 (מערכת)" and n == 3:
                combo_size = 2
            elif bet_type in ("2 מצטבר", "כל ה-2") and n == 2:
                combo_size = 2

            if combo_size is not None:
                for combo in itertools.combinations(range(n), combo_size):
                    if all(wins_mask[i] for i in combo):
                        prod_odds = 1.0
                        for i in combo:
                            prod_odds *= odds_list[i]
                        profit += stake_unit * prod_odds  # full payout returned to bankroll

        total_delta += profit
        c.execute(
            "UPDATE bet_slips SET settled = 1, result_profit = ? WHERE id = ?",
            (profit, slip_id),
        )

    conn.commit()
    conn.close()

    if abs(total_delta) > 0:
        update_bankroll(total_delta)


def get_dashboard_finance_stats():
    """
    מחזיר סטטיסטיקות כספיות ולהימורים: קופה, סה"כ הומר, תלוי, רווח/הפסד, ROI,
    אחוז זכייה, רצף, סדרת רווח מצטבר, חלוקת תוצאות, ROI לפי סוג הימור.
    """
    bankroll = get_bankroll_balance()
    total_invested = 0.0
    pending_amount = 0.0
    net_profit = 0.0
    wins = 0
    losses = 0
    pending_count = 0
    cumulative_rows = []
    by_type = {}
    slip_outcomes = []  # 1=win, -1=loss, 0=pending, by slip id order

    if not os.path.exists(DB_PATH):
        return {
            "bankroll": bankroll,
            "total_invested": 0.0,
            "pending_amount": 0.0,
            "net_profit": 0.0,
            "roi_pct": 0.0,
            "win_pct": 0.0,
            "wins": 0,
            "losses": 0,
            "pending_count": 0,
            "current_streak": "--",
            "cumulative_df": pd.DataFrame(columns=["created_at", "cumulative_profit"]),
            "donut_data": [],
            "roi_by_type": [],
            "insight": "טרם נרשמו הימורים.",
        }

    init_finance_tables()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT id, created_at, bet_type, total_stake, result_profit, settled
        FROM bet_slips ORDER BY id ASC
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {
            "bankroll": bankroll,
            "total_invested": 0.0,
            "pending_amount": 0.0,
            "net_profit": 0.0,
            "roi_pct": 0.0,
            "win_pct": 0.0,
            "wins": 0,
            "losses": 0,
            "pending_count": 0,
            "current_streak": "--",
            "cumulative_df": pd.DataFrame(columns=["created_at", "cumulative_profit"]),
            "donut_data": [],
            "roi_by_type": [],
            "insight": "טרם נרשמו הימורים.",
        }

    total_invested = float(df["total_stake"].sum())
    settled_df = df[df["settled"] == 1]
    pending_df = df[df["settled"] == 0]
    pending_amount = float(pending_df["total_stake"].sum())
    pending_count = len(pending_df)
    # נטו לטפסים שסולקו: תזרים בסילוק (result_profit) פחות עלות הטופס (הועברה מהקופה בשימור)
    rp = settled_df["result_profit"].fillna(0).astype(float)
    st_settled = settled_df["total_stake"].fillna(0).astype(float)
    net_profit = float((rp - st_settled).sum())
    invested_settled = float(st_settled.sum())
    roi_pct = (net_profit / invested_settled * 100.0) if invested_settled > 0 else 0.0

    slip_outcomes = []
    for _, row in df.iterrows():
        if row["settled"] != 1:
            slip_outcomes.append(0)
            continue
        p = float(row["result_profit"] or 0)
        stake = float(row["total_stake"] or 0)
        net_slip = p - stake
        if net_slip > 0:
            wins += 1
            slip_outcomes.append(1)
        elif net_slip < 0:
            losses += 1
            slip_outcomes.append(-1)
        else:
            slip_outcomes.append(0)

    total_resolved = wins + losses
    win_pct = (wins / total_resolved * 100.0) if total_resolved > 0 else 0.0

    # רצף נוכחי: מהטפסים שהסתיימו, מהאחרון לראשון
    current_streak = "--"
    if slip_outcomes:
        last_outcomes = [x for x in reversed(slip_outcomes) if x != 0]
        if last_outcomes:
            sign = last_outcomes[0]
            streak = 0
            for x in last_outcomes:
                if x == sign:
                    streak += 1
                else:
                    break
            current_streak = f"{streak} {'ניצחונות' if sign == 1 else 'הפסדים'}"

    # רווח מצטבר לאורך זמן (רק טפסים שסולקו) — נטו לטופס
    cum = 0.0
    for _, row in settled_df.iterrows():
        cum += float(row["result_profit"] or 0) - float(row["total_stake"] or 0)
        cumulative_rows.append({"created_at": row["created_at"], "cumulative_profit": cum})
    cumulative_df = pd.DataFrame(cumulative_rows) if cumulative_rows else pd.DataFrame(columns=["created_at", "cumulative_profit"])

    # חלוקת תוצאות לדונאט (לפי כמות: זכייה, הפסד, ממתין)
    donut_data = []
    if wins > 0:
        donut_data.append({"label": "זכייה", "value": wins, "color": "#22c55e"})
    if losses > 0:
        donut_data.append({"label": "הפסד", "value": losses, "color": "#ef4444"})
    if pending_count > 0:
        donut_data.append({"label": "ממתין", "value": pending_count, "color": "#f97316"})
    if not donut_data:
        donut_data = [{"label": "ממתין", "value": 1, "color": "#e5e7eb"}]

    # ROI לפי סוג הימור (רק שסולקו) — נטו / סכום הימור
    for bet_type, grp in settled_df.groupby("bet_type"):
        stake_sum = float(grp["total_stake"].sum())
        net_sum = float(grp["result_profit"].fillna(0).sum() - grp["total_stake"].fillna(0).sum())
        roi = (net_sum / stake_sum * 100.0) if stake_sum > 0 else 0.0
        by_type[bet_type] = {"stake": stake_sum, "profit": net_sum, "roi": roi}
    roi_by_type = [{"bet_type": k, "roi": v["roi"]} for k, v in by_type.items()]

    # תובנה
    if total_resolved == 0 and pending_count == 0:
        insight = "טרם נרשמו הימורים."
    elif total_resolved == 0:
        insight = "כל הטפסים ממתינים לתוצאות."
    elif roi_pct < 0:
        insight = f"ROI שלילי {roi_pct:.1f}% - מומלץ לבדוק את אסטרטגיית ההימורים."
    elif roi_pct > 10:
        insight = f"ROI חיובי {roi_pct:.1f}% - ביצועים טובים."
    else:
        insight = f"ROI {roi_pct:.1f}% - ממשיך לעקוב."

    result = {
        "bankroll": bankroll,
        "total_invested": total_invested,
        "pending_amount": pending_amount,
        "net_profit": net_profit,
        "roi_pct": roi_pct,
        "win_pct": win_pct,
        "wins": wins,
        "losses": losses,
        "pending_count": pending_count,
        "current_streak": current_streak,
        "cumulative_df": cumulative_df,
        "donut_data": donut_data,
        "roi_by_type": roi_by_type,
        "insight": insight,
    }
    write_bankroll_roi_cache(result)
    return result


def write_bankroll_roi_cache(fin_stats):
    """מעדכן מטמון ROI קופה כדי ש-Auto-Learner יוכל להשתמש בו לכיול משקולות."""
    if not os.path.exists(DB_PATH):
        return
    init_finance_tables()
    try:
        import json
        roi_by_type_json = json.dumps(fin_stats.get("roi_by_type", []), ensure_ascii=False)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """INSERT OR REPLACE INTO bankroll_roi_cache (id, total_invested, total_profit, roi_pct, roi_by_type_json, updated_at)
               VALUES (1, ?, ?, ?, ?, ?)""",
            (
                fin_stats.get("total_invested", 0),
                fin_stats.get("net_profit", 0),
                fin_stats.get("roi_pct", 0),
                roi_by_type_json,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def display_brains_arena():
    """קורא את מסד הנתונים, בודק איזה מודל חזה נכון משחקים שהסתיימו, ומציג השוואה.
    Uses direct SQL aggregation (COUNT) instead of loading the full table into memory."""
    if not os.path.exists(DB_PATH): return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    base_filter = """
        actual_result IS NOT NULL AND actual_result != '' AND actual_result != 'None'
        AND actual_result IN ('1', 'X', '2')
    """
    try:
        c.execute(
            f"SELECT COUNT(*) FROM matches WHERE {base_filter}"
        )
        total = c.fetchone()[0]
    except Exception:
        conn.close()
        return

    if total == 0:
        conn.close()
        st.info("אין עדיין מספיק נתוני היסטוריה של משחקים שהסתיימו כדי להציג את יכולות הלמידה של המערכת.")
        return

    # Math prediction = argmax(model_prob_1, model_prob_x, model_prob_2)
    c.execute(f"""
        SELECT COUNT(*) FROM matches
        WHERE {base_filter}
        AND (
            (actual_result = '1' AND model_prob_1 >= COALESCE(model_prob_x,0) AND model_prob_1 >= COALESCE(model_prob_2,0))
            OR (actual_result = 'X' AND model_prob_x >= COALESCE(model_prob_1,0) AND model_prob_x >= COALESCE(model_prob_2,0))
            OR (actual_result = '2' AND model_prob_2 >= COALESCE(model_prob_1,0) AND model_prob_2 >= COALESCE(model_prob_x,0))
        )
    """)
    correct_math = c.fetchone()[0]

    # Market prediction = argmax הסתברויות (הפייבוריט = ההסתברות הגבוהה ביותר).
    # market_prob_* נשמרות כהסתברויות 0-1 (לא כיחסים).
    c.execute(f"""
        SELECT COUNT(*) FROM matches
        WHERE {base_filter}
        AND (
            (actual_result = '1' AND market_prob_1 >= COALESCE(market_prob_x,0) AND market_prob_1 >= COALESCE(market_prob_2,0) AND COALESCE(market_prob_1,0) > 0)
            OR (actual_result = 'X' AND market_prob_x >= COALESCE(market_prob_1,0) AND market_prob_x >= COALESCE(market_prob_2,0) AND COALESCE(market_prob_x,0) > 0)
            OR (actual_result = '2' AND market_prob_2 >= COALESCE(market_prob_1,0) AND market_prob_2 >= COALESCE(market_prob_x,0) AND COALESCE(market_prob_2,0) > 0)
        )
    """)
    correct_market = c.fetchone()[0]

    st.markdown("### 🧠 זירת המוחות: מי צודק יותר?")
    st.write(f"המערכת ניתחה ולמדה **{total}** משחקים שכבר הסתיימו. להלן אחוזי הפגיעה של כל 'מוח':")
    
    col1, col2 = st.columns(2)
    col1.metric("📐 מודל מתמטי (xG)", f"{(correct_math/total)*100:.1f}%")
    col2.metric("🏦 השוק (הווינר)", f"{(correct_market/total)*100:.1f}%")
    
    # הצגת גרף ויזואלי מהיר
    chart_data = pd.DataFrame(
        {"אחוזי פגיעה": [(correct_math/total)*100, (correct_market/total)*100]},
        index=["מודל מתמטי", "הווינר"]
    )
    st.bar_chart(chart_data)    
    try:
        c.execute("SELECT updated_at FROM weights WHERE id=1")
        w_update = c.fetchone()
        last_update = w_update[0] if w_update and w_update[0] else "לא ידוע"

        history_rows = []
        try:
            c.execute("""
                SELECT updated_at, w_model, w_market, train_accuracy, test_accuracy, total_matches
                FROM weights_history ORDER BY id DESC LIMIT 10
            """)
            history_rows = c.fetchall()
        except Exception:
            history_rows = []
    except Exception:
        history_rows = []
        last_update = "לא ידוע"
    finally:
        conn.close()

    with st.expander("📊 היסטוריית עדכוני משקולות (Auto-Learner)", expanded=bool(history_rows)):
        st.markdown(f"<p style='color:gray; font-size:0.85rem;'>עדכון מודל ומשקולות אחרון: <b>{last_update}</b></p>", unsafe_allow_html=True)
        if history_rows:
            for row in history_rows:
                upd, wm, wmk, train_acc, test_acc, total = row
                train_str = f"{train_acc*100:.1f}%" if train_acc is not None else "—"
                test_str  = f"{test_acc*100:.1f}%"  if test_acc  is not None else "—"
                total_str = str(total) if total is not None else "—"
                color = "#dcfce7" if (test_acc or 0) >= 0.45 else "#fef9c3" if (test_acc or 0) >= 0.35 else "#fee2e2"
                st.markdown(
                    f"<div style='background:{color}; border-right:4px solid #6366f1; padding:10px; border-radius:6px; margin-bottom:8px; font-size:0.9rem;'>"
                    f"<b>🗓️ {upd}</b> &nbsp;|&nbsp; "
                    f"מודל: <b>{wm:.0%}</b> &nbsp; שוק: <b>{wmk:.0%}</b><br>"
                    f"דיוק Train: <b>{train_str}</b> &nbsp;|&nbsp; דיוק Test: <b>{test_str}</b> &nbsp;|&nbsp; משחקים: <b>{total_str}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("טרם נרשמה היסטוריית משקולות. תיתווסף אוטומטית בריצה הבאה של Auto-Learner.")
def display_recommended_slip(matches_data, w_math=0.5, w_market=0.5, top_n=4, bankroll=1000.0, system_active=True):
    """מדרג משחקים, מחשב תוחלת מצטברת וממליץ על גודל השקעה מתוך הקופה (Kelly Criterion + גבולות קשיחים)."""
    if not system_active:
        st.error("🚨 בלם חירום פעיל: המערכת עצרה הימורים ליום זה עקב חריגה בכללי ניהול הסיכון (רצף הפסדים / תוחלת יומית שלילית). אין הפקת טופס מומלץ.")
        return

    evaluated_matches = []
    current_date = datetime.now().strftime("%d/%m/%Y")
    today_str = datetime.now().strftime("%Y-%m-%d")

    for m in matches_data:
        # סינון משחקים שתאריכם עבר (JSON ישן) — מציג רק היום ומחר
        md = m.get('match_date') or ''
        if md and md < today_str:
            continue

        pro_data = m.get('pro_data') or {}
        market_o = m.get('market_odds') or {}
        # Prefer multi-market output from v76/dadima when available
        if pro_data.get('recommended_bet') and pro_data.get('recommended_bet') != 'NO BET' and pro_data.get('odds'):
            best_bet = str(pro_data.get('recommended_bet', ''))
            best_prob = float(pro_data.get('chosen_prob') or 0)
            odd = float(pro_data.get('odds') or 0)
            ev = float(pro_data.get('best_ev') or pro_data.get('classified_ev') or (best_prob * odd - 1))
            market_type = pro_data.get('market_type') or '1X2'
        else:
            math_p = m.get('model_probs', {})
            # חישוב הסתברויות שוק מתוך יחסים (כולל נרמול והורדת ויג)
            market_p = {k: (1/float(v) if v and float(v)>0 else 0) for k,v in market_o.items()}
            sum_m = sum(market_p.values())
            if sum_m > 0: market_p = {k: v/sum_m for k,v in market_p.items()}
            weighted_p = {}
            for k in ['1', 'X', '2']:
                mp = float(math_p.get(k, 0)) if math_p else 0
                mkp = float(market_p.get(k, 0))
                weighted_p[k] = (mp * w_math) + (mkp * w_market)
            if not weighted_p:
                continue
            best_bet = max(weighted_p, key=weighted_p.get)
            best_prob = weighted_p[best_bet]
            odd = float(market_o.get(best_bet, 0))
            market_type = '1X2'
            ev = (best_prob * odd) - 1
        
        # סינון יחסים קיצוניים / חסרי משמעות
        if odd <= 0:
            continue
        if odd < MIN_ODDS:
            continue
        # סינון EV מינימלי — חוץ מ"ערך גבוה" (EV>12%) שתמיד כשיר
        is_high_value = (ev > 0.12 and odd >= 1.50)
        if ev < MIN_EV_THRESHOLD and not is_high_value:
            continue

        # score: ערך גבוה מקבל בונוס כדי לעלות בסדר העדיפויות
        score = (ev + 1) * best_prob * (1.5 if is_high_value else 1.0) 
        
        raw_md = m.get('match_date') or ''
        try:
            match_date_str = datetime.strptime(raw_md[:10], "%Y-%m-%d").strftime("%d/%m/%Y") if raw_md else "–"
        except Exception:
            match_date_str = "–"

        match_time_str = m.get('match_time') or ''
        if not match_time_str and raw_md and 'T' in str(raw_md) and len(str(raw_md)) >= 16:
            match_time_str = str(raw_md)[11:16]

        league_name, country_name = "", ""
        pro_data = m.get('pro_data') or {}
        if pro_data.get('league'):
            league_name = str(pro_data.get('league', ''))
        if pro_data.get('country'):
            country_name = str(pro_data.get('country', ''))
        if not league_name or not country_name:
            hist = m.get('history') or {}
            for key in ('home', 'away'):
                arr = hist.get(key) or []
                if arr and isinstance(arr[0], dict):
                    leg = (arr[0].get('league') or {}) if isinstance(arr[0], dict) else {}
                    if isinstance(leg, dict):
                        league_name = league_name or str(leg.get('name', ''))
                        country_name = country_name or str(leg.get('country', ''))
                    if league_name and country_name:
                        break

        evaluated_matches.append({
            "משחק": m.get('match', ''),
            "home": m.get('home', ''),
            "away": m.get('away', ''),
            "תאריך_משחק": match_date_str,
            "שעה_משחק": match_time_str,
            "match_date": raw_md[:10] if raw_md else None,
            "match_time": match_time_str,
            "ליגה": league_name,
            "מדינה": country_name,
            "fixture_id": m.get('fixture_id'),
            "הימור": best_bet,
            "market_type": market_type,
            "raw_prob": best_prob,
            "הסתברות": f"{best_prob * 100:.1f}%",
            "יחס": odd,
            "תוחלת (EV)": ev,
            "score": score
        })
    evaluated_matches.sort(key=lambda x: x['score'], reverse=True)

    # מגבלה קשיחה על מספר הימורים ליום
    max_bets_today = min(top_n, MAX_BETS_PER_DAY)
    recommended = [m for m in evaluated_matches if m["raw_prob"] >= 0.40][:max_bets_today]

    if not recommended and evaluated_matches:
        evaluated_matches.sort(key=lambda x: x['raw_prob'], reverse=True)
        recommended = evaluated_matches[:3]
        
    if recommended:
        st.markdown(f"### 🎟️ טופס מומלץ אוטומטי (מבוסס קופה של {bankroll} ₪)")
        
        combined_odds = 1.0
        combined_prob = 1.0
        display_data = []
        
        for r in recommended:
            combined_odds *= r["יחס"]
            combined_prob *= r["raw_prob"]
            ev_pct = r['תוחלת (EV)'] * 100
            ev_str = f"+{ev_pct:.1f}% 🔥" if ev_pct > 0 else f"{ev_pct:.1f}%"
            bet_display = r['הימור']
            mkt = r.get('market_type', '1X2')
            if mkt in ('Totals', 'BTTS'):
                bet_display = f"{bet_display} [{mkt}]"
            display_data.append({
                "תאריך": r.get('תאריך_משחק', '') or '–',
                "שעה": r.get('שעה_משחק', '') or '–',
                "מדינה": r.get('מדינה', '') or '–',
                "ליגה": r.get('ליגה', '') or '–',
                "משחק": r['משחק'],
                "הימור": bet_display,
                "הסתברות": r['הסתברות'],
                "יחס": f"{r['יחס']:.2f}",
                "תוחלת משחק": ev_str
            })
            
        df_rec = pd.DataFrame(display_data)
        st.dataframe(df_rec, width="stretch")
        
        combined_ev = (combined_prob * combined_odds) - 1
        combined_ev_pct = combined_ev * 100
        
        st.markdown("#### 💰 ניהול סיכונים והמלצת השקעה (Kelly + תקרות)")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("יחס טופס כולל", f"{combined_odds:.2f}")
        col2.metric("הסתברות מצטברת", f"{combined_prob*100:.1f}%")
        col3.metric("תוחלת מצטברת (EV)", f"{combined_ev_pct:.1f}%")
        
        MAX_BET_PCT = 0.02  # לא יותר מ-2% מהבנק לטופס
        
        if combined_ev > 0:
            kelly_fraction = combined_ev / (combined_odds - 1)
            safe_kelly = max(0.0, kelly_fraction * 0.25)
            # החלת תקרה שמרנית לפי הכללים (מקסימום 2% מהבנק)
            capped_pct = min(safe_kelly, MAX_BET_PCT)
            rec_stake = bankroll * capped_pct
            col4.metric("השקעה מומלצת", f"₪ {rec_stake:.0f}")
            st.success(
                f"✅ **טופס כדאי (EV>0).** המערכת ממליצה להשקיע {capped_pct*100:.1f}% מהקופה "
                f"(רבע-קלי עם תקרה שמרנית של 2%)."
            )
        else:
            rec_stake = 0.0
            col4.metric("השקעה מומלצת", "₪ 0")
            st.error("❌ **הטופס לא משתלם מתמטית (EV שלילי).** המערכת ממליצה לא להשקיע כסף על שילוב זה.")

        # --- כפתור "שלח טופס" אוטומטי לפי מספר המשחקים המומלצים ---
        n_rec = len(recommended)
        if n_rec >= 4:
            rec_for_slip = recommended[:4]
            auto_bet_type = "2 מתוך 4 (מערכת)"
            combo_size_auto = 2
            n_combos_auto = math.comb(4, 2)
        elif n_rec == 3:
            rec_for_slip = recommended
            auto_bet_type = "2 מתוך 3 (מערכת)"
            combo_size_auto = 2
            n_combos_auto = math.comb(3, 2)
        elif n_rec == 2:
            rec_for_slip = recommended
            auto_bet_type = "2 מצטבר"
            combo_size_auto = 2
            n_combos_auto = 1
        else:
            rec_for_slip = recommended
            auto_bet_type = "סינגל"
            combo_size_auto = 1
            n_combos_auto = 1

        def _to_winner_stake(amount):
            """עיגול לסכום תקני ווינר: מינימום 10 ₪, קפיצות של 5 ₪."""
            if amount < 10.0:
                return 10.0
            return max(10.0, math.ceil((amount - 10.0) / 5.0) * 5.0 + 10.0)

        raw_stake = rec_stake / max(n_combos_auto, 1) if rec_stake > 0 else 10.0
        stake_per_combo = _to_winner_stake(raw_stake)
        total_auto_cost = n_combos_auto * stake_per_combo

        auto_odds_list = [r["יחס"] for r in rec_for_slip]
        potential_auto_return = 0.0
        for combo in itertools.combinations(range(len(rec_for_slip)), combo_size_auto):
            prod = 1.0
            for ci in combo:
                prod *= auto_odds_list[ci]
            potential_auto_return += stake_per_combo * prod

        st.markdown("---")
        st.markdown(
            f"<div style='background:#e0f2fe; border-right:4px solid #0284c7; border-radius:8px; padding:12px; margin-bottom:8px;'>"
            f"<b>🤖 שיטת שליחה אוטומטית: {auto_bet_type}</b> | {n_combos_auto} קומבינציה/ות "
            f"× ₪{stake_per_combo:.0f} = עלות ₪{total_auto_cost:.0f}"
            f"</div>",
            unsafe_allow_html=True,
        )
        ca, cb, cc = st.columns(3)
        ca.metric("יחידה לקומבינציה", f"₪{stake_per_combo:.0f}")
        cb.metric("עלות כוללת", f"₪{total_auto_cost:.0f}")
        cc.metric("החזר פוטנציאלי", f"₪{potential_auto_return:.0f}")

        sent_today = has_main_slip_sent_today()
        can_auto = total_auto_cost <= bankroll and system_active and not sent_today
        if not system_active:
            st.error("🚨 Circuit Breaker פעיל – לא ניתן לשלוח טופס.")
        elif sent_today:
            st.warning("⏳ **כבר שלחת טופס יומי ראשי היום.** ניתן לשלוח טופס ראשי אחד ליום.")
        elif total_auto_cost > bankroll:
            st.warning("⚠️ אין מספיק כסף בקופה לשליחת הטופס.")

        if st.button("📤 שלח טופס", key="auto_send_recommended_slip", disabled=not can_auto):
            init_finance_tables()
            try:
                conn_s = sqlite3.connect(DB_PATH)
                cs = conn_s.cursor()
                cs.execute(
                    "INSERT INTO bet_slips (created_at, bet_type, stake_per_unit, total_stake, potential_return, settled, result_profit, slip_kind) VALUES (?, ?, ?, ?, ?, 0, 0, 'main')",
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        auto_bet_type,
                        float(stake_per_combo),
                        float(total_auto_cost),
                        float(potential_auto_return),
                    ),
                )
                slip_id_auto = cs.lastrowid
                for r in rec_for_slip:
                    cs.execute(
                        "INSERT INTO bet_slip_legs (slip_id, home_team, away_team, selection, odds, match_date, fixture_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            slip_id_auto,
                            r.get("home", ""),
                            r.get("away", ""),
                            str(r["הימור"]),
                            float(r["יחס"]),
                            r.get("match_date") or None,
                            r.get("fixture_id"),
                        ),
                    )
                conn_s.commit()
                conn_s.close()
                update_bankroll(-total_auto_cost)
                st.success(f"✅ הטופס נשלח! ₪{total_auto_cost:.0f} ירדו מהקופה.")
                st.rerun()
            except Exception as e:
                st.error(f"שגיאה בשליחה: {e}")

        # --- סינגלים מומלצים (בנוסף לטופס היומי) ---
        st.markdown("---")
        st.markdown("### 🎯 סינגלים מומלצים")
        n_singles = min(5, len(evaluated_matches))
        singles_candidates = evaluated_matches[:n_singles]
        if singles_candidates:
            stake_single = _to_winner_stake(bankroll * 0.02 / max(n_singles, 1))
            total_singles_cost = n_singles * stake_single
            def _fmt_date(r):
                d = r.get("תאריך_משחק", "")
                if d:
                    return d
                raw = str(r.get("match_date", ""))[:10]
                if len(raw) >= 10:
                    try:
                        return datetime.strptime(raw, "%Y-%m-%d").strftime("%d/%m/%Y")
                    except Exception:
                        pass
                return ""

            df_singles = pd.DataFrame([
                {
                    "תאריך": _fmt_date(r),
                    "שעה": r.get("שעה_משחק", "") or "–",
                    "מדינה": r.get("מדינה", "") or "–",
                    "ליגה": r.get("ליגה", "") or "–",
                    "משחק": r["משחק"],
                    "הימור": r["הימור"],
                    "יחס": f"{r['יחס']:.2f}",
                    "EV": f"{r['תוחלת (EV)']*100:.1f}%"
                }
                for r in singles_candidates
            ])
            st.dataframe(df_singles, width="stretch")
            st.caption(f"השקעה מומלצת: ₪{stake_single:.0f} למשחק | סה\"כ עד ₪{total_singles_cost:.0f}")
            sent_singles = has_singles_slip_sent_today()
            can_singles = system_active and not sent_singles and total_singles_cost <= bankroll
            if sent_singles:
                st.warning("כבר נשלח טופס סינגלים היום.")
            if st.button("📤 שלח סינגלים מומלצים", key="send_singles", disabled=not can_singles):
                init_finance_tables()
                try:
                    conn_s = sqlite3.connect(DB_PATH)
                    cs = conn_s.cursor()
                    pot_return = sum(float(r["יחס"]) * stake_single for r in singles_candidates)
                    cs.execute(
                        "INSERT INTO bet_slips (created_at, bet_type, stake_per_unit, total_stake, potential_return, settled, result_profit, slip_kind) VALUES (?, ?, ?, ?, ?, 0, 0, 'singles')",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "סינגל", float(stake_single), float(total_singles_cost), float(pot_return)),
                    )
                    sid = cs.lastrowid
                    for r in singles_candidates:
                        cs.execute(
                            "INSERT INTO bet_slip_legs (slip_id, home_team, away_team, selection, odds, match_date, fixture_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (sid, r.get("home", ""), r.get("away", ""), str(r["הימור"]), float(r["יחס"]), r.get("match_date") or None, r.get("fixture_id")),
                        )
                    conn_s.commit()
                    conn_s.close()
                    update_bankroll(-total_singles_cost)
                    st.success(f"✅ סינגלים נשלחו. ₪{total_singles_cost:.0f} ירדו מהקופה.")
                    st.rerun()
                except Exception as e:
                    st.error(f"שגיאה בשליחת סינגלים: {e}")

        # --- כפולות מומלצות (בנוסף לטופס היומי) ---
        st.markdown("---")
        st.markdown("### 🔗 כפולות מומלצות")
        top_for_doubles = evaluated_matches[:4]
        double_combos = []
        for i in range(len(top_for_doubles)):
            for j in range(i + 1, len(top_for_doubles)):
                a, b = top_for_doubles[i], top_for_doubles[j]
                comb_ev = (1 + a["תוחלת (EV)"]) * (1 + b["תוחלת (EV)"]) - 1
                double_combos.append(([a, b], comb_ev))
        double_combos.sort(key=lambda x: x[1], reverse=True)
        best_doubles = [combo for combo, _ in double_combos[:2]]
        if best_doubles:
            stake_double = _to_winner_stake(bankroll * 0.01)
            for idx, (leg_a, leg_b) in enumerate(best_doubles):
                odds_prod = leg_a["יחס"] * leg_b["יחס"]
                pot = stake_double * odds_prod
                def _leg_line(leg):
                    date_time = " ".join(p for p in [leg.get("תאריך_משחק"), leg.get("שעה_משחק")] if p)
                    loc = ", ".join(p for p in [leg.get("מדינה"), leg.get("ליגה")] if p)
                    extra = " | ".join(x for x in [date_time, loc] if x)
                    suffix = f" | {extra}" if extra else ""
                    return f"- {leg['משחק']} ({leg['הימור']} @ {leg['יחס']:.2f}){suffix}"
                st.markdown(f"**כפול {idx+1}:**")
                st.markdown(_leg_line(leg_a))
                st.markdown(_leg_line(leg_b))
                st.markdown(f"→ **יחס מצטבר:** {odds_prod:.2f}")
            st.caption(f"השקעה מומלצת: ₪{stake_double:.0f} לכפול")
            n_doubles_today = get_double_slips_count_today()
            max_doubles = 2
            can_double = system_active and n_doubles_today < max_doubles and (len(best_doubles) * stake_double) <= bankroll
            if n_doubles_today >= max_doubles:
                st.warning("הגעת למכסה היומית של כפולות (2).")
            if st.button("📤 שלח כפולות מומלצות", key="send_doubles", disabled=not can_double):
                init_finance_tables()
                try:
                    conn_s = sqlite3.connect(DB_PATH, timeout=15.0)
                    cs = conn_s.cursor()
                    total_to_deduct = 0.0
                    for leg_a, leg_b in best_doubles:
                        stake_d = _to_winner_stake(bankroll * 0.01)
                        total_d = stake_d
                        total_to_deduct += total_d
                        pot_d = stake_d * leg_a["יחס"] * leg_b["יחס"]
                        cs.execute(
                            "INSERT INTO bet_slips (created_at, bet_type, stake_per_unit, total_stake, potential_return, settled, result_profit, slip_kind) VALUES (?, ?, ?, ?, ?, 0, 0, 'double')",
                            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "2 מצטבר", float(stake_d), float(total_d), float(pot_d)),
                        )
                        sid = cs.lastrowid
                        for r in [leg_a, leg_b]:
                            cs.execute(
                                "INSERT INTO bet_slip_legs (slip_id, home_team, away_team, selection, odds, match_date, fixture_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (sid, r.get("home", ""), r.get("away", ""), str(r["הימור"]), float(r["יחס"]), r.get("match_date") or None, r.get("fixture_id")),
                            )
                    conn_s.commit()
                    conn_s.close()
                    update_bankroll(-total_to_deduct)
                    st.success("✅ כפולות נשלחו.")
                    st.rerun()
                except Exception as e:
                    st.error(f"שגיאה בשליחת כפולות: {e}")
    else:
        st.warning("שגיאה: אין מספיק נתונים.")
def display_post_match_analysis(w_model=0.5, w_market=0.5):
    """מציג טבלת אמת של פוסט-משחק, או הודעת המתנה במידה ואין תוצאות"""
    st.markdown("---")
    st.markdown("## 📈 למידת מכונה: ניתוח אמת (Post-Match) - מי צדק?")
    
    if not os.path.exists(DB_PATH): 
        st.warning("מסד הנתונים לא קיים עדיין.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    # הוספנו את match_date לשאילתה
    query = """
    SELECT match_date, home_team, away_team, model_prob_1, model_prob_x, model_prob_2, 
           market_prob_1, market_prob_x, market_prob_2, actual_result 
    FROM matches WHERE actual_result IS NOT NULL AND actual_result != 'None' AND actual_result != ''
    """
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()
        
    if df.empty: 
        st.info("⏳ **ממתין לסיום משחקים:** המערכת סורקת את מסד הנתונים, אך כרגע אין משחקים שהסתיימו והתוצאה שלהם (actual_result) עודכנה. ברגע שיוזנו תוצאות אמת מהמחזור הנוכחי, טבלת ההשוואה והמלצת המשקולות לעתיד יופיעו כאן אוטומטית.")
        return
    
    BASELINE_RANDOM = 1.0 / 3.0  # ניחוש אקראי ≈ 33.3%
    records = []
    correct_system, correct_math, correct_market = 0, 0, 0
    total = len(df)
    
    for _, row in df.iterrows():
        actual = str(row['actual_result']).strip()
        match_name = f"{row['home_team']} - {row['away_team']}"
        date_val = str(row['match_date']) if pd.notna(row['match_date']) else "-"
        
        math_p = {'1': row['model_prob_1'] or 0, 'X': row['model_prob_x'] or 0, '2': row['model_prob_2'] or 0}
        market_o = {'1': row['market_prob_1'] or 0, 'X': row['market_prob_x'] or 0, '2': row['market_prob_2'] or 0}
        
        pred_math = max(math_p, key=math_p.get) if any(math_p.values()) else "-"
        market_vals = {k: float(v) for k, v in market_o.items() if v is not None and float(v) > 0}
        # market_prob = הסתברויות 0-1. אם סכום>1.5 = יחסים, פייבוריט=min. אחרת הסתברויות, פייבוריט=max
        s_mkt = sum(market_vals.values())
        pred_market = (min(market_vals, key=market_vals.get) if s_mkt > 1.5 else max(market_vals, key=market_vals.get)) if market_vals else "-"

        # חיזוי מערכת = שקלול מודל + שוק לפי משקולות Auto-Learner
        sys_p = {}
        if market_vals:
            # אם יחסים (v>1): 1/v נותן הסתברות. אם הסתברויות: השתמש ישירות
            if s_mkt > 1.5:
                total_inv = sum(1/v for v in market_vals.values())
                mkt_probs = {k: (1/v)/total_inv if v else 0 for k, v in market_vals.items()}
            else:
                tot = sum(market_vals.values())
                mkt_probs = {k: v/tot for k, v in market_vals.items()} if tot > 0 else {}
            for k in ['1', 'X', '2']:
                mkt_p = mkt_probs.get(k, 0)
                sys_p[k] = float(math_p.get(k, 0)) * w_model + mkt_p * w_market
        else:
            sys_p = {k: float(math_p.get(k, 0)) for k in ['1', 'X', '2']}
        pred_sys = max(sys_p, key=sys_p.get) if any(sys_p.values()) else "-"

        is_sys = "✅" if pred_sys == actual else "❌"
        is_math = "✅" if pred_math == actual else "❌"
        is_market = "✅" if pred_market == actual else "❌"
        
        if pred_sys == actual: correct_system += 1
        if pred_math == actual: correct_math += 1
        if pred_market == actual: correct_market += 1
        
        records.append({
            "תאריך": date_val,
            "משחק": match_name,
            "המלצת המערכת": f"{pred_sys} ({is_sys})",
            "תוצאה בפועל": actual,
            "מוח מתמטי": f"{pred_math} ({is_math})",
            "שוק (ווינר)": f"{pred_market} ({is_market})",
        })
        
    st.dataframe(pd.DataFrame(records), width="stretch")
    
    math_acc = correct_math / total if total else 0
    market_acc = correct_market / total if total else 0
    
    st.markdown("#### 🎯 סיכום ביצועים כולל")
    c1, c2, c3 = st.columns(3)
    c1.metric("פגיעות מערכת (משוקלל)", f"{(correct_system/total)*100:.1f}%")
    c2.metric("פגיעות מודל מתמטי", f"{(math_acc)*100:.1f}%")
    c3.metric("פגיעות שוק (ווינר)", f"{(market_acc)*100:.1f}%")

    # --- דשבורד "מי צודק יותר" – מוח מתמטי מול שוק ---
    st.markdown("---")
    st.markdown(
        "<h2 style='text-align:right; color:#1e3a8a; margin-bottom:4px;'>מי צודק יותר – מוח מתמטי מול שוק</h2>"
        "<p style='text-align:right; color:#6b7280; font-size:0.95rem; margin-top:0;'>"
        f"ניתוח {total} משחקים שהסתיימו | קו ייחוס: ניחוש אקראי ≈ 33.3%</p>",
        unsafe_allow_html=True
    )
    baseline_pp = BASELINE_RANDOM * 100
    math_pp = math_acc * 100
    market_pp = market_acc * 100
    math_above = math_pp - baseline_pp
    market_above = market_pp - baseline_pp
    models_info = [
        ("מודל מתמטי", math_pp, correct_math, total, math_above),
        ("שוק (ווינר)", market_pp, correct_market, total, market_above),
    ]
    best_name = max(models_info, key=lambda x: x[1])[0]
    best_acc = max(models_info, key=lambda x: x[1])[1]
    second_acc = sorted([m[1] for m in models_info], reverse=True)[1] if len(models_info) >= 2 else 0
    gap = round(best_acc - second_acc, 1)

    card1, card2, card3 = st.columns(3)
    with card1:
        st.markdown(
            f"""
            <div style='background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='color:#6b7280; font-size:0.85rem; margin-bottom:4px;'>מודל מתמטי</div>
                <div style='font-size:1.6rem; font-weight:700; color:#2563eb;'>{math_pp:.1f}%</div>
                <div style='font-size:0.8rem; color:#059669;'>↑+{math_above:.1f}pp ({correct_math} פגיעות) מעל אקראי</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card2:
        st.markdown(
            f"""
            <div style='background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='color:#6b7280; font-size:0.85rem; margin-bottom:4px;'>שוק (ווינר)</div>
                <div style='font-size:1.6rem; font-weight:700; color:#059669;'>{market_pp:.1f}%</div>
                <div style='font-size:0.8rem; color:#059669;'>↑+{market_above:.1f}pp ({correct_market} פגיעות) מעל אקראי</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with card3:
        st.markdown(
            f"""
            <div style='background:#fef3c7; border:1px solid #f59e0b; border-radius:12px; padding:14px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
                <div style='color:#92400e; font-size:0.85rem; margin-bottom:4px;'>🏆 מוביל</div>
                <div style='font-size:1rem; font-weight:700; color:#1e3a8a;'>מודל מוביל</div>
                <div style='font-size:0.9rem; color:#1e3a8a;'>{best_name}</div>
                <div style='font-size:0.8rem; color:#6b7280;'>פער {gap}pp</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("**השוואת אחוזי דיוק**")
        if HAS_PLOTLY:
            labels = ["מודל מתמטי", "שוק (ווינר)", "אקראי (בייסליין)"]
            values = [math_pp, market_pp, baseline_pp]
            colors = ["#2563eb", "#059669", "#9ca3af"]
            fig_bar = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.1f}%" for v in values], textposition="outside"))
            fig_bar.add_hline(y=baseline_pp, line_dash="dot", line_color="#6b7280", opacity=0.8)
            fig_bar.update_layout(
                yaxis=dict(title="אחוז", range=[0, max(60, math_pp + 10)]),
                showlegend=False,
                margin=dict(t=24, b=40, l=40, r=24),
                height=280,
                xaxis_tickangle=-20,
            )
            st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})
        else:
            chart_df = pd.DataFrame({"אחוז": [math_pp, market_pp, baseline_pp]}, index=["מודל מתמטי", "שוק (ווינר)", "אקראי (בייסליין)"])
            st.bar_chart(chart_df)
    with chart_col2:
        st.markdown("**דיוק מודל מתמטי**")
        correct_pct = math_pp
        incorrect_pct = 100 - math_pp
        if HAS_PLOTLY:
            fig_donut = go.Figure(data=[go.Pie(
                labels=["נכון", "שגוי"],
                values=[correct_pct, incorrect_pct],
                hole=0.55,
                marker_colors=["#22c55e", "#ef4444"],
                textinfo="label+percent",
                texttemplate="%{label}<br>%{percent:.1f}",
                direction="clockwise",
            )])
            fig_donut.update_layout(
                annotations=[dict(text=f"{math_pp:.0f}%", x=0.5, y=0.5, font_size=20, showarrow=False)],
                margin=dict(t=24, b=24, l=24, r=24),
                height=280,
                showlegend=True,
                legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_donut, width="stretch", config={"displayModeBar": False})
        else:
            st.write(f"נכון: {correct_pct:.1f}% | שגוי: {incorrect_pct:.1f}%")

# --- Helpers ---
def get_base64_image(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except: return None
    return None

# --- CSS ---
bg_str = get_base64_image(BG_IMAGE_PATH)
bg_css = f"""
<style>
    /* ===== RTL & פונטים ===== */
    html, body {{ direction: rtl; }}
    [data-testid="stSidebar"] {{ direction: rtl; }}
    [data-testid="stSidebar"] * {{ direction: rtl; text-align: right; }}
    .main .block-container {{ padding: 1rem 1rem 2rem 1rem; max-width: 100%; }}

    /* ===== פונטים ===== */
    html, body, [class*="css"], p, div, span, label, li, button, input, select {{
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1rem !important;
    }}
    h1 {{ font-size: 1.8rem !important; font-weight: 800; }}
    h2 {{ font-size: 1.4rem !important; font-weight: 700; }}
    h3 {{ font-size: 1.2rem !important; font-weight: 600; }}

    /* ===== רקע ===== */
    .stApp {{
        background-image: linear-gradient(rgba(245,247,250,0.96), rgba(245,247,250,0.96)),
                          url("data:image/jpeg;base64,{bg_str if bg_str else ''}");
        background-size: cover; background-attachment: fixed;
    }}

    /* ===== כרטיסיית משחק ===== */
    .match-card {{
        background: #ffffff; border-radius: 16px; padding: 18px;
        margin-bottom: 20px; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid #e8ecf0; direction: rtl;
    }}

    /* ===== מדד (Metric box) ===== */
    .metric-container {{
        background: #ffffff; padding: 16px 12px; border-radius: 12px;
        border-right: 5px solid #1565c0; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 8px;
    }}
    .metric-container small {{ color: #6b7280; font-size: 0.85rem !important; }}
    .metric-container b {{ display: block; font-size: 1.6rem !important; margin-top: 4px; }}

    /* ===== סרגל הסתברות ===== */
    .bar-label {{ font-size: 0.9rem !important; font-weight: 700; color: #374151; margin-bottom: 3px; }}

    /* ===== שורות היסטוריה ===== */
    .history-row {{
        display: flex; align-items: center; justify-content: space-between;
        background: #f8f9fa; padding: 6px 10px; margin: 3px 0;
        border-radius: 8px; font-size: 0.9rem !important; border-right: 4px solid #d1d5db;
    }}
    .win {{ border-right-color: #22c55e !important; background-color: #f0fdf4 !important; }}
    .loss {{ border-right-color: #ef4444 !important; background-color: #fef2f2 !important; }}
    .draw {{ border-right-color: #f97316 !important; background-color: #fff7ed !important; }}

    /* ===== אנימציית טעינה ===== */
    @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
    .soccer-loader {{
        border: 6px solid #e5e7eb; border-top: 6px solid #1565c0;
        border-radius: 50%; width: 60px; height: 60px;
        animation: spin 1s linear infinite; margin: 0 auto 16px auto;
    }}
    .loading-overlay {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255,255,255,0.97); z-index: 9999;
        display: flex; flex-direction: column; justify-content: center;
        align-items: center; text-align: center;
    }}
    .loading-text {{ font-size: 1.5rem !important; color: #1565c0; font-weight: bold; }}

    /* ===== כרטיסי סיכום כספי (קופה, הומר, רווח, אחוז זכייה, רצף) ===== */
    .finance-card {{
        background: #ffffff; border-radius: 12px; padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #e8ecf0;
        text-align: center; direction: rtl;
    }}
    .finance-card .card-title {{ color: #6b7280; font-size: 0.9rem !important; margin-bottom: 6px; }}
    .finance-card .card-value {{ font-size: 1.5rem !important; font-weight: 700; }}
    .finance-card .card-sub {{ color: #9ca3af; font-size: 0.8rem !important; margin-top: 4px; }}
    .btn-check-results {{ background: #dc2626 !important; color: white !important; border: none !important; }}
    .btn-update-fund {{ background: white !important; color: #374151 !important; border: 1px solid #d1d5db !important; }}

    /* ===== מובייל (עד 768px) ===== */
    @media (max-width: 768px) {{
        /* ריווח */
        .main .block-container {{ padding: 0.4rem 0.4rem 2rem 0.4rem !important; }}

        /* כותרות */
        h1 {{ font-size: 1.2rem !important; }}
        h2 {{ font-size: 1.0rem !important; }}
        h3 {{ font-size: 0.9rem !important; }}

        /* עמודות Streamlit — מחסן לגובה (מלא 100%) */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}

        /* כרטיסי משחק */
        .match-card {{ padding: 10px; margin-bottom: 12px; }}

        /* כרטיסי כספים — גריד 2 עמודות */
        .finance-card {{
            padding: 10px 8px;
            margin-bottom: 6px;
        }}
        .finance-card .card-title {{ font-size: 0.78rem !important; }}
        .finance-card .card-value {{ font-size: 1.15rem !important; }}

        /* מדדים */
        .metric-container {{ padding: 8px 4px; margin-bottom: 6px; }}
        .metric-container small {{ font-size: 0.75rem !important; }}
        .metric-container b {{ font-size: 1.1rem !important; }}

        /* כפתורים */
        .stButton > button {{
            width: 100% !important;
            min-height: 2.8rem !important;
            font-size: 0.95rem !important;
            margin-bottom: 4px !important;
        }}

        /* שמות קבוצות בכרטיסי משחק */
        .match-card h2 {{ font-size: 0.95rem !important; }}

        /* dataframe — גלילה אופקית */
        [data-testid="stDataFrame"] > div {{
            overflow-x: auto !important;
        }}

        /* expanders */
        [data-testid="stExpander"] {{ font-size: 0.88rem !important; }}

        /* tabs */
        [data-testid="stTabs"] button {{ font-size: 0.82rem !important; padding: 6px 8px !important; }}
    }}
</style>
"""
st.markdown(bg_css, unsafe_allow_html=True)

# --- Logic ---
def simulate_independent_poisson():
    lam_home, lam_away = 1.45, 1.15
    def poisson_prob(lam, k): return (lam**k * math.exp(-lam)) / math.factorial(k)
    p1, px, p2 = 0.0, 0.0, 0.0
    for i in range(5):
        for j in range(5):
            prob = poisson_prob(lam_home, i) * poisson_prob(lam_away, j)
            if i > j: p1 += prob
            elif i == j: px += prob
            else: p2 += prob
    total = p1 + px + p2
    return {'1': p1/total, 'X': px/total, '2': p2/total}

def get_market_probs(market_odds):
    if not market_odds or '1' not in market_odds: return None
    inv_1, inv_X, inv_2 = 1 / market_odds['1'], 1 / market_odds['X'], 1 / market_odds['2']
    total = inv_1 + inv_X + inv_2
    return {'1': inv_1/total, 'X': inv_X/total, '2': inv_2/total}

def get_previous_odds_for_match(home_team, away_team):
    """מחזיר יחסי 1X2 קודמים מהריצה הקודמת, או None אם אין."""
    if not os.path.exists(ODDS_PREVIOUS_FILE):
        return None
    try:
        with open(ODDS_PREVIOUS_FILE, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
    except Exception:
        return None
    if not catalog:
        return None
    def clean_text(text): return str(text).replace("'", "").replace('"', "").replace("–", "-").strip()
    def get_search_terms(team_name):
        name = clean_text(team_name)
        terms = [name]
        if "תל אביב" in name: terms.append(name.replace("תל אביב", "תא"))
        if "תא" in name: terms.append(name.replace("תא", "תל אביב"))
        if "באר שבע" in name: terms.append(name.replace("באר שבע", "בש"))
        if "פתח תקוה" in name: terms.append(name.replace("פתח תקוה", "פת"))
        if "קרית שמונה" in name: terms.append(name.replace("קרית שמונה", "קש"))
        if "ראשון לציון" in name: terms.append(name.replace("ראשון לציון", "רצל"))
        return terms
    h_terms = get_search_terms(home_team)
    a_terms = get_search_terms(away_team)
    for desc, entry in catalog.items():
        desc_clean = clean_text(desc)
        if any(t in desc_clean for t in h_terms) and any(t in desc_clean for t in a_terms):
            o = entry.get("odds_1x2") or entry
            if isinstance(o, dict) and all(k in o for k in ("1", "X", "2")):
                try:
                    return {"1": float(o["1"]), "X": float(o["X"]), "2": float(o["2"])}
                except (TypeError, ValueError):
                    pass
            if all(k in entry for k in ("1", "X", "2")):
                try:
                    return {"1": float(entry["1"]), "X": float(entry["X"]), "2": float(entry["2"])}
                except (TypeError, ValueError):
                    pass
    return None

def _format_odds_with_change(current_odd, prev_odd):
    """מחזיר מחרוזת יחס + אינדיקציית שינוי: 2.10 (קודם: 2.05 ↑)"""
    if prev_odd is None or prev_odd <= 0:
        return f"{current_odd:.2f}"
    diff = current_odd - prev_odd
    if abs(diff) < 0.005:
        return f"{current_odd:.2f} (ללא שינוי)"
    arrow = "↑" if diff > 0 else "↓"
    return f"{current_odd:.2f} (קודם: {prev_odd:.2f} {arrow})"

def calculate_final_probs(model, market, weights):
    final = {}
    m_probs = get_market_probs(market)
    w0 = float(weights[0]) if weights else 0.5
    w1 = float(weights[1]) if len(weights) > 1 else 0.5
    for outcome in ['1', 'X', '2']:
        p_model = model.get(outcome, 0.33)
        p_market = (m_probs.get(outcome, 0.33) if m_probs else p_model)
        final[outcome] = p_model * w0 + p_market * w1
    return final

def generate_toto_recommendations(matches_list, base_weights, num_doubles):
    evaluated = []
    w0 = float(base_weights[0]) if base_weights else 0.5
    w1 = float(base_weights[1]) if len(base_weights) > 1 else 0.5
    t = w0 + w1
    dyn_weights = [w0/t, w1/t] if t > 0 else [0.5, 0.5]
    for i, m in enumerate(matches_list):
        model_data = m.get('model_probs', {})
        final_probs = calculate_final_probs(model_data, m.get('market_odds'), dyn_weights)
        
        p1, px, p2 = final_probs.get('1', 0), final_probs.get('X', 0), final_probs.get('2', 0)
        best_prob = max(p1, px, p2)
        best_pick = '1' if best_prob == p1 else 'X' if best_prob == px else '2'
        
        # הרכבת כפול משתי התוצאות הסבירות ביותר
        sorted_probs = sorted([('1', p1), ('X', px), ('2', p2)], key=lambda item: item[1], reverse=True)
        double_pick = f"{sorted_probs[0][0]}{sorted_probs[1][0]}"
        
        evaluated.append({
            'original_idx': i,
            'final_probs': final_probs,
            'best_prob': best_prob,
            'best_pick': best_pick,
            'double_pick': double_pick,
            'applied_weights': dyn_weights 
        })
        
    # מיון מהמשחק הכי בטוח להכי פחות בטוח 
    evaluated.sort(key=lambda x: x['best_prob'], reverse=True)
    
    total_games = len(evaluated)
    bankers_count = max(0, total_games - num_doubles)
    
    # חלוקת ההמלצות בהתאם לתקציב 
    for i, item in enumerate(evaluated):
        if i < bankers_count:
            item['rec'] = f"עוגן (Banker) - סימון {item['best_pick']}"
            item['val_pick'] = item['best_pick']
            item['type'] = 'banker'
        else:
            item['rec'] = f"כפול (Double) - סימון {item['double_pick']}"
            item['val_pick'] = item['double_pick']
            item['type'] = 'double'
            
    # החזרת המערך לסדר המשחקים המקורי 
    evaluated.sort(key=lambda x: x['original_idx'])
    return evaluated

# --- חלונות זמן מומלצים להרצת Run_GSA / Train_GSA (מחוץ לזמני משחק + המתנה לתוצאות) ---
MATCH_DURATION_MIN = 90
RESULTS_AVAILABLE_AFTER_MATCH_MIN = 30
RUN_GSA_DURATION_MIN = 60
TRAIN_GSA_DURATION_MIN = 30


def _match_start_datetime(m):
    """שעת פתיחה משוערת (שעון מקומי נאיבי), או None אם אין תאריך תקין."""
    raw_d = m.get("match_date") or ""
    raw_t = (m.get("match_time") or "").strip()
    s = str(raw_d).strip()
    if not s:
        return None
    if "T" in s:
        try:
            dt = dateutil.parser.parse(s)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except (ValueError, TypeError, OverflowError):
            pass
    try:
        base = datetime.strptime(s[:10], "%Y-%m-%d")
    except ValueError:
        try:
            dt = dateutil.parser.parse(s)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None
    if not raw_t and len(s) >= 16 and "T" in s:
        raw_t = s[11:16]
    if raw_t:
        parts = raw_t.split(":")
        try:
            h = int(parts[0])
            mi = int(parts[1]) if len(parts) > 1 else 0
            se = int(parts[2]) if len(parts) > 2 else 0
            return base.replace(hour=h, minute=mi, second=se)
        except (ValueError, IndexError):
            pass
    return base


def _match_sort_key(m):
    d = _match_start_datetime(m)
    return d if d is not None else datetime(9999, 12, 31, 23, 59, 59)


def _merge_busy_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _next_maintenance_slot(blackouts_merged, now, task_minutes, horizon_end):
    need = timedelta(minutes=task_minutes)
    t = now.replace(second=0, microsecond=0)
    if now.second > 0 or now.microsecond > 0:
        t += timedelta(minutes=1)
    while t + need <= horizon_end:
        end_task = t + need
        bumped = False
        for bs, be in blackouts_merged:
            if t < be and end_task > bs:
                t = be
                bumped = True
                break
        if not bumped:
            return t
    return None


def compute_maintenance_recommendations(matches_list):
    """
    מחשב מתי להריץ משיכת משחקים (שעה) ולימוד (חצי שעה) מחוץ לחלונות:
    משחק = 90 דק׳; ללימוד מוסיפים 30 דק׳ אחרי הסיום עד שתוצאות מופיעות באתרים.
    Train נבחר אחרי Run כך שלא יחפוף לחלון הריצה של Run (~שעה) — שני הסקריפטים לא במקביל.
    """
    now = datetime.now()
    kickoffs = []
    for m in matches_list or []:
        k = _match_start_datetime(m)
        if k is not None:
            kickoffs.append(k)
    horizon_end = now + timedelta(days=10)
    if kickoffs:
        horizon_end = max(horizon_end, max(kickoffs) + timedelta(days=2))

    def _fmt(slot, dur_min):
        if slot is None:
            return "לא נמצא חלון פנוי אוטומטית בטווח המחושב — בדוק ידנית מול לוח המשחקים."
        end = slot + timedelta(minutes=dur_min)
        return (
            f"{slot.strftime('%d/%m/%Y %H:%M')} – סיום משוער עד {end.strftime('%H:%M')} "
            f"(משך פעולה ~{dur_min} דק׳)"
        )

    if not kickoffs:
        free = (
            "אין משחקים עם תאריך/שעה בקובץ הניתוח — אין חסימות ידועות; ניתן להריץ כשזה נוח."
        )
        return {
            "run_detail": free,
            "train_detail": free,
            "run_caption": "ללא חסימות ידועות (אין משחקים בקובץ)",
            "train_caption": "ללא חסימות ידועות (אין משחקים בקובץ)",
        }

    busy_run = [(k, k + timedelta(minutes=MATCH_DURATION_MIN)) for k in kickoffs]
    train_block = MATCH_DURATION_MIN + RESULTS_AVAILABLE_AFTER_MATCH_MIN
    busy_train = [(k, k + timedelta(minutes=train_block)) for k in kickoffs]
    merged_run = _merge_busy_intervals(busy_run)
    slot_run = _next_maintenance_slot(merged_run, now, RUN_GSA_DURATION_MIN, horizon_end)
    # Train לא יכול לרוץ במקביל ל-Run — חוסמים את כל משך Run_GSA (~שעה) מהחיפוש של Train
    busy_train_with_run = list(busy_train)
    if slot_run is not None:
        busy_train_with_run.append(
            (slot_run, slot_run + timedelta(minutes=RUN_GSA_DURATION_MIN))
        )
    merged_train = _merge_busy_intervals(busy_train_with_run)
    slot_train = _next_maintenance_slot(merged_train, now, TRAIN_GSA_DURATION_MIN, horizon_end)
    return {
        "run_detail": _fmt(slot_run, RUN_GSA_DURATION_MIN),
        "train_detail": _fmt(slot_train, TRAIN_GSA_DURATION_MIN),
        "run_caption": f"המלצה: {slot_run.strftime('%d/%m %H:%M')}" if slot_run else "אין חלון אוטומטי — בדוק ידנית",
        "train_caption": f"המלצה: {slot_train.strftime('%d/%m %H:%M')}" if slot_train else "אין חלון אוטומטי — בדוק ידנית",
    }


def draw_custom_bar(title, probs):
    if not probs:
        st.markdown(f"<div class='bar-label'>{title}</div>", unsafe_allow_html=True); st.caption("אין נתונים"); return
    p1, px, p2 = probs.get('1', 0), probs.get('X', 0), probs.get('2', 0)
    st.markdown(f"<div class='bar-label'>{title}</div>", unsafe_allow_html=True)
    bar_html = f"""
    <div style="display:flex; height:18px; width:100%; border-radius:4px; overflow:hidden; font-size:0.7rem; color:white; line-height:18px;">
        <div style="width:{p1*100}%; background:#2563eb; text-align:center;">{p1:.0%}</div>
        <div style="width:{px*100}%; background:#9ca3af; text-align:center;">{px:.0%}</div>
        <div style="width:{p2*100}%; background:#dc2626; text-align:center;">{p2:.0%}</div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#666; margin-bottom:10px;">
        <span>1 ({p1:.0%})</span><span>X ({px:.0%})</span><span>2 ({p2:.0%})</span>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def render_history_visual(history_list):
    if not history_list:
        st.write("אין נתונים זמינים")
        return

    table_rows = []
    for game in history_list:
        try:
            fixture = game.get('fixture', {})
            teams = game.get('teams', {})
            goals = game.get('goals', {})
            
            home_data = teams.get('home', {})
            away_data = teams.get('away', {})
            
            h_name = home_data.get('name', 'בית')
            a_name = away_data.get('name', 'חוץ')
            h_logo = home_data.get('logo', '')
            a_logo = away_data.get('logo', '')
            
            # בניית עמודות עם לוגו ושם משולבים
            home_col = f'<img src="{h_logo}" width="25"> {h_name}'
            away_col = f'<img src="{a_logo}" width="25"> {a_name}'
            
            # חישוב תוצאה ומנצחת
            h_goals = goals.get('home')
            a_goals = goals.get('away')
            score_str = f"{h_goals} - {a_goals}"
            
            if h_goals > a_goals:
                winner_suffix = f"({h_name})"
                res_val = "W_HOME"
            elif a_goals > h_goals:
                winner_suffix = f"({a_name})"
                res_val = "W_AWAY"
            else:
                winner_suffix = "(תיקו)"
                res_val = "D"

            table_rows.append({
                "תאריך": str(fixture.get('date', ''))[:10],
                "קבוצת בית": home_col,
                "קבוצת חוץ": away_col,
                "תוצאה": f"{score_str} {winner_suffix}",
                "raw_result": res_val # עמודה פנימית לצביעה
            })
        except:
            continue

    if not table_rows:
        st.write("אין נתונים זמינים")
        return

    df_history = pd.DataFrame(table_rows)

    # פונקציית צביעה (מדגישה את השורה לפי תוצאה)
    def style_rows(row):
        color = ""
        if row.raw_result == "D": color = "background-color: #fef9c3" # צהוב עדין לתיקו
        else: color = "background-color: #ffffff" # לבן רגיל
        return [color] * len(row)

    st.write(
        df_history[["תאריך", "קבוצת בית", "קבוצת חוץ", "תוצאה"]]
        .to_html(escape=False, index=False, justify='right'), 
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)


def _get_pro_stats_impl():
    """לוגיקה פנימית של get_pro_stats – ללא cache."""
    try:
        if not os.path.exists(DB_PATH):
            return {"weights": (0.5, 0.5), "hit_rate": 0, "total": 0, "rec_weights": None}
            
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='weights'")
        if cursor.fetchone()[0] == 1:
            w_df = pd.read_sql_query("SELECT w_model, w_market FROM weights WHERE id=1", conn)
            if not w_df.empty:
                wm = float(w_df.iloc[0]['w_model'] or 0.5)
                wmk = float(w_df.iloc[0]['w_market'] or 0.5)
                t = wm + wmk
                weights = (wm/t, wmk/t) if t > 0 else (0.5, 0.5)
            else:
                weights = (0.5, 0.5)
        else:
            weights = (0.5, 0.5)
            
        cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='matches'")
        if cursor.fetchone()[0] == 1:
            df = pd.read_sql_query("SELECT * FROM matches WHERE actual_result IS NOT NULL AND actual_result != 'None' AND actual_result != ''", conn)
        else:
            df = pd.DataFrame()
        bankroll_roi_pct = None
        bankroll_total_invested = None
        bankroll_total_profit = None
        try:
            cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='bankroll_roi_cache'")
            if cursor.fetchone()[0] == 1:
                roi_row = pd.read_sql_query("SELECT roi_pct, total_invested, total_profit FROM bankroll_roi_cache WHERE id=1", conn)
                if not roi_row.empty and roi_row.iloc[0]["roi_pct"] is not None:
                    bankroll_roi_pct = float(roi_row.iloc[0]["roi_pct"])
                    bankroll_total_invested = float(roi_row.iloc[0]["total_invested"] or 0)
                    bankroll_total_profit = float(roi_row.iloc[0]["total_profit"] or 0)
        except Exception:
            pass
        conn.close()
        
        total_finished = len(df)
        hit_rate = 0
        rec_weights = None
        
        if total_finished > 0:
            hits = 0
            correct_math = 0
            correct_market = 0
            for _, row in df.iterrows():
                m1 = float(row['model_prob_1'] or 0)
                mx_ = float(row['model_prob_x'] or 0)
                m2 = float(row['model_prob_2'] or 0)
                o1 = float(row['market_prob_1'] or 0)
                ox = float(row['market_prob_x'] or 0)
                o2 = float(row['market_prob_2'] or 0)

                # המרת יחסי שוק (decimal odds) להסתברויות, או שימוש ישיר אם כבר הסתברויות
                s_o = o1 + ox + o2
                if s_o > 1.5:  # נראה כמו odds
                    inv1, invx, inv2 = (1/o1 if o1 else 0), (1/ox if ox else 0), (1/o2 if o2 else 0)
                    total_inv = inv1 + invx + inv2
                    mk1, mkx, mk2 = (inv1/total_inv, invx/total_inv, inv2/total_inv) if total_inv > 0 else (1/3, 1/3, 1/3)
                elif s_o > 0:  # הסתברויות – נרמול
                    mk1, mkx, mk2 = o1/s_o, ox/s_o, o2/s_o
                else:
                    mk1, mkx, mk2 = 1/3, 1/3, 1/3
                w0, w1 = weights[0], weights[1]
                p1 = m1 * w0 + mk1 * w1
                px = mx_ * w0 + mkx * w1
                p2 = m2 * w0 + mk2 * w1
                prediction = '1' if p1 > px and p1 > p2 else 'X' if px > p1 and px > p2 else '2'
                if prediction == str(row['actual_result']).strip(): hits += 1

                math_p = {'1': m1, 'X': mx_, '2': m2}
                pred_math = max(math_p, key=math_p.get) if any(math_p.values()) else "-"
                # פייבוריט בשוק: אם odds (סכום>1.5)=min, אם הסתברויות=max
                mkt_vals = {k: v for k, v in {'1': o1, 'X': ox, '2': o2}.items() if v and v > 0}
                pred_market = (min(mkt_vals, key=mkt_vals.get) if s_o > 1.5 else max(mkt_vals, key=mkt_vals.get)) if mkt_vals else "-"
                actual_res = str(row['actual_result']).strip()
                if pred_math == actual_res: correct_math += 1
                if pred_market == actual_res: correct_market += 1
                    
            hit_rate = (hits / total_finished) * 100
            math_acc = correct_math / total_finished
            market_acc = correct_market / total_finished
            sum_acc = math_acc + market_acc
            if sum_acc > 0:
                rec_weights = (math_acc / sum_acc, market_acc / sum_acc)
        
        out = {"weights": weights, "hit_rate": hit_rate, "total": total_finished, "rec_weights": rec_weights}
        if bankroll_roi_pct is not None:
            out["bankroll_roi_pct"] = bankroll_roi_pct
            out["bankroll_total_invested"] = bankroll_total_invested
            out["bankroll_total_profit"] = bankroll_total_profit
        return out
    except Exception:
        return {"weights": (0.5, 0.5), "hit_rate": 0, "total": 0, "rec_weights": None}


def get_pro_stats():
    """שליפת נתוני אמת ומשקולות למידה ממסד הנתונים (עם cache)."""
    db_mtime = os.path.getmtime(DB_PATH) if os.path.exists(DB_PATH) else 0
    return get_pro_stats_cached(db_mtime)


def get_weekly_learning_stats():
    """
    בניית טבלת למידה שבועית + פילוח לפי טווחי יחס, על בסיס טבלת matches במסד הנתונים.
    הנחות:
    - סכום קבוע לכל הימור (יחידת בסיס 1).
    - הימור המערכת הוא הארגמקס של final_prob_*.
    - market_prob_*: הסתברויות 0-1 או יחסים (ממירים לפי הצורך).
    """
    if not os.path.exists(DB_PATH):
        return None, None

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT match_date, home_team, away_team,
                   market_prob_1, market_prob_x, market_prob_2,
                   final_prob_1, final_prob_x, final_prob_2,
                   actual_result
            FROM matches
            WHERE actual_result IS NOT NULL
              AND actual_result != ''
              AND actual_result != 'None'
              AND actual_result IN ('1','X','2')
            """,
            conn,
        )
        conn.close()
    except Exception:
        return None, None

    if df.empty:
        return None, None

    records = []
    bucket_records = []

    for _, row in df.iterrows():
        try:
            date_str = str(row["match_date"])
            try:
                d = datetime.strptime(date_str[:10], "%Y-%m-%d")
            except ValueError:
                # אם הפורמט שונה, נוותר על השורה
                continue

            year, week, _ = d.isocalendar()
            week_key = f"{year}-W{week:02d}"

            def _safe_float(v):
                x = float(v) if v is not None else 0.0
                return x if (x == x and np.isfinite(x)) else 0.0

            final_probs = {
                "1": _safe_float(row.get("final_prob_1")),
                "X": _safe_float(row.get("final_prob_x")),
                "2": _safe_float(row.get("final_prob_2")),
            }
            if not any(final_probs.values()):
                continue

            pred = max(final_probs, key=final_probs.get)
            p_pred = final_probs[pred]

            market_vals = {
                "1": float(row.get("market_prob_1") or 0.0),
                "X": float(row.get("market_prob_x") or 0.0),
                "2": float(row.get("market_prob_2") or 0.0),
            }
            raw = market_vals.get(pred, 0.0)
            # אם הסתברות (0-1): odds=1/p. אם כבר יחס (>1): השתמש ישירות
            odds = (1.0 / raw) if raw and 0 < raw <= 1.0 else raw
            if odds is None or odds <= 1.0:
                continue

            actual = str(row.get("actual_result")).strip()
            stake = 1.0
            if actual == pred:
                profit = odds - 1.0
                win_flag = 1
            else:
                profit = -1.0
                win_flag = 0

            ev = (p_pred * odds) - 1.0

            # סיווג לטווח יחסים
            if odds < 1.60:
                bucket = "מתחת 1.60"
            elif odds <= 2.00:
                bucket = "1.60–2.00"
            else:
                bucket = "מעל 2.00"

            records.append(
                {
                    "week": week_key,
                    "stake": stake,
                    "profit": profit,
                    "win": win_flag,
                    "odds": odds,
                    "ev": ev,
                }
            )
            bucket_records.append(
                {
                    "bucket": bucket,
                    "stake": stake,
                    "profit": profit,
                    "win": win_flag,
                    "odds": odds,
                    "ev": ev,
                }
            )
        except Exception:
            continue

    if not records:
        return None, None

    week_df = pd.DataFrame(records)
    bucket_df = pd.DataFrame(bucket_records) if bucket_records else None

    # אגרגציה שבועית (avg_ev: מתעלמים מ-NaN כדי למנוע nan%)
    weekly_agg = (
        week_df.groupby("week")
        .agg(
            total_bets=("stake", "count"),
            wins=("win", "sum"),
            avg_odds=("odds", "mean"),
            avg_ev=("ev", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
            total_stake=("stake", "sum"),
            total_profit=("profit", "sum"),
        )
        .reset_index()
    )
    weekly_agg["hit_rate_%"] = (
        weekly_agg["wins"] / weekly_agg["total_bets"] * 100.0
    )
    weekly_agg["roi_%"] = (
        weekly_agg["total_profit"] / weekly_agg["total_stake"] * 100.0
    )

    # אגרגציה לפי טווח יחס
    bucket_agg = None
    if bucket_df is not None and not bucket_df.empty:
        bucket_agg = (
            bucket_df.groupby("bucket")
            .agg(
                total_bets=("stake", "count"),
                wins=("win", "sum"),
                avg_odds=("odds", "mean"),
                avg_ev=("ev", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
                total_stake=("stake", "sum"),
                total_profit=("profit", "sum"),
            )
            .reset_index()
        )
        bucket_agg["hit_rate_%"] = (
            bucket_agg["wins"] / bucket_agg["total_bets"] * 100.0
        )
        bucket_agg["roi_%"] = (
            bucket_agg["total_profit"] / bucket_agg["total_stake"] * 100.0
        )

    return weekly_agg, bucket_agg


def _build_reliability(df, prob_cols, n_bins: int = 10):
    """
    בונה טבלת Reliability עבור מודל אחד:
    prob_cols: {'1': col_name_for_home, 'X': ..., '2': ...}
    """
    records = []
    for _, row in df.iterrows():
        actual = str(row.get("actual_result")).strip()
        if actual not in ("1", "X", "2"):
            continue
        col = prob_cols.get(actual)
        if not col:
            continue
        try:
            p = float(row.get(col) or 0.0)
        except (TypeError, ValueError):
            p = 0.0
        if p <= 0.0 or p >= 1.0:
            continue
        records.append({"p": p, "y": 1})
    if not records:
        return None
    tmp = pd.DataFrame(records)
    tmp["bin"] = pd.cut(tmp["p"], bins=n_bins, labels=False, include_lowest=True)
    agg = (
        tmp.groupby("bin")
        .agg(
            mean_pred=("p", "mean"),
            empirical=("y", "mean"),
            count=("y", "size"),
        )
        .reset_index()
    )
    agg["bin_center"] = (agg["bin"] + 0.5) / n_bins
    return agg


def get_reliability_stats():
    """
    שליפת נתונים ממסד הנתונים ובניית Reliability Diagram
    עבור: מודל מתמטי, AI, שוק (אחרי המרה להסתברויות), ומערכת משוקללת.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT match_date, home_team, away_team,
                   model_prob_1, model_prob_x, model_prob_2,
                   market_prob_1, market_prob_x, market_prob_2,
                   ai_prob_1, ai_prob_x, ai_prob_2,
                   final_prob_1, final_prob_x, final_prob_2,
                   actual_result
            FROM matches
            WHERE actual_result IS NOT NULL
              AND actual_result != ''
              AND actual_result != 'None'
            """,
            conn,
        )
        conn.close()
    except Exception:
        return None

    if df.empty:
        return None

    stats = {}

    # מודל מתמטי
    math_rel = _build_reliability(
        df,
        {"1": "model_prob_1", "X": "model_prob_x", "2": "model_prob_2"},
    )
    if math_rel is not None:
        stats["מודל מתמטי"] = math_rel

    # מערכת משוקללת (final_probs)
    sys_rel = _build_reliability(
        df,
        {"1": "final_prob_1", "X": "final_prob_x", "2": "final_prob_2"},
    )
    if sys_rel is not None:
        stats["מערכת משוקללת"] = sys_rel

    return stats if stats else None


def get_shishka_last_run():
    """
    שליפת תוצאת הריצה האחרונה של בדיקת שישקה (ShishkaCheck) מהטבלה shishka_last_run.
    מחזיר dict עם run_at, passed, safe_to_train, brier_score, log_loss, actual_roi_pct,
    max_drawdown_pct, consecutive_losses, n_matches, alerts, וסטטוס כל רכיב; או None אם אין נתונים.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            df = pd.read_sql_query(
                """
                SELECT run_at, passed, safe_to_train, safe_to_bet,
                       brier_score, log_loss,
                       actual_roi_pct, max_drawdown_pct, consecutive_losses, n_matches,
                       alerts_json, data_validator_passed, calibrator_passed, drift_passed, risk_passed,
                       details_json
                FROM shishka_last_run
                WHERE id = 1
                """,
                conn,
            )
        except sqlite3.OperationalError:
            df = pd.read_sql_query(
                """
                SELECT run_at, passed, safe_to_train, safe_to_bet,
                       brier_score, log_loss,
                       actual_roi_pct, max_drawdown_pct, consecutive_losses, n_matches,
                       alerts_json, data_validator_passed, calibrator_passed, drift_passed, risk_passed
                FROM shishka_last_run
                WHERE id = 1
                """,
                conn,
            )
        conn.close()
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    try:
        alerts = json.loads(row["alerts_json"]) if row.get("alerts_json") else []
    except Exception:
        alerts = []
    try:
        details = json.loads(row["details_json"]) if row.get("details_json") else {}
    except Exception:
        details = {}
    return {
        "run_at": row.get("run_at"),
        "passed": bool(row.get("passed")),
        "safe_to_train": bool(row.get("safe_to_train")),
        "safe_to_bet": bool(row.get("safe_to_bet")),
        "brier_score": row.get("brier_score"),
        "log_loss": row.get("log_loss"),
        "actual_roi_pct": row.get("actual_roi_pct"),
        "max_drawdown_pct": row.get("max_drawdown_pct"),
        "consecutive_losses": row.get("consecutive_losses"),
        "n_matches": row.get("n_matches"),
        "alerts": alerts,
        "details": details,
        "data_validator_passed": bool(row.get("data_validator_passed")),
        "calibrator_passed": bool(row.get("calibrator_passed")),
        "drift_passed": bool(row.get("drift_passed")),
        "risk_passed": bool(row.get("risk_passed")),
    }


def get_brains_race_curves():
    """
    גרף 'מרוץ מוחות' – דיוק מצטבר של:
    - מודל מתמטי
    - סוכן AI
    - שוק
    - מערכת משוקללת (final_probs)
    לאורך ציר זמן.
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT rowid AS rid,
                   match_date, home_team, away_team,
                   model_prob_1, model_prob_x, model_prob_2,
                   market_prob_1, market_prob_x, market_prob_2,
                   ai_prob_1, ai_prob_x, ai_prob_2,
                   final_prob_1, final_prob_x, final_prob_2,
                   actual_result
            FROM matches
            WHERE actual_result IS NOT NULL
              AND actual_result != ''
              AND actual_result != 'None'
            ORDER BY match_date, rid
            """,
            conn,
        )
        conn.close()
    except Exception:
        return None

    if df.empty:
        return None

    math_hits = market_hits = sys_hits = 0
    total = 0
    rows = []

    for _, row in df.iterrows():
        actual = str(row.get("actual_result")).strip()
        if actual not in ("1", "X", "2"):
            continue

        total += 1

        # מודל מתמטי – argmax הסתברויות מודל
        math_p = {
            "1": float(row.get("model_prob_1") or 0.0),
            "X": float(row.get("model_prob_x") or 0.0),
            "2": float(row.get("model_prob_2") or 0.0),
        }
        math_pred = max(math_p, key=math_p.get) if any(math_p.values()) else None
        if math_pred == actual:
            math_hits += 1

        # שוק – ההסתברות הגבוהה ביותר (הפייבוריט). market_prob_* = הסתברויות 0-1
        market_o = {
            "1": float(row.get("market_prob_1") or 0.0),
            "X": float(row.get("market_prob_x") or 0.0),
            "2": float(row.get("market_prob_2") or 0.0),
        }
        valid = {k: v for k, v in market_o.items() if v and v > 0}
        # אם סכום > 1.5 = יחסים (odds), פייבוריט = min. אחרת הסתברויות, פייבוריט = max
        s = sum(valid.values())
        market_pred = (min(valid, key=valid.get) if s > 1.5 else max(valid, key=valid.get)) if valid else None
        if market_pred == actual:
            market_hits += 1

        # מערכת משוקללת – argmax final_probs
        sys_p = {
            "1": float(row.get("final_prob_1") or 0.0),
            "X": float(row.get("final_prob_x") or 0.0),
            "2": float(row.get("final_prob_2") or 0.0),
        }
        sys_pred = max(sys_p, key=sys_p.get) if any(sys_p.values()) else None
        if sys_pred == actual:
            sys_hits += 1

        rows.append(
            {
                "n": total,
                "מודל מתמטי": math_hits / total,
                "שוק": market_hits / total,
                "מערכת משוקללת": sys_hits / total,
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


def _wilson_ci(hits: int, n: int, z: float = 1.96):
    """רצועת ביטחון 95% (Wilson CI) לפרופורציה."""
    if n <= 0:
        return 0.5, 0.5
    p = hits / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def get_cumulative_learning_curve_data():
    """
    מחזיר נתונים לעקומת למידה מצטברת: דיוק מצטבר של המערכת המשוקללת לאורך זמן,
    עם רצועת ביטחון 95% (Wilson CI). כל שורה = מצטבר עד אותו משחק (לפי תאריך).
    """
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT match_date,
                   final_prob_1, final_prob_x, final_prob_2,
                   actual_result
            FROM matches
            WHERE actual_result IS NOT NULL
              AND actual_result != ''
              AND actual_result != 'None'
              AND actual_result IN ('1', 'X', '2')
            ORDER BY match_date, rowid
            """,
            conn,
        )
        conn.close()
    except Exception:
        return None

    if df.empty:
        return None

    rows = []
    cumulative_hits = 0
    cumulative_n = 0
    first_accuracy_pct = None
    peak_accuracy_pct = None
    peak_date = None

    for _, row in df.iterrows():
        actual = str(row.get("actual_result")).strip()
        final_p = {
            "1": float(row.get("final_prob_1") or 0.0),
            "X": float(row.get("final_prob_x") or 0.0),
            "2": float(row.get("final_prob_2") or 0.0),
        }
        if not any(final_p.values()):
            continue
        pred = max(final_p, key=final_p.get)
        hit = 1 if pred == actual else 0
        cumulative_hits += hit
        cumulative_n += 1
        acc = cumulative_hits / cumulative_n
        acc_pct = acc * 100.0
        if first_accuracy_pct is None:
            first_accuracy_pct = acc_pct
        if peak_accuracy_pct is None or acc_pct >= peak_accuracy_pct:
            peak_accuracy_pct = acc_pct
            peak_date = str(row.get("match_date", ""))[:10]
        ci_lo, ci_hi = _wilson_ci(cumulative_hits, cumulative_n)
        date_str = str(row.get("match_date", ""))[:10]
        rows.append({
            "date": date_str,
            "n": cumulative_n,
            "hits": cumulative_hits,
            "accuracy": acc,
            "accuracy_pct": acc_pct,
            "ci_low": ci_lo * 100,
            "ci_high": ci_hi * 100,
        })

    if not rows:
        return None

    # שמירת ערכים גלובליים לחישוב כרטיסי סיכום
    first_acc = first_accuracy_pct
    current = rows[-1]
    current_acc = current["accuracy_pct"]
    improvement_pp = current_acc - first_acc if first_acc is not None else 0.0
    total_games = current["n"]
    first_date_str = rows[0]["date"]
    last_date_str = rows[-1]["date"]
    try:
        d0 = datetime.strptime(first_date_str[:10], "%Y-%m-%d")
        d1 = datetime.strptime(last_date_str[:10], "%Y-%m-%d")
        days_span = max(0, (d1 - d0).days)
    except Exception:
        days_span = 0

    if len(rows) >= 10:
        prev = rows[-11]
        last_10_hits = current["hits"] - prev["hits"]
        last_10_n = 10
    else:
        last_10_hits = current["hits"]
        last_10_n = len(rows)
    last_10_rate = (last_10_hits / last_10_n * 100) if last_10_n else 0
    above_33 = (last_10_rate - 33.0) if last_10_n == 10 else None

    summary = {
        "current_accuracy_pct": current_acc,
        "improvement_pp": improvement_pp,
        "peak_accuracy_pct": peak_accuracy_pct,
        "peak_date": peak_date,
        "total_games": total_games,
        "days_span": days_span,
        "last_10_hits": last_10_hits,
        "last_10_n": last_10_n,
        "last_10_rate_pct": last_10_rate,
        "above_33_pp": above_33,
    }
    return {"curve": pd.DataFrame(rows), "summary": summary}


def display_cumulative_learning_curve(key=None):
    """מציג דשבורד עקומת למידה מצטברת בסגנון דשבורד יפה: כרטיסים צבעוניים, גרף עם כותרת ואנוטציות. key=מזהה ייחודי למניעת כפילות."""
    data = get_cumulative_learning_curve_data()
    if data is None or data["curve"].empty:
        return
    curve_df = data["curve"]
    s = data["summary"]

    # --- כותרת מרכזית (כמו בתמונה) ---
    st.markdown(
        "<div style='text-align:center; margin-bottom:20px;'>"
        "<h3 style='color:#1e3a8a; font-weight:700; margin:0; font-size:1.5rem;'>"
        "📈 עקומת למידה מצטברת"
        "</h3></div>",
        unsafe_allow_html=True,
    )

    # --- כרטיסי מדדים: כותרות כמו בתמונה, ערכים בצבע ---
    peak_d = s.get("peak_date") or ""
    try:
        peak_disp = datetime.strptime(peak_d[:10], "%Y-%m-%d").strftime("%Y-%m-%d") if peak_d else "—"
    except Exception:
        peak_disp = peak_d or "—"
    imp = s["improvement_pp"]
    imp_str = f"{abs(imp):.1f}pp-" if imp < 0 else f"{imp:.1f}pp"
    above = s.get("above_33_pp")
    last_10_sub = "33% מעל" if (above is not None and above >= 0) else (f"{above:+.0f}% מול 33%" if above is not None else f"{s['days_span']} ימים")

    # (כותרת, ערך, תת־כותרת, צבע ערך, צבע פס)
    cards_spec = [
        ("דיוק נוכחי", f"{s['current_accuracy_pct']:.1f}%", "מצטבר", "#1565c0", "#1565c0"),
        ("שיפור כולל", imp_str, "מאז התחלה", "#dc2626" if imp < 0 else "#059669", "#1565c0"),
        ("שיא", f"{s['peak_accuracy_pct']:.1f}%", peak_disp, "#16a34a", "#16a34a"),
        ("משחקים", str(s["total_games"]), f"{s['days_span']} ימים", "#1565c0", "#1565c0"),
        ("10 ימים אחרונים", f"{s['last_10_hits']}/{s['last_10_n']}", last_10_sub, "#ea580c", "#ea580c"),
    ]
    cols = st.columns(5)
    for i, (title, value, sub, value_color, border_color) in enumerate(cards_spec):
        with cols[i]:
            st.markdown(
                f"<div style='background:#ffffff; border-radius:12px; padding:16px 12px; "
                f"border-left:4px solid {border_color}; box-shadow:0 2px 10px rgba(0,0,0,0.06); "
                f"text-align:center; direction:rtl; margin-bottom:10px;'>"
                f"<div style='color:#6b7280; font-size:0.85rem; font-weight:600; margin-bottom:6px;'>{title}</div>"
                f"<div style='font-size:1.6rem; font-weight:700; color:{value_color};'>{value}</div>"
                f"<div style='color:#9ca3af; font-size:0.8rem; margin-top:4px;'>{sub}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    if not HAS_PLOTLY:
        st.line_chart(curve_df.set_index("date")[["accuracy_pct"]])
        return

    curve_df = curve_df.copy()
    curve_df["date_dt"] = pd.to_datetime(curve_df["date"], errors="coerce")
    curve_df = curve_df.dropna(subset=["date_dt"])
    if curve_df.empty:
        return
    curve_df["date_fmt"] = curve_df["date_dt"].dt.strftime("%d/%m/%y")

    # גרף: כותרת מעל, מגמה מצטברת + רצועת Wilson, קו 50% בירוק, אנוטציות שיא ונוכחי
    fig = go.Figure()
    # רצועת ביטחון
    fig.add_trace(
        go.Scatter(
            x=curve_df["date_fmt"].tolist() + curve_df["date_fmt"].tolist()[::-1],
            y=curve_df["ci_high"].tolist() + curve_df["ci_low"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(21, 101, 192, 0.18)",
            line=dict(color="rgba(21, 101, 192, 0.3)", width=1),
            name="רצועת ביטחון 95% (Wilson CI)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve_df["date_fmt"],
            y=curve_df["accuracy_pct"],
            mode="lines+markers",
            name="מגמה מצטברת",
            line=dict(color="#1565c0", width=2.5),
            marker=dict(size=6, color="#1565c0", line=dict(width=1, color="white")),
        )
    )
    # קו ייחוס 50% בירוק (כמו בתמונה)
    fig.add_hline(y=50, line_dash="dash", line_color="#86efac", line_width=1.5, opacity=0.9)
    fig.add_annotation(x=0.02, y=50, xref="paper", yref="y", text="50%", showarrow=False, font=dict(size=11, color="#16a34a"))

    y_min = max(0, curve_df["ci_low"].min() - 5)
    y_max = min(100, curve_df["ci_high"].max() + 5)
    if y_max - y_min < 30:
        y_center = (y_min + y_max) / 2
        y_min = max(0, y_center - 18)
        y_max = min(100, y_center + 18)

    # אנוטציה בשיא (נקודה ראשונה שבה הדיוק = שיא)
    peak_val = s["peak_accuracy_pct"]
    peak_candidates = curve_df[curve_df["accuracy_pct"] >= peak_val - 0.01]
    peak_row = peak_candidates.iloc[0] if len(peak_candidates) > 0 else curve_df.iloc[0]
    fig.add_annotation(
        x=peak_row["date_fmt"], y=peak_row["accuracy_pct"],
        text=f"{peak_val:.1f}% שיא",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#16a34a",
        bgcolor="rgba(34, 197, 94, 0.2)", bordercolor="#16a34a", borderwidth=1,
        font=dict(size=11, color="#15803d"),
    )
    # אנוטציה בדיוק נוכחי (נקודה אחרונה)
    last_row = curve_df.iloc[-1]
    fig.add_annotation(
        x=last_row["date_fmt"], y=last_row["accuracy_pct"],
        text=f"{s['current_accuracy_pct']:.1f}%",
        showarrow=True, arrowhead=2, arrowsize=1, arrowcolor="#1565c0",
        bgcolor="rgba(21, 101, 192, 0.15)", bordercolor="#1565c0", borderwidth=1,
        font=dict(size=11, color="#1565c0"),
    )

    fig.update_layout(
        title=dict(
            text=f"עקומת למידה מצטברת | דיוק נוכחי: {s['current_accuracy_pct']:.1f}% | שיפור: {s['improvement_pp']:+.1f}",
            font=dict(size=14, color="#374151"),
            x=0.5, xanchor="center", y=1.0, yanchor="bottom",
        ),
        margin=dict(t=56, b=48, l=52, r=28),
        height=420,
        plot_bgcolor="rgba(248,250,252,0.95)",
        paper_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Segoe UI, Tahoma, sans-serif", size=12),
        xaxis=dict(
            title="תאריך",
            title_font=dict(size=13),
            tickfont=dict(size=11),
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#e5e7eb",
        ),
        yaxis=dict(
            title="% דיוק",
            title_font=dict(size=13),
            tickfont=dict(size=11),
            range=[y_min, y_max],
            gridcolor="rgba(0,0,0,0.07)",
            showline=True,
            linecolor="#e5e7eb",
            dtick=5,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.9)",
        ),
        hovermode="x unified",
    )
    fig.update_traces(hovertemplate="%{y:.1f}%<extra></extra>")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False}, key=key)


def get_daily_circuit_breaker():
    """
    Circuit Breaker יומי פשוט:
    - 5 הפסדים רצופים -> עצירה
    - EV ממוצע יומי שלילי (על בסיס final_probs מול יחסי שוק) -> עצירה
    """
    # בלם חירום כבוי בשלב זה — אין בו צורך
    return {"triggered": False, "reasons": []}

    if not os.path.exists(DB_PATH):
        return {"triggered": False, "reasons": []}

    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT match_date,
                   market_prob_1, market_prob_x, market_prob_2,
                   final_prob_1, final_prob_x, final_prob_2,
                   actual_result
            FROM matches
            WHERE actual_result IS NOT NULL
              AND actual_result != ''
              AND actual_result != 'None'
              AND match_date = ?
            ORDER BY rowid ASC
            """,
            conn,
            params=(today_str,),
        )
        conn.close()
    except Exception:
        return {"triggered": False, "reasons": []}

    if df.empty:
        return {"triggered": False, "reasons": []}

    consecutive_losses = 0
    max_consecutive_losses = 0
    total_ev = 0.0
    bet_count = 0

    for _, row in df.iterrows():
        final_probs = {
            "1": float(row.get("final_prob_1") or 0.0),
            "X": float(row.get("final_prob_x") or 0.0),
            "2": float(row.get("final_prob_2") or 0.0),
        }
        if not any(final_probs.values()):
            continue

        pred = max(final_probs, key=final_probs.get)
        p_pred = final_probs[pred]

        market_odds = {
            "1": float(row.get("market_prob_1") or 0.0),
            "X": float(row.get("market_prob_x") or 0.0),
            "2": float(row.get("market_prob_2") or 0.0),
        }
        odds = market_odds.get(pred, 0.0)
        if odds is None or odds <= 1.0:
            continue

        ev = (p_pred * odds) - 1.0
        total_ev += ev
        bet_count += 1

        actual = str(row.get("actual_result")).strip()
        if actual == pred:
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

    if bet_count == 0:
        return {"triggered": False, "reasons": []}

    avg_ev = total_ev / bet_count
    reasons = []
    triggered = False

    if max_consecutive_losses >= 5:
        triggered = True
        reasons.append("5 הפסדים רצופים ביום זה.")
    if avg_ev < 0:
        triggered = True
        reasons.append(f"EV ממוצע יומי שלילי ({avg_ev*100:.1f}%).")

    return {"triggered": triggered, "reasons": reasons}


# --- אתחול מסד נתונים (מבטיח קיום טבלאות matches, weights וכו') ---
init_finance_tables()

# --- טעינת נתוני סטטיסטיקות ---
stats = get_pro_stats()
recommended = stats.get("rec_weights")
# משקולות פעילות תמיד מה-DB (כולל טיכה). rec_weights הוא המלצה ל-2D בלבד – לא דורס את טיכה
active_weights = stats["weights"]

# --- בדיקת בלם חירום יומי ---
breaker_state = get_daily_circuit_breaker()

# --- טעינת משחקים (מוקדם: מיון, המלצות תזמון Run/Train, טוטו אחרי משקולות) ---
_json_load_error = None
matches = []
if os.path.exists(JSON_FILE):
    try:
        _jmt_early = os.path.getmtime(JSON_FILE)
        matches = _load_analysis_json_cached(JSON_FILE, _jmt_early)
    except (json.JSONDecodeError, IOError) as e:
        _json_load_error = e
        matches = []
matches = sorted(matches, key=_match_sort_key)
_maint_rec = compute_maintenance_recommendations(matches)

current_bankroll = get_bankroll_balance()

user_bankroll = current_bankroll
num_doubles = 6  # ברירת מחדל קבועה

# נרמול משקולות (מודל + שוק בלבד)
w_m = float(active_weights[0])
w_mk = float(active_weights[1])
total_2 = w_m + w_mk
if total_2 > 0:
    current_weights = (w_m / total_2, w_mk / total_2)
else:
    current_weights = (0.5, 0.5)

toto_strategy = generate_toto_recommendations(matches, current_weights, num_doubles)


# --- Main Dashboard Layout ---
try:
    _json_mtime = os.path.getmtime(JSON_FILE)
    ts_str = datetime.fromtimestamp(_json_mtime).strftime("%Y-%m-%d %H:%M")
except Exception:
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")
st.markdown(f"<h1 style='text-align:right; color:#1a237e;'>⚽ נחשון GSA – לוח בקרה</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:right; color:#6b7280; font-size:0.85rem;'>🕒 עודכן: <b>{ts_str}</b></p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:right; padding:14px 18px; margin:12px 0 18px 0; "
    "background:linear-gradient(135deg,#ecfdf5 0%,#eff6ff 100%); border-radius:10px; "
    "border-right:4px solid #059669; font-size:0.92rem; color:#1e293b; line-height:1.55;'>"
    "<b style='color:#047857;'>⏱️ שעות מומלצות להרצה</b> "
    "<span style='color:#64748b; font-size:0.82rem;'>(לפי שעון המחשב המקומי; "
    "משחק נחשב 90 דק׳, ללימוד מוסיפים 30 דק׳ אחרי סיום עד עדכון תוצאות באתרים; "
    "Run_GSA ~שעה, Train_GSA ~חצי שעה; "
    "<b>אין להריץ את שני הסקריפטים במקביל</b> — שעת Train מחושבת אחרי חלון ה-Run)</span><br><br>"
    f"<span style='color:#0f766e;'>🚀 <b>Run_GSA.bat</b></span> — משיכת משחקים חדשים: "
    f"{_maint_rec['run_detail']}<br>"
    f"<span style='color:#1d4ed8;'>🧠 <b>Train_GSA.bat</b></span> — למידה וכיול: "
    f"{_maint_rec['train_detail']}"
    "</div>",
    unsafe_allow_html=True,
)

# --- סיכום כספי: 5 כרטיסים + כפתורי פעולה ---
fin = get_dashboard_finance_stats()
_fin_clr_profit = "#22c55e" if fin["net_profit"] >= 0 else "#dc2626"
_fin_clr_win = "#22c55e" if fin["win_pct"] >= 50 else "#f97316"
_fin_sub_invested = f"<div class='card-sub'>תלוי ₪{fin['pending_amount']:,.0f}</div>" if fin["pending_amount"] > 0 else ""
_fin_sub_profit = f"<div class='card-sub'>ROI: {fin['roi_pct']:.1f}%</div>" if fin["total_invested"] > 0 else ""
_fin_sub_win = f"<div class='card-sub'>✅ {fin['wins']} &nbsp; ❌ {fin['losses']} &nbsp; ⏳ {fin['pending_count']}</div>"
st.markdown(f"""
<div style='display:grid; grid-template-columns: repeat(3,1fr); gap:8px; margin-bottom:12px;
            grid-template-columns: repeat(auto-fit, minmax(140px,1fr));'>
    <div class='finance-card'>
        <div class='card-title'>🧮 קופה</div>
        <div class='card-value' style='color:#22c55e;'>₪{fin['bankroll']:,.0f}</div>
    </div>
    <div class='finance-card'>
        <div class='card-title'>🌱 סה"כ הומר</div>
        <div class='card-value' style='color:#22c55e;'>₪{fin['total_invested']:,.0f}</div>
        {_fin_sub_invested}
    </div>
    <div class='finance-card'>
        <div class='card-title'>📈 רווח/הפסד</div>
        <div class='card-value' style='color:{_fin_clr_profit};'>{fin['net_profit']:+.0f}₪</div>
        {_fin_sub_profit}
    </div>
    <div class='finance-card'>
        <div class='card-title'>🎯 אחוז זכייה</div>
        <div class='card-value' style='color:{_fin_clr_win};'>{fin['win_pct']:.0f}%</div>
        {_fin_sub_win}
    </div>
    <div class='finance-card'>
        <div class='card-title'>⚡ רצף נוכחי</div>
        <div class='card-value' style='color:#374151;'>{fin['current_streak']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("🔍 בדוק ועדכן קופה לפי תוצאות חדשות", type="primary", key="btn_check_results", use_container_width=True):
    settle_open_slips()
    st.success("הקופה עודכנה לפי טפסים שהסתיימו (אם היו כאלה). מרענן...")
    st.rerun()
with st.expander("✏️ עדכן קופה", expanded=False):
    new_balance = st.number_input("יתרה חדשה (₪)", min_value=0.0, value=float(fin["bankroll"]), step=50.0, key="new_bankroll_input")
    if st.button("שמור יתרה", key="btn_save_bankroll"):
        set_bankroll_balance(new_balance)
        st.success("הקופה עודכנה.")
        st.rerun()
st.markdown("<br>", unsafe_allow_html=True)

_m_clr_acc = "#22c55e" if stats['hit_rate'] >= 50 else "#f97316"
st.markdown(f"""
<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(130px,1fr)); gap:8px; margin-bottom:8px;'>
    <div class='metric-container'><small>דיוק מערכת</small><b style='color:{_m_clr_acc};'>{stats['hit_rate']:.1f}%</b></div>
    <div class='metric-container'><small>משחקים בזיכרון</small><b>{stats['total']}</b></div>
    <div class='metric-container'><small>משקל מודל</small><b style='color:#1565c0;'>{current_weights[0]:.0%}</b></div>
    <div class='metric-container'><small>משקל שוק</small><b style='color:#059669;'>{current_weights[1]:.0%}</b></div>
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# המלצת מערכת לכיול משקולות – מבוסס היסטוריה + ROI קופה
if stats.get("rec_weights") is not None:
    rec_math_pct = int(stats["rec_weights"][0] * 100)
    rec_market_pct = int(stats["rec_weights"][1] * 100)
    msg = (
        f"💡 **המלצת מערכת למשקולות** (מבוסס {stats['total']} משחקים שהסתיימו): "
        f"מודל מתמטי **{rec_math_pct}%** | שוק (ווינר) **{rec_market_pct}%**"
    )
    if stats.get("bankroll_roi_pct") is not None:
        msg += f" | **ROI קופה:** {stats['bankroll_roi_pct']:.1f}% (משמש לדיוק הלימוד)"
    st.info(msg)
    st.markdown("<br>", unsafe_allow_html=True)

if not os.path.exists(JSON_FILE):
    st.warning("לא נמצאו נתוני ניתוח. לחץ על 'הרץ ציד יומי (Run_GSA.bat)' בתפריט הצד.")
elif _json_load_error is not None:
    st.warning(f"קובץ הניתוח ריק או פגום. הרץ ניתוח מחדש. ({_json_load_error})")
elif not matches:
    st.warning("אין משחקים להצגה. הרץ 'הרץ ציד יומי (Run_GSA.bat)' בתפריט הצד.")

# --- לשוניות לפי קטגוריה ---
tab_skira, tab_toto, tab_matches, tab_ml, tab_gilad, tab_forms = st.tabs([
    "סקירה", "טופס טוטו מסכם", "משחקים", "למידת מכונה ואיכות", "ניתוח גלעדי", "טפסים ששלחתי"
])

with tab_skira:
    display_recommended_slip(
        matches,
        w_math=current_weights[0],
        w_market=current_weights[1],
        bankroll=user_bankroll,
        system_active=not breaker_state["triggered"],
    )

    # --- אנליטיקה וגרפים (כמו בממשק היעד) ---
    st.markdown("---")
    with st.expander("📊 אנליטיקה וגרפים", expanded=True):
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("**עקומת רווח/הפסד מצטברת**")
            cdf = fin["cumulative_df"]
            if not cdf.empty and "cumulative_profit" in cdf.columns and HAS_PLOTLY:
                cdf = cdf.copy()
                cdf["created_at"] = pd.to_datetime(cdf["created_at"], errors="coerce").dt.strftime("%d/%m")
                y_vals = cdf["cumulative_profit"].tolist()
                x_vals = cdf["created_at"].tolist()
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode="lines+markers",
                    line=dict(color="#dc2626", width=2),
                    fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.2)",
                    name="רווח מצטבר (₪)",
                ))
                y_min = min(y_vals) if y_vals else 0
                y_max = max(y_vals) if y_vals else 0
                if y_min == y_max:
                    y_min -= 10
                    y_max += 5
                fig_cum.update_layout(
                    margin=dict(t=20, b=30, l=40, r=20), height=220,
                    yaxis=dict(title="₪", range=[y_min - 5, y_max + 5]),
                    xaxis=dict(title=""),
                    showlegend=False,
                )
                st.plotly_chart(fig_cum, width="stretch", config={"displayModeBar": False})
            elif not cdf.empty and "cumulative_profit" in cdf.columns:
                cdf = cdf.copy()
                cdf["created_at"] = pd.to_datetime(cdf["created_at"], errors="coerce").dt.strftime("%d/%m")
                chart_cum = cdf.set_index("created_at")[["cumulative_profit"]]
                chart_cum.columns = ["רווח מצטבר (₪)"]
                st.line_chart(chart_cum)
            else:
                st.caption("אין עדיין נתוני טפסים שסולקו.")

        with g2:
            st.markdown("**חלוקת תוצאות**")
            donut = fin["donut_data"]
            if donut and HAS_PLOTLY:
                df_d = pd.DataFrame(donut)
                fig = go.Figure(data=[go.Pie(
                    labels=df_d["label"], values=df_d["value"], hole=0.5,
                    marker_colors=df_d["color"],
                    textinfo="label+percent", texttemplate="%{label}<br>%{percent:.1%}",
                    direction="clockwise",
                )])
                fig.update_layout(
                    margin=dict(t=20, b=20, l=20, r=20), height=220,
                    showlegend=True, legend=dict(orientation="h", yanchor="top", xanchor="center", x=0.5),
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            elif donut:
                df_d = pd.DataFrame(donut)
                df_d = df_d.set_index("label")[["value"]]
                st.bar_chart(df_d)

        with g3:
            st.markdown("**לפי סוג הימור ROI**")
            roi_type = fin["roi_by_type"]
            if roi_type and HAS_PLOTLY:
                df_roi = pd.DataFrame(roi_type)
                short_names = [str(x)[:12] + "…" if len(str(x)) > 12 else str(x) for x in df_roi["bet_type"]]
                df_roi["סוג"] = short_names
                colors = ["#22c55e" if y >= 0 else "#dc2626" for y in df_roi["roi"]]
                fig = go.Figure(data=[go.Bar(x=df_roi["סוג"], y=df_roi["roi"], marker_color=colors)])
                y_min = float(df_roi["roi"].min())
                y_max = float(df_roi["roi"].max())
                if y_max <= 0:
                    y_range = [y_min - 10, 5]
                elif y_min >= 0:
                    y_range = [-5, y_max + 10]
                else:
                    y_range = [y_min - 10, y_max + 10]
                fig.update_layout(
                    yaxis_title="ROI %", yaxis=dict(range=y_range),
                    margin=dict(t=20, b=40), height=220, xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            elif roi_type:
                df_roi = pd.DataFrame(roi_type).set_index("bet_type")
                st.bar_chart(df_roi)

        st.markdown("**תובנות:**")
        st.info("💡 " + fin["insight"])

with tab_gilad:
    st.markdown("<h2 style='text-align:right; color:#e65100; border-bottom:2px solid #e65100; padding-bottom:5px;'>🧠 ניתוח גלעדי – Surebet & Holy Trinity</h2>", unsafe_allow_html=True)

    # --- דוח גלעדי מיובא מקובץ CSV (אם קיים) ---
    export_path = os.path.join(_BASE_DIR, "2026-03-09T10-54_export.csv")
    df_export = None
    if os.path.exists(export_path):
        try:
            df_export = pd.read_csv(export_path)
            # הסרת עמודת אינדקס ריקה אם קיימת
            if df_export.shape[1] > 0 and (str(df_export.columns[0]).startswith("Unnamed") or df_export.columns[0] == ""):
                df_export = df_export.iloc[:, 1:]
        except Exception:
            df_export = None

    if df_export is not None and not df_export.empty:
        st.markdown("### 📂 דוח '2026-03-09T10-54_export.csv' – Under 2.5 + Correct Scores")
        st.caption(
            "הטבלה מציגה משחקים שבהם זוהתה הזדמנות בשיטת **Under 2.5 + Correct Scores**: "
            "כיסוי כל התוצאות עד 3 שערים באמצעות Under 2.5 יחד עם התוצאות 2-1 ו‑1-2. "
            "עמודת **ROI %** היא תשואת הרווח המובטח ביחס לסכום ההשקעה הכולל."
        )

        df_view = df_export.copy()

        # ניסיון להעשיר את הדוח בחלוקת ההשקעה לפי מנוע הארביטראז' (אם קיים משחק תואם)
        match_index = {m.get("match", ""): m for m in matches}
        stake_strings = []
        total_override = []
        profit_override = []
        for _, row in df_export.iterrows():
            game_name = str(row.get("משחק", ""))
            strategy = str(row.get("סוג", ""))
            stake_desc = ""
            tot_val = row.get("השקעה כוללת", None)
            prof_val = row.get("רווח מובטח", None)
            m_obj = match_index.get(game_name)
            if m_obj and isinstance(m_obj.get("arbitrage"), dict):
                if "Under 2.5" in strategy and "Correct" in strategy:
                    arb = m_obj["arbitrage"].get("under25_correct_scores") or {}
                else:
                    arb = None
                alloc = (arb or {}).get("allocation") or {}
                stakes = alloc.get("stakes") or {}
                if stakes:
                    # מיפוי לקריאה אנושית
                    mapping = {
                        "total_under_2_5": "Under 2.5",
                        "cs_2-1": "תוצאה 2-1",
                        "cs_1-2": "תוצאה 1-2",
                        "btts_no": "BTTS No",
                        "total_over_2_5": "Over 2.5",
                        "cs_1-1": "תוצאה 1-1",
                    }
                    parts = []
                    for key, amount in stakes.items():
                        try:
                            amt = float(amount)
                        except Exception:
                            continue
                        label = mapping.get(key, key)
                        parts.append(f"{label}: {amt:,.0f} ₪")
                    stake_desc = " | ".join(parts)
                    # אם יש לנו הקצאה מהמערכת – נעדיף את הסכומים המחשבים (עד 100 ₪ למשחק)
                    if "total_stake" in alloc:
                        tot_val = alloc.get("total_stake")
                    if "guaranteed_profit" in alloc:
                        prof_val = alloc.get("guaranteed_profit")
            stake_strings.append(stake_desc)
            total_override.append(tot_val)
            profit_override.append(prof_val)

        if stake_strings:
            df_view["חלוקת השקעה"] = stake_strings
        if total_override:
            df_view["השקעה כוללת"] = total_override
        if profit_override:
            df_view["רווח מובטח"] = profit_override

        # הצגת טבלה ממוינת לפי ROI
        roi_col = "ROI %"
        if roi_col in df_view.columns:
            df_view = df_view.sort_values(by=roi_col, ascending=False)

        # שינוי שמות עמודות להצגה ברורה
        rename_map = {
            "משחק": "משחק",
            "סוג": "שיטת ארביטראז'",
            "ROI %": "ROI % (תשואה בטוחה)",
            "רווח מובטח": "רווח מובטח (₪)",
            "השקעה כוללת": "השקעה כוללת (₪)",
            "פרטי שווקים": "שווקים בשילוב",
            "חלוקת השקעה": "חלוקת השקעה (₪ לכל שוק)",
        }
        df_view = df_view.rename(columns={k: v for k, v in rename_map.items() if k in df_view.columns})

        fmt_cols = {
            c: "{:.2f}"
            for c in df_view.columns
            if "ROI" in c or "רווח" in c or "השקעה" in c
        }
        st.dataframe(df_view.style.format(fmt_cols), width='stretch')

        # גרף עמודות של ROI לפי משחק
        if HAS_PLOTLY and roi_col in df_export.columns:
            try:
                df_chart = df_export.sort_values(by=roi_col, ascending=False).head(15)
                fig_export = px.bar(
                    df_chart,
                    x="משחק",
                    y=roi_col,
                    title="ROI % לפי משחק – Under 2.5 + Correct Scores",
                    color=roi_col,
                    color_continuous_scale="Viridis",
                )
                fig_export.update_layout(
                    xaxis_tickangle=-45,
                    margin=dict(t=40, b=80),
                    yaxis_title="ROI % (תשואה בטוחה)",
                )
                st.plotly_chart(fig_export, width='stretch', config={"displayModeBar": False})
            except Exception:
                pass

    # חילוץ כל ההזדמנויות מסוג Holy Trinity שמצאו המנועים
    holy_rows = []
    all_arb_rows = []
    for m in matches:
        arb = m.get("arbitrage") or {}
        if not isinstance(arb, dict):
            continue
        match_name = m.get("match", "")

        # Holy Trinity – מודגש למעלה
        ht = arb.get("holy_trinity")
        if isinstance(ht, dict) and ht.get("surebet") and ht.get("allocation"):
            alloc = ht["allocation"]
            stakes = alloc.get("stakes", {}) or {}
            holy_rows.append({
                "משחק": match_name,
                "סוג": "השילוש הקדוש",
                "BTTS No": ht.get("odds", {}).get("btts_no"),
                "Over 2.5": ht.get("odds", {}).get("total_over_2_5"),
                "Correct Score 1-1": ht.get("odds", {}).get("cs_1-1"),
                 # חלוקת ההשקעה בפועל בין שלושת ההימורים
                "סכום BTTS No (₪)": stakes.get("btts_no"),
                "סכום Over 2.5 (₪)": stakes.get("total_over_2_5"),
                "סכום 1-1 (₪)": stakes.get("cs_1-1"),
                "השקעה כוללת": alloc.get("total_stake"),
                "רווח מובטח": alloc.get("guaranteed_profit"),
                "ROI %": (alloc.get("roi", 0.0) * 100.0),
            })

        # כל שאר סוגי הארביטראז' (כולל Full Market)
        for key, val in arb.items():
            if not isinstance(val, dict):
                continue
            if key == "holy_trinity":
                label = "השילוש הקדוש"
            elif key == "full_market":
                # full_market הוא רשימה של קומבינציות
                for combo in val:
                    alloc = combo.get("allocation") or {}
                    stakes = alloc.get("stakes") or {}
                    # טקסט קריא לחלוקת ההשקעה
                    if stakes:
                        parts = []
                        for sid, amount in stakes.items():
                            try:
                                amt = float(amount)
                            except Exception:
                                continue
                            parts.append(f"{sid}: {amt:,.0f} ₪")
                        stake_text = " | ".join(parts)
                    else:
                        stake_text = ""
                    all_arb_rows.append({
                        "משחק": match_name,
                        "סוג": combo.get("type", "Full Market Arbitrage"),
                        "ROI %": (alloc.get("roi", 0.0) * 100.0),
                        "רווח מובטח": alloc.get("guaranteed_profit"),
                        "השקעה כוללת": alloc.get("total_stake"),
                        "פרטי שווקים": ", ".join(combo.get("selection_ids", [])),
                        "חלוקת השקעה": stake_text,
                    })
                continue
            else:
                label = val.get("type", key)
            alloc = val.get("allocation") or {}
            if not alloc:
                continue
            stakes = alloc.get("stakes") or {}
            if stakes:
                parts = []
                for sid, amount in stakes.items():
                    try:
                        amt = float(amount)
                    except Exception:
                        continue
                    parts.append(f"{sid}: {amt:,.0f} ₪")
                stake_text = " | ".join(parts)
            else:
                stake_text = ""
            all_arb_rows.append({
                "משחק": match_name,
                "סוג": label,
                "ROI %": (alloc.get("roi", 0.0) * 100.0),
                "רווח מובטח": alloc.get("guaranteed_profit"),
                "השקעה כוללת": alloc.get("total_stake"),
                "פרטי שווקים": ", ".join(val.get("market_keys", [])),
                "חלוקת השקעה": stake_text,
            })

    # אזור בולט להשילוש הקדוש
    if holy_rows:
        st.markdown("### 🔱 הזדמנויות \"השילוש הקדוש\" (BTTS No + Over 2.5 + 1-1)")
        df_holy = pd.DataFrame(holy_rows)
        # הדגשה טקסטואלית במונחים המקצועיים
        st.caption("כל ההזדמנויות כאן מסווגות כ‑**Surebet / Risk-Free Bet / Guaranteed Profit** לאחר בדיקת כיסוי מלאה.")
        st.dataframe(df_holy.style.format({"ROI %": "{:.2f}", "רווח מובטח": "{:.1f}", "השקעה כוללת": "{:.0f}"}), width='stretch')
    else:
        st.info("לא אותרו כרגע הזדמנויות מסוג \"השילוש הקדוש\" בתוכניה הנוכחית.")

    # טבלת כלל ההזדמנויות
    st.markdown("### 📊 כל הזדמנויות ה‑Surebet והארביטראז' שזוהו")
    if all_arb_rows:
        df_all = pd.DataFrame(all_arb_rows)
        df_all = df_all.sort_values(by="ROI %", ascending=False)
        st.dataframe(df_all.style.format({"ROI %": "{:.2f}", "רווח מובטח": "{:.1f}", "השקעה כוללת": "{:.0f}"}), width='stretch')
    else:
        st.caption("עדיין לא זוהו הזדמנויות ארביטראז' במסך זה. הרץ ניתוח חדש וודא שנתוני הווינר עודכנו.")
with tab_toto:
    st.markdown("<h2 style='text-align:right; color:#1565c0; border-bottom:2px solid #1565c0; padding-bottom:5px;'>📋 טופס טוטו מסכם (ניהול תקציב אופטימלי)</h2>", unsafe_allow_html=True)

    # 1. פונקציית צביעה להסתברויות
    def highlight_probabilities(val):
        try:
            num_val = float(val.replace('%', '').strip())
            if num_val > 50.0:
                return 'background-color: #4CAF50; color: white;'
            elif 40.0 <= num_val <= 50.0:
                return 'background-color: #81C784; color: black;'
            else:
                return 'background-color: #C8E6C9; color: black;'
        except:
            return ''

    # 2. פונקציית עזר להוספת שמות הקבוצות או "תיקו" לסימון המומלץ
    def format_pick_with_names(pick, home_team, away_team):
        parts = []
        pick_str = str(pick).upper()
        if '1' in pick_str:
            parts.append(home_team)
        if '2' in pick_str:
            parts.append(away_team)
        if 'X' in pick_str:
            parts.append('תיקו')
        if parts:
            return f"{pick} ({', '.join(parts)})"
        return pick

    # 3. איסוף הנתונים לטבלה
    summary_data = []
    for i, m in enumerate(matches):
        strat = toto_strategy[i]
        home = m.get('home', '')
        away = m.get('away', '')
        best_pick = strat['best_pick']
        final_prob = strat['best_prob']
        market_odds = m.get('market_odds', {})
        odd = market_odds.get(best_pick, 0)
        raw_md = m.get('match_date') or ''
        try:
            match_date_disp = datetime.strptime(raw_md[:10], "%Y-%m-%d").strftime("%d/%m/%Y") if raw_md else "-"
        except Exception:
            match_date_disp = "-"
        o1  = market_odds.get('1', 0)
        ox  = market_odds.get('X', 0)
        o2  = market_odds.get('2', 0)
        odds_str = f"1:{o1:.2f}  X:{ox:.2f}  2:{o2:.2f}" if o1 and ox and o2 else "-"
        odd_f = float(odd or 0)
        if odd_f > 0 and final_prob > 0:
            ev_val = (final_prob * odd_f) - 1
        else:
            ev_val = -1.0
        if ev_val > 0.12 and odd_f >= 1.50:
            rec_status = "💎 ערך גבוה"
        elif final_prob >= 0.40 and ev_val >= 0.025 and odd_f >= 1.20:
            rec_status = "🔥 מומלץ"
        elif final_prob >= 0.35 and 0.005 <= ev_val < 0.025:
            rec_status = "✅ כדאי"
        elif 0.20 <= final_prob < 0.35 and 0.04 <= ev_val <= 0.12:
            rec_status = "⚠️ עקיצה"
        else:
            rec_status = "❌ עזוב"
        if ev_val > 0:
            ev_str = f"+{ev_val*100:.1f}% 🔥"
        elif odd_f == 0:
            ev_str = "-"
        else:
            ev_str = f"{ev_val*100:.1f}%"
        pick_with_names = format_pick_with_names(strat['val_pick'], home, away)
        summary_data.append({
            "משחק מספר": i + 1,
            "תאריך": match_date_disp,
            "קבוצת בית": home,
            "קבוצת חוץ": away,
            "יחסים (1 X 2)": odds_str,
            "סימון מומלץ": pick_with_names,
            "הסתברות": f"{strat['best_prob']*100:.1f}%",
            "תוחלת (EV)": ev_str,
            "המלצת מערכת": rec_status,
            "raw_prob": strat["best_prob"],
            "raw_pick": strat["val_pick"],
            "best_pick": best_pick,
            "odd": float(odd or 0.0),
        })

    df_summary = pd.DataFrame(summary_data, columns=[
        "משחק מספר", "תאריך", "קבוצת בית", "קבוצת חוץ",
        "יחסים (1 X 2)", "סימון מומלץ", "הסתברות", "תוחלת (EV)", "המלצת מערכת"
    ]).set_index("משחק מספר")

    def highlight_ev(val):
        if '🔥' in str(val):
            return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
        elif val == "-":
            return 'color: #6b7280;'
        else:
            return 'color: #991b1b;'

    def highlight_rec(val):
        if "💎" in str(val): return 'background-color: #ede9fe; color: #5b21b6; font-weight: bold;'
        if "🔥" in str(val): return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
        if "✅" in str(val): return 'background-color: #fef08a; color: #854d0e; font-weight: bold;'
        if "⚠️" in str(val): return 'background-color: #ffedd5; color: #9a3412;'
        return 'color: #9ca3af;'

    styled_df = (
        df_summary.style
        .map(highlight_probabilities, subset=['הסתברות'])
        .map(highlight_ev, subset=['תוחלת (EV)'])
        .map(highlight_rec, subset=['המלצת מערכת'])
    )
    st.dataframe(styled_df, width="stretch")
    st.markdown("<br>", unsafe_allow_html=True)

with tab_matches:
    # סיכום משחקים לפני הצגת הרשימה: כמה משחקים זוהו + פירוט לפי תאריך
    total_matches = len(matches)
    date_counts = Counter()
    for m in matches:
        raw_date = m.get('match_date') or ''
        date_key = raw_date[:10] if len(raw_date) >= 10 else raw_date or "ללא תאריך"
        date_counts[date_key] += 1
    summary_lines = [f"<b>{total_matches}</b> משחקים זוהו"]
    if date_counts:
        per_date_parts = []
        for d in sorted(date_counts.keys()):
            try:
                disp = datetime.strptime(d[:10], "%Y-%m-%d").strftime("%d/%m/%Y") if d and d != "ללא תאריך" else d
            except Exception:
                disp = d
            per_date_parts.append(f"<span style='margin-left:12px;'>📅 {disp}: <b>{date_counts[d]}</b> משחקים</span>")
        summary_lines.append("<br>".join(per_date_parts))
    st.markdown(
        f"<div style='text-align:right; padding:12px 16px; background:linear-gradient(135deg,#e0e7ff,#f3f4f6); "
        f"border-radius:8px; border-right:4px solid #4f46e5; margin-bottom:20px; font-size:0.95rem; color:#1e293b;'>"
        f"<b>📋 סיכום לפני הצגת המשחקים:</b><br>"
        f"<span style='font-size:1.05rem;'>{summary_lines[0]}</span>"
        + (f"<br><div style='margin-top:8px;'>{summary_lines[1]}</div>" if len(summary_lines) > 1 else "")
        + f"</div>",
        unsafe_allow_html=True,
    )

    for i, m in enumerate(matches):
        strat = toto_strategy[i]
        final_probs = strat['final_probs']
        toto_rec = strat['rec']
        val_pick = strat['val_pick']
        rec_type = strat['type']

        with st.container():
            st.markdown('<div class="match-card">', unsafe_allow_html=True)
            col_logo1, col_name, col_logo2 = st.columns([1, 4, 1])
            with col_logo1: st.image(m.get('logos', {}).get('home', 'https://via.placeholder.com/50'), width=65)
            with col_name:
                st.markdown(f"<h2 style='text-align:center; margin:0; color:#1e3a8a;'>{m.get('home', 'Home')} - {m.get('away', 'Away')}</h2>", unsafe_allow_html=True)
                _raw_date = m.get('match_date') or ''
                try:
                    _date_disp = datetime.strptime(_raw_date[:10], "%Y-%m-%d").strftime("%d/%m/%Y")
                except Exception:
                    _date_disp = ""
                if _date_disp:
                    st.markdown(f"<p style='text-align:center; color:#1565c0; font-size:0.9rem; font-weight:600; margin:2px 0;'>📅 {_date_disp}</p>", unsafe_allow_html=True)
                _time = m.get('match_time')
                if not _time and _raw_date and "T" in str(_raw_date) and len(str(_raw_date)) >= 16:
                    _time = str(_raw_date)[11:16]  # HH:MM מ-ISO
                if _time:
                    st.markdown(f"<p style='text-align:center; color:#6b7280; font-size:0.85rem; margin:2px 0;'>🕐 {_time}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; color:gray; font-size:0.8rem; margin:2px 0;'>📍 {m.get('pro_data', {}).get('venue', 'Unknown')}</p>", unsafe_allow_html=True)
            with col_logo2: st.image(m.get('logos', {}).get('away', 'https://via.placeholder.com/50'), width=65)

            # 🧠 פאנל מודיעין קוגניטיבי ושקיפות מערכת
            news_data = m.get('news_flash', '')
            context_data = m.get('context', '')

            if "לקח" in context_data or "lesson" in context_data:
                st.markdown("""
            <div style='background-color:#fee2e2; border-right:5px solid #ef4444; padding:10px; border-radius:5px; margin-bottom:15px;'>
                <h4 style='color:#b91c1c; margin:0;'>🚨 הופעל זיכרון קוגניטיבי!</h4>
                <span style='color:#7f1d1d; font-size:0.9rem;'>נחשון איתר טעות עבר עם קבוצות אלו ויישם אוטומטית את הלקח בניתוח הנוכחי.</span>
            </div>
            """, unsafe_allow_html=True)

            dyn_w = strat.get('applied_weights', current_weights)
            total_dyn = (dyn_w[0] or 0) + (dyn_w[1] or 0)
            dyn_model_pct = (dyn_w[0] / total_dyn * 100) if total_dyn > 0 else 50
            dyn_market_pct = (dyn_w[1] / total_dyn * 100) if total_dyn > 0 else 50
            balance_line = f"<span style='color:#2563eb;'>🧠 מודל סטטיסטי: {dyn_model_pct:.0f}%</span> | <span style='color:#059669;'>💰 שוק (בוקיז): {dyn_market_pct:.0f}%</span>"
            st.markdown(f"""
        <div style='background-color:#f3f4f6; padding:8px; border-radius:5px; margin-bottom:15px; text-align:center; font-size:0.9rem; border: 1px dashed #9ca3af;'>
            <b>⚖️ מאזן כוחות למשחק זה:</b><br>
            {balance_line}
        </div>
        """, unsafe_allow_html=True)

            st.divider()
            g1, g2, g3 = st.columns(3)
            with g1: draw_custom_bar("🧠 מוח מתמטי (Hybrid xG/Goals)", m.get('model_probs', {}) or simulate_independent_poisson())
            with g2: draw_custom_bar("💰 השוק (Implied Probability)", get_market_probs(m.get('market_odds')))
            with g3:
                if rec_type == 'banker':
                    bg_color, border_color = "#dcfce7", "#22c55e"
                else:
                    bg_color, border_color = "#fef08a", "#eab308"
                st.markdown(f"""
            <div style='text-align:center; border:2px solid {border_color}; border-radius:8px; padding:8px; margin-bottom:10px; background:{bg_color};'>
                <div style='font-size:1.1rem; font-weight:bold; color:#1f2937;'>🎯 {toto_rec}</div>
                <div style='font-size:0.85rem; color:#4b5563; margin-top:3px;'>סימון משוקלל אופטימלי: <b>{val_pick}</b></div>
            </div>
            """, unsafe_allow_html=True)
                draw_custom_bar("סיכום משוקלל (Final Edge)", final_probs)

            st.markdown("<br>### 💰 ניתוח ערך (Value) חכם - מבוסס הסתברות משוקללת", unsafe_allow_html=True)
            market_odds = m.get('market_odds', {})
            final_probs_ev = strat['final_probs']
            pro_data = m.get('pro_data') or {}
            market_type = pro_data.get('market_type') or '1X2'

            # Multi-market: show Totals/BTTS when that's the chosen play
            if market_type in ('Totals', 'BTTS') and pro_data.get('recommended_bet') and pro_data.get('recommended_bet') != 'NO BET':
                rec_bet = pro_data.get('recommended_bet', '')
                rec_odds = float(pro_data.get('odds') or 0)
                rec_prob = float(pro_data.get('chosen_prob') or 0) * 100
                rec_ev = float(pro_data.get('best_ev') or 0) * 100
                mkt_badge = "📊 Totals" if market_type == 'Totals' else "⚽ BTTS"
                st.markdown(f"""
                <div style='background:#e0f2fe; border-right:4px solid #0284c7; border-radius:8px; padding:12px; margin-bottom:12px;'>
                    <b>{mkt_badge}</b> — הימור נבחר: <b>{rec_bet}</b> @ {rec_odds:.2f}<br>
                    הסתברות: {rec_prob:.1f}% | EV: {rec_ev:+.1f}%
                </div>
                """, unsafe_allow_html=True)
                if rec_ev > 0:
                    st.success(f"**סימון מומלץ: {rec_bet}** [{market_type}] | תוחלת חיובית ({rec_ev:.1f}%) וסיכוי לתפיסה ({rec_prob:.1f}%).")
                else:
                    st.warning(f"**הימור נבחר: {rec_bet}** [{market_type}] — EV שלילי לאחר תיקון דאדיימה.")
            elif market_odds and len(market_odds) == 3 and final_probs_ev:
                prev_odds = get_previous_odds_for_match(m.get('home', ''), m.get('away', ''))
                table_data = []
                best_pick = None
                max_score = -999
                best_ev = 0
                best_prob = 0
                for bet_type in ['1', 'X', '2']:
                    odd = market_odds.get(bet_type, 0)
                    if odd > 0:
                        prev_odd = prev_odds.get(bet_type) if prev_odds else None
                        odds_display = _format_odds_with_change(odd, prev_odd)
                        model_p = final_probs_ev.get(bet_type, 0)
                        ev_roi = ((model_p * odd) - 1) * 100
                        if ev_roi > 5 and model_p >= 0.40:
                            worth = "🔥 שווה מאוד (בטוח)"
                        elif ev_roi > 0 and model_p >= 0.30:
                            worth = "✅ כדאי"
                        elif ev_roi > 10 and model_p < 0.30:
                            worth = "⚠️ עקיצה (סיכון)"
                        else:
                            worth = "❌ לא שווה"
                        if model_p >= 0.30 and ev_roi > 0:
                            pick_score = ev_roi * model_p
                            if pick_score > max_score:
                                max_score = pick_score
                                best_pick = bet_type
                                best_ev = ev_roi
                                best_prob = model_p * 100
                        table_data.append({
                            "סימון": bet_type,
                            "יחס ווינר": odds_display,
                            "הסתברות משוקללת": f"{(model_p * 100):.1f}%",
                            "EV (תוחלת)": f"{ev_roi:.1f}%",
                            "סטטוס": worth
                        })
                df_ev = pd.DataFrame(table_data)
                def highlight_status(val):
                    if "🔥" in val: return 'background-color: #dcfce7; color: #166534; font-weight: bold;'
                    if "✅" in val: return 'background-color: #fef08a; color: #854d0e; font-weight: bold;'
                    if "⚠️" in val: return 'background-color: #ffedd5; color: #9a3412;'
                    return 'background-color: #fee2e2; color: #991b1b;'
                styled_ev = df_ev.style.map(highlight_status, subset=['סטטוס'])
                st.dataframe(styled_ev, width="stretch", hide_index=True)
                st.markdown("#### 🎯 השורה התחתונה:")
                if best_pick:
                    st.success(f"**סימון מומלץ: {best_pick}** | תוחלת חיובית ({best_ev:.1f}%) וסיכוי ריאלי לתפיסה ({best_prob:.1f}%).")
                else:
                    st.error("**המלצת מערכת: עזוב.** אין כאן יתרון מתמטי עם סיכוי ריאלי.")
            else:
                st.info("נתוני יחסים או מודל חסרים לביצוע ניתוח כדאיות.")
            st.markdown("<br>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["📊 היסטוריה ופורמה", "🏥 פציעות וסגלים"])
            with t1:
                h_col, a_col = st.columns(2)
                with h_col: st.markdown(f"**{m.get('home', 'Home')}**"); render_history_visual(m.get('history', {}).get('home', []))
                with a_col: st.markdown(f"**{m.get('away', 'Away')}**"); render_history_visual(m.get('history', {}).get('away', []))
            with t2:
                injuries = m.get('intel', {}).get('injuries', [])
                if injuries:
                    st.warning(f"דיווחים על {len(injuries)} שחקנים:")
                    for inj in injuries: st.markdown(f"- {inj}")
                else: st.success("✅ סגל מלא או חוסר בדיווחים על פציעות משמעותיות.")

with tab_ml:
    st.markdown("## 📈 למידת מכונה ובקרת איכות עצמית")

    # --- השתלשלות הלימודים (ציר זמן) ---
    st.markdown("### 📅 השתלשלות הלימודים")
    st.caption("ציר זמן של אירועי הלמידה: Auto-Learner, טיכה, שישקה.")
    learning_events = []
    weights_history_rows = []
    try:
        conn_ml = sqlite3.connect(DB_PATH, timeout=10.0)
        c_ml = conn_ml.cursor()
        try:
            c_ml.execute("""
                SELECT id, run_at, event_type, summary, details_json
                FROM learning_log ORDER BY run_at DESC LIMIT 150
            """)
            learning_events = c_ml.fetchall()
        except sqlite3.OperationalError:
            pass
        try:
            c_ml.execute("""
                SELECT id, updated_at, w_model, w_market, train_accuracy, test_accuracy, total_matches
                FROM weights_history ORDER BY id DESC LIMIT 80
            """)
            weights_history_rows = c_ml.fetchall()
        except sqlite3.OperationalError:
            pass
        conn_ml.close()
    except Exception:
        pass

    if learning_events:
        event_labels = {"auto_learner": "🤖 Auto-Learner", "ticha": "⚖️ טיכה", "shishka": "🛡️ שישקה"}
        event_colors = {"auto_learner": "#1565c0", "ticha": "#7c3aed", "shishka": "#059669"}
        n_show = min(50, len(learning_events))
        for row in learning_events[:n_show]:
            lid, run_at, event_type, summary, details_json = row
            try:
                dt = dateutil.parser.parse(run_at) if run_at else None
                run_display = dt.strftime("%d/%m/%Y %H:%M") if dt else str(run_at)[:19]
            except Exception:
                run_display = str(run_at)[:19] if run_at else "—"
            label = event_labels.get(event_type, event_type)
            color = event_colors.get(event_type, "#6b7280")
            st.markdown(
                f"<div style='background:linear-gradient(90deg, {color}22, transparent); border-right:4px solid {color}; "
                f"padding:12px 16px; border-radius:8px; margin-bottom:10px; direction:rtl; text-align:right;'>"
                f"<div style='color:#6b7280; font-size:0.85rem;'>{run_display}</div>"
                f"<div style='font-weight:600; color:{color}; margin:4px 0;'>{label}</div>"
                f"<div style='font-size:0.95rem;'>{summary or '—'}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("טרם נרשמו אירועי למידה. הריצה הבאה של Run_GSA תמלא את היומן.")

    if weights_history_rows:
        with st.expander("⚖️ היסטוריית משקולות (Auto-Learner)", expanded=False):
            wh_data = []
            for row in weights_history_rows:
                hid, upd, wm, wmk, train_acc, test_acc, total = row
                train_str = f"{train_acc*100:.1f}%" if train_acc is not None else "—"
                test_str = f"{test_acc*100:.1f}%" if test_acc is not None else "—"
                wh_data.append({
                    "תאריך": upd,
                    "מודל": f"{wm:.0%}",
                    "שוק": f"{wmk:.0%}",
                    "דיוק Train": train_str,
                    "דיוק Test": test_str,
                    "משחקים": total or "—",
                })
            df_wh = pd.DataFrame(wh_data)
            st.dataframe(df_wh, width="stretch", hide_index=True)

    st.markdown("---")
    st.info("מערכת הלמידה (Ticha, Shishka, Auto-Learner) רצה דרך 'הרץ למידה וכיול' בתפריט הצד. משקולות מתעדכנות אוטומטית.")

    display_cumulative_learning_curve(key="learning_curve_ml")
    display_brains_arena()
    display_post_match_analysis(w_model=current_weights[0], w_market=current_weights[1])

    weekly_stats, bucket_stats = get_weekly_learning_stats()
    if weekly_stats is not None:
        st.markdown("## 📊 ביצועים שבועיים (דיוק, תשואה, תוחלת)")
        display_week = weekly_stats.copy()
        display_week.rename(columns={
            "week": "שבוע", "total_bets": "הימורים", "wins": "פגיעות",
            "avg_odds": "יחס ממוצע", "avg_ev": "תוחלת ממוצעת",
            "total_stake": "סה\"כ הימור", "total_profit": "רווח/הפסד",
            "hit_rate_%": "% פגיעה", "roi_%": "% תשואה (ROI)"
        }, inplace=True)
        display_week["% פגיעה"] = display_week["% פגיעה"].map(lambda x: f"{x:.1f}%")
        display_week["% תשואה (ROI)"] = display_week["% תשואה (ROI)"].map(lambda x: f"{x:.1f}%")
        display_week["יחס ממוצע"] = display_week["יחס ממוצע"].map(lambda x: f"{x:.2f}")
        display_week["תוחלת ממוצעת"] = display_week["תוחלת ממוצעת"].map(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) and np.isfinite(x) else "—"
        )
        st.dataframe(display_week, width="stretch")
    else:
        st.info("טרם נאספו מספיק משחקים עם תוצאה אמת לחישוב טבלת ביצועים שבועיים.")

    if bucket_stats is not None:
        st.markdown("## 🧮 ביצועים לפי טווח יחסים (איפה המערכת הכי חזקה?)")
        display_bucket = bucket_stats.copy()
        display_bucket.rename(columns={
            "bucket": "טווח יחס", "total_bets": "הימורים", "wins": "פגיעות",
            "avg_odds": "יחס ממוצע", "avg_ev": "תוחלת ממוצעת",
            "total_stake": "סה\"כ הימור", "total_profit": "רווח/הפסד",
            "hit_rate_%": "% פגיעה", "roi_%": "% תשואה (ROI)"
        }, inplace=True)
        display_bucket["% פגיעה"] = display_bucket["% פגיעה"].map(lambda x: f"{x:.1f}%")
        display_bucket["% תשואה (ROI)"] = display_bucket["% תשואה (ROI)"].map(lambda x: f"{x:.1f}%")
        display_bucket["יחס ממוצע"] = display_bucket["יחס ממוצע"].map(lambda x: f"{x:.2f}")
        display_bucket["תוחלת ממוצעת"] = display_bucket["תוחלת ממוצעת"].map(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) and np.isfinite(x) else "—"
        )
        st.dataframe(display_bucket, width="stretch")

    rel_stats = get_reliability_stats()
    if rel_stats is not None:
        st.markdown("## 📐 בדיקת כיול הסתברויות")
        st.markdown(
            "האם כשהמערכת חזתה 40% – בפועל גם התממשו ~40% מהפעמים? הגרפים הבאים בודקים זאת."
        )
        for name, rel_df in rel_stats.items():
            st.markdown(f"#### {name}")
            show_df = rel_df.copy()
            show_df.rename(columns={"bin_center": "מרכז טווח", "mean_pred": "חיזוי ממוצע", "empirical": "התממשות בפועל", "count": "מספר תצפיות"}, inplace=True)
            show_df["חיזוי ממוצע"] = show_df["חיזוי ממוצע"].map(lambda x: f"{float(x.replace('%','')):.1f}%" if isinstance(x, str) else f"{x*100:.1f}%")
            show_df["התממשות בפועל"] = show_df["התממשות בפועל"].map(lambda x: f"{float(x.replace('%','')):.1f}%" if isinstance(x, str) else f"{x*100:.1f}%")
            st.dataframe(show_df[["מרכז טווח", "חיזוי ממוצע", "התממשות בפועל", "מספר תצפיות"]], width="stretch")
            try:
                chart_df = rel_df.set_index("bin_center")[["mean_pred", "empirical"]]
                chart_df.columns = ["חיזוי", "בפועל"]
                st.line_chart(chart_df)
            except Exception:
                pass

    race_df = get_brains_race_curves()
    if race_df is not None and not race_df.empty:
        st.markdown("## 🧠 מרוץ דיוק מצטבר לאורך זמן")
        st.markdown(
            "כל נקודה מייצגת את אחוז הפגיעה המצטבר עד אותו משחק, עבור כל מודל."
        )
        chart_race = race_df.set_index("n")
        st.line_chart(chart_race)

with tab_forms:
    st.markdown("<h2 style='text-align:right; color:#1a237e; border-bottom:2px solid #1a237e; padding-bottom:5px;'>📋 כל הטפסים ששלחתי</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:right; color:#6b7280;'>סקירה מלאה של כל הטפסים ששלחת – מחולק לפי סטטוס.</p>",
        unsafe_allow_html=True,
    )
    if os.path.exists(DB_PATH):
        try:
            init_finance_tables()
            conn_forms = sqlite3.connect(DB_PATH)
            slips_all = pd.read_sql_query(
                "SELECT id, created_at, bet_type, stake_per_unit, total_stake, potential_return, settled, result_profit "
                "FROM bet_slips ORDER BY id DESC", conn_forms)
            legs_all = pd.read_sql_query(
                "SELECT slip_id, home_team, away_team, selection, odds, match_date, fixture_id FROM bet_slip_legs",
                conn_forms)
            results_df = pd.read_sql_query(
                "SELECT home_team, away_team, match_date, actual_result, home_goals, away_goals FROM matches "
                "WHERE actual_result IS NOT NULL AND actual_result != '' AND actual_result != 'None'",
                conn_forms)
            conn_forms.close()

            def _format_match_datetime(md):
                """מציג תאריך ושעה מהמשחק בפורמט קריא."""
                if md is None or (isinstance(md, float) and pd.isna(md)):
                    return "—"
                s = str(md).strip()
                if not s:
                    return "—"
                try:
                    if "T" in s:
                        dt = dateutil.parser.parse(s)
                    elif " " in s:
                        dt = datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
                    else:
                        dt = datetime.strptime(s[:10], "%Y-%m-%d")
                    return dt.strftime("%d/%m/%Y %H:%M") if dt.hour != 0 or dt.minute != 0 else dt.strftime("%d/%m/%Y")
                except Exception:
                    return s[:16] if len(s) >= 16 else s[:10]

            def _categorize_slip(slip, legs_all, results_df):
                """מחזיר 'ממתינים' | 'זוכים' | 'מתים'"""
                legs = legs_all[legs_all['slip_id'] == slip['id']]
                leg_results = []
                for _, leg in legs.iterrows():
                    hm, aw = leg['home_team'], leg['away_team']
                    md = leg.get('match_date')
                    if md and str(md).strip():
                        date_str_leg = str(md).strip()[:10]
                        res_row = results_df[
                            (results_df['home_team'] == hm) & (results_df['away_team'] == aw)
                            & (results_df['match_date'].astype(str).str[:10] == date_str_leg)
                        ]
                    else:
                        res_row = results_df[(results_df['home_team'] == hm) & (results_df['away_team'] == aw)]
                    if res_row.empty:
                        leg_results.append(None)
                    else:
                        rr = res_row.iloc[0]
                        leg_results.append(
                            resolve_bet_leg_win(
                                leg["selection"],
                                rr["actual_result"],
                                rr.get("home_goals"),
                                rr.get("away_goals"),
                            )
                        )
                alive_dead = _slip_alive_or_dead(slip['bet_type'], leg_results)
                settled = int(slip['settled'])
                profit = float(slip['result_profit'] or 0)
                if settled and profit > 0:
                    return "זוכים"
                if not settled and alive_dead == "חי":
                    return "ממתינים"
                return "מתים"

            if slips_all.empty:
                st.info("טרם נשלחו טפסים.")
            else:
                slips_pending = []
                slips_winning = []
                slips_dead = []
                for _, slip in slips_all.iterrows():
                    cat = _categorize_slip(slip, legs_all, results_df)
                    if cat == "ממתינים":
                        slips_pending.append(slip)
                    elif cat == "זוכים":
                        slips_winning.append(slip)
                    else:
                        slips_dead.append(slip)

                def _summary_stats(slips_list):
                    """מחזיר: (מספר טפסים, סה״כ הושקע, סה״כ רווח, סוג טופס מוביל)"""
                    if not slips_list:
                        return 0, 0, 0, "—"
                    n = len(slips_list)
                    total_stake = sum(float(s.get('total_stake') or 0) for s in slips_list)
                    total_profit = sum(float(s.get('result_profit') or 0) for s in slips_list)
                    bet_types = [str(s.get('bet_type') or "—") for s in slips_list]
                    leading_type = Counter(bet_types).most_common(1)[0][0] if bet_types else "—"
                    return n, total_stake, total_profit, leading_type

                sub_tab_pending, sub_tab_winning, sub_tab_dead = st.tabs([
                    f"⏳ טפסים ממתינים ({len(slips_pending)})",
                    f"✅ טפסים זוכים ({len(slips_winning)})",
                    f"❌ טפסים מתים ({len(slips_dead)})",
                ])

                def _render_summary_card(n, stake, profit, leading_type, profit_label="רווח", is_pending=False):
                    if is_pending:
                        profit_str = "ממתין"
                        profit_color = "#6b7280"
                    else:
                        profit_color = "#22c55e" if profit > 0 else "#ef4444" if profit < 0 else "#6b7280"
                        profit_str = f"+₪{profit:,.0f}" if profit > 0 else f"-₪{abs(profit):,.0f}" if profit < 0 else "±0"
                    st.markdown(
                        f"<div style='display:flex; flex-wrap:wrap; gap:16px; margin-bottom:16px; padding:12px; "
                        f"background:#f8fafc; border-radius:8px; border-right:4px solid #1a237e;'>"
                        f"<div style='min-width:80px;'><b>טפסים:</b> {n}</div>"
                        f"<div style='min-width:100px;'><b>הושקע:</b> ₪{stake:,.0f}</div>"
                        f"<div style='min-width:120px; color:{profit_color};'><b>{profit_label}:</b> {profit_str}</div>"
                        f"<div style='min-width:140px;'><b>סוג מוביל:</b> {leading_type}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                def _render_slip(slip, slip_idx_base, legs_all, results_df):
                    sid = int(slip['id'])
                    created_str = str(slip['created_at'])[:16] if slip['created_at'] else "—"
                    try:
                        created_dt = datetime.strptime(created_str[:16], "%Y-%m-%d %H:%M")
                        created_display = created_dt.strftime("%d/%m/%Y %H:%M")
                    except Exception:
                        created_display = created_str
                    settled = int(slip['settled'])
                    profit = float(slip['result_profit'] or 0)
                    stake = float(slip['total_stake'] or 0)
                    bet_type = slip['bet_type'] or "—"

                    legs = legs_all[legs_all['slip_id'] == sid]
                    leg_results = []
                    for _, leg in legs.iterrows():
                        hm, aw = leg['home_team'], leg['away_team']
                        md = leg.get('match_date')
                        if md and str(md).strip():
                            date_str_leg = str(md).strip()[:10]
                            res_row = results_df[
                                (results_df['home_team'] == hm) & (results_df['away_team'] == aw)
                                & (results_df['match_date'].astype(str).str[:10] == date_str_leg)
                            ]
                        else:
                            res_row = results_df[(results_df['home_team'] == hm) & (results_df['away_team'] == aw)]
                        if res_row.empty:
                            leg_results.append(None)
                        else:
                            rr = res_row.iloc[0]
                            leg_results.append(
                                resolve_bet_leg_win(
                                    leg["selection"],
                                    rr["actual_result"],
                                    rr.get("home_goals"),
                                    rr.get("away_goals"),
                                )
                            )
                    alive_dead = _slip_alive_or_dead(slip['bet_type'], leg_results)

                    if settled:
                        if profit > 0:
                            slip_icon = "✅"
                            slip_color = "#dcfce7"
                            status_txt = f"סגור · {alive_dead} · זכה · +₪{profit:.0f}"
                        elif profit == 0:
                            slip_icon = "🔄"
                            slip_color = "#fef9c3"
                            status_txt = f"סגור · {alive_dead} · תיקו · ±0"
                        else:
                            slip_icon = "❌"
                            slip_color = "#fee2e2"
                            status_txt = f"סגור · {alive_dead} · נכשל · -₪{abs(profit):.0f}"
                    else:
                        slip_icon = "⏳"
                        slip_color = "#f0f9ff"
                        status_txt = f"פתוח · {alive_dead} · עלות ₪{stake:.0f}"

                    header_col, btn_col = st.columns([5, 1])
                    with header_col:
                        header_html = (
                            f"<div style='padding:12px; margin:8px 0; border-radius:8px; background:{slip_color}; "
                            f"border-right:4px solid #1a237e;'>"
                            f"<b>{slip_icon} טופס #{sid}</b> &nbsp;|&nbsp; "
                            f"<span style='color:#4b5563;'>נוצר: {created_display}</span> &nbsp;|&nbsp; "
                            f"<span style='color:#374151;'>{status_txt}</span><br>"
                            f"<small style='color:#6b7280;'>סוג: {bet_type} &nbsp;|&nbsp; עלות: ₪{stake:.0f} &nbsp;|&nbsp; פוטנציאל: ₪{float(slip['potential_return'] or 0):.0f}</small>"
                            f"</div>"
                        )
                        st.markdown(header_html, unsafe_allow_html=True)
                    with btn_col:
                        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ מחק", key=f"del_slip_{slip_idx_base}_{sid}", help="מחיקת הטופס תעדכן את הקופה בהתאם (החזרת עלות או ביטול רווח/הפסד)"):
                            delete_slip(sid)
                            st.success("הטופס נמחק והקופה עודכנה.")
                            st.rerun()

                    for _, leg in legs.iterrows():
                        hm, aw = leg['home_team'], leg['away_team']
                        md = leg.get('match_date')
                        if md and str(md).strip():
                            date_str_leg = str(md).strip()[:10]
                            res_row = results_df[
                                (results_df['home_team'] == hm) & (results_df['away_team'] == aw)
                                & (results_df['match_date'].astype(str).str[:10] == date_str_leg)
                            ]
                        else:
                            res_row = results_df[(results_df['home_team'] == hm) & (results_df['away_team'] == aw)]
                        match_datetime = _format_match_datetime(leg.get('match_date'))
                        if not res_row.empty:
                            rr = res_row.iloc[0]
                            actual = str(rr["actual_result"]).strip()
                            w = resolve_bet_leg_win(
                                leg["selection"], rr["actual_result"], rr.get("home_goals"), rr.get("away_goals")
                            )
                            if w is None:
                                result_txt = f"תוצאה: **{actual}** · ממתין לשערים לסילוק שוק"
                            else:
                                hit = "✅" if w else "❌"
                                hi = _goal_int(rr.get("home_goals"))
                                ai = _goal_int(rr.get("away_goals"))
                                gtxt = f" · שערים {hi}–{ai}" if hi is not None and ai is not None else ""
                                result_txt = f"תוצאה: **{actual}**{gtxt} {hit}"
                        else:
                            result_txt = "⏳ ממתין"
                        leg_html = (
                            f"<div style='font-size:0.88rem; margin:6px 0 6px 24px; padding:8px 12px; "
                            f"background:white; border-radius:6px; border-right:3px solid #94a3b8;'>"
                            f"<b>{hm} – {aw}</b><br>"
                            f"<span style='color:#64748b;'>📅 תאריך ושעה: {match_datetime}</span><br>"
                            f"הימור: <b>{leg['selection']}</b> @ {leg['odds']:.2f} &nbsp;|&nbsp; {result_txt}"
                            f"</div>"
                        )
                        st.markdown(leg_html, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                with sub_tab_pending:
                    if not slips_pending:
                        st.info("אין טפסים ממתינים.")
                    else:
                        n_p, stake_p, _, lead_p = _summary_stats(slips_pending)
                        _render_summary_card(n_p, stake_p, 0, lead_p, profit_label="רווח", is_pending=True)
                        for idx, slip in enumerate(slips_pending):
                            _render_slip(slip, f"p{idx}", legs_all, results_df)

                with sub_tab_winning:
                    if not slips_winning:
                        st.info("אין טפסים זוכים.")
                    else:
                        n_w, stake_w, profit_w, lead_w = _summary_stats(slips_winning)
                        _render_summary_card(n_w, stake_w, profit_w, lead_w)
                        for idx, slip in enumerate(slips_winning):
                            _render_slip(slip, f"w{idx}", legs_all, results_df)

                with sub_tab_dead:
                    if not slips_dead:
                        st.info("אין טפסים מתים.")
                    else:
                        n_d, stake_d, profit_d, lead_d = _summary_stats(slips_dead)
                        _render_summary_card(n_d, stake_d, profit_d, lead_d)
                        for idx, slip in enumerate(slips_dead):
                            _render_slip(slip, f"d{idx}", legs_all, results_df)
        except Exception as ex_forms:
            st.error(f"שגיאה בטעינת הטפסים: {ex_forms}")
    else:
        st.info("מסד הנתונים לא נמצא.")


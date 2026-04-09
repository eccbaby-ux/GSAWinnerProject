# -*- coding: utf-8 -*-
"""
ELO Rating Updater — מחשב דירוג ELO לכל קבוצה מתוך המשחקים ההיסטוריים ב-DB.

שימוש:
  python elo_updater.py

הדירוג נשמר בטבלת elo_ratings ומשמש את v76_Master_Nachshon כדי לספק
הערכות lambda מדויקות יותר כשאין נתוני xG/goals מה-API.

נוסחת ELO סטנדרטית (K=32):
  E_a = 1 / (1 + 10^((Rb - Ra) / 400))
  Ra_new = Ra + K * (S - E_a)
  S: 1.0=ניצחון, 0.5=תיקו, 0.0=הפסד

המרת ELO => lambda (להצגה בלבד, השימוש ב-v76):
  elo_scale = clamp(1 + elo_diff/1000, 0.60, 1.70)
  lambda_attack  = league_anchor * elo_scale
  lambda_defense = league_anchor / elo_scale
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")

DEFAULT_ELO = 1500
K_FACTOR = 32
LEAGUE_ANCHOR = 1.35   # ממוצע גול עולמי (זהה ל-v76)
ELO_SCALE_DIVISOR = 1000.0  # +1000 ELO = scale x2, -1000 = scale x0 (clamped)
ELO_SCALE_MIN = 0.60
ELO_SCALE_MAX = 1.70


# ---------------------------------------------------------------------------
# ELO core
# ---------------------------------------------------------------------------

def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _update(ra: float, rb: float, score_a: float, k: float = K_FACTOR):
    """מחזיר (ra_new, rb_new). score_a: 1.0=win, 0.5=draw, 0.0=loss."""
    ea = _expected(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            team_name TEXT PRIMARY KEY,
            elo       REAL    NOT NULL DEFAULT 1500,
            games_played INTEGER NOT NULL DEFAULT 0,
            updated_at   TEXT
        )
    """)
    conn.commit()


def load_elo_from_db(conn: sqlite3.Connection) -> dict:
    """טוען dict {team_name: elo} מהטבלה. מחזיר {} אם הטבלה חסרה."""
    try:
        rows = conn.execute("SELECT team_name, elo FROM elo_ratings").fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main update logic
# ---------------------------------------------------------------------------

def run_elo_update(conn: sqlite3.Connection = None) -> dict:
    """
    מחשב ELO מאפס מכל המשחקים הסגורים, שומר ל-DB ומחזיר dict {team_name: elo}.
    """
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        conn.execute("PRAGMA journal_mode=WAL;")

    _ensure_table(conn)

    rows = conn.execute("""
        SELECT home_team, away_team, actual_result
        FROM   matches
        WHERE  actual_result IN ('1', 'X', '2')
        ORDER  BY match_date ASC, id ASC
    """).fetchall()

    if not rows:
        print("[ELO] אין משחקים סגורים — הטבלה נשארת ריקה.")
        if own_conn:
            conn.close()
        return {}

    elo: dict = {}   # team_name -> float
    games: dict = {} # team_name -> int

    for home, away, result in rows:
        if not home or not away:
            continue
        ra = elo.setdefault(home, float(DEFAULT_ELO))
        rb = elo.setdefault(away, float(DEFAULT_ELO))
        games.setdefault(home, 0)
        games.setdefault(away, 0)

        score_a = 1.0 if result == '1' else (0.5 if result == 'X' else 0.0)
        new_ra, new_rb = _update(ra, rb, score_a)
        elo[home], elo[away] = new_ra, new_rb
        games[home] += 1
        games[away] += 1

    # שמירה ל-DB
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    c = conn.cursor()
    c.execute("DELETE FROM elo_ratings")
    c.executemany(
        "INSERT INTO elo_ratings (team_name, elo, games_played, updated_at) VALUES (?,?,?,?)",
        [(name, round(elo[name], 2), games[name], now) for name in elo]
    )
    conn.commit()

    # סטטיסטיקה
    sorted_teams = sorted(elo.items(), key=lambda x: x[1], reverse=True)
    print(f"[ELO] עודכנו {len(elo)} קבוצות מתוך {len(rows)} משחקים.")
    print(f"[ELO] חזקות ביותר : {[(n, int(v)) for n, v in sorted_teams[:5]]}")
    print(f"[ELO] חלשות ביותר : {[(n, int(v)) for n, v in sorted_teams[-5:]]}")
    spread = sorted_teams[0][1] - sorted_teams[-1][1] if sorted_teams else 0
    print(f"[ELO] טווח דירוגים : {int(sorted_teams[-1][1]) if sorted_teams else DEFAULT_ELO}"
          f" – {int(sorted_teams[0][1]) if sorted_teams else DEFAULT_ELO} (פרש={int(spread)})")

    if own_conn:
        conn.close()

    return elo


# ---------------------------------------------------------------------------
# Lambda helper — בשימוש v76
# ---------------------------------------------------------------------------

def elo_to_lambda_scale(elo_home: float, elo_away: float) -> float:
    """
    מחזיר elo_scale > 1 אם הבית חזק יותר, < 1 אם האורח חזק יותר.
    elo_scale = clamp(1 + elo_diff / ELO_SCALE_DIVISOR, MIN, MAX)
    """
    diff = elo_home - elo_away
    scale = 1.0 + diff / ELO_SCALE_DIVISOR
    return max(ELO_SCALE_MIN, min(ELO_SCALE_MAX, scale))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 52)
    print("  ELO Rating Updater — GSA WinnerProject")
    print("=" * 52)
    run_elo_update()
    print("[ELO] סיום.")

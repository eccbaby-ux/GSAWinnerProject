# -*- coding: utf-8 -*-
"""
Result Updater: Fetches actual match results from API-Sports and updates gsa_history.db.
Run this as Step 1 of the Training pipeline before Auto-Learner.
Stores 1X2 outcome plus home_goals/away_goals for Totals/BTTS settlement.
"""
import os
import sqlite3
import requests
import time
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")
FOOTBALL_API_KEY = os.environ.get("FOOTBALL_API_KEY", "e49cdc2ba079c654d1dbc88fb16bfa75")
BASE_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# Delay between API calls (seconds) - avoids rate limits (~10 req/min on free tier)
API_DELAY_SEC = 6.5
MAX_RETRIES = 3
RETRY_DELAY_SEC = 60

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _ensure_goal_columns(conn):
    for col in ("home_goals", "away_goals"):
        try:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} INTEGER")
        except sqlite3.OperationalError:
            pass


def get_pending_matches(conn):
    """Return list of (id, fixture_id, home_team, away_team) for matches with NULL actual_result and valid fixture_id."""
    c = conn.cursor()
    c.execute(
        """
        SELECT id, fixture_id, home_team, away_team
        FROM matches
        WHERE actual_result IS NULL AND fixture_id IS NOT NULL
        ORDER BY id ASC
        """
    )
    return c.fetchall()


def get_matches_missing_goals(conn, limit=40):
    """מילוי שערים למשחקים ישנים (לפני העמודות) — מוגבל כדי לא להציף את ה-API."""
    c = conn.cursor()
    c.execute(
        """
        SELECT id, fixture_id, home_team, away_team
        FROM matches
        WHERE fixture_id IS NOT NULL
          AND actual_result IS NOT NULL
          AND actual_result != ''
          AND actual_result != 'None'
          AND (home_goals IS NULL OR away_goals IS NULL)
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return c.fetchall()


def fetch_fixture_outcome(fixture_id):
    """
    Call API-Sports fixtures endpoint.
    Returns dict: outcome_1x2 ('1'|'X'|'2'), home_goals, away_goals — or None if not finished / error.
    """
    url = f"{BASE_URL}?id={fixture_id}"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()

            if r.status_code == 429:
                print(
                    f"  [RATE LIMIT] API throttled. Waiting {RETRY_DELAY_SEC}s before retry {attempt + 1}/{MAX_RETRIES}..."
                )
                time.sleep(RETRY_DELAY_SEC)
                continue

            if r.status_code != 200:
                print(f"  [ERROR] HTTP {r.status_code} for fixture {fixture_id}")
                return None

            if "errors" in data and data["errors"]:
                print(f"  [ERROR] API error for fixture {fixture_id}: {data['errors']}")
                return None

            response = data.get("response")
            if not response:
                print(f"  [WARN] No response data for fixture {fixture_id}")
                return None

            fixture_data = response[0]
            status = fixture_data.get("fixture", {}).get("status", {}).get("short", "")

            if status not in ("FT", "AET", "PEN"):
                return None  # Match not finished

            goals = fixture_data.get("goals", {})
            goals_h = goals.get("home")
            goals_a = goals.get("away")

            if goals_h is None or goals_a is None:
                print(f"  [WARN] Fixture {fixture_id} finished but goals missing")
                return None

            goals_h = int(goals_h)
            goals_a = int(goals_a)

            if goals_h > goals_a:
                outcome = "1"
            elif goals_a > goals_h:
                outcome = "2"
            else:
                outcome = "X"

            return {"outcome_1x2": outcome, "home_goals": goals_h, "away_goals": goals_a}

        except requests.RequestException as e:
            print(f"  [ERROR] Request failed for fixture {fixture_id}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SEC)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            print(f"  [ERROR] Parse error for fixture {fixture_id}: {e}")
            return None

    return None


def main():
    print("=" * 50)
    print("Result Updater - Fetching actual results from API-Sports")
    print("=" * 50)

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH, timeout=20.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    _ensure_goal_columns(conn)
    conn.commit()

    pending = get_pending_matches(conn)
    missing_goals = get_matches_missing_goals(conn)

    if not pending and not missing_goals:
        print("[OK] No pending results and no rows needing goal backfill. Nothing to update.")
        conn.close()
        return

    updated = 0
    skipped = 0
    idx = 0
    total_tasks = len(pending) + len(missing_goals)

    for row_id, fixture_id, home_team, away_team in pending:
        idx += 1
        match_label = f"{home_team or '?'} vs {away_team or '?'}" if home_team or away_team else f"fixture {fixture_id}"
        print(f"\n[{idx}/{total_tasks}] (new result) {match_label} (fixture_id={fixture_id})...")

        out = fetch_fixture_outcome(fixture_id)
        if out is not None:
            conn.execute(
                "UPDATE matches SET actual_result = ?, home_goals = ?, away_goals = ? WHERE id = ?",
                (out["outcome_1x2"], out["home_goals"], out["away_goals"], row_id),
            )
            updated += 1
            print(
                f"  [UPDATED] 1X2={out['outcome_1x2']}, goals {out['home_goals']}-{out['away_goals']}"
            )
        else:
            skipped += 1
            print("  [SKIP] Match not finished or unavailable")

        if idx < total_tasks:
            time.sleep(API_DELAY_SEC)

    for row_id, fixture_id, home_team, away_team in missing_goals:
        idx += 1
        match_label = f"{home_team or '?'} vs {away_team or '?'}" if home_team or away_team else f"fixture {fixture_id}"
        print(f"\n[{idx}/{total_tasks}] (backfill goals) {match_label} (fixture_id={fixture_id})...")

        out = fetch_fixture_outcome(fixture_id)
        if out is not None:
            conn.execute(
                "UPDATE matches SET home_goals = ?, away_goals = ? WHERE id = ?",
                (out["home_goals"], out["away_goals"], row_id),
            )
            updated += 1
            print(f"  [UPDATED] goals {out['home_goals']}-{out['away_goals']}")
        else:
            skipped += 1
            print("  [SKIP] Match not finished or unavailable")

        if idx < total_tasks:
            time.sleep(API_DELAY_SEC)

    conn.commit()
    conn.close()

    print("\n" + "=" * 50)
    print(f"[DONE] Updated {updated} rows. Skipped {skipped}.")
    print("=" * 50)


if __name__ == "__main__":
    main()

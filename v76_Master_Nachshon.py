# -*- coding: utf-8 -*-
import requests, json, os, time, math, sys, re, sqlite3, itertools
import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOOTBALL_API_KEY = "e49cdc2ba079c654d1dbc88fb16bfa75"

try:
    from calibration_layer import load_calibration_params, apply_calibration
except ImportError:
    load_calibration_params = None
    apply_calibration = None

from mapper import (
    get_team_id as mapper_get_team_id,
    init_mapper as mapper_init,
    prefetch_fixtures_for_dates,
    try_resolve_pair_via_fixtures,
    flush_translation_cache,
)

mapper_init()

DB_PATH = os.path.join(BASE_DIR, "gsa_history.db")
MATCHES_FILE = os.path.join(BASE_DIR, "matches.txt")
OUTPUT_JSON = os.path.join(BASE_DIR, "analysis_results_v76.json")
TEAMS_CSV = os.path.join(BASE_DIR, "teams_list.csv")

BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

try:
    sys.stdout.reconfigure(encoding='utf-8')
except: 
    pass

def normalize_probs(probs):
    """
    מקבל מילון הסתברויות גולמיות {'1': x, 'X': y, '2': z} ומחזיר גרסה מתוקננת שסכומה 1.
    במקרה שכל הערכים לא חוקיים/אפסיים – חוזר לחלוקה אחידה.
    """
    if not isinstance(probs, dict):
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    cleaned = {}
    total = 0.0
    for key in ['1', 'X', '2']:
        try:
            val = float(probs.get(key, 0) or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val < 0:
            val = 0.0
        cleaned[key] = val
        total += val
    if total <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    return {k: (v / total) for k, v in cleaned.items()}

def _remove_vig_proportional(implied_probs):
    """
    Proportional (additive) Vig removal: scale implied probabilities so they sum to 1.0.
    True probability for outcome i = implied_i / sum(implied). Standard and robust.
    """
    total = sum(implied_probs.values())
    if total <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    return {k: v / total for k, v in implied_probs.items()}


def _remove_vig_shin(implied_probs, tol=1e-6, max_iter=100):
    """
    Shin's method: solves for z (bookmaker's margin) so that true probabilities sum to 1.
    True p_i = (sqrt(z^2 + 4*(1-z)*implied_i^2) - z) / (2*(1-z)).
    Better for high-vig books (e.g. Winner ~12% overround).
    """
    total_implied = sum(implied_probs.values())
    if total_implied <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    # z in (0, 1); binary search
    z_lo, z_hi = 0.0, 1.0
    for _ in range(max_iter):
        z = (z_lo + z_hi) / 2
        if z >= 1 - 1e-9:
            return _remove_vig_proportional(implied_probs)
        s = 0.0
        for imp in implied_probs.values():
            if imp > 0:
                s += (math.sqrt(z * z + 4 * (1 - z) * imp * imp) - z) / (2 * (1 - z))
        if abs(s - 1.0) < tol:
            return {
                k: (math.sqrt(z * z + 4 * (1 - z) * v * v) - z) / (2 * (1 - z))
                for k, v in implied_probs.items()
            }
        if s > 1.0:
            z_lo = z
        else:
            z_hi = z
    return _remove_vig_proportional(implied_probs)


def odds_to_probs(market_odds):
    """
    Converts market decimal odds to true implied probabilities with Vig removed.
    Raw 1/odd gives overround > 1; blending with model would depress EV. We use
    proportional normalization (or Shin for high-vig) so probabilities sum exactly to 1.0.
    """
    if not isinstance(market_odds, dict):
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    values = []
    for key in ['1', 'X', '2']:
        v = market_odds.get(key)
        try:
            v = float(v) if v is not None else None
        except (TypeError, ValueError):
            v = None
        if v is not None and v > 0:
            values.append(v)
    if not values:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    if all(0 < v <= 1.0 for v in values):
        raw = {k: float(market_odds.get(k) or 0) for k in ['1', 'X', '2']}
        return normalize_probs(raw)
    # Decimal odds: implied probability per outcome = 1/odd (sum > 1 due to Vig)
    implied = {}
    for key in ['1', 'X', '2']:
        v = market_odds.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        implied[key] = (1.0 / v) if v > 0 else 0.0
    overround = sum(implied.values())
    # Shin's method for high overround (e.g. Winner); otherwise proportional
    if overround > 1.08:
        return _remove_vig_shin(implied)
    return _remove_vig_proportional(implied)

_WEIGHTS_CACHE = {"loaded": False, "w_model": 0.5, "w_market": 0.5}

def get_weight_profile(profile: str = "default"):
    """
    טוען משקולות מודל + שוק בלבד (ללא טיכה/AI). מחזיר (w_model, w_market, w_ticha=0).
    """
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE["loaded"]:
        return (_WEIGHTS_CACHE["w_model"], _WEIGHTS_CACHE["w_market"], 0.0)

    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("SELECT w_model, w_market FROM weights WHERE id = 1 LIMIT 1")
            row = c.fetchone()
        except Exception:
            row = None
        if row:
            w_model = float(row[0] or 0.5)
            w_market = float(row[1] or 0.5)
        else:
            w_model, w_market = 0.5, 0.5
        total = w_model + w_market
        if total <= 0:
            w_model, w_market = 0.5, 0.5
        else:
            w_model /= total
            w_market /= total
        conn.close()
        _WEIGHTS_CACHE.update({"w_model": w_model, "w_market": w_market, "loaded": True})
    except Exception:
        _WEIGHTS_CACHE["loaded"] = True

    return (_WEIGHTS_CACHE["w_model"], _WEIGHTS_CACHE["w_market"], 0.0)

def classify_match(final_probs, market_odds):
    """
    סיווג משחקים מציאותי לשוק עם Vig גבוה (הווינר ~12%).
    כדי להראות EV של +6% אחרי Vig, המודל צריך Edge של ~18% מהשוק –
    דבר שלא קיים. לכן סף Gold הוא +2.5% בלבד, ו-EV מעל 12% = מלכודת.
    מחזיר (tier, ev, risk_category, recommended_bet).
    """
    if not final_probs or not market_odds:
        return "skip", -1.0, "no_data", None

    best_pick = None
    best_prob = 0.0
    best_ev = -1e9
    best_score = -1e9
    for bt in ["1", "X", "2"]:
        p = float(final_probs.get(bt, 0.0) or 0.0)
        try:
            odd = float(market_odds.get(bt, 0.0) or 0.0)
        except (TypeError, ValueError):
            odd = 0.0
        if p <= 0 or odd <= 0:
            continue
        ev = (p * odd) - 1.0
        score = ev * p  # שילוב EV × הסתברות – עדיפות לביטחון
        if score > best_score:
            best_score = score
            best_ev = ev
            best_pick = bt
            best_prob = p

    if best_pick is None:
        return "skip", -1.0, "no_bet", None

    ev = best_ev
    odd = float(market_odds.get(best_pick, 0.0) or 0.0)

    # High Value: EV > 12% — יחס גבוה + יתרון גדול — הזדמנות נדירה
    if ev > 0.12 and odd >= 1.50:
        return "high_value", ev, "high", best_pick

    # Gold: +2.5% EV מול הווינר הוא יתרון אמיתי נדיר
    if best_prob >= 0.45 and ev >= 0.025 and odd >= 1.20:
        return "gold", ev, "medium", best_pick

    # Value: יתרון קל אבל קיים
    if best_prob >= 0.35 and 0.005 <= ev < 0.025:
        return "value", ev, "medium", best_pick

    # Sting: סיכוי נמוך, EV חיובי מוגבל
    if 0.20 <= best_prob < 0.35 and 0.04 <= ev <= 0.12:
        return "sting", ev, "high", best_pick

    return "skip", ev, "none", best_pick

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fixture_id INTEGER, match_date TEXT, 
        home_team TEXT, away_team TEXT, model_prob_1 REAL, model_prob_x REAL, 
        model_prob_2 REAL, market_prob_1 REAL, market_prob_x REAL, market_prob_2 REAL, 
        actual_result TEXT,
        final_prob_1 REAL, final_prob_x REAL, final_prob_2 REAL,
        tier TEXT, classified_ev REAL, risk_category TEXT)''')
    for col in ("recommended_bet_market", "market_type"):
        try:
            cursor.execute(f"ALTER TABLE matches ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    for col in ("home_goals", "away_goals"):
        try:
            cursor.execute(f"ALTER TABLE matches ADD COLUMN {col} INTEGER")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn



class Network:
    @staticmethod
    def fetch(endpoint, params):
        time.sleep(0.2)
        try:
            r = requests.get(f"{BASE_URL}/{endpoint}", headers=HEADERS, params=params, timeout=10)
            return r.json().get("response", []) if r.status_code == 200 else []
        except: return []

class LiveWinnerAPI:
    _global_odds_catalog = None 
    ODDS_CACHE_FILE = os.path.join(BASE_DIR, "winner_odds_cache.json")

    @staticmethod
    def build_catalog():
        if LiveWinnerAPI._global_odds_catalog is not None:
            return
            
        print("   🛸 טוען יחסי ווינר מקובץ המטמון המקומי (Lightning Speed)...")
        import json, os
        if os.path.exists(LiveWinnerAPI.ODDS_CACHE_FILE):
            try:
                with open(LiveWinnerAPI.ODDS_CACHE_FILE, 'r', encoding='utf-8') as f:
                    LiveWinnerAPI._global_odds_catalog = json.load(f)
                print(f"      ✅ נטענו {len(LiveWinnerAPI._global_odds_catalog)} משחקים זמינים מהמטמון בהצלחה.")
            except Exception as e:
                print(f"      ⚠️ שגיאה בקריאת קובץ המטמון: {e}")
                LiveWinnerAPI._global_odds_catalog = {}
        else:
            print(f"      ⚠️ קובץ המטמון אינו קיים בנתיב: {LiveWinnerAPI.ODDS_CACHE_FILE}")
            LiveWinnerAPI._global_odds_catalog = {}

    @staticmethod
    def _find_match_entry(home_team, away_team):
        if LiveWinnerAPI._global_odds_catalog is None:
            LiveWinnerAPI.build_catalog()

        if not LiveWinnerAPI._global_odds_catalog:
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

            words = name.split()
            if words and words[-1] not in ["אביב", "תקוה", "שבע", "לציון", "שמונה", "יונייטד", "סיטי"]:
                terms.append(words[-1])
            return terms

        h_terms = get_search_terms(home_team)
        a_terms = get_search_terms(away_team)

        for desc, odds in LiveWinnerAPI._global_odds_catalog.items():
            desc_clean = clean_text(desc)

            h_match = any(term in desc_clean for term in h_terms)
            a_match = any(term in desc_clean for term in a_terms)

            if h_match and a_match:
                return odds

        return None

    @staticmethod
    def get_1x2_odds(home_team, away_team):
        """
        מחזיר יחסי 1X2 בלבד (תאימות לאחור).
        תומך גם במבנה החדש של winner_odds_cache.json שבו 1X2 יושב בתוך אובייקט מורחב.
        """
        entry = LiveWinnerAPI._find_match_entry(home_team, away_team)
        if not isinstance(entry, dict):
            return None

        # מבנה חדש: יש שדה odds_1x2
        if "odds_1x2" in entry and isinstance(entry["odds_1x2"], dict):
            o = entry["odds_1x2"]
            if all(k in o for k in ("1", "X", "2")):
                return {"1": float(o["1"]), "X": float(o["X"]), "2": float(o["2"])}

        # תאימות למבנה ישן / כפול: השדות עצמם ברמה העליונה
        if all(k in entry for k in ("1", "X", "2")):
            try:
                return {
                    "1": float(entry["1"]),
                    "X": float(entry["X"]),
                    "2": float(entry["2"]),
                }
            except Exception:
                return None

        return None

    @staticmethod
    def get_extended_odds(home_team, away_team):
        """
        מחזיר את אובייקט השווקים המורחב למשחק (אם קיים).
        אם נשמר רק 1X2 בפורמט ישן – עוטף אותו למבנה מורחב מינימלי.
        """
        entry = LiveWinnerAPI._find_match_entry(home_team, away_team)
        if not isinstance(entry, dict):
            return None

        # אם זה כבר במבנה החדש – דואגים שכל השדות קיימים ומחזירים כפי שהוא
        extended = dict(entry)  # העתק רדוד
        extended.setdefault("odds_1x2", {})
        extended.setdefault("totals", {})
        extended.setdefault("btts", {})
        extended.setdefault("correct_score", {})
        extended.setdefault("ht_ft", {})
        extended.setdefault("double_chance", {})
        extended.setdefault("team_totals", {})

        # אם odds_1x2 ריק אבל יש שדות 1/X/2 – בונים אותו
        if (not extended["odds_1x2"]) and all(k in extended for k in ("1", "X", "2")):
            try:
                extended["odds_1x2"] = {
                    "1": float(extended["1"]),
                    "X": float(extended["X"]),
                    "2": float(extended["2"]),
                }
            except Exception:
                pass

        return extended

class WeatherStation:
    WEATHER_CODES = {
        0: "שמיים בהירים ☀️", 1: "מעונן חלקית 🌤️", 2: "מעונן ⛅", 3: "מעונן לחלוטין ☁️",
        45: "ערפל 🌫️", 48: "ערפל כבד 🌫️", 51: "טפטוף קל 🌧️", 53: "טפטוף מתון 🌧️",
        55: "טפטוף כבד 🌧️", 61: "גשם קל ☔", 63: "גשם מתון ☔", 65: "גשם כבד ☔",
        71: "שלג קל ❄️", 73: "שלג מתון ❄️", 75: "שלג כבד ❄️",
        95: "סופת רעמים קלה ⛈️", 99: "סופת רעמים קשה ⛈️"
    }

    @staticmethod
    def get_forecast(city):
        if not city: return None
        try:
            clean_city = city.split("(")[0].split(",")[0].strip()
            geo = requests.get(GEO_URL, params={"name": clean_city, "count": 1}).json()
            if geo.get('results'):
                lat, lon = geo['results'][0]['latitude'], geo['results'][0]['longitude']
                w = requests.get(WEATHER_URL, params={"latitude": lat, "longitude": lon, "current": "temperature_2m,weather_code"}).json()
                temp = w['current']['temperature_2m']
                code = w['current']['weather_code']
                desc = WeatherStation.WEATHER_CODES.get(code, f"קוד מזג אוויר: {code}")
                return {"temp": temp, "desc": desc}
        except: return None

# כיוונון Dixon-Coles: rho (תלות תיקו) – ספרות ~0.10–0.18; יתרון בית/חוץ – ספרות ~1.08–1.20 / 0.82–0.92
DIXON_COLES_RHO = 0.12
HOME_ADVANTAGE_FACTOR = 1.12
AWAY_DISADVANTAGE_FACTOR = 0.88

def calculate_dixon_coles_probs(h_lambda, a_lambda, rho=None):
    """
    חישוב הסתברויות 1X2 אנליטי עם תיקון Dixon-Coles.
    מדויק יותר ממונטה-קרלו ומהיר פי ~100.
    rho – תיקון תלות תוצאות נמוכות (ברירת מחדל מהקבוע DIXON_COLES_RHO).
    """
    if rho is None:
        rho = DIXON_COLES_RHO
    h_lambda = max(0.1, h_lambda)
    a_lambda = max(0.1, a_lambda)
    max_goals = 7

    # 1. וקטורי הסתברות Poisson מדויקים
    h_probs = scipy_poisson.pmf(np.arange(max_goals), h_lambda)
    a_probs = scipy_poisson.pmf(np.arange(max_goals), a_lambda)

    # 2. מטריצת הסתברות (עצמאות בסיסית)
    prob_matrix = np.outer(h_probs, a_probs)

    # 3. תיקון Dixon-Coles לתוצאות נמוכות (שם Poisson קלאסי מנפח)
    prob_matrix[0, 0] *= (1 - h_lambda * a_lambda * rho)
    prob_matrix[0, 1] *= (1 + h_lambda * rho)
    prob_matrix[1, 0] *= (1 + a_lambda * rho)
    prob_matrix[1, 1] *= (1 - rho)

    # 4. חסימת ערכים שליליים (מקרי קצה של xG גבוה מאוד)
    prob_matrix = np.clip(prob_matrix, 0, 1)

    # 5. נרמול חזרה ל-100%
    total = np.sum(prob_matrix)
    if total > 0:
        prob_matrix /= total

    # 6. חיתוך ל-1X2
    p_home = float(np.sum(np.tril(prob_matrix, -1)))  # בית > חוץ
    p_draw = float(np.sum(np.diag(prob_matrix)))      # בית == חוץ
    p_away = float(np.sum(np.triu(prob_matrix, 1)))   # חוץ > בית

    return {"1": p_home, "X": p_draw, "2": p_away}


def build_poisson_score_matrix(h_lambda, a_lambda, max_goals=5, rho=None):
    """
    Builds a Dixon-Coles corrected score matrix (0-0 up to max_goals-max_goals).
    Returns the matrix and derived probabilities for Over/Under 2.5 and BTTS.
    """
    if rho is None:
        rho = DIXON_COLES_RHO
    h_lambda = max(0.1, h_lambda)
    a_lambda = max(0.1, a_lambda)
    n = max_goals + 1  # 0..max_goals

    h_probs = scipy_poisson.pmf(np.arange(n), h_lambda)
    a_probs = scipy_poisson.pmf(np.arange(n), a_lambda)
    prob_matrix = np.outer(h_probs, a_probs)

    # Dixon-Coles correction for low scores
    prob_matrix[0, 0] *= (1 - h_lambda * a_lambda * rho)
    prob_matrix[0, 1] *= (1 + h_lambda * rho)
    prob_matrix[1, 0] *= (1 + a_lambda * rho)
    prob_matrix[1, 1] *= (1 - rho)

    prob_matrix = np.clip(prob_matrix, 0, 1)
    total = np.sum(prob_matrix)
    if total > 0:
        prob_matrix /= total

    prob_under_2_5 = 0.0
    prob_btts_yes = 0.0
    for i in range(n):
        for j in range(n):
            p = float(prob_matrix[i, j])
            total_goals = i + j
            if total_goals < 2.5:  # 0, 1, 2 goals
                prob_under_2_5 += p
            if i > 0 and j > 0:
                prob_btts_yes += p

    prob_over_2_5 = 1.0 - prob_under_2_5
    prob_btts_no = 1.0 - prob_btts_yes

    return {
        "matrix": prob_matrix,
        "prob_under_2_5": prob_under_2_5,
        "prob_over_2_5": prob_over_2_5,
        "prob_btts_yes": prob_btts_yes,
        "prob_btts_no": prob_btts_no,
    }


# --- מנוע ארביטראז' / Surebet – פונקציות עזר מתמטיות ---

# מגבלות השקעה לפי ווינר:
# - סכום כולל מומלץ למשחק: עד 100 ₪
# - יחידות הימור נהוגות: קפיצות של 5 ₪, מינימום 10 ₪ לכל שוק פעיל
ARBITRAGE_MAX_STAKE_PER_MATCH = 100.0
ARBITRAGE_STAKE_STEP = 5.0
ARBITRAGE_MIN_STAKE_PER_BET = 10.0
MAX_GOALS_FOR_ARBITRAGE = 6  # טווח תוצאות למטריצת הכיסוי (0-0 עד 6-6)


def decimal_to_implied_prob(odds):
    try:
        o = float(odds)
        if o <= 1.0:
            return 0.0
        return 1.0 / o
    except Exception:
        return 0.0


def _round_stakes_to_winner_units(raw_stakes, odds_by_key, max_total=ARBITRAGE_MAX_STAKE_PER_MATCH):
    """
    מקבל חלוקת הימור רציפה ומעגל אותה:
    - קפיצות של 5 ₪
    - מינימום 10 ₪ לכל שוק פעיל
    - סכום כולל לא יעלה על max_total
    מחשב מחדש רווח מובטח ו-ROI לפי הסכומים המעוגלים.
    """
    if not raw_stakes:
        return None

    # שלב 1: עיגול ראשוני ליחידות 5 ₪ ומינימום 10 ₪
    rounded = {}
    for k, v in raw_stakes.items():
        try:
            amt = float(v)
        except Exception:
            amt = 0.0
        if amt <= 0:
            continue
        # עיגול ליחידת צעד
        step_units = round(amt / ARBITRAGE_STAKE_STEP)
        amt_rounded = step_units * ARBITRAGE_STAKE_STEP
        if 0 < amt_rounded < ARBITRAGE_MIN_STAKE_PER_BET:
            amt_rounded = ARBITRAGE_MIN_STAKE_PER_BET
        if amt_rounded > 0:
            rounded[k] = amt_rounded

    if not rounded:
        return None

    # שלב 2: התאמת סכום כולל לתקרה למשחק
    total = sum(rounded.values())
    if total > max_total and total > 0:
        factor = max_total / total
        for k in list(rounded.keys()):
            amt = rounded[k] * factor
            step_units = max(0, int(round(amt / ARBITRAGE_STAKE_STEP)))
            amt_rounded = step_units * ARBITRAGE_STAKE_STEP
            if amt_rounded > 0 and amt_rounded < ARBITRAGE_MIN_STAKE_PER_BET:
                amt_rounded = ARBITRAGE_MIN_STAKE_PER_BET
            rounded[k] = amt_rounded
        # ייתכן שהגענו מעט מעל max_total בגלל עיגול – חיתוך מהגדולים
        while sum(rounded.values()) > max_total + 1e-6:
            # מוצאים את ההימור עם הסכום הגבוה ביותר ומפחיתים ממנו צעד
            k_max = max(rounded, key=lambda kk: rounded[kk])
            if rounded[k_max] > ARBITRAGE_MIN_STAKE_PER_BET:
                rounded[k_max] -= ARBITRAGE_STAKE_STEP
            else:
                break  # אי אפשר להקטין יותר בלי לרדת מתחת למינימום

    # שלב 3: חישוב רווח מובטח ו-ROI לפי הסכומים המעוגלים
    total_rounded = sum(rounded.values())
    if total_rounded <= 0:
        return None

    min_return = None
    for k, stake in rounded.items():
        try:
            odd = float(odds_by_key.get(k, 0))
        except Exception:
            odd = 0.0
        if not odd or odd <= 1.0 or stake <= 0:
            continue
        ret = stake * odd
        if min_return is None or ret < min_return:
            min_return = ret

    if min_return is None:
        return None

    guaranteed_profit = float(min_return - total_rounded)
    roi = guaranteed_profit / float(total_rounded) if total_rounded > 0 else 0.0

    return {
        "total_stake": float(total_rounded),
        "stakes": rounded,
        "guaranteed_profit": guaranteed_profit,
        "roi": roi,
    }


def compute_surebet_allocation(odds_by_key, target_total=ARBITRAGE_MAX_STAKE_PER_MATCH):
    """
    מקבל מילון {key: odds} ומחשב:
    - סכום הסתברויות מרומזות Σ(1/odds) בחלוקה רציפה
    - חלוקת הימור מעוגלת ליחידות ווינר (5 ₪, מינימום 10 ₪) עד לתקרה למשחק
    - רווח מובטח ו-ROI בפועל אחרי העיגול
    """
    inv_sum = 0.0
    inv_map = {}
    for k, o in odds_by_key.items():
        ip = decimal_to_implied_prob(o)
        if ip <= 0:
            return None
        inv_map[k] = ip
        inv_sum += ip

    if inv_sum >= 1.0:
        return None

    # חלוקת בסיס רציפה לתקציב היעד (לפני עיגול)
    raw_stakes = {}
    for k, ip in inv_map.items():
        raw_stakes[k] = float(target_total * (ip / inv_sum))

    rounded_alloc = _round_stakes_to_winner_units(raw_stakes, odds_by_key, max_total=target_total)
    if not rounded_alloc:
        return None

    rounded_alloc["implied_prob_sum"] = inv_sum
    return rounded_alloc


def build_outcomes(max_goals=MAX_GOALS_FOR_ARBITRAGE):
    """מייצר את טבלת התוצאות האפשריות (Outcome Nodes) עד N שערים לכל קבוצה."""
    outcomes = []
    for hg in range(0, max_goals + 1):
        for ag in range(0, max_goals + 1):
            outcomes.append((hg, ag))
    return outcomes


def _parse_total_key(key):
    """
    ממיר מפתח כמו 'over_2_5' ל-(kind='over', threshold=2.5).
    """
    if not isinstance(key, str):
        return None, None
    parts = key.split("_", 1)
    if len(parts) != 2:
        return None, None
    kind = parts[0]
    try:
        threshold = float(parts[1].replace("_", "."))
    except Exception:
        return None, None
    return kind, threshold


def build_selection_map(extended_odds, outcomes):
    """
    בונה מילון של כל הימור בודד במשחק למבנה אחיד:
    {selection_id: {'id', 'kind', 'code', 'odds', 'covers': set(indices)}}
    """
    selections = {}
    if not isinstance(extended_odds, dict):
        return selections

    # 1X2
    odds_1x2 = extended_odds.get("odds_1x2") or {
        "1": extended_odds.get("1"),
        "X": extended_odds.get("X"),
        "2": extended_odds.get("2"),
    }
    if isinstance(odds_1x2, dict):
        for res in ("1", "X", "2"):
            o = odds_1x2.get(res)
            try:
                o = float(o)
            except Exception:
                o = 0.0
            if o and o > 1.0:
                sel_id = f"1x2_{res}"
                covers = set()
                for idx, (hg, ag) in enumerate(outcomes):
                    if res == "1" and hg > ag:
                        covers.add(idx)
                    elif res == "X" and hg == ag:
                        covers.add(idx)
                    elif res == "2" and ag > hg:
                        covers.add(idx)
                selections[sel_id] = {
                    "id": sel_id,
                    "kind": "1x2",
                    "code": res,
                    "odds": o,
                    "covers": covers,
                }

    # Totals (Over/Under)
    totals = extended_odds.get("totals") or {}
    if isinstance(totals, dict):
        for t_key, o in totals.items():
            try:
                o = float(o)
            except Exception:
                o = 0.0
            if not o or o <= 1.0:
                continue
            kind, threshold = _parse_total_key(t_key)
            if kind not in ("over", "under") or threshold is None:
                continue
            sel_id = f"total_{t_key}"
            covers = set()
            for idx, (hg, ag) in enumerate(outcomes):
                total_goals = hg + ag
                if kind == "over" and total_goals > threshold:
                    covers.add(idx)
                elif kind == "under" and total_goals < threshold:
                    covers.add(idx)
            selections[sel_id] = {
                "id": sel_id,
                "kind": "total",
                "code": t_key,
                "odds": o,
                "covers": covers,
            }

    # BTTS
    btts = extended_odds.get("btts") or {}
    if isinstance(btts, dict):
        for code in ("yes", "no"):
            o = btts.get(code)
            try:
                o = float(o)
            except Exception:
                o = 0.0
            if not o or o <= 1.0:
                continue
            sel_id = f"btts_{code}"
            covers = set()
            for idx, (hg, ag) in enumerate(outcomes):
                if code == "yes":
                    if hg > 0 and ag > 0:
                        covers.add(idx)
                else:  # no
                    if not (hg > 0 and ag > 0):
                        covers.add(idx)
            selections[sel_id] = {
                "id": sel_id,
                "kind": "btts",
                "code": code,
                "odds": o,
                "covers": covers,
            }

    # Correct Score
    cs = extended_odds.get("correct_score") or {}
    if isinstance(cs, dict):
        for score_key, o in cs.items():
            try:
                o = float(o)
            except Exception:
                o = 0.0
            if not o or o <= 1.0:
                continue
            m = re.match(r"^(\d+)-(\d+)$", str(score_key).strip())
            if not m:
                continue
            hg_s, ag_s = int(m.group(1)), int(m.group(2))
            sel_id = f"cs_{hg_s}-{ag_s}"
            covers = set()
            for idx, (hg, ag) in enumerate(outcomes):
                if hg == hg_s and ag == ag_s:
                    covers.add(idx)
            selections[sel_id] = {
                "id": sel_id,
                "kind": "correct_score",
                "code": f"{hg_s}-{ag_s}",
                "odds": o,
                "covers": covers,
            }

    # Double Chance
    dc = extended_odds.get("double_chance") or {}
    if isinstance(dc, dict):
        for dc_key, o in dc.items():
            try:
                o = float(o)
            except Exception:
                o = 0.0
            if not o or o <= 1.0:
                continue
            dc_code = str(dc_key).replace(" ", "")
            if not re.match(r"^[12X]{2}$", dc_code):
                continue
            sel_id = f"dc_{dc_code}"
            covers = set()
            for idx, (hg, ag) in enumerate(outcomes):
                # Union של שני תוצאות 1X2
                for res in dc_code:
                    if res == "1" and hg > ag:
                        covers.add(idx)
                    elif res == "X" and hg == ag:
                        covers.add(idx)
                    elif res == "2" and ag > hg:
                        covers.add(idx)
            selections[sel_id] = {
                "id": sel_id,
                "kind": "double_chance",
                "code": dc_code,
                "odds": o,
                "covers": covers,
            }

    return selections


def compute_generic_full_market_arbitrage(selections, outcomes, max_combo_size=3, max_results=5):
    """
    מנוע כללי: מחפש קומבינציות של 2–3 הימורים שמכסות 100% מהתוצאות
    כאשר סכום ההסתברויות המרומזות קטן מ-1 (Surebet).
    מחזיר רשימת קומבינציות עם הקצאת הימור ורווח מובטח.
    """
    results = []
    if not selections or not outcomes:
        return results

    outcome_count = len(outcomes)

    # סינון ראשוני של הימורים קיצוניים/בעייתיים
    candidate_ids = []
    for sid, sel in selections.items():
        o = sel.get("odds")
        try:
            o = float(o)
        except Exception:
            o = 0.0
        if not o or o <= 1.01 or o > 50.0:
            continue
        # חייב להיות כיסוי כלשהו
        if not sel.get("covers"):
            continue
        candidate_ids.append(sid)

    # כדי למנוע פיצוץ קומבינטורי – מגבלה קשיחה על מספר בחירות
    max_candidates = 20
    if len(candidate_ids) > max_candidates:
        # ממיינים לפי רמת כיסוי (כמה outcomes כל הימור מכסה) ובוחרים את המובילים
        candidate_ids.sort(key=lambda sid: len(selections[sid]["covers"]), reverse=True)
        candidate_ids = candidate_ids[:max_candidates]

    for k in range(2, max_combo_size + 1):
        for combo_ids in itertools.combinations(candidate_ids, k):
            # בדיקת כיסוי מלא
            covered = set()
            for sid in combo_ids:
                covered |= selections[sid]["covers"]
            if len(covered) != outcome_count:
                continue

            odds_map = {sid: selections[sid]["odds"] for sid in combo_ids}
            alloc = compute_surebet_allocation(odds_map, ARBITRAGE_MAX_STAKE_PER_MATCH)
            if not alloc:
                continue

            results.append(
                {
                    "type": "Full Market Arbitrage",
                    "selection_ids": list(combo_ids),
                    "odds": odds_map,
                    "allocation": alloc,
                    "coverage_ratio": 1.0,
                    "surebet": True,
                }
            )

    # דירוג לפי ROI ובחירת Top-N
    results.sort(key=lambda r: r.get("allocation", {}).get("roi", 0.0), reverse=True)
    return results[:max_results]

def process_matches():
    if not os.path.exists(MATCHES_FILE): return
    with open(MATCHES_FILE, 'r', encoding='utf-8') as f:
        matches_raw = [line.strip().split("-") for line in f if "-" in line]

    # רשימת משחקים נקייה + ספירת סה"כ להתקדמות
    matches_list = []
    for m in matches_raw:
        if len(m) < 2:
            continue
        h_name, a_name = str(m[0] or "").strip(), str(m[1] or "").strip()
        if not h_name or not a_name:
            continue
        matches_list.append((h_name, a_name))
    total_matches = len(matches_list)

    # טעינת משחקים לפי תאריך (מטמון) — תומך בזיהוי זוגי אחרי תרגום שם הקבוצה
    LiveWinnerAPI.build_catalog()
    winner_dates_prefetch = set()
    for _h, _a in matches_list:
        _wo = LiveWinnerAPI.get_1x2_odds(_h, _a)
        if not _wo:
            continue
        _ext = LiveWinnerAPI.get_extended_odds(_h, _a)
        if isinstance(_ext, dict):
            _wdv = _ext.get("winner_date") or ""
            if isinstance(_wdv, str) and re.match(r"^\d{4}-\d{2}-\d{2}", _wdv):
                winner_dates_prefetch.add(_wdv[:10])
    prefetch_fixtures_for_dates(winner_dates_prefetch, api_key=FOOTBALL_API_KEY)

    # טעינת משקולות פעם אחת לפני הלופ (מודל + שוק בלבד)
    w_model, w_market, _ = get_weight_profile("default")

    # חיבור DB אחד לכל הלופ במקום פתיחה וסגירה לכל משחק
    db_conn = init_db()
    db_cursor = db_conn.cursor()

    # טעינת ELO ratings לשיפור חיזוי כשאין נתוני xG/goals מה-API
    try:
        from elo_updater import load_elo_from_db, elo_to_lambda_scale, DEFAULT_ELO as _ELO_DEFAULT, LEAGUE_ANCHOR as _ELO_ANCHOR
        _elo_ratings = load_elo_from_db(db_conn)
        if _elo_ratings:
            print(f"[ELO] נטענו דירוגים ל-{len(_elo_ratings)} קבוצות.")
        else:
            print("[ELO] אין טבלת ELO — הרץ elo_updater.py כדי לבנות אותה.")
    except Exception:
        _elo_ratings = {}
        _ELO_DEFAULT = 1500
        _ELO_ANCHOR = 1.35
        def elo_to_lambda_scale(a, b): return 1.0

    results = []
    print(f"🚀 Starting V76 Pro Production Analysis...")

    for idx, (h_name, a_name) in enumerate(matches_list, start=1):
        remaining = max(0, total_matches - idx)
        print(f"\n🔎 התקדמות: {idx}/{total_matches} | נשארו: {remaining}")

        # Strict validation: only process 3-way football matches with valid Draw (X) odd from winner_odds_cache
        winner_odds = LiveWinnerAPI.get_1x2_odds(h_name, a_name)
        x_val = winner_odds.get("X") if isinstance(winner_odds, dict) else None
        if winner_odds is None or x_val is None or x_val == "" or (isinstance(x_val, str) and not str(x_val).strip()):
            print(f"      ⏩ Skipping non-football/invalid match")
            continue

        hid, aid = mapper_get_team_id(h_name), mapper_get_team_id(a_name)
        if not hid or not aid:
            wd_fix = None
            ext_fix = LiveWinnerAPI.get_extended_odds(h_name, a_name)
            if isinstance(ext_fix, dict):
                wdv = ext_fix.get("winner_date") or ""
                if isinstance(wdv, str) and re.match(r"^\d{4}-\d{2}-\d{2}", wdv):
                    wd_fix = wdv[:10]
            if wd_fix:
                resolved = try_resolve_pair_via_fixtures(h_name, a_name, wd_fix, api_key=FOOTBALL_API_KEY)
                if resolved:
                    hid, aid = resolved
            if not hid or not aid:
                continue

        print(f" 🧠 Analyzing {h_name} vs {a_name}...")
        
        h2h = Network.fetch("fixtures/headtohead", {"h2h": f"{hid}-{aid}", "next": 1})
        fid = None
        league_id = None
        league_name = "Unknown"
        actual_res_val = None
        venue_name = "Unknown"
        city_name = None
        raw_date = ""
        
        kickoff_date = None
        country_name = ""
        if h2h:
            fix = h2h[0]
            fid = fix['fixture']['id']
            league_id = fix['league']['id']
            league_name = fix['league']['name']
            country_name = fix['league'].get('country', '') or ''

            raw_date = fix['fixture'].get('date', '')
            if raw_date:
                try:
                    kickoff_date = raw_date[:10]  # YYYY-MM-DD
                except Exception:
                    kickoff_date = None

            if fix['fixture']['venue']:
                venue_name = fix['fixture']['venue'].get('name', 'Unknown')
                city_name = fix['fixture']['venue'].get('city')
                
            status = fix['fixture']['status']['short']
            if status in ['FT', 'AET', 'PEN']:
                try:
                    hg = fix['goals']['home']
                    ag = fix['goals']['away']
                    if hg > ag: actual_res_val = '1'
                    elif hg == ag: actual_res_val = 'X'
                    else: actual_res_val = '2'
                except: pass
                # שמור לDB (ללמידה) אבל דלג — המשחק כבר הסתיים, אין טעם להמליץ עליו
                print(f"      ⏩ {h_name} - {a_name}: משחק שהסתיים (status={status}). שומר תוצאה ומדלג.")
                if fid and actual_res_val:
                    try:
                        hg_i, ag_i = int(hg), int(ag)
                        db_cursor.execute("SELECT id FROM matches WHERE fixture_id = ?", (fid,))
                        _row = db_cursor.fetchone()
                        if _row:
                            db_cursor.execute(
                                "UPDATE matches SET actual_result=?, match_date=?, home_goals=?, away_goals=? WHERE fixture_id=?",
                                (actual_res_val, kickoff_date or datetime.now().strftime("%Y-%m-%d"), hg_i, ag_i, fid),
                            )
                        else:
                            db_cursor.execute(
                                "INSERT INTO matches (fixture_id, match_date, home_team, away_team, actual_result, home_goals, away_goals) VALUES (?,?,?,?,?,?,?)",
                                (
                                    fid,
                                    kickoff_date or datetime.now().strftime("%Y-%m-%d"),
                                    h_name,
                                    a_name,
                                    actual_res_val,
                                    hg_i,
                                    ag_i,
                                ),
                            )
                        db_conn.commit()
                    except Exception as _e:
                        print(f"         ⚠️ שגיאה בשמירת תוצאה ל-DB: {_e}")
                continue

            # מסנן משחקים שרק טרם התחילו (NS=Not Started, TBD, PST=Postponed, CANC=Cancelled)
            if status in ['PST', 'CANC', 'ABD', 'AWD', 'WO']:
                print(f"      ⏩ {h_name} - {a_name}: משחק נדחה/בוטל (status={status}). מדלג.")
                continue

        print(f"      ✅ אושר: משחק מליגת '{league_name}' - מתחיל לשלוף נתונים.")
        match_time_str = (raw_date[11:16] if raw_date and len(raw_date) >= 16 else None)  # HH:MM מ-ISO

        # גיבוי תאריך/שעה מהווינר כש-API-Football לא מחזיר (h2h ריק או משחק שהסתיים)
        if not kickoff_date or not match_time_str:
            winner_entry = LiveWinnerAPI._find_match_entry(h_name, a_name)
            if isinstance(winner_entry, dict):
                wd = winner_entry.get("winner_date") or ""
                wt = winner_entry.get("winner_time") or ""
                if wd and re.match(r"^\d{4}-\d{2}-\d{2}", str(wd)):
                    kickoff_date = kickoff_date or str(wd)[:10]
                if wt and re.match(r"^\d{1,2}:\d{2}", str(wt)):
                    match_time_str = match_time_str or str(wt)[:5] if len(str(wt)) >= 5 else str(wt)

        match_obj = {
            "match": f"{h_name} - {a_name}", "home": h_name, "away": a_name,
            "fixture_id": fid,
            "match_date": kickoff_date,
            "match_time": match_time_str,
            "model_probs": {}, "market_odds": {}, 
            "history": {"home": [], "away": []}, "news_flash": "", "weather": None,
            "logos": {"home": f"https://media.api-sports.io/football/teams/{hid}.png", "away": f"https://media.api-sports.io/football/teams/{aid}.png"}, 
            "pro_data": {"venue": venue_name, "league": league_name, "country": country_name},
            "intel": {"injuries": []} 
        }
        
        winner_odds = LiveWinnerAPI.get_1x2_odds(h_name, a_name)
        if winner_odds and len(winner_odds) == 3:
            # שומרים את היחסים הגולמיים, אבל מחשבים גם הסתברויות שוק מתוקננות לשימוש פנימי
            match_obj['market_odds'] = {
                '1': float(winner_odds.get('1')),
                'X': float(winner_odds.get('X')),
                '2': float(winner_odds.get('2'))
            }
            match_obj['market_probs'] = odds_to_probs(match_obj['market_odds'])
        else:
            print(f"      ⚠️ יחסי ווינר לא נמצאו במטמון עבור המשחק הזה. מדלג.")
            continue 
            
        print(f"   ⏳ מחשב מודל מתמטי (xG) לבקרת איכות...")
        try:
            match_obj["history"]["home"] = Network.fetch("fixtures", {"team": hid, "last": 10})
            match_obj["history"]["away"] = Network.fetch("fixtures", {"team": aid, "last": 10})
        except: 
            pass

        def calculate_hybrid_lambdas(team_id, history_list):
            """
            חישוב למבדות מבוסס xG/שערים עם:
            - חלון עמוק יותר (עד 10 משחקים) – שיפור יציבות ואיכות החיזוי
            - משקולות גבוהות יותר למשחקים האחרונים
            - רגולירזציה לעוגן ליגתי/גלובלי כדי למנוע למבדות קיצוניות.
            """
            xg_for, xg_against, goals_for, goals_against = [], [], [], []
            weights = []
            max_games = 10

            for idx, match in enumerate(history_list[:max_games]):
                try:
                    is_home = match['teams']['home']['id'] == team_id
                    gf = match['goals']['home'] if is_home else match['goals']['away']
                    ga = match['goals']['away'] if is_home else match['goals']['home']
                    if gf is not None and ga is not None:
                        goals_for.append(gf)
                        goals_against.append(ga)

                    # משקל גדל ככל שהמשחק קרוב יותר להווה
                    # המשחק האחרון בחלון מקבל משקל 1.0, הראשון 0.3
                    position = idx  # 0 הוא הישן יותר בחלון, 7 החדש ביותר
                    base_w = 0.3 + 0.7 * (1 - position / max(1, max_games - 1))
                    weights.append(base_w)

                    fid_hist = match['fixture']['id']
                    stats = Network.fetch("fixtures/statistics", {"fixture": fid_hist})
                    if stats:
                        t_xg, o_xg = 0.0, 0.0
                        for team_stat in stats:
                            tid = team_stat['team']['id']
                            val = 0.0
                            for s in team_stat['statistics']:
                                if str(s['type']).lower() == 'expected_goals':
                                    val = float(s['value']) if s['value'] else 0.0
                                    break
                            if tid == team_id:
                                t_xg = val
                            else:
                                o_xg = val

                        if t_xg > 0 or o_xg > 0:
                            xg_for.append(t_xg)
                            xg_against.append(o_xg)
                except Exception:
                    continue

            league_anchor = 1.35  # ממוצע גול עולמי סביר למשחק

            if xg_for and xg_against:
                n = len(xg_for)
                if weights and len(weights) == n:
                    w_sum = sum(weights)
                    if w_sum > 0:
                        avg_f = sum(v * w for v, w in zip(xg_for, weights)) / w_sum
                        avg_a = sum(v * w for v, w in zip(xg_against, weights)) / w_sum
                    else:
                        avg_f = sum(xg_for) / n
                        avg_a = sum(xg_against) / n
                else:
                    avg_f = sum(xg_for) / len(xg_for)
                    avg_a = sum(xg_against) / len(xg_against)
                method_used = "xG"
            elif goals_for and goals_against:
                n = len(goals_for)
                if weights and len(weights) == n:
                    w_sum = sum(weights)
                    if w_sum > 0:
                        avg_f = sum(v * w for v, w in zip(goals_for, weights)) / w_sum
                        avg_a = sum(v * w for v, w in zip(goals_against, weights)) / w_sum
                    else:
                        avg_f = sum(goals_for) / n
                        avg_a = sum(goals_against) / n
                else:
                    avg_f = sum(goals_for) / len(goals_for)
                    avg_a = sum(goals_against) / len(goals_against)
                avg_f = max(0.4, avg_f)
                avg_a = max(0.4, avg_a)
                method_used = "Actual_Goals"
            else:
                avg_f, avg_a = league_anchor, league_anchor
                method_used = "Global_Avg"

            # רגולירזציה לעוגן הליגתי – ככל שיש פחות משחקים, כך העוגן חזק יותר
            sample_size = max(len(xg_for), len(goals_for))
            alpha = min(1.0, sample_size / float(max_games))
            avg_f = alpha * avg_f + (1 - alpha) * league_anchor
            avg_a = alpha * avg_a + (1 - alpha) * league_anchor

            return avg_f, avg_a, method_used

        h_xg_f, h_xg_a, h_method = calculate_hybrid_lambdas(hid, match_obj['history'].get('home', []))
        a_xg_f, a_xg_a, a_method = calculate_hybrid_lambdas(aid, match_obj['history'].get('away', []))

        # ELO fallback: כשאין נתוני xG/goals מה-API, משתמש בדירוג ELO במקום ממוצע ליגתי אחיד
        if _elo_ratings:
            h_elo = _elo_ratings.get(h_name, _ELO_DEFAULT)
            a_elo = _elo_ratings.get(a_name, _ELO_DEFAULT)
            _elo_scale = elo_to_lambda_scale(h_elo, a_elo)
            if h_method == "Global_Avg":
                h_xg_f = _ELO_ANCHOR * _elo_scale
                h_xg_a = _ELO_ANCHOR / _elo_scale
            if a_method == "Global_Avg":
                a_xg_f = _ELO_ANCHOR / _elo_scale
                a_xg_a = _ELO_ANCHOR * _elo_scale

        h_lambda = max(0.5, ((h_xg_f + a_xg_a) / 2.0) * HOME_ADVANTAGE_FACTOR)
        a_lambda = max(0.5, ((a_xg_f + h_xg_a) / 2.0) * AWAY_DISADVANTAGE_FACTOR)

        raw_model_probs = calculate_dixon_coles_probs(h_lambda, a_lambda)
        model_probs = normalize_probs(raw_model_probs)
        if load_calibration_params and apply_calibration:
            cal_params = load_calibration_params()
            if cal_params:
                model_probs = apply_calibration(model_probs, cal_params)
        match_obj['model_probs'] = model_probs
        match_obj.setdefault("pro_data", {})
        match_obj["pro_data"]["lambdas"] = {"home": h_lambda, "away": a_lambda}
        
        best_base_ev = -999.0
        for bet_type in ['1', 'X', '2']:
            p = match_obj['model_probs'].get(bet_type, 0)
            odds = match_obj['market_odds'].get(bet_type, 0)
            if p > 0 and odds > 0:
                ev = (p * odds) - 1
                if ev > best_base_ev:
                    best_base_ev = ev
                
        if best_base_ev < -0.08:
            print(f"      ⛔ חסימה מתמטית: התוחלת ההתחלתית גרועה ({best_base_ev:.2%}). מדלג על ניתוח נוסף.")
            match_obj['news_flash'] = "משחק זה נפסל בשלב הסינון המתמטי המהיר עקב חוסר כדאיות."
            match_obj['context'] = "נותח רק מודל מתמטי ושוק."
        else:
            print(f"      🎯 אותר פוטנציאל (EV מתמטי: {best_base_ev:.2f}). ממשיך ללא הפעלת Groq/LLM.")
            match_obj['weather'] = WeatherStation.get_forecast(city_name)
            
            def calc_form(history_list, team_id):
                if not history_list: return "אין נתונים"
                results = []
                for g in history_list[:5]:
                    try:
                        hg = g['goals']['home']
                        ag = g['goals']['away']
                        is_home = g['teams']['home']['id'] == team_id
                        if hg == ag: results.append("D")
                        elif (is_home and hg > ag) or (not is_home and ag > hg): results.append("W")
                        else: results.append("L")
                    except: 
                        pass
                return "-".join(results) if results else "אין נתונים"

            home_form = calc_form(match_obj['history'].get('home', []), hid)
            away_form = calc_form(match_obj['history'].get('away', []), aid)
            form_text = f"פורמה 5 משחקים אחרונים: קבוצת בית [{home_form}], קבוצת חוץ [{away_form}]."
            
            match_obj['news_flash'] = form_text
            match_obj['context'] = form_text

        # חישוב הסתברויות סופיות משולבות (מודל + שוק בלבד)
        model_probs = match_obj.get('model_probs', {}) or {}
        market_probs = match_obj.get('market_probs', {}) or odds_to_probs(match_obj['market_odds'])
        combined_raw = {}
        for k in ['1', 'X', '2']:
            pm = float(model_probs.get(k, 0.0) or 0.0)
            pmarket = float(market_probs.get(k, 0.0) or 0.0)
            combined_raw[k] = (pm * w_model) + (pmarket * w_market)
        final_probs = normalize_probs(combined_raw)
        match_obj['final_probs'] = final_probs

        # סיווג משחק לפי EV ותוחלת
        tier, classified_ev, risk_category, tier_bet = classify_match(final_probs, match_obj['market_odds'])
        match_obj['tier'] = tier
        match_obj['classified_ev'] = classified_ev
        match_obj['risk_category'] = risk_category
        match_obj['pro_data']['tier'] = tier
        match_obj['pro_data']['classified_ev'] = classified_ev
        match_obj['pro_data']['risk_category'] = risk_category

        # --- Multi-market EV: 1X2, Over/Under 2.5, BTTS ---
        score_probs = build_poisson_score_matrix(h_lambda, a_lambda)
        extended_odds = None
        try:
            extended_odds = LiveWinnerAPI.get_extended_odds(h_name, a_name)
        except Exception:
            pass

        ev_candidates = []
        # 1X2
        for bet_type in ['1', 'X', '2']:
            p = final_probs.get(bet_type, 0.0)
            odds = match_obj['market_odds'].get(bet_type, 0)
            try:
                odds = float(odds) if odds else 0
            except (TypeError, ValueError):
                odds = 0
            if p > 0 and odds > 0:
                ev = (p * odds) - 1
                ev_candidates.append((ev, bet_type, "1X2", odds, p))

        # Totals (Over/Under 2.5)
        if extended_odds:
            totals = extended_odds.get("totals") or {}
            o_over = totals.get("over_2_5")
            o_under = totals.get("under_2_5")
            try:
                o_over = float(o_over) if o_over else 0
                o_under = float(o_under) if o_under else 0
            except (TypeError, ValueError):
                o_over, o_under = 0, 0
            if o_over > 0:
                p = score_probs["prob_over_2_5"]
                if p > 0:
                    ev = (p * o_over) - 1
                    ev_candidates.append((ev, "Over 2.5", "Totals", o_over, p))
            if o_under > 0:
                p = score_probs["prob_under_2_5"]
                if p > 0:
                    ev = (p * o_under) - 1
                    ev_candidates.append((ev, "Under 2.5", "Totals", o_under, p))

            # BTTS
            btts = extended_odds.get("btts") or {}
            o_yes = btts.get("yes")
            o_no = btts.get("no")
            try:
                o_yes = float(o_yes) if o_yes else 0
                o_no = float(o_no) if o_no else 0
            except (TypeError, ValueError):
                o_yes, o_no = 0, 0
            if o_yes > 0:
                p = score_probs["prob_btts_yes"]
                if p > 0:
                    ev = (p * o_yes) - 1
                    ev_candidates.append((ev, "BTTS Yes", "BTTS", o_yes, p))
            if o_no > 0:
                p = score_probs["prob_btts_no"]
                if p > 0:
                    ev = (p * o_no) - 1
                    ev_candidates.append((ev, "BTTS No", "BTTS", o_no, p))

        best_final_ev = -999.0
        recommended_bet = None
        market_type = "1X2"
        chosen_odds = 0.0
        chosen_prob = 0.0
        kelly_fraction = 0.0

        if ev_candidates:
            ev_candidates.sort(key=lambda x: x[0], reverse=True)
            best_ev, recommended_bet, market_type, chosen_odds, chosen_prob = ev_candidates[0]
            best_final_ev = best_ev
            b = chosen_odds - 1
            if b > 0 and best_final_ev > 0:
                q = 1 - chosen_prob
                k = (b * chosen_prob - q) / b
                kelly_fraction = max(0, k / 4)

        match_obj['pro_data']['best_ev'] = best_final_ev
        match_obj['pro_data']['kelly'] = kelly_fraction * 100
        match_obj['pro_data']['recommended_bet'] = recommended_bet
        match_obj['pro_data']['market_type'] = market_type
        match_obj['pro_data']['odds'] = chosen_odds
        match_obj['pro_data']['chosen_prob'] = chosen_prob
        if extended_odds:
            match_obj['totals_odds'] = extended_odds.get('totals') or {}
            match_obj['btts_odds'] = extended_odds.get('btts') or {}
        else:
            match_obj['totals_odds'] = {}
            match_obj['btts_odds'] = {}

        # --- ניתוח ארביטראז' / Surebet לפי שווקי ווינר ---
        if not extended_odds:
            try:
                extended_odds = LiveWinnerAPI.get_extended_odds(h_name, a_name)
            except Exception:
                pass

        match_obj["arbitrage"] = {}
        if extended_odds:
            try:
                outcomes = build_outcomes()
                selections = build_selection_map(extended_odds, outcomes)

                arbs = {}

                # 1) Holy Trinity – BTTS No + Over 2.5 + Correct Score 1-1
                holy_ids = ["btts_no", "total_over_2_5", "cs_1-1"]
                if all(sel_id in selections for sel_id in holy_ids):
                    odds_map = {sid: selections[sid]["odds"] for sid in holy_ids}
                    alloc = compute_surebet_allocation(odds_map, ARBITRAGE_MAX_STAKE_PER_MATCH)
                    if alloc:
                        covered = set()
                        for sid in holy_ids:
                            covered |= selections[sid]["covers"]
                        coverage_ratio = len(covered) / float(len(outcomes)) if outcomes else 0.0
                        arbs["holy_trinity"] = {
                            "type": "Holy Trinity",
                            "label": "השילוש הקדוש",
                            "market_keys": holy_ids,
                            "odds": odds_map,
                            "allocation": alloc,
                            "coverage_ratio": coverage_ratio,
                            "surebet": coverage_ratio >= 0.999 and alloc.get("guaranteed_profit", 0.0) > 0,
                            "tags": ["Surebet", "Risk-Free Bet", "Guaranteed Profit"],
                        }

                # 2) Derived Market Arbitrage – Correct Score 1-0 + Over 1.5
                cs10_id = "cs_1-0"
                over15_id = "total_over_1_5"
                if cs10_id in selections and over15_id in selections:
                    o_cs = selections[cs10_id]["odds"]
                    o_over = selections[over15_id]["odds"]
                    try:
                        B = float(ARBITRAGE_MAX_STAKE_PER_MATCH)
                        # חלוקה רציפה ראשונית לפי הנוסחה שנתן גלעד
                        stake_cs_raw = B / (1.0 + (o_cs / o_over))
                        stake_over_raw = B - stake_cs_raw

                        # עיגול לחלוקת ווינר (קפיצות 5 ₪, מינימום 10 ₪)
                        raw_map = {cs10_id: stake_cs_raw, over15_id: stake_over_raw}
                        rounded = _round_stakes_to_winner_units(raw_map, {cs10_id: o_cs, over15_id: o_over}, max_total=B)
                        if not rounded:
                            raise ValueError("failed to round stakes")

                        stake_cs = rounded["stakes"].get(cs10_id, 0.0)
                        stake_over = rounded["stakes"].get(over15_id, 0.0)
                        total_stake = rounded["total_stake"]
                        profit_1_0 = stake_cs * o_cs - total_stake
                        profit_over = stake_over * o_over - total_stake
                        min_profit = min(profit_1_0, profit_over, rounded.get("guaranteed_profit", profit_1_0))
                        if min_profit > 0:
                            arbs["derived_1_0_over_1_5"] = {
                                "type": "Derived Market Arbitrage",
                                "label": "1-0 + Over 1.5",
                                "market_keys": [cs10_id, over15_id],
                                "odds": {cs10_id: o_cs, over15_id: o_over},
                                "stakes": {cs10_id: stake_cs, over15_id: stake_over},
                                "total_stake": total_stake,
                                "profit_if_1_0": profit_1_0,
                                "profit_if_over_hits": profit_over,
                                "guaranteed_profit_over_covered": min_profit,
                                "roi": min_profit / total_stake if total_stake > 0 else 0.0,
                                "coverage_ratio": len(
                                    selections[cs10_id]["covers"] | selections[over15_id]["covers"]
                                )
                                / float(len(outcomes))
                                if outcomes
                                else 0.0,
                            }
                    except Exception:
                        pass

                # 3) Under 2.5 + Correct Score 2-1 + 1-2
                under25_id = "total_under_2_5"
                cs21_id = "cs_2-1"
                cs12_id = "cs_1-2"
                under_cluster_ids = [under25_id, cs21_id, cs12_id]
                if all(sel_id in selections for sel_id in under_cluster_ids):
                    odds_map = {sid: selections[sid]["odds"] for sid in under_cluster_ids}
                    alloc = compute_surebet_allocation(odds_map, ARBITRAGE_MAX_STAKE_PER_MATCH)
                    if alloc:
                        covered = set()
                        for sid in under_cluster_ids:
                            covered |= selections[sid]["covers"]
                        coverage_ratio = len(covered) / float(len(outcomes)) if outcomes else 0.0
                        arbs["under25_correct_scores"] = {
                            "type": "Under 2.5 + Correct Scores",
                            "label": "Under 2.5 + 2-1 + 1-2",
                            "market_keys": under_cluster_ids,
                            "odds": odds_map,
                            "allocation": alloc,
                            "coverage_ratio": coverage_ratio,
                            "surebet": coverage_ratio >= 0.999 and alloc["implied_prob_sum"] < 1.0 and alloc.get("guaranteed_profit", 0.0) > 0,
                        }

                # 4) BTTS No + Over 2.5 + Correct Score 1-1 (בהגדרה השנייה – חופף לשילוש הקדוש)
                btts_over_ids = ["btts_no", "total_over_2_5", "cs_1-1"]
                if all(sel_id in selections for sel_id in btts_over_ids):
                    odds_map = {sid: selections[sid]["odds"] for sid in btts_over_ids}
                    alloc = compute_surebet_allocation(odds_map, ARBITRAGE_MAX_STAKE_PER_MATCH)
                    if alloc:
                        covered = set()
                        for sid in btts_over_ids:
                            covered |= selections[sid]["covers"]
                        coverage_ratio = len(covered) / float(len(outcomes)) if outcomes else 0.0
                        arbs["btts_over_cs"] = {
                            "type": "BTTS No + Over 2.5 + 1-1",
                            "label": "BTTS No + Over 2.5 + 1-1",
                            "market_keys": btts_over_ids,
                            "odds": odds_map,
                            "allocation": alloc,
                            "coverage_ratio": coverage_ratio,
                            "surebet": coverage_ratio >= 0.999 and alloc["implied_prob_sum"] < 1.0 and alloc.get("guaranteed_profit", 0.0) > 0,
                        }

                # 5) Half Time Exploit – X/1 + 1/1 מול 1 (1X2)
                try:
                    odds_htft = extended_odds.get("ht_ft") or {}
                    o_x1 = float(odds_htft.get("X/1") or odds_htft.get("X/1".replace(" ", "")) or 0)
                    o_11 = float(odds_htft.get("1/1") or odds_htft.get("1/1".replace(" ", "")) or 0)
                    odds_1 = None
                    if "1x2_1" in selections:
                        odds_1 = selections["1x2_1"]["odds"]
                    if o_x1 and o_11 and odds_1 and odds_1 > 1.0:
                        inv_combo = decimal_to_implied_prob(o_x1) + decimal_to_implied_prob(o_11)
                        p_direct = decimal_to_implied_prob(odds_1)
                        if inv_combo < p_direct and inv_combo > 0:
                            arbs["half_time_exploit"] = {
                                "type": "Half Time Exploit",
                                "label": "X/1 + 1/1 מול 1",
                                "odds": {"X/1": o_x1, "1/1": o_11, "1": odds_1},
                                "implied_sum_combo": inv_combo,
                                "implied_prob_direct_1": p_direct,
                            }
                except Exception:
                    pass

                # 6) מנוע כללי – Full Market Arbitrage (כל השווקים)
                try:
                    full_engine = compute_generic_full_market_arbitrage(selections, outcomes, max_combo_size=3, max_results=5)
                    if full_engine:
                        arbs["full_market"] = full_engine
                except Exception:
                    pass

                # TODO: ניתן להרחיב כאן את שאר 10 הסריקות המתקדמות אחת לאחת, בהתאם לזמינות השווקים,
                # מעבר לשילובים המיוחדים שכבר מחושבים למעלה.

                match_obj["arbitrage"] = arbs
            except Exception as _arb_e:
                match_obj["arbitrage_error"] = str(_arb_e)

        if best_final_ev > 0:
            print(f"      📈 סופי (Quarter Kelly): {kelly_fraction*100:.2f}% מהקופה על הימור '{recommended_bet}' (תוחלת: {best_final_ev:.2f})")
        else:
            print(f"      ⛔ לא נמצא ערך חיובי בסוף הניתוח - המלצת מערכת: עזוב את המשחק.")

        if fid:
            try:
                db_cursor.execute("SELECT id, actual_result FROM matches WHERE fixture_id = ?", (fid,))
                existing = db_cursor.fetchone()
                if existing:
                    existing_actual = existing[1]
                    resolved_actual = existing_actual if existing_actual else actual_res_val
                    pro = match_obj.get('pro_data') or {}
                    rec_bet = str(pro.get('recommended_bet') or '') if pro.get('recommended_bet') else None
                    mkt_type = str(pro.get('market_type') or '1X2')
                    db_cursor.execute('''UPDATE matches SET
                                        match_date=?, home_team=?, away_team=?,
                                        model_prob_1=?, model_prob_x=?, model_prob_2=?,
                                        market_prob_1=?, market_prob_x=?, market_prob_2=?,
                                        actual_result=?,
                                        final_prob_1=?, final_prob_x=?, final_prob_2=?,
                                        tier=?, classified_ev=?, risk_category=?,
                                        recommended_bet_market=?, market_type=?
                                      WHERE fixture_id=?''',
                                   (kickoff_date or datetime.now().strftime("%Y-%m-%d"), h_name, a_name,
                                    match_obj['model_probs'].get('1'), match_obj['model_probs'].get('X'), match_obj['model_probs'].get('2'),
                                    match_obj.get('market_probs', {}).get('1'), match_obj.get('market_probs', {}).get('X'), match_obj.get('market_probs', {}).get('2'),
                                    resolved_actual,
                                    final_probs.get('1'), final_probs.get('X'), final_probs.get('2'),
                                    tier, classified_ev, risk_category,
                                    rec_bet, mkt_type,
                                    fid))
                else:
                    pro = match_obj.get('pro_data') or {}
                    rec_bet = str(pro.get('recommended_bet') or '') if pro.get('recommended_bet') else None
                    mkt_type = str(pro.get('market_type') or '1X2')
                    db_cursor.execute('''INSERT INTO matches (
                                        fixture_id, match_date, home_team, away_team,
                                        model_prob_1, model_prob_x, model_prob_2,
                                        market_prob_1, market_prob_x, market_prob_2,
                                        actual_result,
                                        final_prob_1, final_prob_x, final_prob_2,
                                        tier, classified_ev, risk_category,
                                        recommended_bet_market, market_type)
                                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                   (fid, kickoff_date or datetime.now().strftime("%Y-%m-%d"), h_name, a_name,
                                    match_obj['model_probs'].get('1'), match_obj['model_probs'].get('X'), match_obj['model_probs'].get('2'),
                                    match_obj.get('market_probs', {}).get('1'), match_obj.get('market_probs', {}).get('X'), match_obj.get('market_probs', {}).get('2'),
                                    actual_res_val,
                                    final_probs.get('1'), final_probs.get('X'), final_probs.get('2'),
                                    tier, classified_ev, risk_category,
                                    rec_bet, mkt_type))
                db_conn.commit()
            except Exception as _e:
                print(f"      ⚠️ שגיאה בשמירת משחק ל-DB: {_e}")

        results.append(match_obj)

    flush_translation_cache()
    db_conn.close()

    results.sort(key=lambda x: x['pro_data'].get('best_ev', -999), reverse=True)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"✅ Analysis Complete. Sorted {len(results)} high-tier matches by value.")

if __name__ == "__main__":
    process_matches()
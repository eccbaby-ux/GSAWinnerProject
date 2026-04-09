# -*- coding: utf-8 -*-
"""
תיקון דאדיימה (Dadima Correction)
===================================
Final, aggressive calibration and risk-management layer.
Runs after v76_Master_Nachshon.py and before the Streamlit dashboard.

Addresses: Overconfidence, overfitting, and Winner's high Vig (Overround).
Applies: Vig penalty, Bayesian shrinkage, strict EV recalc, fractional Kelly, brutal filtering.
"""

import json
import math
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "analysis_results_v76.json")

# --- Dadima Parameters ---
SYSTEM_WEIGHT = 0.60       # 60% system probability
MARKET_WEIGHT = 0.40       # 40% market true probability (Bayesian shrinkage)
KELLY_FRACTION = 0.25      # 1/4 Kelly Criterion
MAX_BET_FRACTION = 0.02    # Cap: 2% of bankroll max
EV_MIN_THRESHOLD = 0.02    # Minimum EV (2% edge) - below this = NO BET (strict production)


def _remove_vig_proportional(implied_probs, keys=None):
    """
    Scale implied probabilities so they sum to 1.0 (remove Vig).
    For 1X2: keys=['1','X','2']. For Totals/BTTS: use only the 2 related odds.
    """
    if keys is None:
        keys = list(implied_probs.keys())
    total = sum(implied_probs.get(k, 0) for k in keys)
    if total <= 0:
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: implied_probs.get(k, 0) / total for k in keys}


def _remove_vig_shin(implied_probs, tol=1e-6, max_iter=100):
    """Shin's method for high-vig books (e.g. Winner ~12% overround)."""
    total_implied = sum(implied_probs.values())
    if total_implied <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
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


def odds_to_market_true_probs(market_odds, keys=None):
    """
    Convert market decimal odds to true probabilities (Vig removed).
    For 1X2: keys=['1','X','2'] (default). For Totals/BTTS: use keys for the 2 outcomes only.
    """
    if keys is None:
        keys = ['1', 'X', '2']
    if not isinstance(market_odds, dict):
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    implied = {}
    for key in keys:
        try:
            v = float(market_odds.get(key) or 0)
        except (TypeError, ValueError):
            v = 0.0
        implied[key] = (1.0 / v) if v > 0 else 0.0
    overround = sum(implied.values())
    if overround > 1.08 and len(keys) == 3:
        return _remove_vig_shin(implied)
    return _remove_vig_proportional(implied, keys)


def normalize_probs(probs):
    """Ensure probabilities sum to 1.0."""
    if not isinstance(probs, dict):
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    total = sum(float(probs.get(k, 0) or 0) for k in ['1', 'X', '2'])
    if total <= 0:
        return {'1': 1/3, 'X': 1/3, '2': 1/3}
    return {k: max(0.0, float(probs.get(k, 0) or 0) / total) for k in ['1', 'X', '2']}


def apply_dadima_shrinkage(system_probs, market_odds, market_probs=None, keys=None):
    """
    Bayesian Shrinkage: blend 60% system + 40% market true probability.
    Pulls overconfident model predictions toward the market.
    keys: for 1X2 use ['1','X','2'], for 2-way use the two outcome keys.
    """
    if keys is None:
        keys = ['1', 'X', '2']
    market_true = market_probs if market_probs else odds_to_market_true_probs(market_odds, keys)
    if not isinstance(market_true, dict):
        market_true = odds_to_market_true_probs(market_odds, keys)
    shrunk = {}
    for k in keys:
        ps = float(system_probs.get(k, 0) or 0)
        pm = float(market_true.get(k, 0) or 0)
        shrunk[k] = (SYSTEM_WEIGHT * ps) + (MARKET_WEIGHT * pm)
    total = sum(shrunk.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: v / total for k, v in shrunk.items()}


def recalc_ev_and_kelly_single(shrunk_prob, odds):
    """EV and Kelly for a single outcome."""
    if shrunk_prob <= 0 or odds <= 0:
        return -999.0, 0.0
    ev = (shrunk_prob * odds) - 1.0
    b = odds - 1
    kelly_raw = 0.0
    if b > 0 and ev > 0:
        q = 1 - shrunk_prob
        kelly_raw = max(0.0, (b * shrunk_prob - q) / b)
    kelly_fraction = min(KELLY_FRACTION * kelly_raw, MAX_BET_FRACTION)
    return ev, kelly_fraction


def process_match(match_obj):
    """
    Apply Dadima correction to a single match.
    Reads market_type and recommended_bet from pro_data (set by v76).
    For Totals/BTTS: uses only the 2 related odds for Vig removal (not 3).
    Modifies match_obj in place; returns True if processed successfully.
    """
    try:
        pro_data = match_obj.get('pro_data') or {}
        market_type = pro_data.get('market_type') or '1X2'
        recommended_bet = pro_data.get('recommended_bet')
        chosen_prob = float(pro_data.get('chosen_prob') or 0)
        chosen_odds = float(pro_data.get('odds') or 0)

        final_probs = match_obj.get('final_probs') or {}
        market_odds = match_obj.get('market_odds') or {}

        if not market_odds and market_type == '1X2':
            return False

        if market_type == '1X2':
            return _process_match_1x2(match_obj, final_probs, market_odds, pro_data)

        # Totals or BTTS: 2-way market
        if market_type == 'Totals':
            two_way_odds = match_obj.get('totals_odds') or {}
            keys = ['over_2_5', 'under_2_5']
            bet_key = 'over_2_5' if recommended_bet == 'Over 2.5' else 'under_2_5'
        elif market_type == 'BTTS':
            two_way_odds = match_obj.get('btts_odds') or {}
            keys = ['yes', 'no']
            bet_key = 'yes' if recommended_bet == 'BTTS Yes' else 'no'
        else:
            return _process_match_1x2(match_obj, final_probs, market_odds, pro_data)

        o1 = float(two_way_odds.get(keys[0]) or 0)
        o2 = float(two_way_odds.get(keys[1]) or 0)
        if o1 <= 0 or o2 <= 0:
            return False

        # Vig removal: use ONLY the 2 odds for this market
        implied = {keys[0]: 1.0 / o1, keys[1]: 1.0 / o2}
        market_true = _remove_vig_proportional(implied, keys)

        # Bayesian shrinkage on chosen outcome
        system_probs = {bet_key: chosen_prob, keys[1 - keys.index(bet_key)]: 1 - chosen_prob}
        shrunk_probs = apply_dadima_shrinkage(system_probs, two_way_odds, market_true, keys)
        shrunk_prob = float(shrunk_probs.get(bet_key, 0) or 0)

        best_ev, kelly_fraction = recalc_ev_and_kelly_single(shrunk_prob, chosen_odds)

        if best_ev < EV_MIN_THRESHOLD:
            recommended_bet = "NO BET"
            kelly_fraction = 0.0

        pro_data['best_ev'] = best_ev
        pro_data['classified_ev'] = best_ev
        pro_data['kelly'] = kelly_fraction * 100
        pro_data['recommended_bet'] = recommended_bet
        pro_data['risk_category'] = "none" if recommended_bet == "NO BET" else pro_data.get('risk_category', 'medium')
        pro_data['tier'] = "skip" if recommended_bet == "NO BET" else pro_data.get('tier', 'skip')
        match_obj['pro_data'] = pro_data
        match_obj['tier'] = pro_data['tier']
        match_obj['classified_ev'] = best_ev
        match_obj['risk_category'] = pro_data['risk_category']
        return True
    except Exception:
        return False


def _process_match_1x2(match_obj, final_probs, market_odds, pro_data):
    """Original 1X2 logic."""
    market_probs = match_obj.get('market_probs')
    if not final_probs or not market_odds:
        return False

    shrunk_probs = apply_dadima_shrinkage(final_probs, market_odds, market_probs)
    match_obj['final_probs'] = shrunk_probs

    best_ev = -999.0
    recommended_bet = None
    kelly_fraction = 0.0
    for bet_type in ['1', 'X', '2']:
        p = float(shrunk_probs.get(bet_type, 0) or 0)
        odds = float(market_odds.get(bet_type, 0) or 0)
        if p > 0 and odds > 0:
            ev = (p * odds) - 1.0
            if ev > best_ev:
                best_ev = ev
                recommended_bet = bet_type
                b = odds - 1
                if b > 0 and ev > 0:
                    q = 1 - p
                    kelly_raw = max(0.0, (b * p - q) / b)
                    kelly_fraction = min(KELLY_FRACTION * kelly_raw, MAX_BET_FRACTION)

    if best_ev < EV_MIN_THRESHOLD:
        recommended_bet = "NO BET"
        kelly_fraction = 0.0

    pro_data['best_ev'] = best_ev
    pro_data['classified_ev'] = best_ev
    pro_data['kelly'] = kelly_fraction * 100
    pro_data['recommended_bet'] = recommended_bet
    pro_data['risk_category'] = "none" if recommended_bet == "NO BET" else pro_data.get('risk_category', 'medium')
    pro_data['tier'] = "skip" if recommended_bet == "NO BET" else pro_data.get('tier', 'skip')
    match_obj['pro_data'] = pro_data
    match_obj['tier'] = pro_data['tier']
    match_obj['classified_ev'] = best_ev
    match_obj['risk_category'] = pro_data['risk_category']
    return True


def main():
    print("=" * 50)
    print("  תיקון דאדיימה (Dadima Correction)")
    print("=" * 50)

    if not os.path.exists(JSON_PATH):
        print(f"[ERROR] File not found: {JSON_PATH}")
        sys.exit(1)

    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        sys.exit(1)

    if not isinstance(data, list):
        print("[ERROR] Expected JSON array of matches.")
        sys.exit(1)

    if len(data) == 0:
        print("[WARNING] JSON is empty - nothing to correct.")
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        print("[OK] Wrote empty array. Exiting.")
        sys.exit(0)

    processed = 0
    no_bet_count = 0
    for match_obj in data:
        if not isinstance(match_obj, dict):
            continue
        if process_match(match_obj):
            processed += 1
            if match_obj.get('pro_data', {}).get('recommended_bet') == "NO BET":
                no_bet_count += 1

    try:
        with open(JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"[ERROR] Failed to write file: {e}")
        sys.exit(1)

    print(f"[OK] Processed {processed}/{len(data)} matches.")
    print(f"[OK] Dadima filtered {no_bet_count} to NO BET (EV < {EV_MIN_THRESHOLD:.0%}).")
    print("=" * 50)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Local, fast team mapping: Hebrew name -> Team ID.
Uses translation_cache.json, hebrew_to_id_mapping_new.csv, ID.csv.
Fuzzy matching via rapidfuzz (or thefuzz) with 85% confidence threshold.
When AUTO_ADD_100_PERCENT=True: 100% matches not in hebrew CSV are auto-appended to grow the mapper.
Fallback: translate Hebrew (Lingva) + match the pair against API-Football fixtures for that date
(few API calls per day, good coverage for international names on Winner).
"""
import json
import os
import re
import logging
import time
import urllib.parse

import requests

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    try:
        from thefuzz import fuzz
        HAS_RAPIDFUZZ = True
    except ImportError:
        HAS_RAPIDFUZZ = False

from difflib import SequenceMatcher

# Paths relative to this file
_BASE = os.path.dirname(os.path.abspath(__file__))
TRANSLATION_CACHE_PATH = os.path.join(_BASE, "translation_cache.json")
HEBREW_CSV_PATH = os.path.join(_BASE, "hebrew_to_id_mapping_new.csv")
ID_CSV_PATH = os.path.join(_BASE, "ID.csv")

FUZZY_THRESHOLD = 85  # Minimum confidence (0-100) for fuzzy match
AUTO_ADD_100_PERCENT = True  # When match is exactly 100%, append to hebrew_to_id_mapping_new.csv
FAILED_TEAMS = set()
_CACHE = {}
_HEBREW_MAP = {}      # hebrew_name -> team_id (from hebrew_to_id_mapping_new.csv)
_ID_CSV_ROWS = []     # [(team_name, team_id, country), ...]
_TRANSLATION_CACHE = {}
_TRANSLATION_DIRTY = False

_FIXTURES_BY_DATE = {}  # YYYY-MM-DD -> raw API response list (cached per process)

# Same default key as v76_Master_Nachshon.py — override with env FOOTBALL_API_KEY
_DEFAULT_FOOTBALL_API_KEY = "e49cdc2ba079c654d1dbc88fb16bfa75"

# Pair must match both club names vs translated Hebrew (WRatio)
_FIXTURE_PAIR_MIN = 72
_FIXTURE_PAIR_AVG = 80


def _clean_name(name):
    if not name:
        return ""
    s = str(name).strip()
    s = s.replace("–", "-").replace("—", "-").replace("'", "").replace('"', "")
    return re.sub(r"\s+", " ", s).strip()


def _normalize_hebrew(name):
    s = _clean_name(name)
    s = re.sub(r"[^\w\s\u0590-\u05FF-]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip().lower()


def _load_translation_cache():
    global _TRANSLATION_CACHE
    if _TRANSLATION_CACHE:
        return
    if os.path.exists(TRANSLATION_CACHE_PATH):
        try:
            with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
                _TRANSLATION_CACHE = json.load(f)
        except Exception:
            pass


def _load_hebrew_csv():
    global _HEBREW_MAP
    if _HEBREW_MAP:
        return
    if not os.path.exists(HEBREW_CSV_PATH):
        return
    try:
        import csv
        with open(HEBREW_CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                heb = _clean_name(row.get("Hebrew Name", ""))
                tid = row.get("Team ID", "")
                if heb and tid:
                    try:
                        _HEBREW_MAP[heb] = int(tid)
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        logging.warning(f"mapper: failed to load hebrew_to_id_mapping_new.csv: {e}")


def _load_id_csv():
    global _ID_CSV_ROWS
    if _ID_CSV_ROWS:
        return
    if not os.path.exists(ID_CSV_PATH):
        return
    try:
        import csv
        with open(ID_CSV_PATH, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = str(row.get("Team Name", "") or "").strip()
                tid = row.get("Team ID", "")
                country = str(row.get("Country", "") or "").strip()
                if name and tid:
                    try:
                        _ID_CSV_ROWS.append((name, int(tid), country))
                    except (TypeError, ValueError):
                        pass
    except Exception as e:
        logging.warning(f"mapper: failed to load ID.csv: {e}")


def football_api_key():
    return (
        os.environ.get("FOOTBALL_API_KEY", "").strip()
        or os.environ.get("APISPORTS_KEY", "").strip()
        or _DEFAULT_FOOTBALL_API_KEY
    )


def flush_translation_cache():
    """Persist translation_cache.json if new keys were added (call once at end of a batch run)."""
    global _TRANSLATION_DIRTY
    if not _TRANSLATION_DIRTY:
        return
    try:
        with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)
        _TRANSLATION_DIRTY = False
    except Exception as e:
        logging.warning(f"mapper: failed to write translation_cache.json: {e}")


def _translation_store(hebrew_clean, english):
    global _TRANSLATION_DIRTY
    if not hebrew_clean or not english:
        return
    _TRANSLATION_CACHE[hebrew_clean] = english.strip()
    _TRANSLATION_DIRTY = True


def _lingva_translate_he_to_en(text):
    q = urllib.parse.quote((text or "")[:500], safe="")
    if not q:
        return None
    url = f"https://lingva.ml/api/v1/iw/en/{q}"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        tr = (data.get("translation") or "").strip()
        return tr if tr else None
    except Exception:
        return None


def _mymemory_translate_he_to_en(text):
    q = urllib.parse.quote((text or "")[:400], safe="")
    if not q:
        return None
    url = f"https://api.mymemory.translated.net/get?q={q}&langpair=iw|en"
    try:
        r = requests.get(url, timeout=18)
        data = r.json()
        if data.get("responseStatus") != 200:
            return None
        tr = ((data.get("responseData") or {}).get("translatedText") or "").strip()
        if not tr or tr.lower() == (text or "").lower():
            return None
        return tr
    except Exception:
        return None


def translate_hebrew_team_name(clean_hebrew):
    """
    Hebrew -> English for matching against API-Football names. Uses translation_cache, then Lingva, then MyMemory.
    """
    if not clean_hebrew:
        return None
    _load_translation_cache()
    if clean_hebrew in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[clean_hebrew]
    if not re.search(r"[\u0590-\u05FF]", clean_hebrew):
        if clean_hebrew not in _TRANSLATION_CACHE:
            _translation_store(clean_hebrew, clean_hebrew)
        return clean_hebrew
    tr = _lingva_translate_he_to_en(clean_hebrew)
    time.sleep(0.02)
    if not tr:
        tr = _mymemory_translate_he_to_en(clean_hebrew)
        time.sleep(0.02)
    if tr:
        _translation_store(clean_hebrew, tr)
    return tr


def prefetch_fixtures_for_dates(dates_iterable, api_key=None):
    """One fixtures?date= request per distinct day (cached in-process)."""
    key = api_key or football_api_key()
    if not key:
        return
    for d in dates_iterable:
        if not d or len(str(d)) < 10:
            continue
        ds = str(d)[:10]
        if ds in _FIXTURES_BY_DATE:
            continue
        try:
            time.sleep(0.22)
            r = requests.get(
                "https://v3.football.api-sports.io/fixtures",
                headers={"x-apisports-key": key},
                params={"date": ds},
                timeout=45,
            )
            if r.status_code != 200:
                _FIXTURES_BY_DATE[ds] = []
                continue
            payload = r.json() or {}
            _FIXTURES_BY_DATE[ds] = payload.get("response") or []
        except Exception as e:
            logging.warning(f"mapper: fixtures fetch failed for {ds}: {e}")
            _FIXTURES_BY_DATE[ds] = []


def try_resolve_pair_via_fixtures(h_heb, a_heb, date_iso, api_key=None):
    """
    Match Winner Hebrew home/away to a single fixture on date_iso (YYYY-MM-DD).
    On success: appends to hebrew CSV, removes names from FAILED_TEAMS, returns (home_id, away_id) as in API.
    """
    if not date_iso or len(str(date_iso)) < 10:
        return None
    ds = str(date_iso)[:10]
    key = api_key or football_api_key()
    if not key:
        return None

    prefetch_fixtures_for_dates([ds], api_key=key)
    fixtures = _FIXTURES_BY_DATE.get(ds) or []
    if not fixtures:
        return None

    h_clean = _clean_name(h_heb)
    a_clean = _clean_name(a_heb)
    tr_h = translate_hebrew_team_name(h_clean)
    tr_a = translate_hebrew_team_name(a_clean)
    if not tr_h or not tr_a:
        return None

    tr_h = tr_h.lower()
    tr_a = tr_a.lower()

    best = None  # (avg, minv, id_for_winner_h, id_for_winner_a, en_h, en_a, country)
    for fx in fixtures:
        try:
            teams = fx.get("teams") or {}
            th = teams.get("home") or {}
            ta = teams.get("away") or {}
            hn = str(th.get("name") or "")
            an = str(ta.get("name") or "")
            api_home_id, api_away_id = th.get("id"), ta.get("id")
            if not hn or not an or not api_home_id or not api_away_id:
                continue
            hn_l, an_l = hn.lower(), an.lower()

            rh0 = fuzz.WRatio(tr_h, hn_l) if HAS_RAPIDFUZZ else _fuzzy_ratio(tr_h, hn_l)
            ra0 = fuzz.WRatio(tr_a, an_l) if HAS_RAPIDFUZZ else _fuzzy_ratio(tr_a, an_l)
            avg0 = (rh0 + ra0) / 2.0
            m0 = min(rh0, ra0)

            rh1 = fuzz.WRatio(tr_h, an_l) if HAS_RAPIDFUZZ else _fuzzy_ratio(tr_h, an_l)
            ra1 = fuzz.WRatio(tr_a, hn_l) if HAS_RAPIDFUZZ else _fuzzy_ratio(tr_a, hn_l)
            avg1 = (rh1 + ra1) / 2.0
            m1 = min(rh1, ra1)

            # Map Winner home/away (Hebrew order) to API team IDs + English names for that slot.
            if avg0 >= avg1:
                avg, minv = avg0, m0
                id_wh, en_wh = int(api_home_id), hn
                id_wa, en_wa = int(api_away_id), an
            else:
                avg, minv = avg1, m1
                id_wh, en_wh = int(api_away_id), an
                id_wa, en_wa = int(api_home_id), hn

        except Exception:
            continue

        if minv < _FIXTURE_PAIR_MIN or avg < _FIXTURE_PAIR_AVG:
            continue
        if best is None or avg > best[0] or (avg == best[0] and minv > best[1]):
            country = ""
            try:
                country = str((fx.get("league") or {}).get("country") or "")
            except Exception:
                pass
            best = (avg, minv, id_wh, id_wa, en_wh, en_wa, country)

    if not best:
        return None

    _, _, hid, aid, h_en, a_en, country = best
    FAILED_TEAMS.discard(h_clean)
    FAILED_TEAMS.discard(a_clean)
    if AUTO_ADD_100_PERCENT:
        _append_to_hebrew_csv(h_clean, hid, h_en, country)
        _append_to_hebrew_csv(a_clean, aid, a_en, country)
    return hid, aid


def _get_team_info_from_id(team_id):
    """Get (english_name, country) from ID.csv for a given Team ID."""
    for eng_name, tid, country in _ID_CSV_ROWS:
        if tid == team_id:
            return eng_name, country
    return "", ""


def _append_to_hebrew_csv(hebrew_name, team_id, matched_english_name="", country=""):
    """
    Append a new mapping to hebrew_to_id_mapping_new.csv (only when AUTO_ADD_100_PERCENT).
    Used to grow the mapper when we get a 100% confidence match from fuzzy/ID.csv.
    """
    if not AUTO_ADD_100_PERCENT:
        return
    clean = _clean_name(hebrew_name)
    if not clean or not team_id:
        return
    if clean in _HEBREW_MAP:
        return  # Already in map, no need to append
    try:
        import csv
        translated = _TRANSLATION_CACHE.get(clean, "")
        with open(HEBREW_CSV_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([clean, translated, matched_english_name, 100, team_id, country])
        _HEBREW_MAP[clean] = int(team_id)
        logging.info(f"mapper: Auto-added '{clean}' -> Team ID {team_id} to hebrew_to_id_mapping_new.csv")
    except Exception as e:
        logging.warning(f"mapper: Failed to append to hebrew_to_id_mapping_new.csv: {e}")


def _fuzzy_ratio(a, b):
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b)
    return SequenceMatcher(None, a, b).ratio() * 100


def _fuzzy_match_hebrew(hebrew_name, source_rows):
    """
    source_rows: list of (hebrew_or_english_name, team_id) tuples.
    Returns (team_id, confidence) or (None, 0) if below threshold.
    """
    clean_heb = _clean_name(hebrew_name)
    norm_heb = _normalize_hebrew(hebrew_name)
    if not clean_heb:
        return None, 0

    best_id = None
    best_score = 0.0

    for name, tid in source_rows:
        if not name or tid is None:
            continue
        # Compare Hebrew to Hebrew
        norm_cand = _normalize_hebrew(name) if re.search(r"[\u0590-\u05FF]", str(name)) else ""
        if norm_cand:
            score = _fuzzy_ratio(norm_heb, norm_cand)
        else:
            # Hebrew vs Latin: use translation cache if available
            trans = _TRANSLATION_CACHE.get(clean_heb, "")
            if trans:
                score = _fuzzy_ratio(trans.lower(), str(name).lower())
            else:
                score = _fuzzy_ratio(clean_heb, str(name))
        if score > best_score:
            best_score = score
            best_id = tid

    if best_score >= FUZZY_THRESHOLD and best_id is not None:
        return best_id, best_score
    return None, best_score


def get_team_id(hebrew_name):
    """
    Resolve Hebrew team name to API-Sports Team ID.
    1. Check translation_cache.json (keys are Hebrew) -> not directly; cache maps Hebrew->English.
    2. Check hebrew_to_id_mapping_new.csv (column 'Hebrew Name')
    3. Fuzzy match in hebrew_to_id_mapping_new.csv or ID.csv (85% threshold)
    4. If below 85%: return None, log warning, skip stats for this team.
    """
    if not hebrew_name:
        return None

    clean = _clean_name(hebrew_name)
    if not clean:
        return None

    _load_translation_cache()
    _load_hebrew_csv()
    _load_id_csv()

    # Exact CSV hit first so dynamically added mappings (e.g. fixture resolver) always win over FAILED_TEAMS.
    if clean in _HEBREW_MAP:
        return _HEBREW_MAP[clean]

    if clean in FAILED_TEAMS:
        return None

    # Normalized Hebrew match
    norm = _normalize_hebrew(hebrew_name)
    for heb, tid in _HEBREW_MAP.items():
        if _normalize_hebrew(heb) == norm:
            if clean != heb and AUTO_ADD_100_PERCENT:
                eng_name, country = _get_team_info_from_id(tid)
                _append_to_hebrew_csv(clean, tid, eng_name, country)
            return tid

    # 3. Fuzzy match in hebrew_to_id_mapping_new.csv
    heb_rows = [(h, tid) for h, tid in _HEBREW_MAP.items()]
    fid, best_score = _fuzzy_match_hebrew(hebrew_name, heb_rows)
    if fid is not None:
        if best_score >= 99.99 and AUTO_ADD_100_PERCENT:
            eng_name, country = _get_team_info_from_id(fid)
            _append_to_hebrew_csv(clean, fid, eng_name, country)
        return fid

    # 3b. Fuzzy match in ID.csv (by English name; use translation cache to bridge)
    eng_from_cache = _TRANSLATION_CACHE.get(clean, "")
    if eng_from_cache:
        best_id, eng_score = None, 0.0
        best_eng_name, best_country = "", ""
        for name, tid, country in _ID_CSV_ROWS:
            s = _fuzzy_ratio(eng_from_cache.lower(), str(name).lower())
            if s > eng_score:
                eng_score = s
                best_id = tid
                best_eng_name = str(name)
                best_country = str(country)
        if eng_score >= FUZZY_THRESHOLD and best_id is not None:
            if eng_score >= 99.99 and AUTO_ADD_100_PERCENT:
                _append_to_hebrew_csv(clean, best_id, best_eng_name, best_country)
            return best_id
        best_score = max(best_score, eng_score)

    # 4. Below threshold
    logging.warning(f"mapper: No confident match for '{hebrew_name}' (best score {best_score:.1f}% < {FUZZY_THRESHOLD}%). Assigning UNKNOWN, skipping stats.")
    FAILED_TEAMS.add(clean)
    return None


def init_mapper():
    """Preload all data sources."""
    _load_translation_cache()
    _load_hebrew_csv()
    _load_id_csv()

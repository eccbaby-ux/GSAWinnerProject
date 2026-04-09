import json
import os
import re
from datetime import datetime
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHES_FILE = os.path.join(BASE_DIR, "matches.txt")
ODDS_CACHE_FILE = os.path.join(BASE_DIR, "winner_odds_cache.json")
ODDS_PREVIOUS_FILE = os.path.join(BASE_DIR, "winner_odds_previous.json")

def get_winner_data():
    print("🚀 מתחיל בשאיבת נתונים אוטומטית מאתר הווינר...")
    # שמירת היחסים הקודמים לפני עדכון (להצגת שינויים בדשבורד)
    if os.path.exists(ODDS_CACHE_FILE):
        try:
            import shutil
            shutil.copy2(ODDS_CACHE_FILE, ODDS_PREVIOUS_FILE)
            print(f"   📋 יחסים קודמים נשמרו ל-{os.path.basename(ODDS_PREVIOUS_FILE)}")
        except Exception as e:
            print(f"   ⚠️ לא הצלחתי לשמור יחסים קודמים: {e}")
    extracted_matches = {}
    
    _debug_responses = []

    def handle_response(response):
        if response.request.resource_type in ["fetch", "xhr", "document"]:
            try:
                data = response.json()
                _debug_responses.append(f"[JSON] {response.url[:120]}")
                _parse_and_store(data, extracted_matches)
            except:
                _debug_responses.append(f"[non-JSON] {response.url[:120]}")

    def _parse_and_store(data_node, matches_dict):
        # מנגנון ניקוי – מוגדר ראשון כי משמש בסריקות
        def clean_deep(t):
            t = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', str(t))
            return t.replace("'", "").replace('"', "").replace("–", "-").strip()

        # סריקה ראשונה: איתור תאריכים ושעות ברמת אירוע (לפני עיבוד המארקטים)
        date_time_collector = {}
        DATE_KEYS = ("date", "startDate", "eventDate", "start", "dt", "dateTime", "openDate", "e_date")
        TIME_KEYS = ("time", "startTime", "kickoffTime", "t", "m_hour")

        def _parse_winner_e_date(val):
            """e_date: YYMMDD (int) -> YYYY-MM-DD. למשל 260321 -> 2026-03-21"""
            if val is None:
                return ""
            s = str(val).strip()
            if len(s) == 6 and s.isdigit():
                yy, mm, dd = s[0:2], s[2:4], s[4:6]
                yy = "20" + yy if int(yy) < 50 else "19" + yy
                return f"{yy}-{mm}-{dd}"
            return ""

        def _parse_winner_m_hour(val):
            """m_hour: HHMM (str) -> HH:MM. למשל '1714' -> '17:14'"""
            if val is None:
                return ""
            s = str(val).strip()
            if len(s) >= 4 and s[:4].isdigit():
                return f"{s[0:2]}:{s[2:4]}"
            return ""

        def _extract_date_time_from_node(n):
            """מחלץ תאריך ושעה מאובייקט (או מילדים ישירים). תומך ב-e_date (YYMMDD) ו-m_hour (HHMM) של ווינר."""
            out_date, out_time = "", ""
            if not isinstance(n, dict):
                return out_date, out_time
            for k, v in n.items():
                if k == "e_date" and v is not None:
                    parsed = _parse_winner_e_date(v)
                    if parsed:
                        out_date = parsed
                elif k == "m_hour" and v is not None:
                    parsed = _parse_winner_m_hour(v)
                    if parsed:
                        out_time = parsed
                elif k in DATE_KEYS and isinstance(v, str) and v.strip():
                    out_date = v.strip()
                elif k in TIME_KEYS and isinstance(v, str) and v.strip():
                    out_time = v.strip()
                elif k in ("dateTime", "startDateTime", "datetime") and isinstance(v, str) and "T" in v:
                    iso = v.strip()
                    if re.match(r"\d{4}-\d{2}-\d{2}T\d", iso):
                        out_date = out_date or iso[:10]
                        out_time = out_time or (iso[11:16] if len(iso) >= 16 else "")
            if out_date or out_time:
                return out_date, out_time
            for v in n.values():
                if isinstance(v, dict):
                    d, t = _extract_date_time_from_node(v)
                    if d or t:
                        return d or out_date, t or out_time
            return out_date, out_time

        def _extract_desc_from_node(n):
            """מחלץ מחרוזת משחק (קבוצה א - קבוצה ב) מאובייקט."""
            for k in ("desc", "d", "name", "n", "description", "title"):
                v = n.get(k) if isinstance(n, dict) else None
                if isinstance(v, str) and "-" in v and len(v) > 5:
                    parts = v.split("-", 1)
                    if len(parts) == 2 and len(parts[0].strip()) > 1 and len(parts[1].strip()) > 1:
                        return clean_deep(v)
            return None

        def _scan_event_datetime(node, path="", parent_date="", parent_time=""):
            if isinstance(node, dict):
                desc_val = _extract_desc_from_node(node)
                date_val = parent_date
                time_val = parent_time
                for k, v in node.items():
                    if k == "e_date" and v is not None:
                        pd = _parse_winner_e_date(v)
                        if pd:
                            date_val = pd
                    elif k == "m_hour" and v is not None:
                        pt = _parse_winner_m_hour(v)
                        if pt:
                            time_val = pt
                    elif k in DATE_KEYS and isinstance(v, str) and v.strip():
                        date_val = v.strip()
                    elif k in TIME_KEYS and isinstance(v, str) and v.strip():
                        time_val = v.strip()
                    elif k in ("dateTime", "startDateTime", "datetime") and isinstance(v, str) and "T" in v:
                        iso = v.strip()
                        if re.match(r"\d{4}-\d{2}-\d{2}T\d", iso):
                            date_val = date_val or iso[:10]
                            time_val = time_val or (iso[11:16] if len(iso) >= 16 else "")
                if desc_val and (date_val or time_val) and desc_val not in date_time_collector:
                    date_time_collector[desc_val] = {"date": date_val or "", "time": time_val or ""}
                for k, v in node.items():
                    _scan_event_datetime(v, f"{path}.{k}", date_val, time_val)
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    _scan_event_datetime(item, f"{path}[{i}]", parent_date, parent_time)

        _scan_event_datetime(data_node)

        # Only keep matches with valid numerical Draw (X) odd - we want 3-way football matches only
        def _has_valid_x_odd(entry):
            x_val = entry.get("X") if isinstance(entry, dict) else None
            if x_val is None:
                return False
            if isinstance(x_val, str) and (not x_val or not x_val.strip()):
                return False
            try:
                return float(x_val) > 1.0
            except (TypeError, ValueError):
                return False

        def _get_or_create_match(desc_key):
            """
            יוצר/מחזיר מבנה נתונים מורחב למשחק מסוים.
            שומר תאימות אחורה: שדות '1','X','2' נשארים ברמה העליונה.
            """
            existing = matches_dict.get(desc_key)
            if isinstance(existing, dict):
                # ודא שכל השדות החדשים קיימים
                existing.setdefault("odds_1x2", {})
                existing.setdefault("totals", {})
                existing.setdefault("btts", {})
                existing.setdefault("correct_score", {})
                existing.setdefault("ht_ft", {})
                existing.setdefault("double_chance", {})
                existing.setdefault("team_totals", {})
                return existing

            match_entry = {
                # תאימות אחורה – יחסי 1X2 ברמה העליונה
                "1": None,
                "X": None,
                "2": None,
                # שדות מורחבים לפי הגדרת גלעד
                "odds_1x2": {},
                "totals": {},
                "btts": {},
                "correct_score": {},
                "ht_ft": {},
                "double_chance": {},
                "team_totals": {},
            }
            matches_dict[desc_key] = match_entry
            return match_entry
        
        found = []
        def _scan(node, parent=None):
            if isinstance(node, dict):
                d_key = next((k for k in node.keys() if k in ["desc", "d", "name", "n", "description"]), None)
                o_key = next((k for k in node.keys() if k in ["outcomes", "o", "markets", "selections", "options"]), None)
                
                if d_key and o_key and isinstance(node.get(o_key), list):
                    found.append((node, parent, d_key, o_key))
                for v in node.values(): 
                    if isinstance(v, (dict, list)): _scan(v, node)
            elif isinstance(node, list):
                for i in node: 
                    if isinstance(i, (dict, list)): _scan(i, parent)
                
        _scan(data_node)

        def _extract_datetime_from_obj(obj):
            """מחלץ תאריך ושעה מאובייקט (או None). תומך ב-e_date (YYMMDD int) ו-m_hour (HHMM str) של ווינר."""
            if not isinstance(obj, dict):
                return None, None
            raw_d, raw_t = "", ""
            for k, v in obj.items():
                if k == "e_date" and v is not None:
                    raw_d = raw_d or _parse_winner_e_date(v)
                elif k == "m_hour" and v is not None:
                    raw_t = raw_t or _parse_winner_m_hour(v)
                elif k in DATE_KEYS and isinstance(v, str) and v.strip():
                    raw_d = raw_d or v.strip()
                elif k in TIME_KEYS and isinstance(v, str) and v.strip():
                    raw_t = raw_t or v.strip()
                elif k in ("dateTime", "startDateTime", "datetime") and isinstance(v, str) and "T" in v and re.match(r"\d{4}-\d{2}-\d{2}T\d", v.strip()):
                    raw_d = raw_d or v.strip()[:10]
                    raw_t = raw_t or (v.strip()[11:16] if len(v.strip()) >= 16 else "")
            return (raw_d, raw_t) if (raw_d or raw_t) else (None, None)

        def _apply_winner_datetime(match_entry, raw_date, raw_time):
            """מעדכן winner_date ו-winner_time במשחק."""
            if not raw_date and not raw_time:
                return
            if raw_date:
                m = re.search(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?", str(raw_date))
                if m:
                    d, mo, yr = m.group(1), m.group(2), m.group(3) or str(datetime.now().year)
                    if len(yr) == 2:
                        yr = "20" + yr
                    match_entry["winner_date"] = f"{yr}-{mo.zfill(2)}-{d.zfill(2)}"
                elif re.match(r"^\d{4}-\d{2}-\d{2}", str(raw_date)):
                    match_entry["winner_date"] = str(raw_date)[:10]
                else:
                    match_entry["winner_date"] = str(raw_date)
            if raw_time:
                m_t = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?", str(raw_time))
                match_entry["winner_time"] = f"{int(m_t.group(1)):02d}:{m_t.group(2)}" if m_t else str(raw_time)
            elif "winner_date" in match_entry or "winner_time" in match_entry:
                match_entry["winner_time"] = match_entry.get("winner_time", "")

        for ev, parent, d_key, o_key in found:
            desc = clean_deep(ev.get(d_key, ""))
            market = clean_deep(str(ev.get("mp", ev.get("m", ev.get("marketName", "")))))

            # מתעניינים רק במשחקי כדורגל בפורמט "קבוצה א - קבוצה ב"
            if "-" not in desc:
                continue

            # רק תוצאת סיום (Full-Time) – לא מחצית, לא הארכה
            is_ht_or_other = any(x in market for x in ["מחצית", "הארכה", "פנדלים", "Half", "HT", "Extra"])
            is_1x2_market = (
                any(x in market for x in ["תוצאת סיום", "תוצאה סופית", "1X2", "2X1", "תוצאת משחק"])
                and not is_ht_or_other
            )
            # For non-1X2 markets: only process if match already exists with valid Draw (X) odd
            if not is_1x2_market:
                match_entry = matches_dict.get(desc)
                if match_entry is None or not _has_valid_x_odd(match_entry):
                    continue  # Ignore match - no valid X odd, we only want 3-way football matches

            match_entry = _get_or_create_match(desc)

            # חילוץ תאריך ושעה ישירות מאובייקט האירוע ו/או מההורה (מבנה הווינר משתנה)
            raw_date, raw_time = _extract_datetime_from_obj(ev)
            if (not raw_date and not raw_time) and parent:
                raw_date, raw_time = _extract_datetime_from_obj(parent)
            # סריקה עמוקה באירוע עצמו – תאריך עשוי להיות בילדים מקוננים
            if not raw_date and not raw_time:
                raw_date, raw_time = _extract_date_time_from_node(ev)
            # סריקה ברמה עמוקה יותר – תאריך עשוי להימצא בתוך ילדים (למשל בסביבת event)
            if (not raw_date and not raw_time) and parent and isinstance(parent, dict):
                for v in parent.values():
                    if isinstance(v, dict):
                        raw_date, raw_time = _extract_date_time_from_node(v)
                        if raw_date or raw_time:
                            break
            if raw_date or raw_time:
                _apply_winner_datetime(match_entry, raw_date, raw_time)

            parts = desc.split("-")
            h_hint = parts[0].strip() if len(parts) > 1 else "---"
            a_hint = parts[-1].strip() if len(parts) > 1 else "---"

            # איסוף כל האאוטקאם של המארקט הנוכחי
            outcomes = []
            for o in ev.get(o_key, []):
                o_d_key = next((k for k in o.keys() if k in ["desc", "d", "name", "n", "title"]), None)
                p_key = next((k for k in o.keys() if k in ["price", "p", "odds", "odd"]), None)
                if not o_d_key or not p_key:
                    continue
                try:
                    d = clean_deep(o.get(o_d_key, ""))
                    p_val = float(o.get(p_key, 0))
                except Exception:
                    continue
                if p_val <= 1.0:
                    continue
                outcomes.append((d, p_val))

            if not outcomes:
                continue

            # --- 1X2 (תוצאת סיום) – לוגיקה כפי שהייתה, עם ולידציה ---
            if is_1x2_market:
                odds_1x2 = {}
                for d, p_val in outcomes:
                    if d == "1" or h_hint in d:
                        odds_1x2["1"] = p_val
                    elif d == "X" or "תיקו" in d:
                        odds_1x2["X"] = p_val
                    elif d == "2" or a_hint in d:
                        odds_1x2["2"] = p_val

                if len(odds_1x2) == 3 and "X" in odds_1x2:
                    o1, ox, o2 = odds_1x2["1"], odds_1x2["X"], odds_1x2["2"]
                    # ולידציה: תיקו בד"כ 2.0–5.5, overround סביר 1.02–1.28 (ווינר עד ~28%)
                    overround = (1.0/o1 + 1.0/ox + 1.0/o2) if all(x > 0 for x in [o1,ox,o2]) else 999
                    x_valid = 2.0 <= ox <= 5.5  # תיקו בכדורגל בד"כ בטווח זה
                    overround_ok = 1.02 <= overround <= 1.28  # overround סביר (ווינר עד ~28%)
                    if x_valid and overround_ok:
                        match_entry["1"] = odds_1x2["1"]
                        match_entry["X"] = odds_1x2["X"]
                        match_entry["2"] = odds_1x2["2"]
                        match_entry["odds_1x2"] = {
                            "1": odds_1x2["1"],
                            "X": odds_1x2["X"],
                            "2": odds_1x2["2"],
                        }

            # --- שווקי שערים (Over / Under) ---
            is_goals_market = any(x in market for x in ["שערים", "גולים", "Over", "Under"])
            if is_goals_market:
                for d, p_val in outcomes:
                    text = d
                    # דוגמאות: "מעל 1.5 שערים", "Over 2.5", "מתחת 3.5"
                    m_over = re.search(r"(?:מעל|Over)\s*([0-9]+(?:\.[0-9]+)?)", text)
                    m_under = re.search(r"(?:מתחת|Under)\s*([0-9]+(?:\.[0-9]+)?)", text)
                    if m_over:
                        num = m_over.group(1).replace(".", "_")
                        key = f"over_{num}"
                        match_entry["totals"][key] = p_val
                    elif m_under:
                        num = m_under.group(1).replace(".", "_")
                        key = f"under_{num}"
                        match_entry["totals"][key] = p_val

            # --- שתי הקבוצות יבקיעו (BTTS) ---
            is_btts_market = any(x in market for x in ["שתי הקבוצות יבקיעו", "שתי הקבוצות מבקיעות", "Both Teams To Score", "BTTS"])
            if is_btts_market:
                for d, p_val in outcomes:
                    txt = d.lower()
                    if any(x in txt for x in ["yes", "כן"]):
                        match_entry["btts"]["yes"] = p_val
                    elif any(x in txt for x in ["no", "לא"]):
                        match_entry["btts"]["no"] = p_val

            # --- תוצאה מדויקת ---
            is_cs_market = "תוצאה מדויקת" in market or "Correct Score" in market
            if is_cs_market:
                for d, p_val in outcomes:
                    # דפוס כללי: 2-1, 0-0 וכו'
                    if re.match(r"^\d+\s*[-:]\s*\d+$", d):
                        key = re.sub(r"\s*", "", d.replace(":", "-"))
                        match_entry["correct_score"][key] = p_val

            # --- מחצית / סיום (HT/FT) ---
            is_htft_market = any(x in market for x in ["מחצית/סיום", "Half Time/Full Time", "HT/FT"])
            if is_htft_market:
                for d, p_val in outcomes:
                    # דפוס: 1/1, X/1, 1/2 וכו'
                    if re.match(r"^[12X]\s*/\s*[12X]$", d):
                        key = d.replace(" ", "")
                        match_entry["ht_ft"][key] = p_val

            # --- דאבל צ'אנס (Double Chance) ---
            is_double_chance_market = any(x in market for x in ["דאבל", "Double Chance"])
            if is_double_chance_market:
                for d, p_val in outcomes:
                    # דפוס: 1X, X2, 12
                    if re.match(r"^[12X]{2}$", d):
                        key = d.replace(" ", "")
                        match_entry["double_chance"][key] = p_val

        def _normalize_match_key(s):
            """נרמול להשוואה: רווחים, מקפים."""
            if not s:
                return ""
            return re.sub(r"\s+", " ", clean_deep(s))

        # מיזוג תאריכים ושעות מהווינר – גיבוי כש-API-Football לא מחזיר (כולל התאמה מרוככת)
        for match_key, match_entry in matches_dict.items():
            if not isinstance(match_entry, dict):
                continue
            for dt_key, dt_val in date_time_collector.items():
                keys_match = match_key == dt_key or _normalize_match_key(match_key) == _normalize_match_key(dt_key)
                if keys_match:
                    raw_date = (dt_val.get("date") or "").strip()
                    raw_time = (dt_val.get("time") or "").strip()
                    # ניקוי תאריך מטקסט כמו "יום ו' 20.03" – לחלץ 20.03
                    m = re.search(r"(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?", raw_date)
                    if m:
                        d, mo, yr = m.group(1), m.group(2), m.group(3) or str(datetime.now().year)
                        if len(yr) == 2:
                            yr = "20" + yr
                        match_entry["winner_date"] = f"{yr}-{mo.zfill(2)}-{d.zfill(2)}"
                    elif re.match(r"^\d{4}-\d{2}-\d{2}", raw_date):
                        match_entry["winner_date"] = raw_date[:10]
                    else:
                        match_entry["winner_date"] = raw_date
                    # שעה – פורמט HH:MM או HH:MM:SS
                    if raw_time:
                        m_t = re.search(r"(\d{1,2}):(\d{2})(?::\d{2})?", raw_time)
                        if m_t:
                            match_entry["winner_time"] = f"{int(m_t.group(1)):02d}:{m_t.group(2)}"
                        else:
                            match_entry["winner_time"] = raw_time
                    else:
                        match_entry["winner_time"] = ""
                    break

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ]
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="he-IL",
            timezone_id="Asia/Jerusalem",
        )
        # הסתרת סימני אוטומציה לפני טעינת הדף
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3]});
            Object.defineProperty(navigator, 'languages', {get: () => ['he-IL','he','en-US']});
        """)
        page = context.new_page()
        page.on("response", handle_response)

        try:
            print("   ⏳ פותח את אתר הווינר, מאזין לרשת וגולל...")
            page.goto("https://www.winner.co.il/", wait_until="domcontentloaded", timeout=40000)
            page.wait_for_timeout(3000)  # המתנה לטעינת API ראשוני

            # גלילה מרובה – האתר טוען ליגות/משחקים בהדרגה (lazy loading)
            for i in range(10):
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(2500)
            page.wait_for_timeout(1500)  # המתנה סופית לתגובות רשת

        except Exception:
            pass  # התעלמות שקטה מ-Timeout, המידע כבר נאסף
            
        browser.close()

    # Only keep 3-way football matches with a valid Draw (X) odd
    def _has_valid_x_odd(match_entry):
        x_val = match_entry.get("X")
        if x_val is None:
            return False
        if isinstance(x_val, str) and (not x_val or not x_val.strip()):
            return False
        try:
            num = float(x_val)
            return num > 1.0
        except (TypeError, ValueError):
            return False

    extracted_matches = {k: v for k, v in extracted_matches.items() if _has_valid_x_odd(v)}

    if not extracted_matches:
        print(f"   [DEBUG] סה\"כ responses שהתקבלו: {len(_debug_responses)}")
        for r in _debug_responses[:30]:
            print(f"   {r}")
        print("❌ לא נמצאו משחקים.")
        return

    print(f"\n✅ משימה הושלמה! נשאבו {len(extracted_matches)} משחקי כדורגל חוקיים.")
    
    os.makedirs(os.path.dirname(MATCHES_FILE), exist_ok=True)
    with open(MATCHES_FILE, "w", encoding="utf-8") as f:
        for match_str in extracted_matches.keys():
            f.write(f"{match_str}\n")
    print(f"   💾 קובץ {MATCHES_FILE} עודכן בהצלחה.")
            
    os.makedirs(os.path.dirname(ODDS_CACHE_FILE), exist_ok=True)
    with open(ODDS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_matches, f, ensure_ascii=False, indent=4)
    print(f"   💾 קובץ יחסים {ODDS_CACHE_FILE} הופק לשימוש עתידי.")

if __name__ == "__main__":
    get_winner_data()
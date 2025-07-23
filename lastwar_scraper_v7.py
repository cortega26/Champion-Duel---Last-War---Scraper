
"""
lastwar_scraper_v7.py
---------------------
Automate grabbing the Top 10 players for Groups A–P from "Last War: Survival" (Windows client),
OCR the text, and save everything to an Excel file.

Usage:
  python lastwar_scraper_v7.py --calibrate
  python lastwar_scraper_v7.py --run leaderboard.xlsx --debug
"""

import os
import sys
import time
import json
import re
import signal
from typing import Dict, List, Tuple, Optional

import mss
import numpy as np
import cv2
from PIL import Image
import pytesseract
from pytesseract import Output
import pyautogui as pag
import keyboard
import pandas as pd

try:
    import pygetwindow as gw
except Exception:
    gw = None

# --------- FAILSAFE ---------
STOP = False


def _stop_handler(*_):  # Ctrl+C / Ctrl+Break
    global STOP
    STOP = True


signal.signal(signal.SIGINT, _stop_handler)
try:
    signal.signal(signal.SIGBREAK, _stop_handler)
except Exception:
    pass


def check_abort():
    if STOP or keyboard.is_pressed("esc"):
        print("Abort requested. Exiting…")
        sys.exit(0)


# --------- CONFIG ----------
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "window_title_regex": "Last War",
    "tesseract_exe": "",
    # enable Asian languages by default; ensure you have the
    # corresponding Tesseract language data installed
    "lang": "eng+jpn+kor+chi_sim",
    "groups": [chr(ord('A')+i) for i in range(16)],  # A..P
    "sleep_after_click": 0.35,
    "sleep_after_group_change": 0.9,
    "extra_wait_after_group": 0.5,
    "sleep_after_scroll": 0.45,
    "scroll_method": "drag",         # "drag" or "wheel"
    "max_scroll_attempts_top": 3,
    "max_scroll_attempts_bottom": 1,
    "expected_total_rows": 10,
    "ocr_psm": 6,
    "threshold": 180,
    "debug": False,
    "ui": {
        "group_button": [0, 0],
        "dropdown_tl": [0, 0],
        "dropdown_br": [0, 0],
        "board_tl":    [0, 0],
        "board_br":    [0, 0],
        # wheel
        "scroll_point": [0, 0],
        "scroll_top_ticks": 6,
        "scroll_bottom_ticks": -3,
        # drag
        "drag_from": [0, 0],
        "drag_to":   [0, 0],
    }
}

# --------- UTILS ----------


def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        def merge(a, b):
            for k, v in b.items():
                if k not in a:
                    a[k] = v
                elif isinstance(v, dict):
                    merge(a[k], v)
        merge(cfg, DEFAULT_CONFIG)
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def bring_window_to_front(rx: str):
    if gw is None:
        return
    titles = [t for t in gw.getAllTitles() if re.search(rx, t, re.I)]
    if not titles:
        print("[WARN] Window not found.")
        return
    w = gw.getWindowsWithTitle(titles[0])[0]
    try:
        w.activate()
    except Exception:
        w.minimize()
        time.sleep(0.4)
        w.restore()
    time.sleep(0.4)


def grab(region: Tuple[int, int, int, int]) -> Image.Image:
    with mss.mss() as sct:
        s = sct.grab(
            {"left": region[0], "top": region[1], "width": region[2], "height": region[3]})
    return Image.frombytes("RGB", s.size, s.rgb)


def preprocess(img: Image.Image, thr: int) -> Image.Image:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    # small morphological closing helps connect broken glyphs
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(th)


def ocr_tsv(img: Image.Image, psm: int, lang: str, exe: str) -> pd.DataFrame:
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    df = pytesseract.image_to_data(
        img,
        config=f"--oem 3 --psm {psm}",
        lang=lang,
        output_type=Output.DATAFRAME,
    )
    df = df[df.conf != -1].fillna("")
    return df


def click(xy, sleep=0.2):
    pag.moveTo(*xy)
    pag.click()
    time.sleep(sleep)


def scroll_wheel(pt, ticks, sleep):
    pag.moveTo(*pt)
    pag.scroll(ticks)
    time.sleep(sleep)


def scroll_drag(fr, to, reps, sleep):
    for _ in range(reps):
        check_abort()
        pag.moveTo(*fr)
        pag.dragTo(to[0], to[1], duration=0.25)
        time.sleep(sleep)


def wait_point(label: str, key: str) -> Tuple[int, int]:
    print(f"Hover {label} → press {key.upper()}  (ESC abort)")
    while True:
        check_abort()
        if keyboard.is_pressed(key):
            p = pag.position()
            print(f"{label}: {p}")
            time.sleep(0.25)
            return p
        time.sleep(0.05)

# --------- CALIBRATION ----------


def calibrate(cfg: Dict):
    bring_window_to_front(cfg["window_title_regex"])
    ui = cfg["ui"]

    print("=== Calibration ===")
    ui["group_button"] = list(wait_point("blue Group arrow", "f1"))

    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    ui["dropdown_tl"] = list(wait_point("dropdown TOP-LEFT", "f2"))
    ui["dropdown_br"] = list(wait_point("dropdown BOTTOM-RIGHT", "f3"))

    ui["board_tl"] = list(wait_point(
        "Leaderboard TOP-LEFT (inside white panel)", "f4"))
    ui["board_br"] = list(wait_point(
        "Leaderboard BOTTOM-RIGHT (inside white panel)", "f5"))

    if cfg["scroll_method"] == "wheel":
        ui["scroll_point"] = list(wait_point(
            "point where wheel scrolls", "f6"))
    else:
        ui["drag_from"] = list(wait_point("drag START (inside list)", "f6"))
        ui["drag_to"] = list(wait_point("drag END (pull upward)", "f7"))

    save_config(cfg)
    print("Saved config.json.\nRun:\n  python lastwar_scraper_v7.py --run --debug")

# --------- DROPDOWN OCR CLICK ----------


def ocr_click_group(cfg: Dict, letter: str):
    ui = cfg["ui"]
    tlx, tly = ui["dropdown_tl"]
    brx, bry = ui["dropdown_br"]
    img = grab((tlx, tly, brx - tlx, bry - tly))
    pre = preprocess(img, cfg["threshold"])
    df = ocr_tsv(pre, 6, cfg["lang"], cfg["tesseract_exe"])

    lines = {}
    for _, r in df.iterrows():
        key = (r.block_num, r.par_num, r.line_num)
        lines.setdefault(key, []).append(r)

    patt = rf"Group\s*{letter}\b"
    alt = {"I": "[I1]", "O": "[O0]"}
    if letter in alt:
        patt = rf"Group\s*{alt[letter]}\b"

    target_xy = None
    for _, words in lines.items():
        ws = sorted(words, key=lambda r: r.left)
        text = " ".join(w.text for w in ws).strip()
        if re.search(patt, text, re.I):
            xs = [w.left for w in ws]
            ys = [w.top for w in ws]
            ws_ = [w.width for w in ws]
            hs_ = [w.height for w in ws]
            x1, y1 = min(xs), min(ys)
            x2 = max(x+w for x, w in zip(xs, ws_))
            y2 = max(y+h for y, h in zip(ys, hs_))
            target_xy = (tlx + (x1+x2)//2, tly + (y1+y2)//2)
            break

    if target_xy is None:
        print(f"[WARN] Couldn't OCR 'Group {letter}'. Fallback math.")
        total_h = (ui["dropdown_br"][1] - ui["dropdown_tl"][1])
        lh = total_h / len(cfg["groups"])
        gx = ui["dropdown_tl"][0] + 40
        gy = int(ui["dropdown_tl"][1] + lh *
                 cfg["groups"].index(letter) + lh/2)
        target_xy = (gx, gy)

    click(target_xy, cfg["sleep_after_group_change"])
    time.sleep(cfg["extra_wait_after_group"])  # dropdown auto-closes

# --------- SCROLL ENSURE (probe bands) ----------


def _grab_probe(cfg: Dict, rel_y: float, band_h: int = 120) -> np.ndarray:
    """Grab a horizontal band at rel_y (0..1) height inside the board."""
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    w = brx - tlx
    h = bry - tly
    y0 = tly + int(h * rel_y)
    y0 = max(tly, min(bry - band_h, y0))
    img = grab((tlx, y0, w, band_h))
    return np.array(img)


def _changed(a: np.ndarray, b: np.ndarray, tol: int = 5) -> bool:
    if a is None or b is None:
        return True
    diff = np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16)))
    return diff > tol


def ensure_top(cfg: Dict):
    prev = None
    for _ in range(cfg["max_scroll_attempts_top"]):
        check_abort()
        probe = _grab_probe(cfg, rel_y=0.35)   # 35% down from top
        if not _changed(probe, prev):
            break
        # scroll up once
        if cfg["scroll_method"] == "wheel":
            scroll_wheel(tuple(cfg["ui"]["scroll_point"]),
                         cfg["ui"]["scroll_top_ticks"],
                         cfg["sleep_after_scroll"])
        else:
            scroll_drag(tuple(cfg["ui"]["drag_to"]),
                        tuple(cfg["ui"]["drag_from"]),
                        1, cfg["sleep_after_scroll"])
        prev = probe


def ensure_bottom(cfg: Dict):
    prev = None
    for _ in range(cfg["max_scroll_attempts_bottom"]):
        check_abort()
        probe = _grab_probe(cfg, rel_y=0.80)   # near bottom
        if not _changed(probe, prev):
            break
        # scroll down once
        if cfg["scroll_method"] == "wheel":
            scroll_wheel(tuple(cfg["ui"]["scroll_point"]),
                         cfg["ui"]["scroll_bottom_ticks"],
                         cfg["sleep_after_scroll"])
        else:
            scroll_drag(tuple(cfg["ui"]["drag_from"]),
                        tuple(cfg["ui"]["drag_to"]),
                        1, cfg["sleep_after_scroll"])
        prev = probe

# --------- OCR PARSING ----------


def df_to_clusters(df: pd.DataFrame) -> List[pd.DataFrame]:
    if df.empty:
        return []
    df = df.copy()
    df["mid_y"] = df["top"] + df["height"]/2
    df.sort_values("mid_y", inplace=True)
    gap = max(40, int(df["height"].median()*2.0))
    clusters, cur, last_y = [], [], None
    for _, w in df.iterrows():
        if last_y is None or (w.mid_y - last_y) <= gap:
            cur.append(w)
        else:
            clusters.append(pd.DataFrame(cur))
            cur = [w]
        last_y = w.mid_y
    if cur:
        clusters.append(pd.DataFrame(cur))
    return clusters


def parse_row_tokens(tokens: List[str], regexes: List[str] = None) -> Optional[Dict]:
    """Parse OCR tokens for a single leaderboard row.

    The function first tries user supplied regular expressions (from the
    configuration) and falls back to heuristic parsing when no regex matches.
    """
    text = " ".join(tokens)
    if regexes:
        for rx in regexes:
            m = re.search(rx, text, re.I)
            if m:
                d = m.groupdict()
                try:
                    d["kill_score"] = int(d["kill_score"].replace(",", ""))
                    d["warzone"] = int(d["warzone"])
                except Exception:
                    continue
                d["player"] = d.get("player", "").strip(" -|:;\"',.")
                d["alliance"] = d.get("alliance", "").strip()
                return {
                    "alliance": d.get("alliance", ""),
                    "player": d.get("player", ""),
                    "warzone": d.get("warzone"),
                    "kill_score": d.get("kill_score"),
                }
    if "Group " in text:
        return None
    if "[" not in text or "]" not in text:
        return None

    m = re.search(r"\[([^\]]+)\]", text)
    if not m:
        return None
    alliance = m.group(1).strip()
    after = text[m.end():].strip()

    wz_match = re.search(r"(?:WZ|Warzone)\s*#?\s*(\d+)", after, re.I)
    if wz_match:
        wz = int(wz_match.group(1))
        player_part = after[:wz_match.start()].strip()
        tail = after[wz_match.end():]
    else:
        num = re.search(r"(\d{3,5})", after)
        if not num:
            return None
        wz = int(num.group(1))
        player_part = after[:num.start()].strip()
        tail = after[num.end():]

    nums = re.findall(
        r"(\d[\d,\.]+)", tail) or re.findall(r"(\d[\d,\.]+)", after)
    if not nums:
        return None
    ks = nums[-1].replace(",", "").replace(".", "")
    try:
        ks = int(ks)
    except ValueError:
        return None

    player = player_part.strip(" -|:;\"',.")
    if not player:
        return None

    return {"alliance": alliance, "player": player, "warzone": wz, "kill_score": ks}


def capture_board(cfg: Dict) -> Image.Image:
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    return grab((tlx, tly, brx - tlx, bry - tly))


def scrape_group(cfg: Dict, gname: str) -> List[Dict]:
    ensure_top(cfg)
    img_top = capture_board(cfg)

    ensure_bottom(cfg)
    img_bot = capture_board(cfg)

    rows_out = []
    for pos, img in [("top", img_top), ("bot", img_bot)]:
        pre = preprocess(img, cfg["threshold"])
        df = ocr_tsv(pre, cfg["ocr_psm"], cfg["lang"], cfg["tesseract_exe"])

        if cfg.get("debug"):
            os.makedirs("debug", exist_ok=True)
            img.save(f"debug/{gname}_raw_{pos}.png")
            pre.save(f"debug/{gname}_bin_{pos}.png")
            df.to_csv(f"debug/{gname}_{pos}.csv", index=False)

        clusters = df_to_clusters(df)
        for c in clusters:
            c_sorted = c.sort_values("left")
            tokens = [t for t in c_sorted.text.tolist() if t]
            rec = parse_row_tokens(tokens, cfg.get("regexes"))
            if rec:
                rows_out.append(rec)

    # dedupe & cap at 10
    seen, clean = set(), []
    for r in rows_out:
        key = (r["alliance"], r["player"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)
        if len(clean) >= cfg["expected_total_rows"]:
            break

    for i, r in enumerate(clean, 1):
        r["group"] = gname
        r["rank"] = i

    if cfg.get("debug"):
        with open(f"debug/{gname}_final.json", "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)

    return clean

# --------- ORCHESTRATION ----------


def select_group(cfg: Dict, idx: int):
    check_abort()
    click(tuple(cfg["ui"]["group_button"]), cfg["sleep_after_click"])
    ocr_click_group(cfg, cfg["groups"][idx])


def run(cfg: Dict, out_file: str, debug: bool):
    if debug:
        cfg["debug"] = True
    bring_window_to_front(cfg["window_title_regex"])

    all_rows: List[Dict] = []
    for i, g in enumerate(cfg["groups"]):
        check_abort()
        print(f"Processing Group {g}...")
        select_group(cfg, i)
        rows = scrape_group(cfg, g)
        if len(rows) < cfg["expected_total_rows"]:
            print(f"  [WARN] Only got {len(rows)} rows for group {g}")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
                      "group", "rank", "alliance", "player", "kill_score", "warzone"])
    df.sort_values(["group", "rank"], inplace=True)
    df.to_excel(out_file, index=False)
    print(f"Saved to {out_file}")


# --------- MAIN ----------
if __name__ == "__main__":
    cfg = load_config()
    if "--calibrate" in sys.argv:
        calibrate(cfg)
        sys.exit(0)
    if "--run" in sys.argv:
        out = "leaderboard.xlsx"
        i = sys.argv.index("--run")
        if len(sys.argv) > i+1 and not sys.argv[i+1].startswith("-"):
            out = sys.argv[i+1]
        run(cfg, out, debug="--debug" in sys.argv)
        sys.exit(0)

    print("Usage:")
    print("  python lastwar_scraper_v7.py --calibrate")
    print("  python lastwar_scraper_v7.py --run [outfile.xlsx] [--debug]")

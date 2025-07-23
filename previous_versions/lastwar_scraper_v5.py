"""
lastwar_scraper_v5.py
---------------------
- Multi-drag scroll reset (no drift between groups)
- OCR TSV -> cluster by Y to build whole player rows
- Alliance = [brackets], player = text right after ], warzone & kill score located flexibly
- Dropdown auto-closes; no ESC needed

Usage:
  python lastwar_scraper_v5.py --calibrate
  python lastwar_scraper_v5.py --run leaderboard.xlsx --debug
"""

import os
import sys
import time
import json
import re
from typing import Dict, List, Tuple

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

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "window_title_regex": "Last War",
    "tesseract_exe": "",
    "lang": "eng+kor+jpn",                      # add like "eng+kor+jpn" if needed
    "groups": [chr(ord('A') + i) for i in range(16)],  # A..P
    "sleep_after_click": 0.35,
    "sleep_after_group_change": 0.9,
    "extra_wait_after_group": 0.5,
    "sleep_after_scroll": 0.45,
    "scroll_method": "drag",            # "drag" or "wheel"
    "scroll_top_repeats": 6,
    "scroll_bottom_repeats": 1,
    "expected_total_rows": 10,
    "ocr_psm": 6,
    "threshold": 180,
    "debug": False,
    "ui": {
        "group_button": [0, 0],         # F1
        "dropdown_tl": [0, 0],          # F2
        "dropdown_br": [0, 0],          # F3
        "board_tl":    [0, 0],          # F4
        "board_br":    [0, 0],          # F5
        # wheel
        "scroll_point": [0, 0],
        "scroll_top_ticks": 6,
        "scroll_bottom_ticks": -3,
        # drag
        "drag_from": [0, 0],            # F6
        "drag_to":   [0, 0],            # F7
    }
}

# ------------------------ small utils ------------------------


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
    return Image.fromarray(th)


def ocr_tsv(img: Image.Image, psm: int, lang: str, exe: str) -> pd.DataFrame:
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    df = pytesseract.image_to_data(
        img, config=f"--psm {psm}", lang=lang, output_type=Output.DATAFRAME)
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
        pag.moveTo(*fr)
        pag.dragTo(to[0], to[1], duration=0.25)
        time.sleep(sleep)


def wait_point(label: str, key: str) -> Tuple[int, int]:
    print(f"Hover {label} → press {key.upper()}  (ESC abort)")
    while True:
        if keyboard.is_pressed("esc"):
            print("Aborted.")
            sys.exit(0)
        if keyboard.is_pressed(key):
            p = pag.position()
            print(f"{label}: {p}")
            time.sleep(0.25)
            return p
        time.sleep(0.05)

# ------------------------ calibration ------------------------


def calibrate(cfg: Dict):
    bring_window_to_front(cfg["window_title_regex"])
    ui = cfg["ui"]

    print("=== Calibration ===")
    ui["group_button"] = list(wait_point("blue Group arrow", "f1"))

    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    ui["dropdown_tl"] = list(wait_point("dropdown TOP-LEFT", "f2"))
    ui["dropdown_br"] = list(wait_point("dropdown BOTTOM-RIGHT", "f3"))
    # dropdown auto-closes after next click; do nothing else (ESC not needed)

    ui["board_tl"] = list(wait_point(
        "Leaderboard TOP-LEFT (inside white area)", "f4"))
    ui["board_br"] = list(wait_point(
        "Leaderboard BOTTOM-RIGHT (inside white area)", "f5"))

    if cfg["scroll_method"] == "wheel":
        ui["scroll_point"] = list(wait_point(
            "point where wheel scrolls", "f6"))
    else:
        ui["drag_from"] = list(wait_point("drag START (inside list)", "f6"))
        ui["drag_to"] = list(wait_point("drag END (pull upward)", "f7"))

    save_config(cfg)
    print("Saved config.json. Run: python lastwar_scraper_v5.py --run --debug")

# ------------------------ dropdown OCR click ------------------------


def ocr_click_group(cfg: Dict, letter: str):
    ui = cfg["ui"]
    tlx, tly = ui["dropdown_tl"]
    brx, bry = ui["dropdown_br"]
    img = grab((tlx, tly, brx - tlx, bry - tly))
    pre = preprocess(img, cfg["threshold"])
    df = ocr_tsv(pre, 6, cfg["lang"], cfg["tesseract_exe"])

    # group words by line
    lines = {}
    for _, r in df.iterrows():
        key = (r.block_num, r.par_num, r.line_num)
        lines.setdefault(key, []).append(r)

    target_xy = None
    patt = rf"Group\s*{letter}\b"
    # allow OCR confusion I <-> 1, O <-> 0
    alt = {"I": "[I1]", "O": "[O0]"}
    if letter in alt:
        patt = rf"Group\s*{alt[letter]}\b"

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
            target_xy = (tlx + (x1 + x2)//2, tly + (y1 + y2)//2)
            break

    if target_xy is None:
        print(f"[WARN] Couldn't OCR 'Group {letter}'. Fallback to math.")
        total_h = (ui["dropdown_br"][1] - ui["dropdown_tl"][1])
        lh_guess = total_h / len(cfg["groups"])
        gx = ui["dropdown_tl"][0] + 40
        gy = int(ui["dropdown_tl"][1] + lh_guess *
                 cfg["groups"].index(letter) + lh_guess/2)
        target_xy = (gx, gy)

    click(target_xy, cfg["sleep_after_group_change"])
    time.sleep(cfg["extra_wait_after_group"])  # let panel paint

# ------------------------ scrolling ------------------------


def goto_top(cfg: Dict):
    if cfg["scroll_method"] == "wheel":
        for _ in range(cfg["scroll_top_repeats"]):
            scroll_wheel(tuple(cfg["ui"]["scroll_point"]), cfg["ui"]
                         ["scroll_top_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(cfg["ui"]["drag_to"]), tuple(cfg["ui"]["drag_from"]),
                    cfg["scroll_top_repeats"], cfg["sleep_after_scroll"])


def goto_bottom(cfg: Dict):
    if cfg["scroll_method"] == "wheel":
        for _ in range(cfg["scroll_bottom_repeats"]):
            scroll_wheel(tuple(cfg["ui"]["scroll_point"]), cfg["ui"]
                         ["scroll_bottom_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(cfg["ui"]["drag_from"]), tuple(cfg["ui"]["drag_to"]),
                    cfg["scroll_bottom_repeats"], cfg["sleep_after_scroll"])

# ------------------------ OCR board parsing ------------------------


def capture_board(cfg: Dict) -> Image.Image:
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    return grab((tlx, tly, brx - tlx, bry - tly))


def df_to_row_clusters(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Cluster words into rows by their top Y coordinate."""
    df = df.copy()
    df["mid_y"] = df["top"] + df["height"]/2
    df.sort_values("mid_y", inplace=True)

    # dynamic gap: median height * 2
    gap = max(40, int(df["height"].median()*2.2))
    clusters = []
    cur = []
    last_y = None
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


def parse_row_tokens(tokens: List[str]) -> Dict:
    """
    tokens: tokens of a single visual row (sorted by x).
    Returns dict or None.
    """
    text = " ".join(tokens)
    if "Group " in text:   # junk
        return None
    if "[" not in text or "]" not in text:
        return None

    # alliance
    m = re.search(r"\[([^\]]+)\]", text)
    if not m:
        return None
    alliance = m.group(1).strip()

    after = text[m.end():].strip()

    # find warzone
    wz = None
    wz_match = re.search(r"(?:WZ|Warzone)\\s*#?\\s*(\\d+)", after, re.I)
    if wz_match:
        wz = int(wz_match.group(1))
        player_part = after[:wz_match.start()].strip()
        tail = after[wz_match.end():]
    else:
        # fallback: first 3-5 digit number
        num = re.search(r"(\\d{3,5})", after)
        if not num:
            return None
        wz = int(num.group(1))
        player_part = after[:num.start()].strip()
        tail = after[num.end():]

    # kill score: largest number with commas
    nums = re.findall(r"(\\d[\\d,\\.]+)", tail)
    if not nums:
        # maybe it's before warzone on same row (rare)
        nums = re.findall(r"(\\d[\\d,\\.]+)", after)
    if not nums:
        return None
    ks = nums[-1].replace(",", "").replace(".", "")
    try:
        ks = int(ks)
    except ValueError:
        return None

    # clean player (remove stray punctuation)
    player = player_part.strip(" -|:;\"'.,")
    if not player:
        return None

    return {"alliance": alliance, "player": player, "warzone": wz, "kill_score": ks}


def scrape_group(cfg: Dict, gname: str) -> List[Dict]:
    # hard reset to top first
    goto_top(cfg)
    img_top = capture_board(cfg)

    goto_bottom(cfg)
    img_bot = capture_board(cfg)

    # OCR both
    rows_out = []
    for pos, img in [("top", img_top), ("bot", img_bot)]:
        pre = preprocess(img, cfg["threshold"])
        df = ocr_tsv(pre, cfg["ocr_psm"], cfg["lang"], cfg["tesseract_exe"])
        # save debug
        if cfg.get("debug"):
            os.makedirs("debug", exist_ok=True)
            img.save(f"debug/{gname}_raw_{pos}.png")
            pre.save(f"debug/{gname}_bin_{pos}.png")
            df.to_csv(f"debug/{gname}_{pos}.csv", index=False)

        # build clusters -> tokens -> parse
        clusters = df_to_row_clusters(df)
        for c in clusters:
            c_sorted = c.sort_values("left")
            tokens = [t for t in c_sorted.text.tolist() if t]
            rec = parse_row_tokens(tokens)
            if rec:
                rows_out.append(rec)

    # dedupe + top 10
    seen = set()
    cleaned = []
    for r in rows_out:
        key = (r["alliance"], r["player"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)
        if len(cleaned) >= cfg["expected_total_rows"]:
            break

    for i, r in enumerate(cleaned, 1):
        r["group"] = gname
        r["rank"] = i

    if cfg.get("debug"):
        with open(f"debug/{gname}_final.json", "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

    return cleaned

# ------------------------ orchestrators ------------------------


def select_group(cfg: Dict, idx: int):
    click(tuple(cfg["ui"]["group_button"]), cfg["sleep_after_click"])
    ocr_click_group(cfg, cfg["groups"][idx])


def run(cfg: Dict, out_file: str, debug: bool):
    if debug:
        cfg["debug"] = True
    bring_window_to_front(cfg["window_title_regex"])

    all_rows: List[Dict] = []
    for i, g in enumerate(cfg["groups"]):
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

# ------------------------ main ------------------------


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
    print("  python lastwar_scraper_v5.py --calibrate")
    print("  python lastwar_scraper_v5.py --run [outfile.xlsx] [--debug]")

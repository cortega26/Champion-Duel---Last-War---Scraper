"""
lastwar_scraper_v3.py
---------------------
Key changes:
- Uses pytesseract.image_to_data (TSV) to parse text line-by-line (no more 0-row regex fails).
- OCR-based dropdown clicking: it screenshots the list and clicks the word "Group X".
- Optional drag scrolling.
- Rich debug output (PNG, BIN, TSV CSV, and parsed_lines.txt).

Hotkeys in calibration (ESC to abort):
  F1: Blue up-arrow (open group list)
  F2: Top-left of the DROPDOWN area (a little above 'Group A')
  F3: Bottom-right of the DROPDOWN area (so we can OCR the full list)
  F4: TOP-LEFT inside leaderboard white panel
  F5: BOTTOM-RIGHT inside leaderboard white panel
  F6: (if scroll_method == "wheel") point where wheel works
      (if scroll_method == "drag") DRAG START
  F7: (drag only) DRAG END

Run:
  python lastwar_scraper_v3.py --calibrate
  python lastwar_scraper_v3.py --run leaderboard.xlsx --debug
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
    "lang": "eng",               # add: eng+kor+jpn+chi_sim etc if needed
    "groups": [chr(ord('A') + i) for i in range(16)],  # A..P
    "sleep_after_click": 0.30,
    "sleep_after_group_change": 0.70,
    "sleep_after_scroll": 0.35,
    "expected_total_rows": 10,
    "rows_per_screen": 7,        # for cropping by row if needed
    "ocr_psm": 6,
    "threshold": 180,
    "scroll_method": "drag",     # "drag" or "wheel"
    "debug": False,
    "ui": {
        "group_button": [0, 0],          # F1
        "dropdown_tl": [0, 0],           # F2  (dropdown region top-left)
        "dropdown_br": [0, 0],           # F3  (dropdown region bottom-right)
        "board_tl": [0, 0],              # F4
        "board_br": [0, 0],              # F5
        # wheel method
        "scroll_point": [0, 0],
        "scroll_top_ticks": 6,
        "scroll_bottom_ticks": -3,
        # drag method
        "drag_from": [0, 0],             # F6
        "drag_to":   [0, 0],             # F7
        "drag_repeats_bottom": 2
    }
}

# -------------------- Utils --------------------


def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # merge defaults

        def merge(d, default):
            for k, v in default.items():
                if k not in d:
                    d[k] = v
                elif isinstance(v, dict):
                    merge(d[k], v)
        merge(cfg, DEFAULT_CONFIG)
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def bring_window_to_front(pattern: str):
    if gw is None:
        return
    titles = [t for t in gw.getAllTitles() if re.search(pattern, t, re.I)]
    if not titles:
        print("[WARN] Window title not found.")
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
    df = df[df.conf != -1].copy()
    df["text"] = df["text"].fillna("").astype(str)
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
    print(f"Hover {label} and press {key.upper()} (ESC abort)")
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

# -------------------- Calibration --------------------


def calibrate(cfg: Dict):
    bring_window_to_front(cfg["window_title_regex"])
    ui = cfg["ui"]

    print("=== Calibration ===")
    ui["group_button"] = list(wait_point(
        "blue up-arrow (open group list)", "f1"))

    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    ui["dropdown_tl"] = list(wait_point(
        "DROPDOWN top-left (a bit above 'Group A')", "f2"))
    ui["dropdown_br"] = list(wait_point(
        "DROPDOWN bottom-right (below 'Group P')", "f3"))
    keyboard.press_and_release("esc")
    time.sleep(0.2)

    ui["board_tl"] = list(wait_point(
        "Leaderboard TOP-LEFT inside white frame", "f4"))
    ui["board_br"] = list(wait_point(
        "Leaderboard BOTTOM-RIGHT inside white frame", "f5"))

    if cfg["scroll_method"] == "wheel":
        ui["scroll_point"] = list(wait_point(
            "point where mouse wheel scrolls", "f6"))
    else:
        ui["drag_from"] = list(wait_point(
            "drag START point (inside list)", "f6"))
        ui["drag_to"] = list(wait_point("drag END point (pull upward)", "f7"))

    save_config(cfg)
    print("Saved config.json\nRun:  python lastwar_scraper_v3.py --run --debug")

# -------------------- Dropdown OCR Click --------------------


def ocr_click_group(cfg: Dict, target_letter: str):
    """Open dropdown (already open), OCR it, click 'Group X' line."""
    ui = cfg["ui"]
    tlx, tly = ui["dropdown_tl"]
    brx, bry = ui["dropdown_br"]
    img = grab((tlx, tly, brx - tlx, bry - tly))
    pre = preprocess(img, cfg["threshold"])
    df = ocr_tsv(pre, 6, cfg["lang"], cfg["tesseract_exe"])

    # Combine words per line
    lines = {}
    for i, row in df.iterrows():
        key = (row.block_num, row.par_num, row.line_num)
        lines.setdefault(key, []).append(row)
    target_xy = None

    for key, words in lines.items():
        words_sorted = sorted(words, key=lambda r: r.left)
        line_text = " ".join(w.text for w in words_sorted).strip()
        # Look for "Group X"
        m = re.search(rf"Group\s*{target_letter}\b", line_text, re.I)
        if m:
            # union bbox of that line
            xs = [w.left for w in words_sorted]
            ys = [w.top for w in words_sorted]
            ws = [w.width for w in words_sorted]
            hs = [w.height for w in words_sorted]
            x1, y1 = min(xs), min(ys)
            x2 = max(x+w for x, w in zip(xs, ws))
            y2 = max(y+h for y, h in zip(ys, hs))
            cx = tlx + (x1 + x2)//2
            cy = tly + (y1 + y2)//2
            target_xy = (cx, cy)
            break

    if target_xy is None:
        # fallback to old math
        print(
            f"[WARN] Couldn't OCR 'Group {target_letter}'. Using fallback math.")
        # fallback guess spacing:
        lh_guess = (ui["dropdown_br"][1] - ui["dropdown_tl"]
                    [1]) / len(cfg["groups"])
        gx = ui["dropdown_tl"][0] + 30
        gy = int(ui["dropdown_tl"][1] + lh_guess *
                 cfg['groups'].index(target_letter) + lh_guess/2)
        target_xy = (gx, gy)

    click(target_xy, cfg["sleep_after_group_change"])

# -------------------- Board OCR & Parse --------------------


ROW_REGEX = re.compile(
    r"\[(?P<alliance>[^\]]+)\]\s*(?P<player>.+?)\s+(?:WZ|Warzone)\s*#?\s*(?P<warzone>\d+).*?(?P<score>\d[\d,\.]*)",
    re.I
)


def parse_lines(lines: List[str]) -> List[Dict]:
    out = []
    for ln in lines:
        m = ROW_REGEX.search(ln)
        if not m:
            continue
        d = m.groupdict()
        d["alliance"] = d["alliance"].strip()
        d["player"] = d["player"].strip()
        d["kill_score"] = int(d["score"].replace(",", "").replace(".", ""))
        d["warzone"] = int(d["warzone"])
        del d["score"]
        out.append(d)
    return out


def board_text_lines(cfg: Dict, img: Image.Image) -> List[str]:
    pre = preprocess(img, cfg["threshold"])
    df = ocr_tsv(pre, cfg["ocr_psm"], cfg["lang"], cfg["tesseract_exe"])
    # group by line
    lines = {}
    for i, row in df.iterrows():
        key = (row.block_num, row.par_num, row.line_num)
        lines.setdefault(key, []).append(row)
    out_lines = []
    for key, words in sorted(lines.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
        ws = sorted(words, key=lambda r: r.left)
        out_lines.append(" ".join(w.text for w in ws).strip())
    return out_lines, pre, df


def capture_board(cfg: Dict) -> Image.Image:
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    return grab((tlx, tly, brx - tlx, bry - tly))


def goto_top(cfg: Dict):
    ui = cfg["ui"]
    if cfg["scroll_method"] == "wheel":
        scroll_wheel(tuple(ui["scroll_point"]),
                     ui["scroll_top_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(ui["drag_to"]), tuple(
            ui["drag_from"]), 2, cfg["sleep_after_scroll"])


def goto_bottom(cfg: Dict):
    ui = cfg["ui"]
    if cfg["scroll_method"] == "wheel":
        scroll_wheel(tuple(ui["scroll_point"]),
                     ui["scroll_bottom_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(ui["drag_from"]), tuple(
            ui["drag_to"]), ui["drag_repeats_bottom"], cfg["sleep_after_scroll"])


def scrape_group(cfg: Dict, gname: str) -> List[Dict]:
    # top
    goto_top(cfg)
    img_top = capture_board(cfg)
    # bottom
    goto_bottom(cfg)
    img_bot = capture_board(cfg)

    lines_top, pre_top, df_top = board_text_lines(cfg, img_top)
    lines_bot, pre_bot, df_bot = board_text_lines(cfg, img_bot)

    # Combine and parse
    all_lines = lines_top + lines_bot
    rows = parse_lines(all_lines)

    # dedupe & rank
    seen = set()
    clean = []
    for r in rows:
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

    # Debug dump
    if cfg.get("debug"):
        os.makedirs("debug", exist_ok=True)
        img_top.save(f"debug/{gname}_raw_top.png")
        img_bot.save(f"debug/{gname}_raw_bottom.png")
        pre_top.save(f"debug/{gname}_bin_top.png")
        pre_bot.save(f"debug/{gname}_bin_bottom.png")
        pd.DataFrame(df_top).to_csv(f"debug/{gname}_top_tsv.csv", index=False)
        pd.DataFrame(df_bot).to_csv(f"debug/{gname}_bot_tsv.csv", index=False)
        with open(f"debug/{gname}_parsed_lines.txt", "w", encoding="utf-8") as f:
            for ln in all_lines:
                f.write(ln + "\n")

    return clean

# -------------------- Selecting group --------------------


def select_group(cfg: Dict, idx: int):
    click(tuple(cfg["ui"]["group_button"]), cfg["sleep_after_click"])
    letter = cfg["groups"][idx]
    ocr_click_group(cfg, letter)

# -------------------- Run --------------------


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

# -------------------- Main --------------------


if __name__ == "__main__":
    cfg = load_config()
    if "--calibrate" in sys.argv:
        calibrate(cfg)
        sys.exit(0)

    if "--run" in sys.argv:
        out = "leaderboard.xlsx"
        i = sys.argv.index("--run")
        if len(sys.argv) > i + 1 and not sys.argv[i+1].startswith("-"):
            out = sys.argv[i+1]
        run(cfg, out, debug="--debug" in sys.argv)
        sys.exit(0)

    print("Usage:")
    print("  python lastwar_scraper_v3.py --calibrate")
    print("  python lastwar_scraper_v3.py --run [outfile.xlsx] [--debug]")

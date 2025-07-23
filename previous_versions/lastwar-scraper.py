"""
lastwar_scraper.py
-------------------
Automate grabbing the Top 10 players for Groups A–P from "Last War: Survival" (Windows client),
OCR the text, and save everything to an Excel file.

Run it in two phases:
1) Calibrate once:  python lastwar_scraper.py --calibrate
2) Scrape:          python lastwar_scraper.py --run  (outputs leaderboard.xlsx)

Dependencies
------------
Python 3.9+
> pip install -U pyautogui keyboard mss pillow pytesseract opencv-python pandas openpyxl pygetwindow

Tesseract OCR (Windows):
- Download the installer: https://github.com/UB-Mannheim/tesseract/wiki
- Install to default:  C:\\Program Files\\Tesseract-OCR
- Add that folder to your PATH or set TESSERACT_EXE in the config (below).

Hotkeys used during calibration: F1..F8 (shown on screen). ESC to abort.

What you will mark in calibration:
  F1  = click arrow button (blue up arrow next to "Group X")
  F2  = top-left text point of "Group A" in dropdown list
  F3  = top-left text point of "Group B" in dropdown list  (used to measure row height)
  F4  = top-left corner of the leaderboard box (just inside the white frame)
  F5  = bottom-right corner of the leaderboard box
  F6  = a point INSIDE the leaderboard box where scrolling works (mouse wheel)
  F7  = top-most position (run this after you've scrolled all the way UP once; saves the number of scroll ticks)
  F8  = bottom-most position (scroll down to the very bottom once; saves ticks)

If you can't reach absolute top/bottom with wheel, set scroll_top_ticks/scroll_bottom_ticks manually in config.json later.

After calibration, a config.json is created. You can edit it any time.

Output columns: group, rank, alliance, player, kill_score, warzone.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import keyboard
import mss
import numpy as np
import pandas as pd
import pyautogui as pag
import pytesseract
from PIL import Image
import cv2

try:
    import pygetwindow as gw
except Exception:
    gw = None

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    # Part of the game window title. Adjust if needed.
    "window_title_regex": "Last War",
    "tesseract_exe": "",                # Leave empty if it's already on PATH
    "groups": [chr(ord('A') + i) for i in range(16)],  # A..P
    "rows_per_screen": 7,                # How many rows visible before scrolling
    "expected_total_rows": 10,
    "sleep_after_click": 0.35,
    "sleep_after_group_change": 0.7,
    "sleep_after_scroll": 0.35,
    "ocr_psm": 6,
    # binarization threshold (0-255). tweak if OCR is bad
    "threshold": 180,
    "regex": r"\[(?P<alliance>[^\]]+)\](?P<player>[^\n]+?)\\s*Warzone\\s*#(?P<warzone>\\d+).*?(?P<score>\\d[\\d,]+)",
    "ui": {
        # absolute pixel coords filled by calibration
        "group_button": [0, 0],
        "group_list_top_left": [0, 0],   # top-left of Group A text
        "group_list_line_height": 0,     # pixel distance between items in dropdown list
        "board_tl": [0, 0],              # leaderboard crop top-left
        "board_br": [0, 0],              # leaderboard crop bottom-right
        # point to place the mouse for wheel scroll
        "scroll_point": [0, 0],
        # wheel ticks to force to top (positive values)
        "scroll_top_ticks": 6,
        # wheel ticks to show last rows (negative)
        "scroll_bottom_ticks": -3
    }
}

# ---------------------------- Utility helpers ---------------------------- #


def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()


def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def bring_window_to_front(title_regex: str):
    if gw is None:
        return
    wins = [w for w in gw.getAllTitles() if re.search(title_regex, w, re.I)]
    if not wins:
        print("[WARN] Could not find window matching regex.")
        return
    win = gw.getWindowsWithTitle(wins[0])[0]
    try:
        win.activate()
    except Exception:
        win.minimize()
        time.sleep(0.5)
        win.restore()
    time.sleep(0.4)


def grab(region: Tuple[int, int, int, int]) -> Image.Image:
    """Grab region (left, top, width, height) using mss and return PIL image."""
    with mss.mss() as sct:
        sshot = sct.grab(
            {"left": region[0], "top": region[1], "width": region[2], "height": region[3]})
    img = Image.frombytes("RGB", sshot.size, sshot.rgb)
    return img


def preprocess_for_ocr(img: Image.Image, threshold: int) -> Image.Image:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(th)


def ocr_text(img: Image.Image, psm: int, tess_exe: str = "") -> str:
    if tess_exe:
        pytesseract.pytesseract.tesseract_cmd = tess_exe
    cfg = f"--psm {psm}"  # add --oem 1 if you need LSTM only
    return pytesseract.image_to_string(img, config=cfg, lang="eng")


def parse_entries(text: str, pattern: str) -> List[Dict]:
    compiled = re.compile(pattern, re.I | re.S)
    out = []
    for m in compiled.finditer(text):
        d = m.groupdict()
        # cleanup
        d["score"] = d["score"].replace(",", "")
        d["player"] = d["player"].strip()
        d["alliance"] = d["alliance"].strip()
        out.append(d)
    return out


def click(xy: Tuple[int, int], sleep: float = 0.2):
    pag.moveTo(*xy)
    pag.click()
    time.sleep(sleep)


def scroll_to_top(point: Tuple[int, int], ticks: int, sleep: float):
    pag.moveTo(*point)
    pag.scroll(ticks)  # positive = up
    time.sleep(sleep)


def scroll_to_bottom(point: Tuple[int, int], ticks: int, sleep: float):
    pag.moveTo(*point)
    pag.scroll(ticks)  # negative = down
    time.sleep(sleep)


# ---------------------------- Calibration ---------------------------- #

def wait_and_capture_point(label: str, key: str = "f1") -> Tuple[int, int]:
    print(
        f"Hover mouse over {label} and press {key.upper()} ... (ESC to abort)")
    while True:
        if keyboard.is_pressed("esc"):
            print("Calibration aborted.")
            sys.exit(0)
        if keyboard.is_pressed(key):
            pos = pag.position()
            print(f"Captured {label}: {pos}")
            time.sleep(0.3)
            return pos
        time.sleep(0.05)


def calibrate(cfg: Dict):
    bring_window_to_front(cfg["window_title_regex"])

    ui = cfg["ui"]
    print("\n=== Calibration ===")
    print("Make sure the leaderboard screen is visible (Group dropdown closed).")
    time.sleep(1.0)

    ui["group_button"] = list(wait_and_capture_point(
        "the blue up-arrow Group button", "f1"))

    # Open dropdown to capture group list coordinates
    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    ui["group_list_top_left"] = list(wait_and_capture_point(
        "the text of 'Group A' in the dropdown", "f2"))
    ui_b = wait_and_capture_point(
        "the text of 'Group B' in the dropdown", "f3")
    ui["group_list_line_height"] = ui_b[1] - ui["group_list_top_left"][1]
    # Close dropdown by pressing ESC or clicking outside
    keyboard.press_and_release("esc")
    time.sleep(0.3)

    print("Now mark the leaderboard box (white panel) corners.")
    ui["board_tl"] = list(wait_and_capture_point(
        "TOP-LEFT inside the leaderboard frame", "f4"))
    ui["board_br"] = list(wait_and_capture_point(
        "BOTTOM-RIGHT inside the leaderboard frame", "f5"))

    ui["scroll_point"] = list(wait_and_capture_point(
        "any point inside that box where mouse wheel scrolls", "f6"))

    # Ask user to manually scroll all the way up then capture ticks? We'll measure by letting user scroll themselves and press F7/F8 to store numbers? Simpler: ask user to scroll up/down and we just store default in config.
    print("Optionally record scroll ticks. Scroll to the VERY TOP of the list now, then press F7.")
    wait_and_capture_point("(just press F7 when ready)", "f7")
    # We'll ask how many ticks? We can't capture ticks. We'll use default values.
    print("Scroll to the VERY BOTTOM of the list now, then press F8.")
    wait_and_capture_point("(press F8 when ready)", "f8")
    print("If you know how many wheel notches you needed, set scroll_top_ticks / scroll_bottom_ticks in config.json manually. Defaults are fine for 7 visible rows.")

    save_config(cfg)
    print("\nCalibration saved to config.json.\nRun:  python lastwar_scraper.py --run\n")


# ---------------------------- Scraping ---------------------------- #

def select_group(cfg: Dict, idx: int):
    ui = cfg["ui"]
    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    # compute target position for item idx
    gx = ui["group_list_top_left"][0]
    gy = ui["group_list_top_left"][1] + int(ui["group_list_line_height"] * idx)
    click((gx, gy), cfg["sleep_after_group_change"])


def capture_board(cfg) -> Image.Image:
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    w, h = brx - tlx, bry - tly
    return grab((tlx, tly, w, h))


def scrape_group(cfg: Dict, group_name: str) -> List[Dict]:
    # Ensure top part visible
    scroll_to_top(tuple(cfg["ui"]["scroll_point"]), cfg["ui"]
                  ["scroll_top_ticks"], cfg["sleep_after_scroll"])
    img_top = capture_board(cfg)

    # Scroll to show last rows
    scroll_to_bottom(tuple(cfg["ui"]["scroll_point"]), cfg["ui"]
                     ["scroll_bottom_ticks"], cfg["sleep_after_scroll"])
    img_bottom = capture_board(cfg)

    # OCR
    pre_top = preprocess_for_ocr(img_top, cfg["threshold"])
    pre_bot = preprocess_for_ocr(img_bottom, cfg["threshold"])

    text_top = ocr_text(pre_top, cfg["ocr_psm"], cfg.get("tesseract_exe", ""))
    text_bot = ocr_text(pre_bot, cfg["ocr_psm"], cfg.get("tesseract_exe", ""))

    # Parse
    entries = parse_entries(text_top + "\n" + text_bot, cfg["regex"])

    # Deduplicate and trim to 10
    seen = set()
    cleaned = []
    for e in entries:
        key = (e['alliance'], e['player'])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(e)
        if len(cleaned) >= cfg["expected_total_rows"]:
            break

    # add group & rank
    for i, e in enumerate(cleaned, start=1):
        e["group"] = group_name
        e["rank"] = i
        e["kill_score"] = int(e.pop("score"))
        e["warzone"] = int(e["warzone"])
    return cleaned


def run(cfg: Dict, out_file: str = "leaderboard.xlsx"):
    bring_window_to_front(cfg["window_title_regex"])
    all_rows: List[Dict] = []

    for i, g in enumerate(cfg["groups"]):
        print(f"Processing Group {g}...")
        select_group(cfg, i)
        rows = scrape_group(cfg, g)
        # if OCR misses, you can log raw text or save images
        if len(rows) < cfg["expected_total_rows"]:
            print(
                f"  [WARN] Only got {len(rows)} rows for group {g} — check OCR / regex / scroll settings.")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows, columns=[
                      "group", "rank", "alliance", "player", "kill_score", "warzone"])
    df.sort_values(["group", "rank"], inplace=True)
    df.to_excel(out_file, index=False)
    print(f"\nDone! Saved to {out_file}")


# ---------------------------- Main entry ---------------------------- #
if __name__ == "__main__":
    cfg = load_config()
    if len(sys.argv) == 1:
        print("Use --calibrate or --run")
        sys.exit(0)
    if "--calibrate" in sys.argv:
        calibrate(cfg)
    elif "--run" in sys.argv:
        out_idx = sys.argv.index("--run") + 1
        out_file = sys.argv[out_idx] if len(
            sys.argv) > out_idx and not sys.argv[out_idx].startswith("-") else "leaderboard.xlsx"
        run(cfg, out_file)
    else:
        print("Unknown option. Use --calibrate or --run")

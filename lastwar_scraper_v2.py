"""
lastwar_scraper_v2.py
---------------------
- Works when the mouse wheel DOESN'T scroll (uses click-drag).
- Adds debug images/text so you can see what OCR saw.
- More tolerant regex.
- Safer group clicking (clicks mid‑row).
"""

import json
import os
import re
import sys
import time
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
    "window_title_regex": "Last War",
    "tesseract_exe": "",                        # set if not in PATH
    "groups": [chr(ord("A") + i) for i in range(16)],  # A..P
    "sleep_after_click": 0.30,
    "sleep_after_group_change": 0.70,
    "sleep_after_scroll": 0.35,
    "expected_total_rows": 10,
    "ocr_psm": 6,
    "threshold": 180,
    "regex": r"\[(?P<alliance>[^\]]+)\]\s*(?P<player>[^\n]+?)\s*Warzone\s*#(?P<warzone>\d+)\D*(?P<score>\d[\d,]*)",
    "debug": False,
    "ui": {
        "group_button": [0, 0],
        "group_list_top_left": [0, 0],
        "group_list_line_height": 0,
        "group_list_click_offset": 0.5,            # 0 = top, 1 = bottom of row
        "board_tl": [0, 0],
        "board_br": [0, 0],
        "scroll_method": "drag",                    # "drag" or "wheel"
        # wheel method
        "scroll_point": [0, 0],
        "scroll_top_ticks": 6,
        "scroll_bottom_ticks": -3,
        # drag method
        "drag_from": [0, 0],
        "drag_to": [0, 0],
        "drag_repeats_bottom": 2
    }
}

# ---------- helpers ----------


def load_config() -> Dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # merge any new keys

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


def bring_window_to_front(regex_title: str):
    if gw is None:
        return
    titles = [t for t in gw.getAllTitles() if re.search(regex_title, t, re.I)]
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


def ocr(img: Image.Image, psm: int, exe: str) -> str:
    if exe:
        pytesseract.pytesseract.tesseract_cmd = exe
    return pytesseract.image_to_string(img, config=f"--psm {psm}", lang="eng")


def parse(text: str, pattern: str) -> List[Dict]:
    rx = re.compile(pattern, re.I | re.S)
    out = []
    for m in rx.finditer(text):
        d = m.groupdict()
        d["alliance"] = d["alliance"].strip()
        d["player"] = d["player"].strip()
        d["kill_score"] = int(d["score"].replace(",", ""))
        d["warzone"] = int(d["warzone"])
        del d["score"]
        out.append(d)
    return out


def click(xy, sleep=0.2):
    pag.moveTo(*xy)
    pag.click()
    time.sleep(sleep)


def scroll_wheel(point, ticks, sleep):
    pag.moveTo(*point)
    pag.scroll(ticks)
    time.sleep(sleep)


def scroll_drag(fr, to, reps, sleep):
    for _ in range(reps):
        pag.moveTo(*fr)
        pag.dragTo(to[0], to[1], duration=0.25)
        time.sleep(sleep)

# ---------- calibration ----------


def wait_point(label: str, key: str) -> Tuple[int, int]:
    print(f"Hover {label} → press {key.upper()}   (ESC abort)")
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


def calibrate(cfg: Dict):
    bring_window_to_front(cfg["window_title_regex"])
    ui = cfg["ui"]
    print("=== Calibration ===")

    ui["group_button"] = list(wait_point("blue Group arrow", "f1"))
    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    ui["group_list_top_left"] = list(wait_point("'Group A' text", "f2"))
    p_b = wait_point("'Group B' text", "f3")
    ui["group_list_line_height"] = p_b[1] - ui["group_list_top_left"][1]
    keyboard.press_and_release("esc")
    time.sleep(0.2)

    print("Leaderboard crop corners:")
    ui["board_tl"] = list(wait_point("TOP-LEFT inside white panel", "f4"))
    ui["board_br"] = list(wait_point("BOTTOM-RIGHT inside white panel", "f5"))

    if ui["scroll_method"] == "wheel":
        ui["scroll_point"] = list(wait_point(
            "point where wheel scrolls", "f6"))
    else:
        ui["drag_from"] = list(wait_point(
            "drag START point (inside list)", "f6"))
        ui["drag_to"] = list(wait_point("drag END point (pull upward)", "f7"))

    save_config(cfg)
    print("Saved config.json. Run:  python lastwar_scraper_v2.py --run --debug")

# ---------- scraping ----------


def select_group(cfg: Dict, idx: int):
    ui = cfg["ui"]
    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    topx, topy = ui["group_list_top_left"]
    lh = ui["group_list_line_height"]
    gy = topy + int(lh * idx + lh * ui.get("group_list_click_offset", 0.5))
    gx = topx + 10
    click((gx, gy), cfg["sleep_after_group_change"])


def capture_board(cfg: Dict) -> Image.Image:
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    return grab((tlx, tly, brx - tlx, bry - tly))


def scrape_group(cfg: Dict, gname: str) -> List[Dict]:
    ui = cfg["ui"]

    # show top
    if ui["scroll_method"] == "wheel":
        scroll_wheel(tuple(ui["scroll_point"]),
                     ui["scroll_top_ticks"], cfg["sleep_after_scroll"])
    else:
        # reverse drag to go top
        scroll_drag(tuple(ui["drag_to"]), tuple(
            ui["drag_from"]), 2, cfg["sleep_after_scroll"])
    img_top = capture_board(cfg)

    # show bottom
    if ui["scroll_method"] == "wheel":
        scroll_wheel(tuple(ui["scroll_point"]),
                     ui["scroll_bottom_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(ui["drag_from"]), tuple(
            ui["drag_to"]), ui["drag_repeats_bottom"], cfg["sleep_after_scroll"])
    img_bottom = capture_board(cfg)

    pre_top = preprocess(img_top, cfg["threshold"])
    pre_bot = preprocess(img_bottom, cfg["threshold"])
    text_top = ocr(pre_top, cfg["ocr_psm"], cfg.get("tesseract_exe", ""))
    text_bot = ocr(pre_bot, cfg["ocr_psm"], cfg.get("tesseract_exe", ""))

    if cfg.get("debug"):
        os.makedirs("debug", exist_ok=True)
        img_top.save(f"debug/{gname}_raw_top.png")
        img_bottom.save(f"debug/{gname}_raw_bottom.png")
        pre_top.save(f"debug/{gname}_bin_top.png")
        pre_bot.save(f"debug/{gname}_bin_bottom.png")
        with open(f"debug/{gname}_ocr.txt", "w", encoding="utf-8") as f:
            f.write(text_top + "\n---\n" + text_bot)

    rows = parse(text_top + "\n" + text_bot, cfg["regex"])

    # dedupe & rank
    seen = set()
    out = []
    for r in rows:
        key = (r["alliance"], r["player"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= cfg["expected_total_rows"]:
            break

    for i, r in enumerate(out, 1):
        r["group"] = gname
        r["rank"] = i
    return out

# ---------- runner ----------


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

# ---------- main ----------


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
    print("  python lastwar_scraper_v2.py --calibrate")
    print("  python lastwar_scraper_v2.py --run [outfile.xlsx] [--debug]")

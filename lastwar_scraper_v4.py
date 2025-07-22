"""
lastwar_scraper_v4.py
---------------------
Fixes:
1. Take board screenshots ONLY after the dropdown is closed.
2. Extra scroll repeats / pauses to really hit top & bottom.
3. Alliance = text in [brackets], Player = text immediately after ] until Warzone/WZ/#digits.

Run:
  python lastwar_scraper_v4.py --calibrate
  python lastwar_scraper_v4.py --run leaderboard.xlsx --debug
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
    "lang": "eng",                          # add: "eng+kor" etc if needed
    "groups": [chr(ord('A') + i) for i in range(16)],  # A..P
    "sleep_after_click": 0.35,
    "sleep_after_group_change": 0.9,
    "extra_wait_after_group": 0.6,          # NEW: pause after dropdown close
    "sleep_after_scroll": 0.4,
    "expected_total_rows": 10,
    "ocr_psm": 6,
    "threshold": 180,
    "scroll_method": "drag",                # "drag" or "wheel"
    "scroll_top_repeats": 3,                # NEW
    "scroll_bottom_repeats": 3,             # NEW
    "debug": False,
    "ui": {
        "group_button": [0, 0],             # F1
        "dropdown_tl": [0, 0],              # F2
        "dropdown_br": [0, 0],              # F3
        "board_tl":    [0, 0],              # F4
        "board_br":    [0, 0],              # F5
        # wheel
        "scroll_point": [0, 0],
        "scroll_top_ticks": 6,
        "scroll_bottom_ticks": -3,
        # drag
        "drag_from": [0, 0],                # F6
        "drag_to":   [0, 0],                # F7
    }
}

# ---------- basic utils ----------


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


def bring_window_to_front(pattern: str):
    if gw is None:
        return
    titles = [t for t in gw.getAllTitles() if re.search(pattern, t, re.I)]
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
    return pytesseract.image_to_data(img, config=f"--psm {psm}", lang=lang, output_type=Output.DATAFRAME)


def click(xy, sleep=0.2):
    pag.moveTo(*xy)
    pag.click()
    time.sleep(sleep)


def press_esc():
    keyboard.press_and_release("esc")
    time.sleep(0.2)


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

# ---------- calibration ----------


def calibrate(cfg: Dict):
    bring_window_to_front(cfg["window_title_regex"])
    ui = cfg["ui"]
    print("=== Calibration ===")
    ui["group_button"] = list(wait_point(
        "blue up-arrow (open group list)", "f1"))

    click(tuple(ui["group_button"]), cfg["sleep_after_click"])
    ui["dropdown_tl"] = list(wait_point("dropdown TOP-LEFT", "f2"))
    ui["dropdown_br"] = list(wait_point("dropdown BOTTOM-RIGHT", "f3"))
    press_esc()

    ui["board_tl"] = list(wait_point(
        "Leaderboard TOP-LEFT (inside white panel)", "f4"))
    ui["board_br"] = list(wait_point(
        "Leaderboard BOTTOM-RIGHT (inside white panel)", "f5"))

    if cfg["scroll_method"] == "wheel":
        ui["scroll_point"] = list(wait_point(
            "point where mouse wheel scrolls", "f6"))
    else:
        ui["drag_from"] = list(wait_point(
            "drag START point (inside list)", "f6"))
        ui["drag_to"] = list(wait_point("drag END point (pull upwards)", "f7"))

    save_config(cfg)
    print("Saved config.json\nRun: python lastwar_scraper_v4.py --run --debug")

# ---------- dropdown OCR click ----------


def ocr_click_group(cfg: Dict, letter: str):
    """assumes dropdown already opened"""
    ui = cfg["ui"]
    tlx, tly = ui["dropdown_tl"]
    brx, bry = ui["dropdown_br"]
    img = grab((tlx, tly, brx - tlx, bry - tly))
    pre = preprocess(img, cfg["threshold"])
    df = ocr_tsv(pre, 6, cfg["lang"], cfg["tesseract_exe"])
    df = df[df.conf != -1].fillna("")

    # build lines
    lines = {}
    for _, r in df.iterrows():
        key = (r.block_num, r.par_num, r.line_num)
        lines.setdefault(key, []).append(r)

    target_xy = None
    for key, words in lines.items():
        ws = sorted(words, key=lambda r: r.left)
        line_text = " ".join(w.text for w in ws).strip()
        if re.search(rf"Group\s*{letter}\b", line_text, re.I):
            xs = [w.left for w in ws]
            ys = [w.top for w in ws]
            ws_ = [w.width for w in ws]
            hs_ = [w.height for w in ws]
            x1, y1 = min(xs), min(ys)
            x2 = max(x + w for x, w in zip(xs, ws_))
            y2 = max(y + h for y, h in zip(ys, hs_))
            target_xy = (tlx + (x1 + x2) // 2, tly + (y1 + y2) // 2)
            break

    if target_xy is None:
        print(f"[WARN] Couldn't OCR 'Group {letter}'. Fallback to math.")
        total_h = (ui["dropdown_br"][1] - ui["dropdown_tl"][1])
        lh_guess = total_h / len(cfg["groups"])
        gx = ui["dropdown_tl"][0] + 40
        gy = int(ui["dropdown_tl"][1] + lh_guess *
                 cfg["groups"].index(letter) + lh_guess / 2)
        target_xy = (gx, gy)

    click(target_xy, cfg["sleep_after_group_change"])

    # ensure dropdown closed (press ESC just in case)
    # press_esc()
    time.sleep(cfg["extra_wait_after_group"])

# ---------- board OCR ----------


def capture_board(cfg: Dict) -> Image.Image:
    tlx, tly = cfg["ui"]["board_tl"]
    brx, bry = cfg["ui"]["board_br"]
    return grab((tlx, tly, brx - tlx, bry - tly))


def goto_top(cfg: Dict):
    if cfg["scroll_method"] == "wheel":
        for _ in range(cfg["scroll_top_repeats"]):
            scroll_wheel(tuple(cfg["ui"]["scroll_point"]), cfg["ui"]
                         ["scroll_top_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(cfg["ui"]["drag_to"]), tuple(
            cfg["ui"]["drag_from"]), cfg["scroll_top_repeats"], cfg["sleep_after_scroll"])


def goto_bottom(cfg: Dict):
    if cfg["scroll_method"] == "wheel":
        for _ in range(cfg["scroll_bottom_repeats"]):
            scroll_wheel(tuple(cfg["ui"]["scroll_point"]), cfg["ui"]
                         ["scroll_bottom_ticks"], cfg["sleep_after_scroll"])
    else:
        scroll_drag(tuple(cfg["ui"]["drag_from"]), tuple(
            cfg["ui"]["drag_to"]), cfg["scroll_bottom_repeats"], cfg["sleep_after_scroll"])


def board_lines(cfg: Dict, img: Image.Image):
    pre = preprocess(img, cfg["threshold"])
    df = ocr_tsv(pre, cfg["ocr_psm"], cfg["lang"], cfg["tesseract_exe"])
    df = df[df.conf != -1].fillna("")
    lines = {}
    for _, r in df.iterrows():
        key = (r.block_num, r.par_num, r.line_num)
        lines.setdefault(key, []).append(r)
    out_lines = []
    for key, words in sorted(lines.items(), key=lambda k: (k[0][0], k[0][1], k[0][2])):
        ws = sorted(words, key=lambda r: r.left)
        out_lines.append(" ".join(w.text for w in ws).strip())
    return out_lines, pre, df


def parse_rows_from_lines(lines: List[str]) -> List[Dict]:
    results = []
    for ln in lines:
        if "Group " in ln:           # filter out dropdown text leftovers
            continue
        if "[" not in ln or "]" not in ln:
            continue
        # alliance
        m = re.search(r"\[([^\]]+)\]", ln)
        if not m:
            continue
        alliance = m.group(1).strip()
        after = ln[m.end():].strip()

        # split tokens
        # We expect something like: PlayerName ... Warzone #1234 ... KillScore
        # We'll look for warzone number first (#digits)
        wz_match = re.search(r"(?:WZ|Warzone)?\\s*#\\s*(\\d+)", after, re.I)
        warzone = None
        player = None
        if wz_match:
            warzone = int(wz_match.group(1))
            player = after[:wz_match.start()].strip()
            tail = after[wz_match.end():]
        else:
            # fallback: grab first number of 3+ digits as warzone
            nmatch = re.search(r"(\\d{3,})", after)
            if nmatch:
                warzone = int(nmatch.group(1))
                player = after[:nmatch.start()].strip()
                tail = after[nmatch.end():]
            else:
                # can't parse warzone, skip
                continue

        # kill score: last big number
        kmatch = re.findall(r"(\\d[\\d,\\.]+)", tail)
        kill_score = None
        if kmatch:
            kill_score = kmatch[-1].replace(",", "").replace(".", "")
            try:
                kill_score = int(kill_score)
            except ValueError:
                kill_score = None
        if kill_score is None:
            # sometimes score stays near beginning
            kmatch2 = re.findall(r"(\\d[\\d,\\.]+)", after)
            if kmatch2:
                kill_score = kmatch2[-1].replace(",", "").replace(".", "")
                try:
                    kill_score = int(kill_score)
                except ValueError:
                    continue
            else:
                continue

        results.append({
            "alliance": alliance,
            "player": player,
            "warzone": warzone,
            "kill_score": kill_score
        })
    return results


def scrape_group(cfg: Dict, gname: str) -> List[Dict]:
    # ensure dropdown closed before we start grabbing
    # press_esc()
    time.sleep(0.1)

    goto_top(cfg)
    img_top = capture_board(cfg)

    goto_bottom(cfg)
    img_bot = capture_board(cfg)

    lines_top, pre_top, df_top = board_lines(cfg, img_top)
    lines_bot, pre_bot, df_bot = board_lines(cfg, img_bot)

    all_lines = lines_top + lines_bot
    rows = parse_rows_from_lines(all_lines)

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

    if cfg.get("debug"):
        os.makedirs("debug", exist_ok=True)
        img_top.save(f"debug/{gname}_raw_top.png")
        img_bot.save(f"debug/{gname}_raw_bottom.png")
        pre_top.save(f"debug/{gname}_bin_top.png")
        pre_bot.save(f"debug/{gname}_bin_bottom.png")
        df_top.to_csv(f"debug/{gname}_top_tsv.csv", index=False)
        df_bot.to_csv(f"debug/{gname}_bot_tsv.csv", index=False)
        with open(f"debug/{gname}_lines.txt", "w", encoding="utf-8") as f:
            for ln in all_lines:
                f.write(ln + "\n")
        with open(f"debug/{gname}_rows.json", "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)

    return clean

# ---------- group selection ----------


def select_group(cfg: Dict, idx: int):
    # open list
    click(tuple(cfg["ui"]["group_button"]), cfg["sleep_after_click"])
    letter = cfg["groups"][idx]
    ocr_click_group(cfg, letter)

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
    print("  python lastwar_scraper_v4.py --calibrate")
    print("  python lastwar_scraper_v4.py --run [outfile.xlsx] [--debug]")

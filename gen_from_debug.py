import os
import pandas as pd
import json
import sys
import re
from typing import List, Dict, Optional

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "window_title_regex": "Last War",
    "tesseract_exe": "",
    "lang": "eng+jpn+kor+chi_sim",
    "groups": [chr(ord('A') + i) for i in range(16)],
    "sleep_after_click": 0.35,
    "sleep_after_group_change": 0.9,
    "extra_wait_after_group": 0.5,
    "sleep_after_scroll": 0.45,
    "scroll_method": "drag",
    "max_scroll_attempts_top": 3,
    "max_scroll_attempts_bottom": 1,
    "expected_total_rows": 10,
    "ocr_psm": 6,
    "threshold": 180,
    "debug": False,
    "regexes": [],
    "ui": {}
}


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


def df_to_clusters(df: pd.DataFrame) -> List[pd.DataFrame]:
    if df.empty:
        return []
    df = df.copy()
    df["mid_y"] = df["top"] + df["height"] / 2
    df.sort_values("mid_y", inplace=True)
    gap = max(40, int(df["height"].median() * 2.0))
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


def parse_group_from_debug(g: str, cfg: Dict, debug_dir: str = "debug") -> List[Dict]:
    rows = []
    for pos in ["top", "bot"]:
        tsv_path = f"{debug_dir}/{g}_{pos}_tsv.csv"
        try:
            df = pd.read_csv(tsv_path)
        except FileNotFoundError:
            continue
        df = df[df.conf != -1].fillna("")
        clusters = df_to_clusters(df)
        for c in clusters:
            tokens = [t for t in c.sort_values("left").text.tolist() if t]
            rec = parse_row_tokens(tokens, cfg.get("regexes"))
            if rec:
                rows.append(rec)
    seen, clean = set(), []
    for r in rows:
        key = (r["alliance"], r["player"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(r)
        if len(clean) >= cfg["expected_total_rows"]:
            break
    for i, r in enumerate(clean, 1):
        r["group"] = g
        r["rank"] = i
    return clean


def main(out_file: str = "leaderboard.xlsx"):
    cfg = load_config()
    all_rows: List[Dict] = []
    for g in cfg["groups"]:
        rows = parse_group_from_debug(g, cfg)
        all_rows.extend(rows)
    if not all_rows:
        print("No rows parsed")
        return
    df = pd.DataFrame(all_rows, columns=[
                      "group", "rank", "alliance", "player", "kill_score", "warzone"])
    df.sort_values(["group", "rank"], inplace=True)
    df.to_excel(out_file, index=False)
    print("Saved", out_file)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "leaderboard.xlsx"
    main(output)

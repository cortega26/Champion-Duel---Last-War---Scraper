# Last War: Survival Leaderboard Scraper

## Goal

Automatically collect the top 10 players for every Group (A–P) from the "Champion Duel → Seeded Player" leaderboard in Last War: Survival (Windows client) and export the results to Excel (.xlsx) with columns:

- `group`
- `rank`
- `alliance`
- `player`
- `kill_score`
- `warzone`

## Process Flow

### One-time Calibration

You point at key UI elements and press hotkeys (F1–F7) so the script learns their screen coordinates:

- **F1**: The blue "Group" arrow button
- **F2/F3**: Top‑left & bottom‑right of the dropdown list area (to OCR group names)
- **F4/F5**: Top‑left & bottom‑right corners of the leaderboard panel (crop for OCR)
- **F6/F7**: Drag start/end points inside the leaderboard list (since mouse wheel doesn't scroll)

A `config.json` is written with these coordinates and timing parameters.

### Run Mode

For each letter A → P:

1. **Initialize Group**
   - Open the dropdown (click F1 point)
   - OCR the dropdown to find the exact "Group X" text and click its center
   - Wait for list to finish drawing

2. **Capture Data**
   - Reset list to top (upward drag cycles or wheel scrolls)
   - Screenshot the panel ("top" image)
   - Scroll to bottom to reveal rows 8–10
   - Screenshot again ("bottom" image)

3. **Process Images**
   - OCR both screenshots (using image_to_data TSV)
   - Group words into visual rows by Y position
   - Build text lines

4. **Parse Rows**
   Each row is parsed according to these rules:
   - Alliance: Text inside `[...]`
   - Player Name: Text between `]` and "Warzone/WZ/#digits"
   - Warzone Number: First integer after "Warzone/WZ/#"
   - Kill Score: Last large number on the row (commas removed)

5. **Clean Data**
   - Remove duplicates (same alliance+player)
   - Keep only first 10 rows
   - Add group and rank information

### Export Process

1. Combine all groups' rows into a DataFrame
2. Sort by group then rank
3. Save to `leaderboard.xlsx`

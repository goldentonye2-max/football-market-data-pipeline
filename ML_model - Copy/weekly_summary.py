"""
=============================================================
  weekly_summary.py
  Reads ALL dated all_picks_YYYY-MM-DD_graded.csv files,
  combines them, and outputs a full analysis in one report.

  USAGE:
    python weekly_summary.py

  Put this script in ML_model - Copy alongside your graded CSVs.
  No database connection needed — reads pre-graded files only.
=============================================================
"""

import glob
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────
PICKS_FOLDER = "."
OUTPUT_FILE  = "weekly_report.txt"
# ─────────────────────────────────────────────────────────────────────────

DISPLAY = {
    "target__home_win":          "Home Win",
    "target__away_win":          "Away Win",
    "target__over_2_5_goals":    "Over 2.5 Goals",
    "target__over_3_5_goals":    "Over 3.5 Goals",
    "target__over_4_5_goals":    "Over 4.5 Goals",
    "target__ht_home_win":       "HT Home Win",
    "target__ht_away_win":       "HT Away Win",
    "target__ht_over_1_5_goals": "HT Over 1.5 Goals",
    "target__home_clean_sheet":  "Home Clean Sheet",
    "target__away_clean_sheet":  "Away Clean Sheet",
}

# ── HELPERS ───────────────────────────────────────────────────────────────
def roi(sub):
    wins  = (sub["result"] == "WON").sum()
    total = len(sub)
    if total == 0:
        return 0, 0, float("nan"), float("nan")
    wr     = wins / total * 100
    profit = sub.apply(
        lambda r: r["decimal_odds"] - 1 if r["result"] == "WON" else -1, axis=1
    ).sum()
    return int(wins), total, wr, profit / total * 100

def section(title, width=65):
    return "\n" + "=" * width + "\n  " + title + "\n" + "=" * width

# ── LOAD ALL GRADED CSVS ──────────────────────────────────────────────────
pattern = os.path.join(PICKS_FOLDER, "all_picks_????-??-??_graded.csv")
files   = sorted(glob.glob(pattern))

if not files:
    sys.exit(
        f"ERROR: No graded files found matching:\n  {pattern}\n"
        f"Run evaluate_picks.py for each day first."
    )

print(f"Found {len(files)} graded files:")
frames = []
for f in files:
    date_str = os.path.basename(f).replace("all_picks_", "").replace("_graded.csv", "")
    try:
        tmp = pd.read_csv(f)
        tmp["pick_date"] = date_str
        frames.append(tmp)
        settled_count = tmp[tmp["result"].isin(["WON","LOST"])].shape[0]
        print(f"  {date_str}  →  {len(tmp)} picks  ({settled_count} settled)")
    except Exception as e:
        print(f"  {date_str}  →  SKIPPED ({e})")

if not frames:
    sys.exit("ERROR: No graded picks could be loaded.")

df        = pd.concat(frames, ignore_index=True)
total_raw = len(df)
print(f"\nTotal rows loaded: {total_raw} across {df['pick_date'].nunique()} days")

settled   = df[df["result"].isin(["WON", "LOST"])].copy()
unsettled = df[df["result"] == "UNSETTLED"]
print(f"Settled: {len(settled)}   Unsettled: {len(unsettled)}")

if len(settled) == 0:
    sys.exit("No settled picks to analyse yet.")

# ── BUILD REPORT ──────────────────────────────────────────────────────────
lines = []
ts         = datetime.now().strftime("%Y-%m-%d %H:%M")
date_range = f"{df['pick_date'].min()} to {df['pick_date'].max()}"

lines.append("=" * 65)
lines.append(f"  WEEKLY PICKS REPORT")
lines.append(f"  Generated : {ts}")
lines.append(f"  Period    : {date_range}")
lines.append(f"  Days      : {df['pick_date'].nunique()}")
lines.append(f"  Total picks loaded   : {total_raw}")
lines.append(f"  Settled picks        : {len(settled)}")
lines.append(f"  Unsettled / pending  : {len(unsettled)}")
lines.append("=" * 65)

# ── OVERALL ───────────────────────────────────────────────────────────────
lines.append(section("OVERALL SUMMARY"))
w, t, wr, r = roi(settled)
lines.append(f"  Picks     : {t}")
lines.append(f"  Won       : {w}")
lines.append(f"  Win rate  : {wr:.1f}%")
lines.append(f"  Flat ROI  : {r:+.1f}%  (1 unit staked per pick)")
lines.append(f"  Avg edge  : {settled['edge_%'].mean():.1f}%")
lines.append(f"  Avg odds  : {settled['decimal_odds'].mean():.2f}")

days_pos = sum(1 for _, g in settled.groupby("pick_date") if roi(g)[3] > 0)
lines.append(f"  +ROI days : {days_pos}/{settled['pick_date'].nunique()}")

# ── PER DAY ───────────────────────────────────────────────────────────────
lines.append(section("BY DAY"))
lines.append(f"  {'Date':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8} {'AvgOdds':>8}  Trend")
lines.append(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*8}  -----")

for day in sorted(settled["pick_date"].unique()):
    grp = settled[settled["pick_date"] == day]
    w, t, wr, r = roi(grp)
    ao    = grp["decimal_odds"].mean()
    trend = "▲ PROFIT" if r > 0 else "▼ LOSS"
    lines.append(f"  {day:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}% {ao:>8.2f}  {trend}")

# ── BY MARKET ─────────────────────────────────────────────────────────────
lines.append(section("BY MARKET"))
lines.append(f"  {'Market':<26} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}  {'AvgOdds':>8}")
lines.append(f"  {'-'*26} {'-'*6} {'-'*5} {'-'*6} {'-'*8}  {'-'*8}")

market_rows = []
for target, grp in settled.groupby("target"):
    name = DISPLAY.get(target, target.replace("target__", ""))
    w, t, wr, r = roi(grp)
    ao = grp["decimal_odds"].mean()
    market_rows.append((r, name, t, w, wr, r, ao))

for _, name, t, w, wr, r, ao in sorted(market_rows, key=lambda x: -x[5]):
    lines.append(f"  {name:<26} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%  {ao:>8.2f}")

# ── BY ODDS BAND ──────────────────────────────────────────────────────────
lines.append(section("BY ODDS BAND"))
lines.append(f"  {'Odds band':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

bins   = [1.0, 1.50, 1.70, 2.00, 2.50, 3.00, 4.00, 99.0]
labels = ["1.01–1.50", "1.51–1.70", "1.71–2.00",
          "2.01–2.50", "2.51–3.00", "3.01–4.00", "4.01+"]
settled["odds_band"] = pd.cut(settled["decimal_odds"], bins=bins, labels=labels)

for band in labels:
    grp = settled[settled["odds_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    lines.append(f"  {band:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── BY EDGE TIER ──────────────────────────────────────────────────────────
lines.append(section("BY EDGE TIER"))
lines.append(f"  {'Edge tier':<10} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

ebins   = [0, 5, 8, 12, 100]
elabels = ["3–5%", "5–8%", "8–12%", "12%+"]
settled["edge_band"] = pd.cut(settled["edge_%"], bins=ebins, labels=elabels)

for band in elabels:
    grp = settled[settled["edge_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    lines.append(f"  {band:<10} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── BY AUC TIER ───────────────────────────────────────────────────────────
lines.append(section("BY MODEL AUC TIER"))
lines.append(f"  {'AUC tier':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

abins   = [0, 0.62, 0.65, 0.70, 1.0]
alabels = ["<0.62", "0.62–0.65", "0.65–0.70", ">0.70"]
settled["auc_band"] = pd.cut(settled["model_auc"], bins=abins, labels=alabels)

for band in alabels:
    grp = settled[settled["auc_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    lines.append(f"  {band:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── MARKET × EDGE CROSS TABLE ─────────────────────────────────────────────
lines.append(section("MARKET x EDGE TIER (best combinations)"))
lines.append(f"  {'Market + Edge':<35} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

cross_rows = []
for (target, eband), grp in settled.groupby(["target", "edge_band"]):
    if len(grp) < 5:
        continue
    name = DISPLAY.get(target, target.replace("target__", ""))
    w, t, wr, r = roi(grp)
    cross_rows.append((r, f"{name} | edge {eband}", t, w, wr, r))

for _, label, t, w, wr, r in sorted(cross_rows, key=lambda x: -x[5]):
    lines.append(f"  {label:<35} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── HIGH EDGE PICKS ───────────────────────────────────────────────────────
lines.append(section("HIGH-EDGE PICKS (edge >= 12%) — summary"))
hi = settled[settled["edge_%"] >= 12]
if len(hi):
    w, t, wr, r = roi(hi)
    lines.append(f"  Win rate: {wr:.1f}%   ROI: {r:+.1f}%   ({t} picks across all days)")
else:
    lines.append("  No picks with edge >= 12%")

# ── CONSISTENCY CHECK ─────────────────────────────────────────────────────
lines.append(section("CONSISTENCY CHECK"))

day_rois = []
for day, grp in settled.groupby("pick_date"):
    _, _, _, r = roi(grp)
    day_rois.append(r)

avg_daily_roi = np.nanmean(day_rois)
std_daily_roi = np.nanstd(day_rois)
days_positive = sum(1 for r in day_rois if r > 0)

lines.append(f"  Average daily ROI : {avg_daily_roi:+.1f}%")
lines.append(f"  Std dev daily ROI : {std_daily_roi:.1f}%  (lower = more consistent)")
lines.append(f"  Days with +ROI    : {days_positive}/{len(day_rois)}")

if std_daily_roi < 15 and avg_daily_roi > 0:
    verdict = "CONSISTENT EDGE — performance is stable across days"
elif avg_daily_roi > 0 and std_daily_roi >= 15:
    verdict = "VOLATILE EDGE — profitable overall but inconsistent day-to-day"
elif days_positive > len(day_rois) // 2 and avg_daily_roi < 0:
    verdict = "MARGINAL — more winning days than losing but overall negative"
else:
    verdict = "NO EDGE CONFIRMED YET — needs more data or filter adjustments"

lines.append(f"\n  Verdict: {verdict}")

# ── RECOMMENDED FILTERS ───────────────────────────────────────────────────
lines.append(section("RECOMMENDED BOOKING GENERATOR COMMANDS"))
lines.append("  Based on this data:\n")

best_market = None
best_roi_val = -999
for target, grp in settled.groupby("target"):
    if len(grp) < 10:
        continue
    _, _, _, r = roi(grp)
    if r > best_roi_val:
        best_roi_val = r
        best_market  = DISPLAY.get(target, target.replace("target__", ""))

if best_market:
    lines.append(f"  # Best market ({best_market}, ROI {best_roi_val:+.1f}%)")
    lines.append(f'  python booking_generator.py --top 10 --min-edge 12 '
                 f'--markets "{best_market}" --min-odds 1.71 --max-odds 3.00 --dry-run')
    lines.append("")
    lines.append(f"  # High-edge only across all markets")
    lines.append(f'  python booking_generator.py --top 20 --min-edge 12 '
                 f'--min-odds 1.71 --max-odds 3.00 --dry-run')

lines.append("\n" + "=" * 65)
lines.append(f"  END OF REPORT — {ts}")
lines.append("=" * 65)

# ── PRINT + SAVE ──────────────────────────────────────────────────────────
report = "\n".join(lines)
print("\n" + report)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nReport saved to {OUTPUT_FILE}")

out_csv = "weekly_graded.csv"
df.to_csv(out_csv, index=False)
print(f"Combined graded data saved to {out_csv}")
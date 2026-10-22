"""
=============================================================
  three_markets_summary.py
  Analyses graded picks for ONLY these three markets:
    - HT Away Win
    - Away Win
    - HT Over 1.5 Goals

  Everything else is ignored completely.

  USAGE:
    python three_markets_summary.py

  Run from inside ML_model - Copy.
=============================================================
"""

import glob
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────
PICKS_FOLDER  = "."
OUTPUT_FILE   = "three_markets_report.txt"

# Only these three targets — everything else ignored
KEEP_TARGETS = {
    "target__ht_away_win":       "HT Away Win",
    "target__away_win":          "Away Win",
    "target__ht_over_1_5_goals": "HT Over 1.5 Goals",
}
# ─────────────────────────────────────────────────────────────────────────

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

# ── LOAD ──────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(PICKS_FOLDER, "all_picks_????-??-??_graded.csv")))
if not files:
    sys.exit("ERROR: No graded files found. Run evaluate_picks.py first.")

print(f"Found {len(files)} graded files. Loading 3 markets only...")
frames = []
for f in files:
    date_str = os.path.basename(f).replace("all_picks_","").replace("_graded.csv","")
    try:
        tmp = pd.read_csv(f)
        tmp = tmp[tmp["target"].isin(KEEP_TARGETS.keys())].copy()
        tmp["pick_date"] = date_str
        if len(tmp) > 0:
            frames.append(tmp)
    except Exception as e:
        print(f"  {date_str} SKIPPED: {e}")

if not frames:
    sys.exit("ERROR: No picks found for these three markets.")

df      = pd.concat(frames, ignore_index=True)
settled = df[df["result"].isin(["WON","LOST"])].copy()

print(f"Total picks (3 markets): {len(df)}")
print(f"Settled: {len(settled)}   Unsettled: {len(df) - len(settled)}")

# ── BUILD REPORT ──────────────────────────────────────────────────────────
lines = []
ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

lines.append("=" * 65)
lines.append("  THREE MARKETS REPORT")
lines.append("  Markets: HT Away Win | Away Win | HT Over 1.5 Goals")
lines.append(f"  Generated : {ts}")
lines.append(f"  Period    : {df['pick_date'].min()} to {df['pick_date'].max()}")
lines.append(f"  Days      : {df['pick_date'].nunique()}")
lines.append(f"  Total picks : {len(df)}   Settled : {len(settled)}")
lines.append("=" * 65)

# ── OVERALL ───────────────────────────────────────────────────────────────
lines.append(section("OVERALL SUMMARY (3 markets only)"))
w, t, wr, r = roi(settled)
lines.append(f"  Picks    : {t}")
lines.append(f"  Won      : {w}")
lines.append(f"  Win rate : {wr:.1f}%")
lines.append(f"  Flat ROI : {r:+.1f}%")
lines.append(f"  Avg edge : {settled['edge_%'].mean():.1f}%")
lines.append(f"  Avg odds : {settled['decimal_odds'].mean():.2f}")

days_pos = sum(1 for _, g in settled.groupby("pick_date") if roi(g)[3] > 0)
lines.append(f"  +ROI days: {days_pos}/{settled['pick_date'].nunique()}")

# ── PER MARKET ────────────────────────────────────────────────────────────
lines.append(section("BY MARKET"))
lines.append(f"  {'Market':<26} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}  {'AvgOdds':>8}")
lines.append(f"  {'-'*26} {'-'*6} {'-'*5} {'-'*6} {'-'*8}  {'-'*8}")

for target, name in KEEP_TARGETS.items():
    grp = settled[settled["target"] == target]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    ao = grp["decimal_odds"].mean()
    lines.append(f"  {name:<26} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%  {ao:>8.2f}")

# ── PER DAY ───────────────────────────────────────────────────────────────
lines.append(section("BY DAY"))
lines.append(f"  {'Date':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}  Trend")
lines.append(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}  -----")

for day in sorted(settled["pick_date"].unique()):
    grp = settled[settled["pick_date"] == day]
    w, t, wr, r = roi(grp)
    trend = "▲ PROFIT" if r > 0 else "▼ LOSS"
    lines.append(f"  {day:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%  {trend}")

# ── BY EDGE TIER ──────────────────────────────────────────────────────────
lines.append(section("BY EDGE TIER"))
lines.append(f"  {'Edge tier':<10} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

ebins   = [0, 8, 12, 100]
elabels = ["<8%", "8–12%", "12%+"]
settled["edge_band"] = pd.cut(settled["edge_%"], bins=ebins, labels=elabels)

for band in elabels:
    grp = settled[settled["edge_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    lines.append(f"  {band:<10} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── BY ODDS BAND ──────────────────────────────────────────────────────────
lines.append(section("BY ODDS BAND"))
lines.append(f"  {'Odds band':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

bins   = [1.0, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80,
          1.90, 2.00, 2.25, 2.50, 2.75, 3.00, 3.50, 4.00, 99.0]
labels = ["1.01–1.20", "1.21–1.30", "1.31–1.40", "1.41–1.50",
          "1.51–1.60", "1.61–1.70", "1.71–1.80", "1.81–1.90",
          "1.91–2.00", "2.01–2.25", "2.26–2.50", "2.51–2.75",
          "2.76–3.00", "3.01–3.50", "3.51–4.00", "4.01+"]
settled["odds_band"] = pd.cut(settled["decimal_odds"], bins=bins, labels=labels)

for band in labels:
    grp = settled[settled["odds_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    lines.append(f"  {band:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── MARKET × EDGE (the key table) ─────────────────────────────────────────
lines.append(section("MARKET x EDGE TIER — the key combinations"))
lines.append(f"  {'Market + Edge':<35} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
lines.append(f"  {'-'*35} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")

cross_rows = []
for target, name in KEEP_TARGETS.items():
    for eband in elabels:
        grp = settled[
            (settled["target"] == target) &
            (settled["edge_band"] == eband)
        ]
        if len(grp) < 5:
            continue
        w, t, wr, r = roi(grp)
        cross_rows.append((r, f"{name} | {eband}", t, w, wr, r))

for _, label, t, w, wr, r in sorted(cross_rows, key=lambda x: -x[5]):
    lines.append(f"  {label:<35} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── CONSISTENCY CHECK ─────────────────────────────────────────────────────
lines.append(section("CONSISTENCY CHECK"))
day_rois = [roi(g)[3] for _, g in settled.groupby("pick_date")]
avg_r = np.nanmean(day_rois)
std_r = np.nanstd(day_rois)
pos   = sum(1 for r in day_rois if r > 0)

lines.append(f"  Avg daily ROI : {avg_r:+.1f}%")
lines.append(f"  Std dev       : {std_r:.1f}%")
lines.append(f"  +ROI days     : {pos}/{len(day_rois)}")

if std_r < 20 and avg_r > 0:
    verdict = "CONSISTENT EDGE"
elif avg_r > 0:
    verdict = "VOLATILE EDGE — profitable but inconsistent"
elif pos > len(day_rois) // 2:
    verdict = "MARGINAL — more winning days than losing"
else:
    verdict = "NO EDGE CONFIRMED — adjust filters"
lines.append(f"\n  Verdict: {verdict}")

lines.append("\n" + "=" * 65)
lines.append(f"  END — {ts}")
lines.append("=" * 65)

report = "\n".join(lines)
print("\n" + report)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nSaved to {OUTPUT_FILE}")

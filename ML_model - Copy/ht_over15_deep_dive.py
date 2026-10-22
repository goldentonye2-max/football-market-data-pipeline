"""
=============================================================
  ht_over15_deep_dive.py
  Deep dive into HT Over 1.5 Goals picks only.

  For every pick, shows the actual halftime score so you
  can see what the real HT goal lines look like at each
  edge tier — and spot if Under 1.5 HT is the real signal
  when the model's edge is highest.

  USAGE:
    python ht_over15_deep_dive.py

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
PICKS_FOLDER = "."
OUTPUT_FILE  = "ht_over15_deep_dive.txt"
TARGET       = "target__ht_over_1_5_goals"
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

def section(title, width=70):
    return "\n" + "=" * width + "\n  " + title + "\n" + "=" * width

# ── LOAD ──────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(PICKS_FOLDER, "all_picks_????-??-??_graded.csv")))
if not files:
    sys.exit("ERROR: No graded files found. Run evaluate_picks.py first.")

frames = []
for f in files:
    date_str = os.path.basename(f).replace("all_picks_","").replace("_graded.csv","")
    try:
        tmp = pd.read_csv(f)
        tmp = tmp[tmp["target"] == TARGET].copy()
        tmp["pick_date"] = date_str
        if len(tmp) > 0:
            frames.append(tmp)
    except:
        pass

if not frames:
    sys.exit("ERROR: No HT Over 1.5 Goals picks found in graded files.")

df = pd.concat(frames, ignore_index=True)
settled = df[df["result"].isin(["WON","LOST"])].copy()

# ── COMPUTE HT TOTALS ─────────────────────────────────────────────────────
# ht_home and ht_away come from the results merge in evaluate_picks.py
if "ht_home" not in settled.columns or "ht_away" not in settled.columns:
    sys.exit(
        "ERROR: ht_home / ht_away columns not found in graded files.\n"
        "Re-run evaluate_picks.py — these columns come from the results table merge."
    )

settled["ht_home"]  = pd.to_numeric(settled["ht_home"],  errors="coerce")
settled["ht_away"]  = pd.to_numeric(settled["ht_away"],  errors="coerce")
settled["ht_total"] = settled["ht_home"] + settled["ht_away"]
settled["ht_score"] = (
    settled["ht_home"].fillna("?").astype(str) + "-" +
    settled["ht_away"].fillna("?").astype(str)
)

# Remove rows where we don't have HT scores
has_score = settled["ht_total"].notna()
no_score  = (~has_score).sum()
settled   = settled[has_score].copy()

# Edge tiers
ebins   = [0, 8, 12, 100]
elabels = ["<8%", "8–12%", "12%+"]
settled["edge_band"] = pd.cut(settled["edge_%"], bins=ebins, labels=elabels)

print(f"HT Over 1.5 Goals picks loaded: {len(settled)} settled")
if no_score > 0:
    print(f"  ({no_score} excluded — no HT score in DB yet)")

# ── BUILD REPORT ──────────────────────────────────────────────────────────
lines = []
ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

lines.append("=" * 70)
lines.append("  HT OVER 1.5 GOALS — DEEP DIVE")
lines.append("  Shows actual HT score for every pick, grouped by edge tier")
lines.append(f"  Generated : {ts}")
lines.append(f"  Period    : {settled['pick_date'].min()} to {settled['pick_date'].max()}")
lines.append(f"  Total settled picks: {len(settled)}")
lines.append("=" * 70)

# ── OVERALL HT SCORE DISTRIBUTION ─────────────────────────────────────────
lines.append(section("OVERALL HT SCORE DISTRIBUTION"))
score_dist = settled["ht_score"].value_counts().head(20)
under_1_5  = (settled["ht_total"] <= 1).sum()
over_1_5   = (settled["ht_total"] >= 2).sum()
lines.append(f"  Under 1.5 HT (0 or 1 goal)  : {under_1_5}  ({under_1_5/len(settled)*100:.1f}%)")
lines.append(f"  Over 1.5 HT  (2+ goals)      : {over_1_5}  ({over_1_5/len(settled)*100:.1f}%)")
lines.append("")
lines.append(f"  {'HT Score':<12} {'Count':>6} {'%':>6}")
lines.append(f"  {'-'*12} {'-'*6} {'-'*6}")
for score, count in score_dist.items():
    pct = count / len(settled) * 100
    lines.append(f"  {score:<12} {count:>6} {pct:>5.1f}%")

# ── PER EDGE TIER — the key section ───────────────────────────────────────
for eband in elabels:
    grp = settled[settled["edge_band"] == eband]
    if len(grp) == 0:
        continue

    w, t, wr, r = roi(grp)
    under = (grp["ht_total"] <= 1).sum()
    over  = (grp["ht_total"] >= 2).sum()

    lines.append(section(f"EDGE TIER: {eband}  ({t} picks)", width=70))
    lines.append(f"  Win rate : {wr:.1f}%   ROI : {r:+.1f}%")
    lines.append(f"  Under 1.5 HT : {under} ({under/t*100:.1f}%)  ← these are LOSSES")
    lines.append(f"  Over 1.5 HT  : {over}  ({over/t*100:.1f}%)  ← these are WINS")
    lines.append("")

    # HT score distribution for this tier
    dist = grp["ht_score"].value_counts().head(15)
    lines.append(f"  {'HT Score':<12} {'Count':>6} {'%':>6}  Outcome")
    lines.append(f"  {'-'*12} {'-'*6} {'-'*6}  -------")
    for score, count in dist.items():
        pct     = count / t * 100
        outcome = "OVER (WON)" if "-" in score and sum(
            int(x) for x in score.split("-") if x.isdigit()
        ) >= 2 else "UNDER (LOST)"
        lines.append(f"  {score:<12} {count:>6} {pct:>5.1f}%  {outcome}")

    lines.append("")
    lines.append(f"  KEY QUESTION: When edge is {eband}, HT Under 1.5 happens"
                 f" {under/t*100:.1f}% of the time.")

    if eband == "12%+" and under/t > 0.55:
        lines.append(f"  *** SIGNAL FOUND: Over 1.5 LOSES {under/t*100:.1f}% when edge is 12%+")
        lines.append(f"      This means UNDER 1.5 HT may be the real play when edge is 12%+")
    elif eband == "12%+" and under/t < 0.45:
        lines.append(f"  *** CONFIRMED: Over 1.5 WINS {over/t*100:.1f}% when edge is 12%+")
        lines.append(f"      Model is correctly identifying high-scoring first halves")

    # List every single pick in this tier
    lines.append("")
    lines.append(f"  ALL PICKS IN THIS TIER ({t} picks):")
    lines.append(
        f"  {'Date':<12} {'Match':<32} {'Edge':>6} {'Odds':>6} "
        f"{'HT':>5} {'Result':<8}"
    )
    lines.append(f"  {'-'*12} {'-'*32} {'-'*6} {'-'*6} {'-'*5} {'-'*8}")

    for _, row in grp.sort_values(["pick_date","edge_%"],
                                   ascending=[True, False]).iterrows():
        m = str(row.get("match","?"))[:31]
        lines.append(
            f"  {row['pick_date']:<12} {m:<32} "
            f"{row['edge_%']:>5.1f}% {row['decimal_odds']:>6.2f} "
            f"{row['ht_score']:>5} {row['result']:<8}"
        )

# ── SUMMARY COMPARISON ────────────────────────────────────────────────────
lines.append(section("SUMMARY — UNDER vs OVER RATE BY EDGE TIER", width=70))
lines.append(f"  {'Edge tier':<10} {'Picks':>6} {'Under%':>8} {'Over%':>8} "
             f"{'Win%':>7} {'ROI':>8}  Implication")
lines.append(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*8}  -----------")

for eband in elabels:
    grp = settled[settled["edge_band"] == eband]
    if len(grp) < 5:
        continue
    w, t, wr, r = roi(grp)
    under_pct = (grp["ht_total"] <= 1).sum() / t * 100
    over_pct  = (grp["ht_total"] >= 2).sum() / t * 100

    if under_pct > 60:
        impl = "Play UNDER 1.5 HT"
    elif over_pct > 60:
        impl = "Play OVER 1.5 HT"
    else:
        impl = "No clear signal"

    lines.append(
        f"  {eband:<10} {t:>6} {under_pct:>7.1f}% {over_pct:>7.1f}% "
        f"{wr:>6.1f}% {r:>+7.1f}%  {impl}"
    )

lines.append("\n" + "=" * 70)
lines.append(f"  END — {ts}")
lines.append("=" * 70)

report = "\n".join(lines)
print("\n" + report)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nSaved to {OUTPUT_FILE}")

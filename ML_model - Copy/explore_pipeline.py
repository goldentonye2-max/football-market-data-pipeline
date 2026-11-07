"""
explore_pipeline.py
===============================================================
Diagnostic script — shows the full shape of your picks pipeline
so we know exactly what to wire the gate into.

USAGE (inside ML_model - Copy):
    python explore_pipeline.py

Shows:
  - Every CSV file in this folder and its shape
  - The picks_enriched.csv column layout
  - Date range, targets, pick counts
  - How many picks per day (so we know when picks are generated)
  - Coverage of the five filter column groups
  - A sample of today's / most recent picks
===============================================================
"""

import os
import sys
import glob

import pandas as pd
import numpy as np

SEP  = "=" * 80
SEP2 = "-" * 80

# ── 1.  Every CSV in this folder ──────────────────────────────────────────────
print(f"\n{SEP}")
print("  CSV FILES IN THIS DIRECTORY")
print(SEP)
csvs = sorted(glob.glob("*.csv"))
if not csvs:
    print("  (none found)")
else:
    for f in csvs:
        try:
            df_tmp = pd.read_csv(f, nrows=0)
            size   = os.path.getsize(f) / 1_048_576
            print(f"  {f:<50}  {len(df_tmp.columns):>4} cols   {size:>7.1f} MB")
        except Exception as e:
            print(f"  {f:<50}  (unreadable: {e})")

# ── 2.  Load picks_enriched.csv ───────────────────────────────────────────────
enr = "picks_enriched.csv"
if not os.path.exists(enr):
    print(f"\nERROR: {enr} not found — run enrich_picks.py first.")
    sys.exit(1)

print(f"\n{SEP}")
print(f"  PICKS_ENRICHED.CSV — STRUCTURE")
print(SEP)
df = pd.read_csv(enr, low_memory=False)
print(f"  Rows : {len(df):,}")
print(f"  Cols : {len(df.columns)}")

# ── 3.  Column groups ─────────────────────────────────────────────────────────
meta_cols   = [c for c in df.columns if c.startswith("_") or c in
               ["fixture_id","game_id","home_team","away_team","kickoff",
                "target","market","result","won","decimal_odds","stake"]]
filter_cols = [c for c in df.columns if any(c.startswith(p) for p in
               ["m900036","m900037","m900038","m900052","m450004","m810002"])]
other_odds  = [c for c in df.columns if c not in meta_cols + filter_cols
               and c.startswith("m")]

print(f"\n  Meta / pick columns   : {len(meta_cols)}")
print(f"  Filter signal columns : {len(filter_cols)}")
print(f"  Other odds columns    : {len(other_odds)}")

print(f"\n  Meta columns found:")
for c in meta_cols:
    print(f"    {c}")

# ── 4.  Date range ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  DATE RANGE & PICK COUNTS")
print(SEP)
date_col = next((c for c in ["_pick_date","kickoff","date"] if c in df.columns), None)
if date_col:
    df["_date_tmp"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    date_counts = df.groupby("_date_tmp").size().sort_index()
    print(f"  Date column : {date_col}")
    print(f"  Earliest    : {date_counts.index[0]}")
    print(f"  Latest      : {date_counts.index[-1]}")
    print(f"  Total days  : {len(date_counts)}")
    print(f"\n  Picks per day (last 10 days):")
    for d, n in date_counts.tail(10).items():
        print(f"    {d}   {n:>5} picks")
else:
    print("  (no date column found)")

# ── 5.  Target breakdown ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TARGET / MARKET BREAKDOWN")
print(SEP)
target_col = "target" if "target" in df.columns else "market"
if target_col in df.columns:
    won_col = "_won" if "_won" in df.columns else "result"
    tc = df.groupby(target_col).agg(
        picks=("_won" if "_won" in df.columns else target_col, "count"),
    )
    if "_won" in df.columns:
        tc["wr_pct"] = df.groupby(target_col)["_won"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean() * 100)
    if "decimal_odds" in df.columns:
        tc["avg_odds"] = df.groupby(target_col)["decimal_odds"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean())
    print(tc.to_string())
else:
    print("  (no target column found)")

# ── 6.  Filter column coverage ────────────────────────────────────────────────
print(f"\n{SEP}")
print("  FILTER SIGNAL COVERAGE  (% of picks with odds available)")
print(SEP)
KEY_FILTERS = {
    "home_win": ["m900036_No","m900037_Yes","m900038_No","m900052_Yes"],
    "away_win": ["m900036_Yes","m900037_No","m900038_Yes","m900052_No"],
    "over_2_5": ["m450004_4","m450004_5+","m810002_3+"],
    "over_3_5": ["m450004_4","m450004_5+","m810002_3+","m18_total=4_5_Under_4_5"],
    "ht_over_1_5": ["m450004_4","m810002_3+"],
}
for tgt, cols in KEY_FILTERS.items():
    if target_col in df.columns:
        matched = [t for t in df[target_col].dropna().unique()
                   if tgt.lower() in str(t).lower()]
        sub = df[df[target_col].isin(matched)] if matched else df
    else:
        sub = df
    if len(sub) == 0:
        continue
    coverages = {}
    for col in cols:
        if col in sub.columns:
            cov = pd.to_numeric(sub[col], errors="coerce").notna().mean() * 100
            coverages[col] = cov
        else:
            coverages[col] = 0.0
    avg_cov = np.mean(list(coverages.values()))
    print(f"  {tgt}  (avg coverage {avg_cov:.0f}%)")
    for col, cov in coverages.items():
        bar = "█" * int(cov / 5)
        print(f"    {col:<35}  {cov:>5.1f}%  {bar}")

# ── 7.  Most recent picks sample ──────────────────────────────────────────────
print(f"\n{SEP}")
print("  MOST RECENT 10 PICKS  (key columns only)")
print(SEP)
show_cols = [c for c in meta_cols
             if c in df.columns and c not in ["_won"]][:10]
if date_col and date_col in df.columns:
    recent = df.sort_values(date_col, ascending=False).head(10)[show_cols]
else:
    recent = df.tail(10)[show_cols]
print(recent.to_string(index=False))

# ── 8.  P75 threshold preview (what the gate will use) ───────────────────────
print(f"\n{SEP}")
print("  P75 THRESHOLD PREVIEW  (from full dataset — gate will use training dates only)")
print(SEP)
GATE_FILTERS = {
    "home_win": [("m900036_No","high"),("m900037_Yes","high"),
                 ("m900038_No","high"),("m900052_Yes","high")],
    "away_win": [("m900036_Yes","high"),("m900037_No","high"),
                 ("m900038_Yes","high"),("m900052_No","high")],
    "over_2_5": [("m450004_4","high"),("m450004_5+","high"),("m810002_3+","high")],
    "over_3_5": [("m450004_4","high"),("m450004_5+","high"),
                 ("m810002_3+","high"),("m18_total=4_5_Under_4_5","high")],
    "ht_over_1_5": [("m450004_4","high"),("m810002_3+","high")],
}
for tgt, cols in GATE_FILTERS.items():
    print(f"  {tgt}")
    for col, direction in cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) >= 10:
                p50 = np.percentile(vals, 50)
                p75 = np.percentile(vals, 75)
                print(f"    {col:<35}  p50={p50:.2f}   p75={p75:.2f}")
            else:
                print(f"    {col:<35}  (insufficient data)")
        else:
            print(f"    {col:<35}  (column missing)")
    print()

print(SEP)
print("  SUMMARY — what the gate needs")
print(SEP)
print("  Gate reads picks_enriched.csv each day.")
print("  Filters to today's picks using the date column above.")
print("  Applies p75 thresholds to Win-to-Nil + Excluded Goals columns.")
print("  Outputs a BET / SKIP decision for each pick.")
print(SEP)

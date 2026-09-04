"""
=============================================================
  TRAINING TABLE DIAGNOSTIC
  Reads master_training_table.csv and shows full summary
  Also identifies additional targets available from
  flashscore_stats data in your database
=============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import os

CSV_PATH = "master_training_table.csv"
DB_PATH  = "sportybet.db"

# ── PART 1: Analyse the existing CSV ────────────────────────
print("\n" + "="*65)
print("  PART 1: MASTER TRAINING TABLE — FULL DIAGNOSTIC")
print("="*65)

if not os.path.exists(CSV_PATH):
    print(f"  ✗ CSV not found at: {CSV_PATH}")
    print("    Make sure you run this from the same folder as the CSV.")
    exit()

df = pd.read_csv(CSV_PATH, low_memory=False)

# Column categories
feature_true    = [c for c in df.columns if c.startswith("true_prob__")]
feature_implied = [c for c in df.columns if c.startswith("implied_prob__")]
feature_vig     = [c for c in df.columns if c.startswith("vig_delta__")]
feature_cross   = [c for c in df.columns if c.startswith("cross__")]
target_cols     = [c for c in df.columns if c.startswith("target__")]
stat_cols       = [c for c in df.columns if c.startswith(("home_stat__", "away_stat__"))]
meta_cols       = [c for c in df.columns if not any(c.startswith(p) for p in
                   ["true_prob__","implied_prob__","vig_delta__","cross__","target__","home_stat__","away_stat__"])]

print(f"\n  SHAPE")
print(f"  {'Total fixtures (rows):':<35} {len(df):>6,}")
print(f"  {'Total columns:':<35} {len(df.columns):>6,}")

print(f"\n  FEATURE COLUMNS")
print(f"  {'true_prob__* (backend prob):':<35} {len(feature_true):>6,}")
print(f"  {'implied_prob__* (1/odds):':<35} {len(feature_implied):>6,}")
print(f"  {'vig_delta__* (margin signal):':<35} {len(feature_vig):>6,}")
print(f"  {'cross__* (engineered):':<35} {len(feature_cross):>6,}")

print(f"\n  TARGET COLUMNS          : {len(target_cols)}")
print(f"  STAT COLUMNS (reference): {len(stat_cols)}")
print(f"  META COLUMNS            : {len(meta_cols)}")

# Missing data in features
if feature_true:
    avg_missing = df[feature_true].isna().mean().mean() * 100
    print(f"\n  FEATURE DATA QUALITY")
    print(f"  {'Avg missing across feature cols:':<35} {avg_missing:>5.1f}%")
    sparse = [c for c in feature_true if df[c].isna().mean() > 0.5]
    print(f"  {'Feature cols >50% empty:':<35} {len(sparse):>5,}")

# Target hit rates
print(f"\n  TARGET HIT RATES")
print(f"  {'TARGET COLUMN':<45} {'MATCHES':>8}  {'HIT%':>6}  {'STATUS'}")
print(f"  {'-'*72}")

GOOD_TARGETS = []
SPARSE_TARGETS = []
SKIP_TARGETS = []

for t in sorted(target_cols):
    valid = int(df[t].notna().sum())
    if valid == 0:
        print(f"  {t:<45} {'NO DATA':>8}")
        SKIP_TARGETS.append(t)
        continue
    hits = int(df[t].sum())
    pct  = hits / valid * 100

    # Categorise
    if pct < 10 or pct > 90:
        status = "⚠ SKIP (too rare/common)"
        SKIP_TARGETS.append(t)
    elif valid < 200:
        status = "⚠ SPARSE (few matches)"
        SPARSE_TARGETS.append(t)
    elif 30 <= pct <= 70:
        status = "✓ IDEAL"
        GOOD_TARGETS.append(t)
    else:
        status = "✓ USABLE"
        GOOD_TARGETS.append(t)

    print(f"  {t:<45} {valid:>8,}  {pct:>5.1f}%  {status}")

print(f"\n  TRAINING RECOMMENDATION")
print(f"  {'Ideal/Usable targets (train these):':<40} {len(GOOD_TARGETS)}")
print(f"  {'Sparse targets (train with caution):':<40} {len(SPARSE_TARGETS)}")
print(f"  {'Skip targets:':<40} {len(SKIP_TARGETS)}")

print(f"\n  TARGETS TO TRAIN FIRST:")
for t in GOOD_TARGETS:
    print(f"    → {t}")


# ── PART 2: Additional targets from flashscore_stats ────────
print("\n" + "="*65)
print("  PART 2: ADDITIONAL TARGETS FROM FLASHSCORE STATS")
print("="*65)

if not os.path.exists(DB_PATH):
    print(f"  DB not found at {DB_PATH} — skipping Part 2")
else:
    conn = sqlite3.connect(DB_PATH)

    # Load stats with numeric values
    fs = pd.read_sql_query("""
        SELECT event_id, stat_name, section, home_value, away_value
        FROM flashscore_stats
        WHERE event_id IS NOT NULL
    """, conn)

    def to_num(v):
        try:
            return float(str(v).replace("%","").strip().split(":")[0])
        except:
            return np.nan

    fs["home_num"] = fs["home_value"].apply(to_num)
    fs["away_num"] = fs["away_value"].apply(to_num)
    fs["total"]    = fs["home_num"] + fs["away_num"]

    # Targets to explore — stat / section / lines
    STAT_TARGETS = [
        # (stat_name,        section,   lines,                  total_or_home)
        ("Corner kicks",     "match",   [7.5, 8.5, 9.5, 10.5, 11.5], "total"),
        ("Corner kicks",     "1st_half",[3.5, 4.5, 5.5],             "total"),
        ("Corner kicks",     "2nd_half",[3.5, 4.5, 5.5],             "total"),
        ("Shots on target",  "match",   [4.5, 5.5, 6.5, 7.5, 8.5],  "total"),
        ("Total shots",      "match",   [18.5, 20.5, 22.5, 24.5],    "total"),
        ("Yellow cards",     "match",   [1.5, 2.5, 3.5, 4.5],        "total"),
        ("Yellow cards",     "1st_half",[0.5, 1.5],                   "total"),
        ("Ball possession",  "match",   [50],                         "home"),
        ("Expected goals (xG)","match", [1.5, 2.0, 2.5, 3.0],        "total"),
        ("Big chances",      "match",   [2.5, 3.5, 4.5],              "total"),
        ("Fouls",            "match",   [18.5, 20.5, 22.5],           "total"),
        ("Offsides",         "match",   [1.5, 2.5, 3.5],              "total"),
        ("Throws ins",       "match",   [25.5, 28.5, 31.5],           "total"),
        ("Passes",           "match",   [700, 800, 900],               "total"),
        ("Shots inside the box","match",[10.5, 12.5, 14.5],           "total"),
        ("Goalkeeper saves", "match",   [3.5, 4.5, 5.5],              "total"),
    ]

    print(f"\n  {'NEW TARGET':<55} {'MATCHES':>8}  {'HIT%':>6}  {'VIABLE?'}")
    print(f"  {'-'*80}")

    new_viable = []

    for stat_name, section, lines, mode in STAT_TARGETS:
        sub = fs[(fs["stat_name"] == stat_name) & (fs["section"] == section)].copy()
        if sub.empty:
            continue

        for line in lines:
            col_safe  = stat_name.lower().replace(" ","_").replace("(","").replace(")","")
            sec_short = section.replace("st_half","h1").replace("nd_half","h2")
            tgt_name  = f"target__{col_safe}__{sec_short}__over_{str(line).replace('.','_')}"

            if mode == "total":
                hits_series = sub["total"]
            else:
                hits_series = sub["home_num"]

            valid = hits_series.notna().sum()
            if valid == 0:
                continue

            hits = (hits_series > line).sum()
            pct  = hits / valid * 100

            viable = "✓ VIABLE" if 20 <= pct <= 80 and valid >= 150 else "⚠ skip"
            if viable == "✓ VIABLE":
                new_viable.append((tgt_name, stat_name, section, line, mode, int(valid), round(pct,1)))

            print(f"  {tgt_name:<55} {int(valid):>8,}  {pct:>5.1f}%  {viable}")

    print(f"\n  SUMMARY: {len(new_viable)} additional viable targets found")
    print(f"  These can be added to your training table with one script update.\n")

    conn.close()

print("="*65)
print("  DIAGNOSTIC COMPLETE")
print("="*65 + "\n")

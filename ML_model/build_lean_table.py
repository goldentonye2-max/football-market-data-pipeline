"""
=============================================================
  LEAN TRAINING TABLE BUILDER
  Fixes the 147,982-column explosion problem
  + Adds 42 new stat-based targets
  + Outputs a clean, XGBoost-ready CSV
=============================================================
  PROBLEM SOLVED:
    Original table had 147,982 columns because every unique
    market/specifier/outcome across all leagues created its
    own column — most were empty for most fixtures.

  SOLUTION:
    Keep only feature columns present in >= MIN_COVERAGE% of
    fixtures. These are the universal core markets that exist
    across all leagues. Everything else is noise.
=============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import os

# ── CONFIG ───────────────────────────────────────────────────
CSV_PATH    = "master_training_table.csv"
DB_PATH     = "sportybet.db"
OUTPUT_CSV  = "lean_training_table.csv"

# Only keep feature columns present in this % of fixtures
MIN_COVERAGE = 0.50   # 50% — adjust down to 0.30 if you want more features

# ─────────────────────────────────────────────────────────────


def to_num(v):
    """Safely parse stat values."""
    try:
        return float(str(v).replace("%", "").strip().split(":")[0])
    except:
        return np.nan


def clean_col(text):
    if pd.isna(text) or text is None:
        return "none"
    return (
        str(text).strip().lower()
        .replace(" ", "_").replace("/", "_").replace("-", "_")
        .replace(".", "_").replace("(", "").replace(")", "")
        .replace(":", "").replace("'", "")
        .replace(">", "over").replace("<", "under")
    )


print("\n" + "="*65)
print("  LEAN TRAINING TABLE BUILDER")
print("="*65)

# ══════════════════════════════════════════════════════════════
# STEP 1 — Load existing CSV
# ══════════════════════════════════════════════════════════════
print("\n[1/6] Loading master_training_table.csv...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"    → Loaded {len(df):,} rows × {len(df.columns):,} columns")

# ══════════════════════════════════════════════════════════════
# STEP 2 — Separate columns by type
# ══════════════════════════════════════════════════════════════
print(f"\n[2/6] Separating column types...")

meta_cols    = [c for c in df.columns if not any(c.startswith(p) for p in
                ["true_prob__","implied_prob__","vig_delta__","cross__",
                 "target__","home_stat__","away_stat__"])]
feature_cols = [c for c in df.columns if c.startswith(
                ("true_prob__","implied_prob__","vig_delta__","cross__"))]
target_cols  = [c for c in df.columns if c.startswith("target__")]
stat_cols    = [c for c in df.columns if c.startswith(("home_stat__","away_stat__"))]

print(f"    Meta columns    : {len(meta_cols)}")
print(f"    Feature columns : {len(feature_cols):,}")
print(f"    Target columns  : {len(target_cols)}")
print(f"    Stat columns    : {len(stat_cols)}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — Filter features by coverage threshold
# ══════════════════════════════════════════════════════════════
print(f"\n[3/6] Filtering features (keeping ≥{MIN_COVERAGE*100:.0f}% coverage)...")

n_rows = len(df)
coverage = df[feature_cols].notna().mean()

keep_features = coverage[coverage >= MIN_COVERAGE].index.tolist()
drop_features = coverage[coverage <  MIN_COVERAGE].index.tolist()

print(f"    Features kept   : {len(keep_features):,}")
print(f"    Features dropped: {len(drop_features):,}")

if len(keep_features) == 0:
    print(f"\n    ⚠ No features survived {MIN_COVERAGE*100:.0f}% threshold.")
    print(f"    Lowering to 10% threshold to find any universal markets...")
    MIN_COVERAGE = 0.10
    keep_features = coverage[coverage >= MIN_COVERAGE].index.tolist()
    print(f"    Features at 10%: {len(keep_features):,}")

# Show what the surviving universal markets are
print(f"\n    UNIVERSAL CORE MARKETS (present in ≥{MIN_COVERAGE*100:.0f}% of fixtures):")
surviving_true = [c for c in keep_features if c.startswith("true_prob__")]
if surviving_true:
    for c in sorted(surviving_true)[:50]:  # show up to 50
        cov_pct = coverage[c] * 100
        print(f"      {c:<70} {cov_pct:>5.1f}%")
    if len(surviving_true) > 50:
        print(f"      ... and {len(surviving_true)-50} more")
else:
    print("      (none found at this threshold)")

# ══════════════════════════════════════════════════════════════
# STEP 4 — Build lean base table
# ══════════════════════════════════════════════════════════════
print(f"\n[4/6] Building lean base table...")

keep_cols = meta_cols + keep_features + target_cols + stat_cols
lean = df[keep_cols].copy()
print(f"    → Shape: {lean.shape[0]:,} rows × {lean.shape[1]:,} columns")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Add 42 new stat-based targets from flashscore_stats
# ══════════════════════════════════════════════════════════════
print(f"\n[5/6] Adding new stat-based targets from FlashScore...")

if not os.path.exists(DB_PATH):
    print(f"    ⚠ DB not found at {DB_PATH} — skipping new targets")
else:
    conn = sqlite3.connect(DB_PATH)

    fs = pd.read_sql_query("""
        SELECT event_id, stat_name, section, home_value, away_value
        FROM flashscore_stats
        WHERE event_id IS NOT NULL
    """, conn)

    fs["home_num"] = fs["home_value"].apply(to_num)
    fs["away_num"] = fs["away_value"].apply(to_num)
    fs["total"]    = fs["home_num"] + fs["away_num"]

    # All viable new targets from diagnostic output
    NEW_TARGETS = [
        # (stat_name,                 section,    line, mode,    col_suffix)
        ("Corner kicks",              "match",     7.5,  "total", "corners__match__over_7_5"),
        ("Corner kicks",              "match",     8.5,  "total", "corners__match__over_8_5"),
        ("Corner kicks",              "match",     9.5,  "total", "corners__match__over_9_5"),
        ("Corner kicks",              "match",    10.5,  "total", "corners__match__over_10_5"),
        ("Corner kicks",              "match",    11.5,  "total", "corners__match__over_11_5"),
        ("Corner kicks",              "1st_half",  3.5,  "total", "corners__h1__over_3_5"),
        ("Corner kicks",              "1st_half",  4.5,  "total", "corners__h1__over_4_5"),
        ("Corner kicks",              "1st_half",  5.5,  "total", "corners__h1__over_5_5"),
        ("Corner kicks",              "2nd_half",  3.5,  "total", "corners__h2__over_3_5"),
        ("Corner kicks",              "2nd_half",  4.5,  "total", "corners__h2__over_4_5"),
        ("Corner kicks",              "2nd_half",  5.5,  "total", "corners__h2__over_5_5"),
        ("Shots on target",           "match",     5.5,  "total", "shots_on_target__match__over_5_5"),
        ("Shots on target",           "match",     6.5,  "total", "shots_on_target__match__over_6_5"),
        ("Shots on target",           "match",     7.5,  "total", "shots_on_target__match__over_7_5"),
        ("Shots on target",           "match",     8.5,  "total", "shots_on_target__match__over_8_5"),
        ("Total shots",               "match",    18.5,  "total", "total_shots__match__over_18_5"),
        ("Total shots",               "match",    20.5,  "total", "total_shots__match__over_20_5"),
        ("Total shots",               "match",    22.5,  "total", "total_shots__match__over_22_5"),
        ("Total shots",               "match",    24.5,  "total", "total_shots__match__over_24_5"),
        ("Yellow cards",              "match",     2.5,  "total", "yellow_cards__match__over_2_5"),
        ("Yellow cards",              "match",     3.5,  "total", "yellow_cards__match__over_3_5"),
        ("Yellow cards",              "match",     4.5,  "total", "yellow_cards__match__over_4_5"),
        ("Yellow cards",              "1st_half",  1.5,  "total", "yellow_cards__h1__over_1_5"),
        ("Ball possession",           "match",    50.0,  "home",  "home_possession__match__over_50"),
        ("Expected goals (xG)",       "match",     2.0,  "total", "xg__match__over_2_0"),
        ("Expected goals (xG)",       "match",     2.5,  "total", "xg__match__over_2_5"),
        ("Expected goals (xG)",       "match",     3.0,  "total", "xg__match__over_3_0"),
        ("Big chances",               "match",     2.5,  "total", "big_chances__match__over_2_5"),
        ("Big chances",               "match",     3.5,  "total", "big_chances__match__over_3_5"),
        ("Big chances",               "match",     4.5,  "total", "big_chances__match__over_4_5"),
        ("Fouls",                     "match",    18.5,  "total", "fouls__match__over_18_5"),
        ("Fouls",                     "match",    20.5,  "total", "fouls__match__over_20_5"),
        ("Fouls",                     "match",    22.5,  "total", "fouls__match__over_22_5"),
        ("Offsides",                  "match",     1.5,  "total", "offsides__match__over_1_5"),
        ("Offsides",                  "match",     2.5,  "total", "offsides__match__over_2_5"),
        ("Offsides",                  "match",     3.5,  "total", "offsides__match__over_3_5"),
        ("Shots inside the box",      "match",    10.5,  "total", "shots_in_box__match__over_10_5"),
        ("Shots inside the box",      "match",    12.5,  "total", "shots_in_box__match__over_12_5"),
        ("Shots inside the box",      "match",    14.5,  "total", "shots_in_box__match__over_14_5"),
        ("Goalkeeper saves",          "match",     3.5,  "total", "gk_saves__match__over_3_5"),
        ("Goalkeeper saves",          "match",     4.5,  "total", "gk_saves__match__over_4_5"),
        ("Goalkeeper saves",          "match",     5.5,  "total", "gk_saves__match__over_5_5"),
    ]

    added = 0
    for stat_name, section, line, mode, suffix in NEW_TARGETS:
        col_name = f"target__new__{suffix}"

        sub = fs[(fs["stat_name"] == stat_name) & (fs["section"] == section)].copy()
        if sub.empty:
            continue

        # Use groupby+mean to safely handle duplicate event_id rows
        # (some matches have multiple stat entries for the same stat/section)
        if mode == "total":
            val_series = sub.groupby("event_id")["total"].mean()
        else:
            val_series = sub.groupby("event_id")["home_num"].mean()

        # Map to lean table
        mapped = lean["event_id"].map(val_series)

        # CRITICAL: preserve NaN for matches with no stat data.
        # (NaN > line) evaluates to False in pandas, not NA.
        # Using .where() keeps NaN as NA instead of converting to 0.
        result = (mapped > line).where(mapped.notna())
        lean[col_name] = result.astype("Int64")  # NA-safe nullable int

        added += 1

    conn.close()
    print(f"    → {added} new target columns added")

# ══════════════════════════════════════════════════════════════
# STEP 6 — Save and report
# ══════════════════════════════════════════════════════════════
print(f"\n[6/6] Saving lean_training_table.csv...")
lean.to_csv(OUTPUT_CSV, index=False)

# Final column audit
all_targets    = [c for c in lean.columns if c.startswith("target__")]
all_features   = [c for c in lean.columns if c.startswith(
                  ("true_prob__","implied_prob__","vig_delta__","cross__"))]

print(f"\n" + "="*65)
print(f"  LEAN TABLE — FINAL SUMMARY")
print(f"="*65)
print(f"  Fixtures (rows)   : {len(lean):,}")
print(f"  Total columns     : {len(lean.columns):,}")
print(f"  Feature columns   : {len(all_features):,}")
print(f"  Target columns    : {len(all_targets):,}")
print(f"    - Original      : {len(target_cols):,}")
print(f"    - New (stats)   : {added}")
print(f"  Avg feature coverage: {lean[all_features].notna().mean().mean()*100:.1f}%")
print(f"\n  Saved to: {OUTPUT_CSV}")

# Show all trainable targets with hit rates
print(f"\n  ALL TRAINABLE TARGETS:")
print(f"  {'TARGET':<55} {'N':>6}  {'HIT%':>6}")
print(f"  {'-'*70}")
for t in sorted(all_targets):
    vals  = lean[t].dropna()
    n     = len(vals)
    if n == 0:
        continue
    pct   = float(vals.sum()) / n * 100
    flag  = "" if 15 <= pct <= 85 else "  ⚠"
    print(f"  {t:<55} {n:>6,}  {pct:>5.1f}%{flag}")

print(f"\n  ✅ Ready for XGBoost training.")
print("="*65 + "\n")
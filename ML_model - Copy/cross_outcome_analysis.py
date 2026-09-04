"""
=============================================================
  cross_outcome_analysis.py
  IDEA 1 — Cross-outcome pattern discovery

  For every market your models predict, this script asks:
  "When this prediction WINS vs LOSES — what do corners,
   cards, goals, BTTS, and other outcomes look like?"

  If Over 2.5 WON fixtures have corners > 9.5 at 68%
  vs a base rate of 44%, that's a derived corner signal
  without ever training a corner model.

  USAGE:
    python cross_outcome_analysis.py

  REQUIRES:
    - weekly_graded.csv  (from weekly_summary.py)
      OR individual all_picks_YYYY-MM-DD_graded.csv files
    - sportybet.db (one level up)
=============================================================
"""

import sqlite3
import glob
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────
DB_PATH       = r"..\sportybet.db"
OUTPUT_FILE   = "cross_outcome_report.txt"
MIN_SAMPLE    = 30      # minimum fixtures in a group to report
DELTA_FLAG    = 0.08    # flag deviations ≥ 8% from base rate as notable
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

# ── LOAD GRADED PICKS ─────────────────────────────────────────────────────
print("Loading graded picks...")

graded = None
if os.path.exists("weekly_graded.csv"):
    graded = pd.read_csv("weekly_graded.csv")
    print(f"  Loaded weekly_graded.csv  → {len(graded)} rows")
else:
    files = sorted(glob.glob("all_picks_????-??-??_graded.csv"))
    if files:
        frames = []
        for f in files:
            tmp = pd.read_csv(f)
            tmp["pick_date"] = os.path.basename(f)[10:20]
            frames.append(tmp)
        graded = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {len(files)} graded files → {len(graded)} rows")
    else:
        sys.exit(
            "ERROR: No graded picks found.\n"
            "Run weekly_summary.py first to produce weekly_graded.csv,\n"
            "or run evaluate_picks.py on each day to produce graded CSVs."
        )

# Keep only settled picks
settled = graded[graded["result"].isin(["WON", "LOST"])].copy()
print(f"  Settled picks: {len(settled)}")

if len(settled) == 0:
    sys.exit("No settled picks to analyse.")

# ── LOAD RESULTS FROM DB ──────────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    sys.exit(f"ERROR: Database not found at {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

event_ids = settled["event_id"].unique().tolist()
placeholders = ",".join(f"'{e}'" for e in event_ids)

results = pd.read_sql(f"""
    SELECT event_id,
           ht_home, ht_away,
           ft_home, ft_away
    FROM results
    WHERE event_id IN ({placeholders})
      AND ft_home IS NOT NULL
""", conn)
print(f"  Results matched: {len(results)} fixtures")

# ── LOAD FLASHSCORE STATS ─────────────────────────────────────────────────
# Join via match_links to get fs_match_id, then pull stats
links = pd.read_sql(f"""
    SELECT event_id, fs_match_id
    FROM match_links
    WHERE event_id IN ({placeholders})
      AND stats_fetched = 1
""", conn)

stats_raw = pd.DataFrame()
if len(links) > 0:
    fs_ids = links["fs_match_id"].unique().tolist()
    fs_placeholders = ",".join(f"'{f}'" for f in fs_ids)
    stats_raw = pd.read_sql(f"""
        SELECT fs_match_id, stat_name, home_value, away_value
        FROM flashscore_stats
        WHERE fs_match_id IN ({fs_placeholders})
          AND stat_name IN (
              'Corner Kicks',
              'Yellow Cards',
              'Red Cards',
              'Total Shots',
              'Shots on Target',
              'Ball possession'
          )
    """, conn)

conn.close()
print(f"  FlashScore stat rows: {len(stats_raw)}")

# ── PARSE STATS INTO WIDE FORMAT ──────────────────────────────────────────
def parse_num(val):
    try:
        return float(str(val).replace("%", "").strip())
    except:
        return np.nan

if len(stats_raw) > 0:
    stats_raw["home_val"] = stats_raw["home_value"].apply(parse_num)
    stats_raw["away_val"] = stats_raw["away_value"].apply(parse_num)
    stats_raw["total"]    = stats_raw["home_val"] + stats_raw["away_val"]

    stats_wide = stats_raw.pivot_table(
        index="fs_match_id", columns="stat_name",
        values="total", aggfunc="first"
    ).reset_index()
    stats_wide.columns.name = None
    stats_wide.columns = (
        ["fs_match_id"] +
        [c.lower().replace(" ", "_") for c in stats_wide.columns[1:]]
    )
    stats_wide = links.merge(stats_wide, on="fs_match_id", how="left")
else:
    stats_wide = links.copy()

# ── BUILD MASTER ANALYSIS TABLE ───────────────────────────────────────────
# results columns may already be in the graded CSV — only merge if missing
if "ft_home" not in settled.columns:
    df = settled.merge(results, on="event_id", how="inner")
else:
    df = settled.copy()
    df = df[df["ft_home"].notna()]   # drop rows with no score

df = df.merge(stats_wide, on="event_id", how="left")

# Derived outcomes from scores
df["ft_goals"]   = df["ft_home"] + df["ft_away"]
df["ht_goals"]   = df["ht_home"] + df["ht_away"]
df["btts"]       = ((df["ft_home"] > 0) & (df["ft_away"] > 0)).astype(float)
df["home_cs"]    = (df["ft_away"] == 0).astype(float)
df["away_cs"]    = (df["ft_home"] == 0).astype(float)

print(f"  Analysis table: {len(df)} rows, "
      f"{df['corner_kicks'].notna().sum()} with corner data\n")

# ── DEFINE WHAT TO CHECK ──────────────────────────────────────────────────
# (stat_column, threshold, label)
CHECKS = [
    # Goals
    ("ft_goals",      1.5, "FT Goals Over 1.5"),
    ("ft_goals",      2.5, "FT Goals Over 2.5"),
    ("ft_goals",      3.5, "FT Goals Over 3.5"),
    ("ft_goals",      4.5, "FT Goals Over 4.5"),
    # HT Goals
    ("ht_goals",      0.5, "HT Goals Over 0.5"),
    ("ht_goals",      1.5, "HT Goals Over 1.5"),
    # BTTS / Clean sheets
    ("btts",          0.5, "BTTS Yes"),
    ("home_cs",       0.5, "Home Clean Sheet"),
    ("away_cs",       0.5, "Away Clean Sheet"),
    # Corners
    ("corner_kicks",  7.5, "Corners Over 7.5"),
    ("corner_kicks",  9.5, "Corners Over 9.5"),
    ("corner_kicks", 11.5, "Corners Over 11.5"),
    # Cards
    ("yellow_cards",  1.5, "Yellow Cards Over 1.5"),
    ("yellow_cards",  2.5, "Yellow Cards Over 2.5"),
    ("yellow_cards",  3.5, "Yellow Cards Over 3.5"),
    # Shots on target
    ("shots_on_target", 4.5, "Shots on Target Over 4.5"),
    ("shots_on_target", 6.5, "Shots on Target Over 6.5"),
    ("shots_on_target", 8.5, "Shots on Target Over 8.5"),
]

# ── BASE RATES (across all settled fixtures) ──────────────────────────────
base_rates = {}
base_n     = {}
for col, thresh, label in CHECKS:
    if col not in df.columns:
        continue
    sub = df[col].dropna()
    if len(sub) >= MIN_SAMPLE:
        base_rates[label] = (sub > thresh).mean()
        base_n[label]     = len(sub)

# ── PER-TARGET ANALYSIS ───────────────────────────────────────────────────
lines = []
ts    = datetime.now().strftime("%Y-%m-%d %H:%M")

lines.append("=" * 70)
lines.append("  CROSS-OUTCOME PATTERN ANALYSIS")
lines.append(f"  Generated : {ts}")
lines.append(f"  Fixtures  : {df['event_id'].nunique()} unique")
lines.append(f"  Settled picks: {len(settled)}")
lines.append(f"  ★ = deviation ≥ {DELTA_FLAG*100:.0f}% from base rate (notable)")
lines.append(f"  ✦ = deviation ≥ {DELTA_FLAG*2*100:.0f}% from base rate (strong)")
lines.append("=" * 70)

# Summary of strongest findings across all targets
all_findings = []

for target in sorted(settled["target"].unique()):
    tname  = DISPLAY.get(target, target.replace("target__", ""))
    sub    = df[df["target"] == target]
    won    = sub[sub["result"] == "WON"]
    lost   = sub[sub["result"] == "LOST"]

    if len(won) < MIN_SAMPLE or len(lost) < MIN_SAMPLE:
        continue

    lines.append("")
    lines.append(f"{'─'*70}")
    lines.append(f"  TARGET: {tname}")
    lines.append(f"  Total predicted: {len(sub)}   Won: {len(won)} "
                 f"({len(won)/len(sub)*100:.1f}%)   Lost: {len(lost)}")
    lines.append(f"{'─'*70}")
    lines.append(
        f"  {'Outcome':<30} {'Base N':>6} {'Base':>6} "
        f"{'Won':>6} {'Lost':>6} {'ΔWon':>8} {'ΔLost':>8}"
    )
    lines.append(
        f"  {'-'*30} {'-'*6} {'-'*6} "
        f"{'-'*6} {'-'*6} {'-'*8} {'-'*8}"
    )

    target_findings = []

    for col, thresh, label in CHECKS:
        if col not in df.columns:
            continue
        if label not in base_rates:
            continue

        base = base_rates[label]
        bn   = base_n[label]

        won_sub  = won[col].dropna()
        lost_sub = lost[col].dropna()

        if len(won_sub) < 10 or len(lost_sub) < 10:
            continue

        w_rate = (won_sub  > thresh).mean()
        l_rate = (lost_sub > thresh).mean()
        dw     = w_rate - base
        dl     = l_rate - base

        # Flag strong deviations
        flag = ""
        if abs(dw) >= DELTA_FLAG * 2:
            flag = " ✦"
        elif abs(dw) >= DELTA_FLAG:
            flag = " ★"

        lines.append(
            f"  {label:<30} {bn:>6} {base:>5.1%} "
            f"{w_rate:>5.1%} {l_rate:>5.1%} "
            f"{dw:>+7.1%} {dl:>+7.1%}{flag}"
        )

        if abs(dw) >= DELTA_FLAG:
            direction = "MORE likely" if dw > 0 else "LESS likely"
            target_findings.append({
                "target"   : tname,
                "outcome"  : label,
                "base"     : base,
                "won_rate" : w_rate,
                "lost_rate": l_rate,
                "delta_won": dw,
                "direction": direction,
                "n_won"    : len(won_sub),
            })
            all_findings.append(target_findings[-1])

    # Plain-English summary for this target
    if target_findings:
        lines.append("")
        lines.append(f"  KEY FINDINGS for {tname}:")
        for f in sorted(target_findings, key=lambda x: abs(x["delta_won"]), reverse=True)[:5]:
            lines.append(
                f"    → When {tname} WINS, {f['outcome']} is "
                f"{f['direction']} ({f['won_rate']:.1%} vs base {f['base']:.1%}, "
                f"Δ{f['delta_won']:+.1%})"
            )
    else:
        lines.append("")
        lines.append(f"  No notable cross-outcome patterns found for {tname}.")

# ── GLOBAL STRONGEST SIGNALS ──────────────────────────────────────────────
lines.append("")
lines.append("=" * 70)
lines.append("  STRONGEST SIGNALS ACROSS ALL TARGETS")
lines.append("  (These are your best derived prediction opportunities)")
lines.append("=" * 70)

if all_findings:
    findings_df = pd.DataFrame(all_findings)
    findings_df = findings_df.sort_values("delta_won",
                                          key=abs, ascending=False)
    lines.append(
        f"\n  {'Target':<25} {'Outcome':<28} {'Base':>6} "
        f"{'WonRate':>8} {'Delta':>8}"
    )
    lines.append(
        f"  {'-'*25} {'-'*28} {'-'*6} {'-'*8} {'-'*8}"
    )
    for _, row in findings_df.head(20).iterrows():
        lines.append(
            f"  {row['target']:<25} {row['outcome']:<28} "
            f"{row['base']:>5.1%} {row['won_rate']:>7.1%} "
            f"{row['delta_won']:>+7.1%}"
        )

    # Idea 2 input: which derived markets appear most consistently
    lines.append("")
    lines.append("=" * 70)
    lines.append("  DERIVED MARKET OPPORTUNITIES — INPUT FOR IDEA 2")
    lines.append("  (Markets worth targeting via derived prediction)")
    lines.append("=" * 70)

    outcome_counts = findings_df.groupby("outcome").agg(
        times_flagged = ("target", "count"),
        avg_delta     = ("delta_won", lambda x: x.abs().mean()),
        max_delta     = ("delta_won", lambda x: x.abs().max()),
    ).sort_values("avg_delta", ascending=False)

    lines.append(
        f"\n  {'Outcome':<30} {'Times flagged':>14} "
        f"{'Avg |Δ|':>8} {'Max |Δ|':>8}"
    )
    lines.append(
        f"  {'-'*30} {'-'*14} {'-'*8} {'-'*8}"
    )
    for outcome, row in outcome_counts.iterrows():
        lines.append(
            f"  {outcome:<30} {row['times_flagged']:>14} "
            f"{row['avg_delta']:>7.1%} {row['max_delta']:>7.1%}"
        )
else:
    lines.append("\n  No significant cross-outcome patterns found yet.")
    lines.append("  Try running with more days of graded data (aim for 7+).")

lines.append("")
lines.append("=" * 70)
lines.append(f"  END OF REPORT — {ts}")
lines.append("=" * 70)

# ── PRINT + SAVE ──────────────────────────────────────────────────────────
report = "\n".join(lines)
print(report)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nReport saved to {OUTPUT_FILE}")

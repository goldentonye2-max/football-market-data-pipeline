"""
=============================================================
  evaluate_picks.py
  Grades every pick in a predictions CSV against actual results
  from sportybet.db and prints a full breakdown.

  Usage:
      python evaluate_picks.py

  Config (edit the two paths below if needed):
      PICKS_CSV  — path to your all_picks_today_v2.csv (or any picks CSV)
      DB_PATH    — path to sportybet.db
=============================================================
"""

import sqlite3
import pandas as pd
import os
import sys

# ── CONFIG ────────────────────────────────────────────────────────────────
PICKS_CSV = r"ML_model -x    Copy\all_picks_today_v2.csv"
DB_PATH   = "sportybet.db"

# ─────────────────────────────────────────────────────────────────────────

# ── RESOLUTION RULES ──────────────────────────────────────────────────────
# Each lambda receives a row with: ft_home, ft_away, ht_home, ht_away
# Returns 1 if the pick WON, 0 if LOST
RESOLVERS = {
    "target__home_win":          lambda r: int(r.ft_home > r.ft_away),
    "target__away_win":          lambda r: int(r.ft_away > r.ft_home),
    "target__over_2_5_goals":    lambda r: int(r.ft_home + r.ft_away > 2),
    "target__over_3_5_goals":    lambda r: int(r.ft_home + r.ft_away > 3),
    "target__over_4_5_goals":    lambda r: int(r.ft_home + r.ft_away > 4),
    "target__ht_home_win":       lambda r: int(r.ht_home > r.ht_away),
    "target__ht_away_win":       lambda r: int(r.ht_away > r.ht_home),
    "target__ht_over_1_5_goals": lambda r: int(r.ht_home + r.ht_away > 1),
    "target__home_clean_sheet":  lambda r: int(r.ft_away == 0),
    "target__away_clean_sheet":  lambda r: int(r.ft_home == 0),
}

# Friendly display name for each target
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

# ── LOAD ──────────────────────────────────────────────────────────────────
if not os.path.exists(PICKS_CSV):
    sys.exit(f"ERROR: picks file not found: {PICKS_CSV}")
if not os.path.exists(DB_PATH):
    sys.exit(f"ERROR: database not found: {DB_PATH}")

picks = pd.read_csv(PICKS_CSV)
print(f"Loaded {len(picks)} picks across {picks['event_id'].nunique()} fixtures.")

conn   = sqlite3.connect(DB_PATH)
results = pd.read_sql("""
    SELECT event_id,
           ht_home, ht_away,
           ft_home, ft_away,
           status
    FROM results
    WHERE event_id IN ({})
""".format(",".join(f"'{e}'" for e in picks["event_id"].unique())), conn)
conn.close()

print(f"Matched {len(results)} fixtures in results table "
      f"({results[results['status']=='finished'].shape[0]} finished).")

# ── MERGE & RESOLVE ───────────────────────────────────────────────────────
df = picks.merge(results, on="event_id", how="left")

def resolve(row):
    if pd.isna(row.get("status")) or row["status"] != "finished":
        return "UNSETTLED"
    if pd.isna(row["ft_home"]) or pd.isna(row["ft_away"]):
        return "UNSETTLED"
    fn = RESOLVERS.get(row["target"])
    if fn is None:
        return f"UNKNOWN_TARGET"
    try:
        return "WON" if fn(row) == 1 else "LOST"
    except Exception as e:
        return f"ERROR:{e}"

df["result"] = df.apply(resolve, axis=1)

# ── SETTLEMENT OVERVIEW ───────────────────────────────────────────────────
settled   = df[df["result"].isin(["WON", "LOST"])].copy()
unsettled = df[df["result"] == "UNSETTLED"]
errors    = df[~df["result"].isin(["WON", "LOST", "UNSETTLED"])]

print()
print("=" * 65)
print("  SETTLEMENT BREAKDOWN")
print("=" * 65)
print(f"  Settled   : {len(settled)}")
print(f"  Unsettled : {len(unsettled)}")
if len(errors):
    print(f"  Errors    : {len(errors)}  ← check these")
    print(errors[["event_id","target","result"]].to_string(index=False))

if len(unsettled) > 0:
    print(f"\n  Not yet settled — fixtures still pending or not in DB:")
    missing = df[df["result"] == "UNSETTLED"][["match", "kickoff", "target"]].drop_duplicates("match").head(15)
    print(missing.to_string(index=False))

if len(settled) == 0:
    print("\n  No settled picks to analyse yet.")
    sys.exit()

# ROI helper
def roi(sub):
    wins  = (sub["result"] == "WON").sum()
    total = len(sub)
    if total == 0:
        return wins, total, float("nan"), float("nan")
    win_rate = wins / total * 100
    # Flat-stake ROI: sum of (odds - 1) for wins minus 1 for losses, divided by total stakes
    profit = sub.apply(lambda r: r["decimal_odds"] - 1 if r["result"] == "WON" else -1, axis=1).sum()
    roi_pct = profit / total * 100
    return wins, total, win_rate, roi_pct

# ── OVERALL SUMMARY ───────────────────────────────────────────────────────
print()
print("=" * 65)
print("  OVERALL SUMMARY (settled picks only)")
print("=" * 65)
wins, total, win_rate, roi_pct = roi(settled)
print(f"  Picks     : {total}")
print(f"  Won       : {wins}")
print(f"  Win rate  : {win_rate:.1f}%")
print(f"  Flat ROI  : {roi_pct:+.1f}%  (each pick staked 1 unit)")
print(f"  Avg edge  : {settled['edge_%'].mean():.1f}%")
print(f"  Avg odds  : {settled['decimal_odds'].mean():.2f}")

# ── BY TARGET ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BY MARKET")
print("=" * 65)
print(f"  {'Market':<25} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8} {'AvgOdds':>8}")
print(f"  {'-'*25} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*8}")

for target, grp in settled.groupby("target"):
    name = DISPLAY.get(target, target.replace("target__",""))
    w, t, wr, r = roi(grp)
    avg_odds = grp["decimal_odds"].mean()
    print(f"  {name:<25} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}% {avg_odds:>8.2f}")

# ── BY ODDS BAND ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BY ODDS BAND")
print("=" * 65)

bins   = [1.0, 1.50, 1.70, 2.00, 2.50, 3.00, 4.00, 99.0]
labels = ["1.01–1.50", "1.51–1.70", "1.71–2.00",
          "2.01–2.50", "2.51–3.00", "3.01–4.00", "4.01+"]
settled["odds_band"] = pd.cut(settled["decimal_odds"], bins=bins, labels=labels)

print(f"  {'Odds band':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
for band in labels:
    grp = settled[settled["odds_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    print(f"  {band:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── BY EDGE TIER ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BY EDGE TIER")
print("=" * 65)

ebins   = [0, 5, 8, 12, 100]
elabels = ["3–5%", "5–8%", "8–12%", "12%+"]
settled["edge_band"] = pd.cut(settled["edge_%"], bins=ebins, labels=elabels)

print(f"  {'Edge tier':<10} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
for band in elabels:
    grp = settled[settled["edge_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    print(f"  {band:<10} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── BY AUC TIER ───────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  BY MODEL AUC TIER")
print("=" * 65)

abins   = [0, 0.62, 0.65, 0.70, 1.0]
alabels = ["<0.62", "0.62–0.65", "0.65–0.70", ">0.70"]
settled["auc_band"] = pd.cut(settled["model_auc"], bins=abins, labels=alabels)

print(f"  {'AUC tier':<12} {'Picks':>6} {'Won':>5} {'Win%':>6} {'ROI':>8}")
print(f"  {'-'*12} {'-'*6} {'-'*5} {'-'*6} {'-'*8}")
for band in alabels:
    grp = settled[settled["auc_band"] == band]
    if len(grp) == 0:
        continue
    w, t, wr, r = roi(grp)
    print(f"  {band:<12} {t:>6} {w:>5} {wr:>5.1f}% {r:>+7.1f}%")

# ── TOP 20 PICKS BY EDGE ──────────────────────────────────────────────────
print()
print("=" * 65)
print("  TOP 20 PICKS BY EDGE (settled only)")
print("=" * 65)
cols_show = ["match", "pick", "edge_%", "decimal_odds", "model_auc", "result"]
top20 = settled.sort_values("edge_%", ascending=False).head(20)[cols_show]
print(top20.to_string(index=False))

# ── HIGH-CONFIDENCE WINS: edge ≥ 10% ─────────────────────────────────────
print()
print("=" * 65)
print("  HIGH-EDGE PICKS (edge ≥ 10%) — every single one")
print("=" * 65)
hi = settled[settled["edge_%"] >= 10].sort_values("edge_%", ascending=False)[cols_show]
if len(hi):
    print(hi.to_string(index=False))
    w, t, wr, r = roi(settled[settled["edge_%"] >= 10])
    print(f"\n  Win rate: {wr:.1f}%   ROI: {r:+.1f}%   ({t} picks)")
else:
    print("  No picks with edge ≥ 10%")

# ── SAVE FULL GRADED FILE ─────────────────────────────────────────────────
out_path = PICKS_CSV.replace(".csv", "_graded.csv")
df.to_csv(out_path, index=False)
print()
print("=" * 65)
print(f"  Full graded file saved: {out_path}")
print("=" * 65)

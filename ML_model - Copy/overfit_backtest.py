"""
=============================================================
  overfit_backtest.py
  Backtests the EXACT filter used by overfit_booking_generator.py
  on every all_picks_YYYY-MM-DD_graded.csv file.

  The filter rules are imported from overfit_booking_generator.py
  (MARKET_RULES, ODDS_ZONES), so if you change them there, this
  backtest follows automatically.

  Per day it does what the generator does:
    edge rules per market -> odds zones -> one pick per fixture
    (highest edge) -> top 50 by edge.

  Reports:
    - flat-stake singles: win%, break-even win%, ROI +/- 95% CI
    - by market, by odds zone, by market x zone
    - first half vs second half of the period (stability)
    - the daily slip as ONE accumulator (what you actually book)

  USAGE (run inside ML_model - Copy):
    python overfit_backtest.py

  CAVEAT: the edge tiers and odds bands were chosen by looking at
  these same graded picks, so this is an IN-SAMPLE test. It can only
  disprove the filter, not prove it. Real proof = new days.
=============================================================
"""

import glob
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from overfit_booking_generator import MARKET_RULES, ODDS_ZONES

PICKS_FOLDER = "."
OUTPUT_FILE  = "overfit_backtest_report.txt"
TOP_N        = 50

NAMES = {
    "target__ht_over_1_5_goals": "HT Over 1.5 Goals",
    "target__away_win":          "Away Win",
    "target__ht_away_win":       "HT Away Win",
}


def day_filter(d):
    """Same logic as overfit_booking_generator.overfit_filter + top N."""
    mask = pd.Series(False, index=d.index)
    for target, (lo, hi) in MARKET_RULES.items():
        m = (d["target"] == target) & (d["edge_%"] >= lo)
        if hi is not None:
            m &= (d["edge_%"] <= hi)
        mask |= m
    d = d[mask]

    zone = pd.Series(False, index=d.index)
    for lo, hi in ODDS_ZONES:
        zone |= (d["decimal_odds"] > lo) & (d["decimal_odds"] <= hi)
    d = d[zone]

    d = d.sort_values("edge_%", ascending=False).drop_duplicates("event_id", keep="first")
    return d.head(TOP_N)


def zone_label(odds):
    for lo, hi in ODDS_ZONES:
        if lo < odds <= hi:
            return f"{lo + 0.01:.2f}-{hi:.2f}"
    return "other"


def stats(sub):
    n = len(sub)
    if n == 0:
        return None
    won = int((sub["result"] == "WON").sum())
    profit = np.where(sub["result"] == "WON", sub["decimal_odds"] - 1.0, -1.0)
    roi = profit.mean() * 100
    se = (profit.std(ddof=1) / np.sqrt(n) * 100) if n > 1 else float("nan")
    be = (1.0 / sub["decimal_odds"]).mean() * 100
    return {"n": n, "won": won, "wr": won / n * 100, "be": be,
            "roi": roi, "ci": 1.96 * se, "odds": sub["decimal_odds"].mean()}


def row(label, s, w=26):
    if s is None:
        return f"  {label:<{w}} {'-':>5}"
    return (f"  {label:<{w}} {s['n']:>5} {s['won']:>5} {s['wr']:>6.1f}% "
            f"{s['be']:>7.1f}% {s['roi']:>+7.1f}% {s['ci']:>6.1f}  {s['odds']:>6.2f}")


HEAD = (f"  {{:<26}} {'Picks':>5} {'Won':>5} {'Win%':>7} {'BrkEvn%':>8} "
        f"{'ROI':>8} {'+/-95%':>6}  {'AvgOdds':>6}")


def header(first="Group"):
    return [HEAD.format(first), "  " + "-" * 82]


def section(title, width=70):
    return "\n" + "=" * width + "\n  " + title + "\n" + "=" * width


# ── LOAD ──────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(PICKS_FOLDER, "all_picks_????-??-??_graded.csv")))
if not files:
    sys.exit("ERROR: No graded files found. Run evaluate_picks.py first.")

print(f"Found {len(files)} graded files. Applying overfit filter per day...")
frames, unsettled_total = [], 0
for f in files:
    date_str = os.path.basename(f).replace("all_picks_", "").replace("_graded.csv", "")
    try:
        tmp = pd.read_csv(f)
        picked = day_filter(tmp)
        picked = picked.copy()
        picked["pick_date"] = date_str
        unsettled_total += int((~picked["result"].isin(["WON", "LOST"])).sum())
        picked = picked[picked["result"].isin(["WON", "LOST"])]
        if len(picked):
            frames.append(picked)
    except Exception as e:
        print(f"  {date_str} SKIPPED: {e}")

if not frames:
    sys.exit("ERROR: the filter selected no settled picks in any file.")

raw = pd.concat(frames, ignore_index=True).sort_values("pick_date")
dups = int(raw.duplicated("event_id").sum())
df = raw.drop_duplicates("event_id", keep="first").copy()   # same fixture on 2 days = count once
df["zone"] = df["decimal_odds"].apply(zone_label)

ts = datetime.now().strftime("%Y-%m-%d %H:%M")
lines = []
lines.append("=" * 70)
lines.append("  OVERFIT FILTER BACKTEST")
lines.append(f"  Generated : {ts}")
lines.append(f"  Period    : {df['pick_date'].min()} to {df['pick_date'].max()}   "
             f"({df['pick_date'].nunique()} days with picks)")
lines.append("  Filter    : " + " | ".join(
    f"{NAMES[t]} edge {lo:g}%" + (f"-{hi:g}%" if hi else "+")
    for t, (lo, hi) in MARKET_RULES.items()))
lines.append("  Odds zones: " + " & ".join(f"{lo + 0.01:.2f}-{hi:.2f}" for lo, hi in ODDS_ZONES))
lines.append(f"  Settled picks: {len(df)}   (unsettled skipped: {unsettled_total}, "
             f"repeat fixtures dropped: {dups})")
lines.append("  IN-SAMPLE: bands/tiers were chosen from this same data.")
lines.append("=" * 70)

# ── OVERALL ───────────────────────────────────────────────────────────────
lines.append(section("OVERALL (flat 1-unit singles)"))
lines += header()
overall = stats(df)
lines.append(row("All filtered picks", overall))
lines.append("  Win% above BrkEvn% = beating the odds. ROI +/-95% is the noise range.")

# ── BY MARKET ─────────────────────────────────────────────────────────────
lines.append(section("BY MARKET"))
lines += header("Market")
for t, name in NAMES.items():
    lines.append(row(name, stats(df[df["target"] == t])))

# ── BY ZONE ───────────────────────────────────────────────────────────────
lines.append(section("BY ODDS ZONE"))
lines += header("Zone")
for lo, hi in ODDS_ZONES:
    z = f"{lo + 0.01:.2f}-{hi:.2f}"
    lines.append(row(z, stats(df[df["zone"] == z])))

# ── MARKET x ZONE ─────────────────────────────────────────────────────────
lines.append(section("MARKET x ODDS ZONE"))
lines += header("Market | Zone")
for t, name in NAMES.items():
    for lo, hi in ODDS_ZONES:
        z = f"{lo + 0.01:.2f}-{hi:.2f}"
        sub = df[(df["target"] == t) & (df["zone"] == z)]
        if len(sub):
            lines.append(row(f"{name} | {z}", stats(sub)))

# ── STABILITY: FIRST HALF vs SECOND HALF ─────────────────────────────────
lines.append(section("STABILITY - first half vs second half of the period"))
lines += header("Half")
dates = sorted(df["pick_date"].unique())
mid = len(dates) // 2
first, second = set(dates[:mid]), set(dates[mid:])
lines.append(row(f"{dates[0]} .. {dates[mid - 1]}", stats(df[df["pick_date"].isin(first)])))
lines.append(row(f"{dates[mid]} .. {dates[-1]}", stats(df[df["pick_date"].isin(second)])))
lines.append("  A real edge should be positive in BOTH halves, not just one.")

# ── DAILY SLIP AS ONE ACCUMULATOR ────────────────────────────────────────
lines.append(section("DAILY SLIP AS ONE ACCUMULATOR (what you actually book)"))
lines.append(f"  {'Date':<12} {'Legs':>5} {'Won':>5} {'Lost':>5} {'AllWon':>7} "
             f"{'SlipOdds':>12} {'Singles ROI':>12}")
lines.append("  " + "-" * 66)

acc_profit, acc_days, acc_hit = [], 0, 0
for day, g in df.groupby("pick_date"):
    legs = len(g)
    wins = int((g["result"] == "WON").sum())
    all_won = wins == legs
    slip_odds = float(np.prod(g["decimal_odds"]))
    s = stats(g)
    acc_days += 1
    acc_hit += int(all_won)
    acc_profit.append(slip_odds - 1 if all_won else -1.0)
    lines.append(f"  {day:<12} {legs:>5} {wins:>5} {legs - wins:>5} "
                 f"{'YES' if all_won else 'no':>7} {slip_odds:>12,.1f} {s['roi']:>+11.1f}%")

lines.append("")
lines.append(f"  Days where EVERY leg won      : {acc_hit}/{acc_days}")
lines.append(f"  Accumulator ROI (1 unit/day)  : {np.mean(acc_profit) * 100:+.1f}%")
lines.append(f"  Avg legs per slip             : {df.groupby('pick_date').size().mean():.1f}")

# ── VERDICT ───────────────────────────────────────────────────────────────
lines.append(section("VERDICT (singles, in-sample)"))
lo_b = overall["roi"] - overall["ci"]
hi_b = overall["roi"] + overall["ci"]
lines.append(f"  ROI {overall['roi']:+.1f}%   95% range {lo_b:+.1f}% to {hi_b:+.1f}%   "
             f"({overall['n']} picks)")
if overall["n"] < 150:
    verdict = "TOO FEW PICKS - result is mostly noise either way"
elif lo_b > 0:
    verdict = "POSITIVE even after noise - but in-sample, so confirm on new days"
elif overall["roi"] > 0:
    verdict = "POSITIVE BUT WITHIN NOISE - not distinguishable from luck"
else:
    verdict = "NEGATIVE - the filter does not hold up on its own data"
lines.append(f"  Verdict: {verdict}")

lines.append("\n" + "=" * 70)
lines.append(f"  END - {ts}")
lines.append("=" * 70)

report = "\n".join(lines)
print("\n" + report)
with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
    fh.write(report)
print(f"\nSaved to {OUTPUT_FILE}")

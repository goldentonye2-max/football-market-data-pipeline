"""
filter_test.py
===============================================================
Tests selection filters on picks_enriched.csv.

For each target type, applies the market-based filters found
in correlation analysis and shows filtered vs unfiltered win rates.

USAGE (inside ML_model - Copy):
    python filter_test.py
    python filter_test.py --percentile 75    # top 25% of games only
    python filter_test.py --target home_win  # one target only
    python filter_test.py --csv picks_enriched.csv
===============================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── Filter definitions ────────────────────────────────────────────────────────
# For each target, the strongest signals found in enrich_picks.py analysis.
#
# direction "high" = take pick when these odds are ABOVE threshold
#                   (expensive outcome = that outcome is unlikely = market agrees with pick)
# direction "low"  = take pick when these odds are BELOW threshold
#
# Win-to-Nil markets (m9000xx):
#   m900036 = HT Home to Win to Nil Yes/No
#   m900037 = HT Away to Win to Nil Yes/No
#   m900038 = FT Home to Win to Nil Yes/No
#   m900052 = FT Away to Win to Nil Yes/No
#
# Excluded Goals markets:
#   m450004 = Excluded Number of Goals - Home  (outcome = goals excluded)
#   m810002 = Excluded Number of Goals - Away

FILTERS = {
    "home_win": [
        ("m900036_No",  "high"),  # HT: home WILL lead clean  (No is expensive)
        ("m900037_Yes", "high"),  # HT: away will NOT lead clean
        ("m900038_No",  "high"),  # FT: home WILL win clean
        ("m900052_Yes", "high"),  # FT: away will NOT win clean
    ],
    "away_win": [
        ("m900036_Yes", "high"),  # HT: home will NOT lead clean
        ("m900037_No",  "high"),  # HT: away WILL lead clean
        ("m900038_Yes", "high"),  # FT: home will NOT win clean
        ("m900052_No",  "high"),  # FT: away WILL win clean
    ],
    "ht_home_win": [
        ("m900036_No",  "high"),
        ("m900037_Yes", "high"),
        ("m900052_Yes", "high"),
    ],
    "ht_away_win": [
        ("m900036_Yes", "high"),
        ("m900037_No",  "high"),
        ("m900052_No",  "high"),
    ],
    "over_2_5": [
        ("m450004_4",   "high"),  # home scoring exactly 4 goals is in play
        ("m450004_5+",  "high"),  # home scoring 5+ is in play
        ("m810002_3+",  "high"),  # away scoring 3+ is in play
        ("m450004_0",   "high"),  # skip when home scoring 0 is near-certain (loser signal)
        ("m810002_0",   "high"),  # skip when away scoring 0 is near-certain
    ],
    "over_3_5": [
        ("m450004_4",   "high"),
        ("m450004_5+",  "high"),
        ("m810002_3+",  "high"),
        ("m18_total=4_5_Under_4_5", "high"),  # under 4.5 is expensive = goals are open
    ],
    "over_4_5": [
        ("m450004_4",   "high"),
        ("m450004_5+",  "high"),
        ("m810002_3+",  "high"),
    ],
    "home_clean_sheet": [
        ("m900052_Yes", "high"),  # away will NOT win clean at FT
        ("m810002_0",   "low"),   # away scoring 0 is cheap (likely) → home keeps clean
    ],
    "away_clean_sheet": [
        ("m900052_No",  "high"),  # away WILL win clean at FT
        ("m450004_0",   "low"),   # home scoring 0 is cheap (likely) → away keeps clean
    ],
    "ht_over_1_5": [
        ("m450004_4",   "high"),
        ("m810002_3+",  "high"),
        ("m450004_0",   "high"),  # skip near-certain home zero
        ("m810002_0",   "high"),  # skip near-certain away zero
    ],
}

# Applied across ALL targets regardless of type
GLOBAL_FILTERS = [
    ("m1_Draw", "high", "FT draw odds HIGH → picks win more"),
    ("m60_Draw", "high", "HT draw odds HIGH → picks win more"),
]


def to_won(df):
    if "_won" in df.columns:
        return pd.to_numeric(df["_won"], errors="coerce")
    for col in ["result", "won", "Won", "Result", "correct"]:
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.upper()
            m = {"WON": 1, "WIN": 1, "YES": 1, "TRUE": 1, "1": 1,
                 "LOST": 0, "LOSS": 0, "NO": 0, "FALSE": 0, "0": 0}
            return s.map(m)
    return None


def split(w, vals, direction, pct):
    valid = vals.notna() & w.notna()
    if valid.sum() < 20:
        return None, None, None
    threshold = float(np.percentile(vals[valid].dropna(), pct))
    if direction == "high":
        take = valid & (vals > threshold)
        skip = valid & (vals <= threshold)
    else:
        take = valid & (vals < threshold)
        skip = valid & (vals >= threshold)
    return take, skip, threshold


def wr(w, mask):
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0
    return float(w[mask].mean() * 100), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        default="picks_enriched.csv")
    ap.add_argument("--target",     default=None)
    ap.add_argument("--percentile", type=float, default=50,
                    help="threshold percentile (50=median split, 75=top 25%% only)")
    ap.add_argument("--min-n",      type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: {args.csv} not found — run enrich_picks.py first.")

    print(f"\nLoading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} picks  ×  {len(df.columns)} columns")

    won = to_won(df)
    if won is None:
        sys.exit("ERROR: cannot locate won/result column")
    df["_won"] = won

    graded = df[won.notna()].copy()
    overall_wr = won[won.notna()].mean() * 100
    print(f"  Graded: {len(graded):,} | Overall win rate: {overall_wr:.1f}%")
    print(f"  Threshold percentile: p{args.percentile:.0f}\n")

    target_col = "target" if "target" in df.columns else "market"
    sep  = "=" * 80
    sep2 = "-" * 80

    # ── Global signals ─────────────────────────────────────────────────────────
    print(sep)
    print("  GLOBAL SIGNALS  (apply regardless of which market you are picking)")
    print(sep)
    for gcol, gdir, glabel in GLOBAL_FILTERS:
        if gcol not in graded.columns:
            continue
        gvals = pd.to_numeric(graded[gcol], errors="coerce")
        tm, sm, thresh = split(graded["_won"], gvals, gdir, args.percentile)
        if tm is None:
            continue
        t_wr, t_n = wr(graded["_won"], tm)
        s_wr, s_n = wr(graded["_won"], sm)
        print(f"  {gcol}  (threshold {thresh:.2f})   —  {glabel}")
        print(f"    TAKE  (above threshold): {t_n:>5} picks  →  {t_wr:.1f}%  "
              f"({t_wr - overall_wr:+.1f}pp vs {overall_wr:.1f}% base)")
        print(f"    SKIP  (at/below):        {s_n:>5} picks  →  {s_wr:.1f}%\n")

    # ── Per-target filters ─────────────────────────────────────────────────────
    all_targets = (sorted(graded[target_col].dropna().unique())
                   if target_col in graded.columns else [])

    for fkey, fcols in FILTERS.items():
        if args.target and args.target.lower() not in fkey.lower():
            continue

        matched_targets = [t for t in all_targets if fkey.lower() in t.lower()]
        if not matched_targets:
            continue

        grp = graded[graded[target_col].isin(matched_targets)].copy()
        w   = grp["_won"]
        n   = int(w.notna().sum())
        if n < args.min_n:
            continue

        base = w.mean() * 100

        print(sep)
        print(f"  TARGET : {fkey}")
        print(f"  Picks  : {n}   Base win rate: {base:.1f}%")
        print(sep)

        available = [(c, d) for c, d in fcols if c in grp.columns]
        if not available:
            print("  (no filter columns present in this CSV)\n")
            continue

        # ── Individual filters ─────────────────────────────────────────────
        hdr = (f"  {'Signal':<42} {'Thresh':>7}  "
               f"{'Take N':>6} {'Take WR%':>9}  "
               f"{'Skip N':>6} {'Skip WR%':>9}  {'Δ':>6}")
        print(hdr)
        print("  " + sep2)

        best_gain = -999.0
        for col, direction in available:
            vals = pd.to_numeric(grp[col], errors="coerce")
            tm, sm, thresh = split(w, vals, direction, args.percentile)
            if tm is None:
                continue
            t_wr, t_n = wr(w, tm)
            s_wr, s_n = wr(w, sm)
            if t_n < args.min_n or s_n < args.min_n:
                continue
            gain = t_wr - base
            star = "  ◄" if gain > best_gain else ""
            if gain > best_gain:
                best_gain = gain
            print(f"  {col:<42} {thresh:>7.2f}  "
                  f"{t_n:>6} {t_wr:>8.1f}%  "
                  f"{s_n:>6} {s_wr:>8.1f}%  {gain:>+5.1f}pp{star}")

        # ── Combo filter ───────────────────────────────────────────────────
        if len(available) >= 2:
            combo = pd.Series(True,  index=grp.index)
            valid = pd.Series(True,  index=grp.index)
            for col, direction in available:
                vals = pd.to_numeric(grp[col], errors="coerce")
                valid &= vals.notna()
                thresh = float(np.percentile(vals.dropna(), args.percentile))
                if direction == "high":
                    combo &= (vals > thresh)
                else:
                    combo &= (vals < thresh)
            combo  &= valid & w.notna()
            reject  = valid & ~combo & w.notna()

            t_wr, t_n = wr(w, combo)
            s_wr, s_n = wr(w, reject)
            if t_n >= args.min_n:
                kept_pct = t_n / n * 100
                print(f"\n  COMBO ({len(available)} signals, p{args.percentile:.0f})"
                      f"  →  {t_n} picks kept ({kept_pct:.0f}%)")
                print(f"    ✓ TAKE  {t_n:>5} picks  →  {t_wr:.1f}% win rate  "
                      f"({t_wr - base:+.1f}pp vs {base:.1f}% base)")
                if s_n >= args.min_n:
                    print(f"    ✗ SKIP  {s_n:>5} picks  →  {s_wr:.1f}% win rate")

        print()

    # ── Summary advice ─────────────────────────────────────────────────────────
    print(sep)
    print("  READING THE RESULTS")
    print(sep)
    print("  Δ = win-rate improvement vs base for that target.")
    print("  COMBO = all signals must agree (fewer picks, usually higher win rate).")
    print()
    print("  Next steps:")
    print("  1. The best single signal per target is marked ◄")
    print("  2. Run with --percentile 75 for a stricter filter")
    print("     (only top 25% of games pass, but higher expected quality).")
    print("  3. Once a filter consistently adds +5pp or more, add it to")
    print("     your pick-selection step as a mandatory gate.")
    print(sep)


if __name__ == "__main__":
    main()

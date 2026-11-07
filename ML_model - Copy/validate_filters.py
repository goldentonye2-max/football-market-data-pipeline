"""
validate_filters.py
===============================================================
OUT-OF-SAMPLE validation of the selection filters.

Splits picks_enriched.csv by DATE into train / test sets,
derives thresholds ONLY from the training dates, then applies
them to the held-out test dates and reports win rate + ROI.

If the numbers hold up out-of-sample, the filter is real.

USAGE (inside ML_model - Copy):
    python validate_filters.py
    python validate_filters.py --split 0.70   # 70% train / 30% test
    python validate_filters.py --percentile 75
    python validate_filters.py --target home_win
===============================================================
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── Filter definitions ────────────────────────────────────────────────────────
# (m450004_0 / m810002_0 removed from over_2_5 — their direction is inverse
#  and they broke the combo in filter_test.py; use only the positive signals)
FILTERS = {
    "home_win": [
        ("m900036_No",  "high"),   # HT Home Win-to-Nil "No" expensive → home WILL lead clean
        ("m900037_Yes", "high"),   # HT Away Win-to-Nil "Yes" expensive → away will NOT lead clean
        ("m900038_No",  "high"),   # FT Home Win-to-Nil "No" expensive → home WILL win clean
        ("m900052_Yes", "high"),   # FT Away Win-to-Nil "Yes" expensive → away will NOT win clean
    ],
    "away_win": [
        ("m900036_Yes", "high"),
        ("m900037_No",  "high"),
        ("m900038_Yes", "high"),
        ("m900052_No",  "high"),
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
        ("m450004_4",   "high"),
        ("m450004_5+",  "high"),
        ("m810002_3+",  "high"),
    ],
    "over_3_5": [
        ("m450004_4",   "high"),
        ("m450004_5+",  "high"),
        ("m810002_3+",  "high"),
        ("m18_total=4_5_Under_4_5", "high"),
    ],
    "over_4_5": [
        ("m450004_4",   "high"),
        ("m450004_5+",  "high"),
        ("m810002_3+",  "high"),
    ],
    "home_clean_sheet": [
        ("m900052_Yes", "high"),
    ],
    "away_clean_sheet": [
        ("m900052_No",  "high"),
    ],
    "ht_over_1_5": [
        ("m450004_4",   "high"),
        ("m810002_3+",  "high"),
    ],
}

GLOBAL_FILTERS = [
    ("m1_Draw",  "high", "FT draw odds HIGH"),
    ("m60_Draw", "high", "HT draw odds HIGH"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
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


def calc_roi(w, odds, mask):
    n = int(mask.sum())
    if n == 0 or odds is None:
        return float("nan"), n
    o = pd.to_numeric(odds, errors="coerce")
    valid = mask & w.notna() & o.notna()
    if valid.sum() == 0:
        return float("nan"), 0
    profits = np.where(w[valid] == 1, o[valid] - 1.0, -1.0)
    return float(profits.mean() * 100), int(valid.sum())


def calc_wr(w, mask):
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    valid = mask & w.notna()
    return float(w[valid].mean() * 100), int(valid.sum())


def get_thresh(series, percentile):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) < 10:
        return None
    return float(np.percentile(vals, percentile))


def make_mask(vals_series, thresh, direction):
    vals = pd.to_numeric(vals_series, errors="coerce")
    if direction == "high":
        return vals.notna() & (vals > thresh)
    else:
        return vals.notna() & (vals < thresh)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        default="picks_enriched.csv")
    ap.add_argument("--split",      type=float, default=0.70,
                    help="fraction of dates used for training (default 0.70)")
    ap.add_argument("--percentile", type=float, default=50,
                    help="threshold percentile (50=median, 75=strict)")
    ap.add_argument("--target",     default=None)
    ap.add_argument("--min-n",      type=int,   default=10)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"ERROR: {args.csv} not found — run enrich_picks.py first.")

    print(f"\nLoading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  {len(df):,} picks  ×  {len(df.columns)} columns")

    won = to_won(df)
    if won is None:
        sys.exit("ERROR: cannot find won/result column")
    df["_won"] = won

    # ── Date split ─────────────────────────────────────────────────────────
    date_col = next((c for c in ["_pick_date", "kickoff", "date"] if c in df.columns), None)
    if date_col is None:
        sys.exit("ERROR: no date column found")

    dates = sorted(df[date_col].dropna().unique())
    n_train = max(1, int(len(dates) * args.split))
    train_dates = set(dates[:n_train])
    test_dates  = set(dates[n_train:])

    if not test_dates:
        sys.exit("ERROR: test set is empty — lower --split (e.g. --split 0.65)")

    train = df[df[date_col].isin(train_dates)].copy()
    test  = df[df[date_col].isin(test_dates) & df["_won"].notna()].copy()

    odds_col = "decimal_odds" if "decimal_odds" in test.columns else None

    sep  = "=" * 80
    sep2 = "-" * 80

    print(f"\n  {'TRAINING':<10} {n_train} days   {dates[0]}  →  {dates[n_train-1]}"
          f"   ({len(train):,} picks)")
    print(f"  {'TEST':<10} {len(test_dates)} days   {dates[n_train]}  →  {dates[-1]}"
          f"   ({len(test):,} picks graded)")

    test_base_wr = test["_won"].mean() * 100
    test_base_roi, _ = calc_roi(test["_won"],
                                test[odds_col] if odds_col else None,
                                pd.Series(True, index=test.index))
    print(f"\n  Test baseline —  WR: {test_base_wr:.1f}%   ROI: {test_base_roi:+.1f}%")

    target_col = "target" if "target" in df.columns else "market"

    # ── Global signals ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  GLOBAL SIGNALS  (out-of-sample, all targets pooled)")
    print(sep)
    for gcol, gdir, glabel in GLOBAL_FILTERS:
        if gcol not in train.columns or gcol not in test.columns:
            continue
        thresh = get_thresh(train[gcol], args.percentile)
        if thresh is None:
            continue
        take = make_mask(test[gcol], thresh, gdir) & test["_won"].notna()
        skip = test["_won"].notna() & ~make_mask(test[gcol], thresh, gdir) & pd.to_numeric(test[gcol], errors="coerce").notna()
        t_wr, t_n = calc_wr(test["_won"], take)
        s_wr, s_n = calc_wr(test["_won"], skip)
        t_roi, _  = calc_roi(test["_won"], test[odds_col] if odds_col else None, take)
        print(f"  {gcol}  train-threshold {thresh:.2f}  —  {glabel}")
        print(f"    TAKE  {t_n:>5} picks  →  WR {t_wr:.1f}%  ({t_wr-test_base_wr:+.1f}pp)   "
              f"ROI {t_roi:+.1f}%")
        print(f"    SKIP  {s_n:>5} picks  →  WR {s_wr:.1f}%\n")

    # ── Per-target ──────────────────────────────────────────────────────────
    all_test_targets = (sorted(test[target_col].dropna().unique())
                        if target_col in test.columns else [])

    for fkey, fcols in FILTERS.items():
        if args.target and args.target.lower() not in fkey.lower():
            continue

        matched = [t for t in all_test_targets if fkey.lower() in t.lower()]
        if not matched:
            continue

        train_grp = (train[train[target_col].isin(matched)].copy()
                     if target_col in train.columns else train)
        test_grp  = test[test[target_col].isin(matched)].copy()

        if len(test_grp) < args.min_n:
            continue

        w    = test_grp["_won"]
        odds = test_grp[odds_col] if odds_col else None
        base_wr  = w.mean() * 100
        base_roi, _ = calc_roi(w, odds, pd.Series(True, index=test_grp.index))

        print(sep)
        print(f"  TARGET : {fkey}")
        print(f"  Test picks: {len(test_grp)}   "
              f"Base WR: {base_wr:.1f}%   Base ROI: {base_roi:+.1f}%")
        print(sep)

        available = [(c, d) for c, d in fcols
                     if c in train_grp.columns and c in test_grp.columns]
        if not available:
            print("  (no filter columns in this CSV)\n")
            continue

        # Derive thresholds from TRAIN only
        thresholds = {}
        for col, _ in available:
            t = get_thresh(train_grp[col], args.percentile)
            if t is not None:
                thresholds[col] = t

        # Individual signals tested on TEST
        hdr = (f"  {'Signal':<42} {'TrainThresh':>12}  "
               f"{'Take N':>6} {'WR%':>6} {'ROI%':>7}  "
               f"{'Skip N':>6} {'SkipWR%':>8}  {'ΔWR':>7}")
        print(hdr)
        print("  " + sep2)

        best_gain = -999.0
        for col, direction in available:
            if col not in thresholds:
                continue
            thresh = thresholds[col]
            take   = make_mask(test_grp[col], thresh, direction) & w.notna()
            skip   = (pd.to_numeric(test_grp[col], errors="coerce").notna()
                      & ~make_mask(test_grp[col], thresh, direction) & w.notna())
            t_wr, t_n = calc_wr(w, take)
            s_wr, s_n = calc_wr(w, skip)
            t_roi, _  = calc_roi(w, odds, take)
            if t_n < args.min_n or s_n < args.min_n:
                continue
            gain = t_wr - base_wr
            star = "  ◄" if gain > best_gain else ""
            if gain > best_gain:
                best_gain = gain
            print(f"  {col:<42} {thresh:>12.2f}  "
                  f"{t_n:>6} {t_wr:>5.1f}% {t_roi:>+6.1f}%  "
                  f"{s_n:>6} {s_wr:>7.1f}%  {gain:>+6.1f}pp{star}")

        # Combo
        if len(thresholds) >= 2:
            combo = pd.Series(True, index=test_grp.index)
            valid = pd.Series(True, index=test_grp.index)
            for col, direction in available:
                if col not in thresholds:
                    continue
                vals = pd.to_numeric(test_grp[col], errors="coerce")
                valid &= vals.notna()
                combo &= (vals > thresholds[col]) if direction == "high" else (vals < thresholds[col])
            combo &= valid & w.notna()
            skip   = valid & ~combo & w.notna()

            t_wr, t_n = calc_wr(w, combo)
            s_wr, s_n = calc_wr(w, skip)
            t_roi, _  = calc_roi(w, odds, combo)
            if t_n >= args.min_n:
                kept_pct = t_n / len(test_grp) * 100
                print(f"\n  COMBO ({len(thresholds)} signals from train p{args.percentile:.0f})"
                      f"  →  {t_n} test picks kept ({kept_pct:.0f}%)")
                print(f"    ✓ TAKE  {t_n:>5}  →  WR {t_wr:.1f}%  "
                      f"({t_wr-base_wr:+.1f}pp)   ROI {t_roi:+.1f}%")
                if s_n >= args.min_n:
                    print(f"    ✗ SKIP  {s_n:>5}  →  WR {s_wr:.1f}%")
        print()

    print(sep)
    print("  VERDICT GUIDE")
    print(sep)
    print("  PASS  : test WR and ROI are positive and within ~5pp of filter_test.py")
    print("          → signal is real, safe to add to production pipeline")
    print("  WARN  : test WR positive but drops >10pp vs filter_test.py")
    print("          → signal has some edge but was partly overfitted; use with caution")
    print("  FAIL  : test WR at or below baseline, ROI negative")
    print("          → signal was entirely in-sample noise; do not use")
    print(sep)


if __name__ == "__main__":
    main()
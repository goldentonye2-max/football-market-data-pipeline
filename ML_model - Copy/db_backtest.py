"""
=============================================================
  db_backtest.py
  Tests TOP-K PER MARKET PER DAY on your FULL database history
  (not just the ~41 graded picks-file days).

  With 25,000+ results this gives a far more reliable test.

  Only SHORT ODDS and LONG ODDS can be tested here — model_prob
  and edge require the ML model output which only exists in your
  picks CSV files.

  USAGE (inside ML_model - Copy):
      python db_backtest.py              # k=2, all markets
      python db_backtest.py --k 1
      python db_backtest.py --k 3
      python db_backtest.py --k 1 --market over_1_5
      python db_backtest.py --min-odds 1.50 --max-odds 2.00
      python db_backtest.py --since 2026-09-01
      python db_backtest.py --db "../../sportybet.db"

  Output: db_backtest_report.txt  and  db_backtest_table.csv
=============================================================
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

DB_PATH = r"..\sportybet.db"

# (market_id, specifier, outcome_desc), needs_ht_score
# win_fn takes the whole DataFrame and returns a boolean Series — vectorised, no apply()
TARGETS = {
    "over_1_5":    (("18", "total=1.5", "Over 1.5"), False, lambda d: (d["ft_home"] + d["ft_away"]) > 1),
    "over_2_5":    (("18", "total=2.5", "Over 2.5"), False, lambda d: (d["ft_home"] + d["ft_away"]) > 2),
    "over_3_5":    (("18", "total=3.5", "Over 3.5"), False, lambda d: (d["ft_home"] + d["ft_away"]) > 3),
    "over_4_5":    (("18", "total=4.5", "Over 4.5"), False, lambda d: (d["ft_home"] + d["ft_away"]) > 4),
    "home_win":    (("1",  "",          "Home"),      False, lambda d: d["ft_home"] > d["ft_away"]),
    "away_win":    (("1",  "",          "Away"),      False, lambda d: d["ft_away"] > d["ft_home"]),
    "draw":        (("1",  "",          "Draw"),      False, lambda d: d["ft_home"] == d["ft_away"]),
    "btts_yes":    (("29", "",          "Yes"),       False, lambda d: (d["ft_home"] > 0) & (d["ft_away"] > 0)),
    "ht_over_1_5": (("68", "total=1.5", "Over 1.5"), True,  lambda d: (d["ht_home"] + d["ht_away"]) > 1),
    "ht_over_0_5": (("68", "total=0.5", "Over 0.5"), True,  lambda d: (d["ht_home"] + d["ht_away"]) > 0),
    "ht_home_win": (("60", "",          "Home"),      True,  lambda d: d["ht_home"] > d["ht_away"]),
    "ht_away_win": (("60", "",          "Away"),      True,  lambda d: d["ht_away"] > d["ht_home"]),
    "ht_draw":     (("60", "",          "Draw"),      True,  lambda d: d["ht_home"] == d["ht_away"]),
}


def to_date(series):
    """Convert kickoff_time column (unix ms/s or datetime string) to YYYY-MM-DD."""
    s = series.astype(str).str.strip()
    if s.str.fullmatch(r"\d+").all():
        v = s.astype("int64")
        dt = pd.to_datetime(v, unit="ms" if v.max() > 1e11 else "s")
    else:
        dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def load_target(conn, tname, args):
    (mid, spec, desc), needs_ht, win_fn = TARGETS[tname]
    df = pd.read_sql_query("""
        SELECT o.event_id,
               CAST(o.odds AS REAL) AS decimal_odds,
               f.kickoff_time,
               r.ht_home, r.ht_away,
               r.ft_home, r.ft_away
        FROM odds o
        JOIN fixtures f ON f.event_id = o.event_id
        JOIN results  r ON r.event_id = o.event_id
        WHERE o.market_id    = ?
          AND COALESCE(o.specifier, '') = ?
          AND o.outcome_desc = ?
          AND r.status       = 'finished'
          AND r.ft_home IS NOT NULL
          AND r.ft_away IS NOT NULL
          AND o.odds IS NOT NULL
          AND CAST(o.odds AS REAL) > 1.0
    """, conn, params=(mid, spec, desc))

    if needs_ht:
        df = df.dropna(subset=["ht_home", "ht_away"])

    df["event_id"] = df["event_id"].astype(str)
    df = df.drop_duplicates("event_id").copy()

    if args.min_odds is not None:
        df = df[df["decimal_odds"] > args.min_odds]
    if args.max_odds is not None:
        df = df[df["decimal_odds"] <= args.max_odds]

    df["pick_date"] = to_date(df["kickoff_time"])
    if args.since:
        df = df[df["pick_date"] >= args.since]

    df = df.dropna(subset=["pick_date"]).copy()
    df["won"] = win_fn(df).astype(int)   # vectorised — no apply()
    return df


def evaluate(df, descending, k, min_pool, draws, rng):
    """
    Per day: sort by decimal_odds (ascending=short, descending=long),
    take top-k, count wins. Compare against random k from same pool.
    Returns dict of stats, or None if no days qualify.
    """
    n_pool = n_pick = 0
    obs_w = exp_w = var_w = 0.0
    obs_p = exp_p = 0.0

    for _, g in df.groupby("pick_date"):
        n = len(g)
        if n < min_pool:
            continue
        s = g["decimal_odds"].to_numpy(float)
        w = g["won"].to_numpy(float)
        o = g["decimal_odds"].to_numpy(float)
        prof = np.where(w == 1, o - 1.0, -1.0)
        kk = min(k, n)
        sc = -s if descending else s       # lexsort sorts ascending; negate for descending
        ow = op = 0.0
        for _ in range(draws):
            order = np.lexsort((rng.random(n), sc))[:kk]
            ow += w[order].sum()
            op += prof[order].sum()
        obs_w += ow / draws
        obs_p += op / draws
        p = w.mean()
        exp_w += kk * p
        exp_p += kk * prof.mean()
        var_w += kk * p * (1 - p) * (n - kk) / (n - 1) if n > 1 else 0.0
        n_pool += 1
        n_pick += kk

    if n_pick == 0:
        return None
    z = (obs_w - exp_w) / np.sqrt(var_w) if var_w > 0 else 0.0
    return dict(
        pools=n_pool, picks=n_pick,
        se=np.sqrt(var_w) / n_pick * 100,
        win=obs_w / n_pick * 100,
        rnd=exp_w / n_pick * 100,
        gap=(obs_w - exp_w) / n_pick * 100,
        z=z, p=float(norm.sf(z)),
        roi=obs_p / n_pick * 100,
        roi_rnd=exp_p / n_pick * 100,
    )


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / (np.arange(m) + 1))[::-1])[::-1].clip(max=1)
    return q


def main():
    ap = argparse.ArgumentParser(description="Full-DB top-k per market backtest")
    ap.add_argument("--k",        type=int,   default=2)
    ap.add_argument("--min-pool", type=int,   default=4,
                    help="only days where the market has at least this many fixtures")
    ap.add_argument("--min-odds", type=float, default=None, help="lower edge (exclusive)")
    ap.add_argument("--max-odds", type=float, default=None, help="upper edge (inclusive)")
    ap.add_argument("--market",   type=str,   default=None,
                    help="substring of target name, e.g. over_1_5")
    ap.add_argument("--since",    type=str,   default=None, help="YYYY-MM-DD")
    ap.add_argument("--draws",    type=int,   default=50,
                    help="random tie-break draws per day (more = slower but smoother)")
    ap.add_argument("--db",       type=str,   default=DB_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"ERROR: database not found: {args.db}\n"
                 f"Try: python db_backtest.py --db \"../sportybet.db\"")

    conn = sqlite3.connect(args.db)
    rng  = np.random.default_rng(0)
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")

    target_names = [t for t in TARGETS
                    if not args.market or args.market.lower() in t.lower()]

    SCORES = [("short_odds", False), ("long_odds", True)]

    rows = []
    print(f"\nLoading from {args.db} ...")
    for tname in target_names:
        try:
            df = load_target(conn, tname, args)
        except Exception as e:
            print(f"  {tname:<16} skipped ({e})")
            continue
        if len(df) < 50:
            print(f"  {tname:<16} only {len(df)} games — skipping (too few)")
            continue
        days = df["pick_date"].nunique()
        print(f"  {tname:<16} {len(df):>6} games  {days:>4} days  "
              f"win rate {df['won'].mean()*100:.1f}%")
        for sname, desc in SCORES:
            r = evaluate(df, desc, args.k, args.min_pool, args.draws, rng)
            if r and r["picks"] >= 20:
                r.update(target=tname, score=sname,
                         games=len(df), days=days,
                         base_wr=df["won"].mean() * 100)
                rows.append(r)
    conn.close()

    if not rows:
        sys.exit("No markets had enough data. Try lowering --min-pool.")

    t = pd.DataFrame(rows)
    t["q"] = bh(t["p"].values)

    L = []
    L.append("=" * 104)
    L.append(f"  FULL-DATABASE  TOP-{args.k} PER MARKET PER DAY  vs  RANDOM {args.k}")
    L.append(f"  Generated : {ts}")
    L.append(f"  Odds band : {args.min_odds or 'any'} — {args.max_odds or 'any'}   "
             f"Min pool : {args.min_pool}   Since : {args.since or 'all time'}")
    L.append("=" * 104)
    L.append("  Win%   = win rate of your top-k picks.")
    L.append("  Rnd%   = win rate of random k picks from the same day-pool (the fair comparison).")
    L.append("  Gap    = Win% minus Rnd%.  z/q: is the gap bigger than luck? (q corrected for many tests).")
    L.append("  NOTE   : model_prob and edge are NOT in the DB — only short_odds / long_odds shown.")

    L.append("\n" + "-" * 104)
    L.append(f"  {'Market':<16} {'Score':<12} {'Games':>6} {'Days':>5} {'Picks':>6} "
             f"{'Win%':>6} {'Rnd%':>6} {'Gap':>6} {'z':>5} {'q':>5} {'ROI':>7} {'RndROI':>7}")
    L.append("  " + "-" * 100)
    for _, r in t.sort_values(["target", "score"]).iterrows():
        L.append(
            f"  {r['target'][:15]:<16} {r['score']:<12} {r['games']:>6} {r['days']:>5} {r['picks']:>6} "
            f"{r['win']:>5.1f}% {r['rnd']:>5.1f}% {r['gap']:>+6.1f} {r['z']:>5.1f} {r['q']:>5.2f} "
            f"{r['roi']:>+6.1f}% {r['roi_rnd']:>+6.1f}%"
        )

    L.append("\n" + "=" * 104)
    L.append("  ALL MARKETS POOLED  (the cleanest single number per score)")
    L.append("=" * 104)
    L.append(f"  {'Score':<12} {'Picks':>6} {'Win%':>6} {'Rnd%':>6} {'Gap':>6} {'z':>5} {'ROI':>7} {'RndROI':>7}")
    pooled_ok = False
    for sname, _ in SCORES:
        s = t[t["score"] == sname]
        if s.empty:
            continue
        picks = s["picks"].sum()
        win   = (s["win"] * s["picks"]).sum() / picks
        rnd   = (s["rnd"] * s["picks"]).sum() / picks
        # pooled z: weight each market by the reciprocal of its SE
        w_z   = 1.0 / s["se"].replace(0, np.nan)
        z_all = ((s["gap"] * w_z).sum() / np.sqrt((w_z ** 2).sum())
                 if w_z.notna().any() else 0.0)
        roi   = (s["roi"]     * s["picks"]).sum() / picks
        rroi  = (s["roi_rnd"] * s["picks"]).sum() / picks
        if z_all > 2.5:
            pooled_ok = True
        L.append(f"  {sname:<12} {picks:>6} {win:>5.1f}% {rnd:>5.1f}% {win-rnd:>+6.1f} "
                 f"{z_all:>5.1f} {roi:>+6.1f}% {rroi:>+6.1f}%")

    L.append("\n" + "=" * 104)
    L.append("  VERDICT")
    L.append("=" * 104)
    surv = t[t["q"] < 0.10]
    if len(surv):
        L.append(f"  {len(surv)} combination(s) beat random after correction (q < 0.10):")
        for _, r in surv.sort_values("gap", ascending=False).iterrows():
            roi_label = f"ROI {r['roi']:+.1f}%" if r["roi"] > 0 else f"ROI {r['roi']:+.1f}% (negative)"
            L.append(f"    {r['target']:<16} by {r['score']:<12}  "
                     f"{r['win']:.1f}% vs {r['rnd']:.1f}% random  "
                     f"(gap {r['gap']:+.1f}, {r['picks']} picks, {r['roi']:+.1f}% ROI)")
    elif pooled_ok:
        L.append("  No single market/score survives correction, but overall short_odds beats random.")
    else:
        L.append("  NO RANKING BEATS RANDOM after correction.")

    det = 1.96 * t["se"].median()
    L.append(f"\n  Detection limit: a gap of ~{det:.0f} points or more is detectable with this data.")
    L.append("  Compare this to the top2_backtest.py results (41 days) — "
             "more data = narrower noise band = more trustworthy.")
    L.append("=" * 104)

    out = "\n".join(L)
    print("\n" + out)
    with open("db_backtest_report.txt", "w", encoding="utf-8") as f:
        f.write(out)
    t.to_csv("db_backtest_table.csv", index=False)
    print("\nSaved db_backtest_report.txt  and  db_backtest_table.csv")
    print(f"\nTip: run with --k 1 to test the single most-favoured pick per market per day.")


if __name__ == "__main__":
    main()
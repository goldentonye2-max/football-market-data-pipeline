"""
=============================================================
  league_compare.py
  For one market (and optionally one odds band), compares every
  league's ACTUAL win rate with the win rate the market EXPECTED
  (bookmaker probability with the margin removed, taken from the
  `probability` column of your odds table).

  It is about win rate, not profit. A league with a positive gap
  wins more often than its odds said it should.

  Because there are many leagues, some will look good by luck. So it:
    1. gives each league a z-score and a p-value,
    2. corrects for testing many leagues at once (Benjamini-Hochberg),
    3. runs a hold-out: pick the leagues that looked best in the
       FIRST half of the period, then check whether they were still
       above expectation in the SECOND half (and the reverse for the
       worst leagues).

  USAGE (inside ML_model - Copy):
    python league_compare.py --target over_1_5
    python league_compare.py --target over_1_5 --min-odds 1.25 --max-odds 1.40
    python league_compare.py --target ht_over_1_5 --min-games 60
    python league_compare.py --target away_win --since 2026-08-01

  Targets: ht_over_1_5 ht_over_0_5 ht_away_win ht_home_win
           away_win home_win over_1_5 over_2_5 btts_yes
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

# key -> (market_id, specifier, outcome_desc), needs HT score, win rule
TARGETS = {
    "ht_over_1_5": (("68", "total=1.5", "Over 1.5"), True,  lambda d: (d.ht_home + d.ht_away) > 1),
    "ht_over_0_5": (("68", "total=0.5", "Over 0.5"), True,  lambda d: (d.ht_home + d.ht_away) > 0),
    "ht_away_win": (("60", "", "Away"),              True,  lambda d: d.ht_away > d.ht_home),
    "ht_home_win": (("60", "", "Home"),              True,  lambda d: d.ht_home > d.ht_away),
    "away_win":    (("1", "", "Away"),               False, lambda d: d.ft_away > d.ft_home),
    "home_win":    (("1", "", "Home"),               False, lambda d: d.ft_home > d.ft_away),
    "over_1_5":    (("18", "total=1.5", "Over 1.5"), False, lambda d: (d.ft_home + d.ft_away) > 1),
    "over_2_5":    (("18", "total=2.5", "Over 2.5"), False, lambda d: (d.ft_home + d.ft_away) > 2),
    "btts_yes":    (("29", "", "Yes"),               False, lambda d: (d.ft_home > 0) & (d.ft_away > 0)),
}


def to_dt(s):
    s = s.astype(str).str.strip()
    if s.str.fullmatch(r"\d+").all():
        v = s.astype("int64")
        return pd.to_datetime(v, unit="ms" if v.max() > 1e11 else "s")
    return pd.to_datetime(s, errors="coerce")


def section(title, w=88):
    return "\n" + "=" * w + "\n  " + title + "\n" + "=" * w


def load(conn, target, lo, hi, since):
    (mid, spec, desc), needs_ht, win = TARGETS[target]
    df = pd.read_sql_query("""
        SELECT o.event_id, o.odds AS decimal_odds, o.probability AS prob,
               f.kickoff_time, f.tournament_name, f.category_name,
               r.ht_home, r.ht_away, r.ft_home, r.ft_away
        FROM odds o
        JOIN fixtures f ON f.event_id = o.event_id
        JOIN results  r ON r.event_id = o.event_id
        WHERE o.market_id = ? AND COALESCE(o.specifier,'') = ? AND o.outcome_desc = ?
          AND r.status = 'finished' AND r.ft_home IS NOT NULL AND r.ft_away IS NOT NULL
          AND o.odds IS NOT NULL
    """, conn, params=(mid, spec, desc))
    if needs_ht:
        df = df.dropna(subset=["ht_home", "ht_away"])
    df["event_id"] = df["event_id"].astype(str)
    df = df.drop_duplicates("event_id")
    if lo is not None:
        df = df[df["decimal_odds"] > lo]
    if hi is not None:
        df = df[df["decimal_odds"] <= hi]
    df = df.copy()
    df["won"] = win(df).astype(int)
    # market-expected win probability with margin removed; fall back to 1/odds
    inv = 1.0 / df["decimal_odds"]
    df["exp"] = df["prob"].where(df["prob"].between(0.001, 0.999), inv)
    df["when"] = to_dt(df["kickoff_time"])
    df["league"] = (df["category_name"].fillna("?").astype(str) + " | "
                    + df["tournament_name"].fillna("?").astype(str))
    if since:
        df = df[df["when"] >= pd.Timestamp(since)]
    return df.dropna(subset=["when"]).reset_index(drop=True)


def stats(g):
    n = len(g)
    if n == 0:
        return None
    w = int(g["won"].sum())
    e = float(g["exp"].sum())
    v = float((g["exp"] * (1 - g["exp"])).sum())
    z = (w - e) / np.sqrt(v) if v > 0 else 0.0
    return dict(n=n, w=w, wr=w / n * 100, ex=e / n * 100, gap=(w - e) / n * 100,
                z=z, p=float(2 * norm.sf(abs(z))),
                ci=(1.96 * np.sqrt(v) / n * 100) if v > 0 else float("nan"))


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    ranked = p[order] * m / (np.arange(m) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


def main():
    ap = argparse.ArgumentParser(description="League win-rate vs market expectation")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--min-odds", type=float, default=None, help="band lower edge (exclusive)")
    ap.add_argument("--max-odds", type=float, default=None, help="band upper edge (inclusive)")
    ap.add_argument("--since", type=str, default=None, help="only games on/after YYYY-MM-DD")
    ap.add_argument("--min-games", type=int, default=40, help="smallest league to show")
    ap.add_argument("--top", type=int, default=15, help="rows to show at each end")
    ap.add_argument("--db", type=str, default=DB_PATH)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"ERROR: database not found: {args.db}")
    conn = sqlite3.connect(args.db)
    d = load(conn, args.target, args.min_odds, args.max_odds, args.since)
    conn.close()
    if len(d) < 200:
        sys.exit(f"Only {len(d)} games - too few. Widen the band or drop --since.")

    cut = d["when"].median()
    d["half"] = np.where(d["when"] <= cut, 1, 2)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    all_s = stats(d)
    L = []
    L.append("=" * 88)
    L.append(f"  LEAGUE COMPARISON  -  {args.target}")
    L.append(f"  Generated : {ts}")
    L.append(f"  Band      : {args.min_odds} - {args.max_odds}")
    L.append(f"  Period    : {d['when'].min():%Y-%m-%d} to {d['when'].max():%Y-%m-%d}   "
             f"(halves split at {cut:%Y-%m-%d})")
    L.append("=" * 88)
    L.append(f"  Games {all_s['n']}   Won {all_s['wr']:.1f}%   Market expected {all_s['ex']:.1f}%   "
             f"Gap {all_s['gap']:+.1f} pts")
    L.append("  Gap = actual win rate minus the win rate the odds implied (margin removed).")

    # per-league table
    rows = []
    for lg, g in d.groupby("league"):
        if len(g) < args.min_games:
            continue
        s = stats(g)
        s1 = stats(g[g["half"] == 1])
        s2 = stats(g[g["half"] == 2])
        s["league"] = lg
        s["gap1"] = s1["gap"] if s1 and s1["n"] >= 10 else np.nan
        s["gap2"] = s2["gap"] if s2 and s2["n"] >= 10 else np.nan
        rows.append(s)
    if not rows:
        sys.exit(f"No league has {args.min_games}+ games. Lower --min-games.")
    t = pd.DataFrame(rows)
    t["q"] = bh(t["p"].values)
    t = t.sort_values("z", ascending=False).reset_index(drop=True)
    m = len(t)

    L.append(section(f"1. LEAGUES WITH {args.min_games}+ GAMES ({m} leagues)"))
    L.append(f"  Raw p < 0.05          : {(t['p'] < 0.05).sum()}   (chance alone gives about {0.05 * m:.0f})")
    L.append(f"  Survive FDR (q < 0.10): {(t['q'] < 0.10).sum()}")

    head = (f"  {'League':<40} {'Games':>5} {'Win%':>6} {'Mkt%':>6} {'Gap':>6} {'z':>5} {'q':>5} "
            f"{'1st half':>8} {'2nd half':>8}")

    def fmt(r):
        g1 = f"{r['gap1']:+.1f}" if pd.notna(r["gap1"]) else "-"
        g2 = f"{r['gap2']:+.1f}" if pd.notna(r["gap2"]) else "-"
        return (f"  {r['league'][:39]:<40} {r['n']:>5} {r['wr']:>5.1f}% {r['ex']:>5.1f}% "
                f"{r['gap']:>+6.1f} {r['z']:>5.1f} {r['q']:>5.2f} {g1:>8} {g2:>8}")

    k = min(args.top, m)
    L.append("")
    L.append(f"  MOST ABOVE the market's expectation (top {k}):")
    L.append(head)
    L.append("  " + "-" * 100)
    for _, r in t.head(k).iterrows():
        L.append(fmt(r))
    L.append("")
    L.append(f"  MOST BELOW the market's expectation (bottom {k}):")
    L.append(head)
    L.append("  " + "-" * 100)
    for _, r in t.tail(k).iloc[::-1].iterrows():
        L.append(fmt(r))
    L.append("  Gap in percentage points. 1st/2nd half = that league's gap in each half of the period.")
    L.append("  A real league effect keeps the same sign in both halves.")

    # hold-out
    L.append(section("2. HOLD-OUT: choose leagues on the first half, test on the second half"))
    min1 = max(15, args.min_games // 2)
    first = d[d["half"] == 1]
    second = d[d["half"] == 2]
    sel = []
    for lg, g in first.groupby("league"):
        if len(g) >= min1:
            s = stats(g)
            sel.append((lg, s["z"]))
    sel.sort(key=lambda x: -x[1])
    base2 = stats(second)
    hold_ok = False
    ci_top = None
    if len(sel) >= 10:
        n_pick = min(10, len(sel) // 2)
        best = [x[0] for x in sel[:n_pick]]
        worst = [x[0] for x in sel[-n_pick:]]
        sb = stats(second[second["league"].isin(best)])
        sw = stats(second[second["league"].isin(worst)])
        L.append(f"  Leagues picked on first half (>= {min1} games there): best {n_pick} and worst {n_pick}.")
        L.append("")
        L.append(f"  {'Group (judged on SECOND half only)':<44} {'Games':>6} {'Win%':>6} {'Mkt%':>6} {'Gap':>7} {'+/-95%':>7}")
        L.append("  " + "-" * 80)
        for lab, s in (("Best leagues from first half", sb),
                       ("Worst leagues from first half", sw),
                       ("All leagues (reference)", base2)):
            if s is None:
                L.append(f"  {lab:<44} {'-':>6}")
            else:
                L.append(f"  {lab:<44} {s['n']:>6} {s['wr']:>5.1f}% {s['ex']:>5.1f}% "
                         f"{s['gap']:>+6.1f} {s['ci']:>7.1f}")
        L.append("  If the best leagues were truly better, their second-half gap stays clearly")
        L.append("  positive and the worst leagues' stays negative. If both drift to the reference")
        L.append("  line, the first-half differences were luck.")
        if sb is not None and sw is not None:
            # judged against the all-leagues reference, not against zero
            ref = base2["gap"]
            hold_ok = ((sb["gap"] - ref) > sb["ci"]) or ((sw["gap"] - ref) < -sw["ci"])
            ci_top = sb["ci"]
    else:
        L.append(f"  Too few leagues with {min1}+ games in the first half ({len(sel)}). "
                 f"Lower --min-games or widen the band.")

    # verdict
    L.append(section("VERDICT"))
    fdr_hits = int((t["q"] < 0.10).sum())
    if hold_ok and fdr_hits > 0:
        v = ("LEAGUE EFFECT FOUND: some leagues win more (or less) often than their odds say, and "
             "it carried into the second half.")
    elif hold_ok or fdr_hits > 0:
        v = ("WEAK/MIXED: one check passes and the other does not. Treat as a hint; collect more days.")
    else:
        v = ("NO LEAGUE EFFECT CONFIRMED: league differences look like luck. The market's odds already "
             "describe each league's win rate about as well as this data can show.")
    L.append("  " + v)
    if ci_top is not None:
        L.append(f"  Detection limit in this hold-out is about +/-{ci_top:.0f} points.")
    L.append("\n" + "=" * 88)

    report = "\n".join(L)
    print(report)
    rep = f"league_report_{args.target}.txt"
    csv = f"league_table_{args.target}.csv"
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write(report)
    t.to_csv(csv, index=False)
    print(f"\nSaved {rep} and {csv}")


if __name__ == "__main__":
    main()

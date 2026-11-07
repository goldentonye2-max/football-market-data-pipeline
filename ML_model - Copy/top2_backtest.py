"""
=============================================================
  top2_backtest.py
  For EVERY market in your graded files: if each day you took
  only the best 2 picks of that market, ranked by some score,
  would they win more often than 2 picks chosen at random from
  that same day's picks of that market?

  Scores tried for ranking (inside one market, one day):
    model_prob   highest model probability
    edge         highest edge
    short_odds   shortest odds (market's favourite)
    long_odds    longest odds
  (model_auc and hit_rate_% are the same for every pick of a
   market, so they cannot rank picks inside it.)

  It grades extra markets itself from the score columns saved in
  the graded files (over 1.5, under, BTTS, draw, double chance,
  HT over 0.5 ...), because evaluate_picks.py leaves those
  UNKNOWN_TARGET.

  Fair comparison: against RANDOM top-2 from the same day-pool,
  with a z-score and a correction for testing many markets.

  USAGE (inside ML_model - Copy):
      python top2_backtest.py
      python top2_backtest.py --k 2 --min-pool 4
      python top2_backtest.py --min-odds 1.30 --max-odds 1.70
      python top2_backtest.py --market over_1_5_goals
=============================================================
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

FOLDER = "."
SCORES = {
    "model_prob": ("model_prob_%", True),
    "edge":       ("edge_%", True),
    "short_odds": ("decimal_odds", False),   # False = lower value ranks first
    "long_odds":  ("decimal_odds", True),
}


# ── GRADING ───────────────────────────────────────────────────────────────
def resolve(target, r):
    """Return 1/0 if the target can be graded from the saved scores, else None."""
    fh, fa, hh, ha = r["ft_home"], r["ft_away"], r["ht_home"], r["ht_away"]
    t = target.replace("target__", "")
    ft, ht = fh + fa, hh + ha
    simple = {
        "home_win": fh > fa, "draw": fh == fa, "away_win": fa > fh,
        "home_or_draw": fh >= fa, "draw_or_away": fa >= fh, "home_or_away": fh != fa,
        "btts_yes": fh > 0 and fa > 0, "btts_no": not (fh > 0 and fa > 0),
        "home_clean_sheet": fa == 0, "away_clean_sheet": fh == 0,
        "ht_home_win": hh > ha, "ht_draw": hh == ha, "ht_away_win": ha > hh,
    }
    if t in simple:
        if t.startswith("ht_") and (pd.isna(hh) or pd.isna(ha)):
            return None
        return int(bool(simple[t]))
    m = re.fullmatch(r"(over|under)_(\d+)_(\d+)_goals", t)
    if m:
        x = float(f"{m.group(2)}.{m.group(3)}")
        return int(ft > x) if m.group(1) == "over" else int(ft < x)
    m = re.fullmatch(r"ht_(over|under)_(\d+)_(\d+)_goals", t)
    if m:
        if pd.isna(hh) or pd.isna(ha):
            return None
        x = float(f"{m.group(2)}.{m.group(3)}")
        return int(ht > x) if m.group(1) == "over" else int(ht < x)
    m = re.fullmatch(r"score_(\d+)_(\d+)", t)
    if m:
        return int(fh == int(m.group(1)) and fa == int(m.group(2)))
    return None


def load(args):
    files = sorted(glob.glob(os.path.join(FOLDER, "all_picks_????-??-??_graded.csv")))
    if not files:
        sys.exit("ERROR: no all_picks_YYYY-MM-DD_graded.csv files found here.")
    frames = []
    for f in files:
        day = os.path.basename(f).replace("all_picks_", "").replace("_graded.csv", "")
        d = pd.read_csv(f)
        need = {"event_id", "target", "decimal_odds", "ft_home", "ft_away", "ht_home", "ht_away"}
        if not need.issubset(d.columns):
            print(f"  {day} SKIPPED: missing columns {sorted(need - set(d.columns))}")
            continue
        if "status" in d.columns:
            d = d[d["status"] == "finished"]
        d = d.dropna(subset=["ft_home", "ft_away", "decimal_odds"]).copy()
        d["pick_date"] = day
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["event_id"] = df["event_id"].astype(str)
    df = df.sort_values("pick_date").drop_duplicates(["event_id", "target"], keep="first")

    if args.market:
        df = df[df["target"].str.contains(args.market)]
    if args.min_odds is not None:
        df = df[df["decimal_odds"] > args.min_odds]
    if args.max_odds is not None:
        df = df[df["decimal_odds"] <= args.max_odds]

    df["won"] = [resolve(t, r) for t, r in zip(df["target"], df.to_dict("records"))]
    ungraded = sorted(df.loc[df["won"].isna(), "target"].unique())
    df = df.dropna(subset=["won"]).copy()
    df["won"] = df["won"].astype(int)
    return df, ungraded, len(files)


# ── TOP-K VS RANDOM ───────────────────────────────────────────────────────
def evaluate(df, score_col, desc, k, min_pool, draws, rng):
    """Per day x market pool: observed top-k (tie-broken at random, averaged)
    against a random pick of k from the same pool."""
    n_pool = n_pick = 0
    obs_w = exp_w = var_w = obs_p = exp_p = 0.0
    for _, g in df.groupby(["pick_date", "target"]):
        n = len(g)
        if n < min_pool:
            continue
        s = g[score_col].to_numpy(float)
        w = g["won"].to_numpy(float)
        o = g["decimal_odds"].to_numpy(float)
        prof = np.where(w == 1, o - 1.0, -1.0)
        kk = min(k, n)
        sc = s if desc else -s
        ow = op = 0.0
        for _ in range(draws):
            order = np.lexsort((rng.random(n), -sc))[:kk]
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
    return dict(pools=n_pool, picks=n_pick, se=np.sqrt(var_w) / n_pick * 100,
                win=obs_w / n_pick * 100, rnd=exp_w / n_pick * 100,
                gap=(obs_w - exp_w) / n_pick * 100, z=z, p=float(norm.sf(z)),
                roi=obs_p / n_pick * 100, roi_rnd=exp_p / n_pick * 100)


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / (np.arange(m) + 1))[::-1])[::-1].clip(max=1)
    return q


def main():
    ap = argparse.ArgumentParser(description="Best-2-per-market-per-day backtest")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--min-pool", type=int, default=4,
                    help="only days where a market has at least this many picks")
    ap.add_argument("--min-odds", type=float, default=None)
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--market", type=str, default=None, help="substring of target name")
    ap.add_argument("--draws", type=int, default=50, help="random tie-break draws")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    df, ungraded, n_files = load(args)
    if df.empty:
        sys.exit("No gradable picks left after filters.")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = []
    for target, g in df.groupby("target"):
        for sname, (col, desc) in SCORES.items():
            if col not in g.columns:
                continue
            r = evaluate(g, col, desc, args.k, args.min_pool, args.draws, rng)
            if r and r["picks"] >= 20:
                r.update(target=target.replace("target__", ""), score=sname)
                rows.append(r)
    if not rows:
        sys.exit(f"No market has 20+ top-{args.k} picks with pool >= {args.min_pool}. "
                 f"Lower --min-pool or --k.")
    t = pd.DataFrame(rows)
    t["q"] = bh(t["p"].values)

    L = []
    L.append("=" * 100)
    L.append(f"  TOP-{args.k} PER MARKET PER DAY vs RANDOM {args.k}")
    L.append(f"  Generated : {ts}     Files: {n_files}     Settled picks used: {len(df)}")
    L.append(f"  Pool rule : a market must have >= {args.min_pool} picks that day. "
             f"Odds filter: {args.min_odds} - {args.max_odds}")
    L.append("=" * 100)
    L.append("  Win% = top picks' win rate. Random% = win rate of random picks from the same days.")
    L.append("  Gap = Win% - Random%. z/q: is the gap bigger than luck (q corrected for all rows).")

    L.append("\n" + "-" * 100)
    L.append(f"  {'Market':<24} {'Score':<11} {'Days':>5} {'Picks':>6} {'Win%':>6} {'Rnd%':>6} "
             f"{'Gap':>6} {'z':>5} {'q':>5} {'ROI':>7} {'RndROI':>7}")
    L.append("  " + "-" * 96)
    for _, r in t.sort_values(["target", "score"]).iterrows():
        L.append(f"  {r['target'][:23]:<24} {r['score']:<11} {r['pools']:>5} {r['picks']:>6} "
                 f"{r['win']:>5.1f}% {r['rnd']:>5.1f}% {r['gap']:>+6.1f} {r['z']:>5.1f} {r['q']:>5.2f} "
                 f"{r['roi']:>+6.1f}% {r['roi_rnd']:>+6.1f}%")

    L.append("\n" + "=" * 100)
    L.append("  ALL MARKETS TOGETHER, by score (the cleanest test: one number per score)")
    L.append("=" * 100)
    L.append(f"  {'Score':<12} {'Picks':>6} {'Win%':>6} {'Rnd%':>6} {'Gap':>6} {'z':>5} {'ROI':>7} {'RndROI':>7}")
    pooled_ok = False
    for sname in SCORES:
        s = t[t["score"] == sname]
        if s.empty:
            continue
        picks = s["picks"].sum()
        win = (s["win"] * s["picks"]).sum() / picks
        rnd = (s["rnd"] * s["picks"]).sum() / picks
        # combine z across markets (weights by sqrt-variance): z_total = sum(gap_i*n_i)/sqrt(sum var_i)
        num = ((s["win"] - s["rnd"]) / 100 * s["picks"]).sum()
        den = np.sqrt(((((s["win"] - s["rnd"]) / 100 * s["picks"]) / s["z"].replace(0, np.nan)) ** 2).sum())
        z_all = num / den if den and not np.isnan(den) else 0.0
        roi = (s["roi"] * s["picks"]).sum() / picks
        rroi = (s["roi_rnd"] * s["picks"]).sum() / picks
        if z_all > 2.5:
            pooled_ok = True
        L.append(f"  {sname:<12} {picks:>6} {win:>5.1f}% {rnd:>5.1f}% {win - rnd:>+6.1f} {z_all:>5.1f} "
                 f"{roi:>+6.1f}% {rroi:>+6.1f}%")

    L.append("\n" + "=" * 100)
    L.append("  VERDICT")
    L.append("=" * 100)
    surv = t[t["q"] < 0.10]
    if len(surv):
        L.append(f"  {len(surv)} market/score combination(s) beat random after correction:")
        for _, r in surv.iterrows():
            L.append(f"    {r['target']} ranked by {r['score']}: {r['win']:.1f}% vs {r['rnd']:.1f}% random "
                     f"(gap {r['gap']:+.1f}, {r['picks']} picks)")
        L.append("  Next step: confirm on days AFTER these (they were found on this same data).")
    elif pooled_ok:
        L.append("  No single market survives correction, but one score is better than random overall.")
    else:
        L.append("  NO RANKING CLEARLY BEATS RANDOM: the best 2 per market did not win more often than")
        L.append(f"  2 random picks by a margin larger than luck. {len(t)} rows tested; about "
                 f"{0.05 * len(t):.0f} would look good by luck.")
    L.append(f"  Detection limit: a typical row has about {int(t['picks'].median())} picks, so a gap needs to be "
             f"about {1.96 * t['se'].median():.0f} points or more to stand out from luck.")
    L.append("  Smaller real advantages cannot be confirmed with this many days.")
    if ungraded:
        L.append(f"\n  Not gradable from the saved scores (skipped): {', '.join(u.replace('target__','') for u in ungraded[:12])}"
                 + (" ..." if len(ungraded) > 12 else ""))
    L.append("=" * 100)

    out = "\n".join(L)
    print(out)
    open("top2_report.txt", "w", encoding="utf-8").write(out)
    t.to_csv("top2_table.csv", index=False)
    print("\nSaved top2_report.txt and top2_table.csv")


if __name__ == "__main__":
    main()

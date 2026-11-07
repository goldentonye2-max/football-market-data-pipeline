"""
=============================================================
  pattern_miner.py
  Looks for pre-match ODDS patterns that separate WINNERS from
  LOSERS for one market inside one odds band.

  Idea: take every game of a market (e.g. HT Over 1.5) whose own
  price was in a band (e.g. 1.91-2.00), pull the full pre-match
  odds sheet for each game from sportybet.db, then ask what
  differs between the ones that won and the ones that lost.

  Two sources of games:
    --source fixtures  (default, recommended)
        every FINISHED fixture in the DB that had this market in
        the band. Big sample, no model-selection bias.
    --source picks
        only games your model picked (graded CSVs). Small sample,
        biased by the model's own selection.

  Checks it runs (so noise is not mistaken for a pattern):
    1. Univariate: each odds column, won vs lost (Mann-Whitney),
       with Benjamini-Hochberg correction for testing hundreds of
       columns at once. Prints how many "hits" chance alone gives.
    2. Hold-out rule test: pick the best columns on the EARLIER 70%
       of games, then test the rule on the LATER 30%.
    3. ML test: gradient boosting trained on earlier games, scored
       on later games, compared with a label-shuffled null.

  USAGE (inside ML_model - Copy):
    python pattern_miner.py --target ht_over_1_5 --min-odds 1.91 --max-odds 2.00
    python pattern_miner.py --target over_1_5 --min-odds 1.30 --max-odds 1.35
    python pattern_miner.py --target away_win --min-odds 1.51 --max-odds 1.70 --source picks

  Targets: ht_over_1_5 ht_over_0_5 ht_away_win ht_home_win
           away_win home_win over_1_5 over_2_5 btts_yes
=============================================================
"""

import argparse
import glob
import os
import sqlite3
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DB_PATH      = r"..\sportybet.db"
PICKS_FOLDER = "."

# key -> graded-csv target, (market_id, specifier, outcome_desc), needs HT score, win rule
TARGETS = {
    "ht_over_1_5": dict(csv="target__ht_over_1_5_goals", market=("68", "total=1.5", "Over 1.5"),
                        ht=True,  win=lambda d: (d.ht_home + d.ht_away) > 1),
    "ht_over_0_5": dict(csv="target__ht_over_0_5_goals", market=("68", "total=0.5", "Over 0.5"),
                        ht=True,  win=lambda d: (d.ht_home + d.ht_away) > 0),
    "ht_away_win": dict(csv="target__ht_away_win",       market=("60", "", "Away"),
                        ht=True,  win=lambda d: d.ht_away > d.ht_home),
    "ht_home_win": dict(csv="target__ht_home_win",       market=("60", "", "Home"),
                        ht=True,  win=lambda d: d.ht_home > d.ht_away),
    "away_win":    dict(csv="target__away_win",          market=("1", "", "Away"),
                        ht=False, win=lambda d: d.ft_away > d.ft_home),
    "home_win":    dict(csv="target__home_win",          market=("1", "", "Home"),
                        ht=False, win=lambda d: d.ft_home > d.ft_away),
    "over_1_5":    dict(csv="target__over_1_5_goals",    market=("18", "total=1.5", "Over 1.5"),
                        ht=False, win=lambda d: (d.ft_home + d.ft_away) > 1),
    "over_2_5":    dict(csv="target__over_2_5_goals",    market=("18", "total=2.5", "Over 2.5"),
                        ht=False, win=lambda d: (d.ft_home + d.ft_away) > 2),
    "btts_yes":    dict(csv="target__btts_yes",          market=("29", "", "Yes"),
                        ht=False, win=lambda d: (d.ft_home > 0) & (d.ft_away > 0)),
}


def clean(text):
    if pd.isna(text) or text is None:
        return "none"
    return (str(text).strip().lower()
            .replace(" ", "_").replace("/", "_").replace("-", "_")
            .replace(".", "_").replace("(", "").replace(")", "")
            .replace(":", "").replace("'", "")
            .replace(">", "over").replace("<", "under"))


def to_dt(s):
    s = s.astype(str).str.strip()
    if s.str.fullmatch(r"\d+").all():
        v = s.astype("int64")
        return pd.to_datetime(v, unit="ms" if v.max() > 1e11 else "s")
    return pd.to_datetime(s, errors="coerce")


def in_band(odds, lo, hi):
    m = pd.Series(True, index=odds.index)
    if lo is not None:
        m &= odds > lo
    if hi is not None:
        m &= odds <= hi
    return m


# ── GAME LISTS ────────────────────────────────────────────────────────────
def load_fixtures(conn, t, lo, hi, since):
    mid, spec, desc = t["market"]
    df = pd.read_sql_query("""
        SELECT o.event_id, o.odds AS decimal_odds, f.kickoff_time,
               r.ht_home, r.ht_away, r.ft_home, r.ft_away
        FROM odds o
        JOIN fixtures f ON f.event_id = o.event_id
        JOIN results  r ON r.event_id = o.event_id
        WHERE o.market_id = ? AND COALESCE(o.specifier,'') = ? AND o.outcome_desc = ?
          AND r.status = 'finished' AND r.ft_home IS NOT NULL AND r.ft_away IS NOT NULL
          AND o.odds IS NOT NULL
    """, conn, params=(mid, spec, desc))
    if t["ht"]:
        df = df.dropna(subset=["ht_home", "ht_away"])
    df["event_id"] = df["event_id"].astype(str)
    df = df.drop_duplicates("event_id")
    df = df[in_band(df["decimal_odds"], lo, hi)].copy()
    df["won"] = t["win"](df).astype(int)
    df["when"] = to_dt(df["kickoff_time"])
    if since:
        df = df[df["when"] >= pd.Timestamp(since)]
    return df[["event_id", "decimal_odds", "won", "when"]]


def load_picks(t, lo, hi, since):
    files = sorted(glob.glob(os.path.join(PICKS_FOLDER, "all_picks_????-??-??_graded.csv")))
    if not files:
        sys.exit("ERROR: no graded files found.")
    frames = []
    for f in files:
        day = os.path.basename(f).replace("all_picks_", "").replace("_graded.csv", "")
        d = pd.read_csv(f)
        d = d[(d["target"] == t["csv"]) & d["result"].isin(["WON", "LOST"])].copy()
        d["when"] = pd.Timestamp(day)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["event_id"] = df["event_id"].astype(str)
    df = df.sort_values("when").drop_duplicates("event_id", keep="first")
    df = df[in_band(df["decimal_odds"], lo, hi)].copy()
    df["won"] = (df["result"] == "WON").astype(int)
    if since:
        df = df[df["when"] >= pd.Timestamp(since)]
    return df[["event_id", "decimal_odds", "won", "when"]]


# ── FEATURES: FULL PRE-MATCH ODDS SHEET ──────────────────────────────────
def build_features(conn, ids, min_cov):
    parts = []
    for i in range(0, len(ids), 500):
        ch = ids[i:i + 500]
        ph = ",".join("?" * len(ch))
        parts.append(pd.read_sql_query(
            f"SELECT event_id, market_name, specifier, outcome_desc, odds "
            f"FROM odds WHERE event_id IN ({ph}) AND odds IS NOT NULL", conn, params=ch))
    raw = pd.concat(parts, ignore_index=True)
    raw["event_id"] = raw["event_id"].astype(str)
    raw["mkt"] = raw["market_name"].map(clean) + "__" + raw["specifier"].fillna("").map(clean)
    raw["col_key"] = raw["mkt"] + "__" + raw["outcome_desc"].map(clean)
    raw = raw.drop_duplicates(["event_id", "col_key"])
    per_event = raw.groupby("event_id").size()
    n_ev = len(ids)

    # Keep only columns present for enough games BEFORE pivoting. Correct-score,
    # handicap and player lines create hundreds of thousands of rare columns that
    # would otherwise blow up memory.
    key_cov = raw.groupby("col_key")["event_id"].nunique() / n_ev
    keep_keys = key_cov[key_cov >= min_cov].index
    mkt_cov = raw.groupby("mkt")["event_id"].nunique() / n_ev
    keep_mkts = mkt_cov[mkt_cov >= min_cov].index
    print(f"  {len(key_cov)} distinct odds columns in DB for these games; "
          f"{len(keep_keys)} present for >= {min_cov:.0%} of games (kept)")

    depth = raw.groupby("event_id")["mkt"].nunique().rename("n_markets")

    rk = raw[raw["col_key"].isin(keep_keys)]
    odds_w = rk.pivot(index="event_id", columns="col_key", values="odds")
    odds_w.columns = ["odds__" + c for c in odds_w.columns]

    rm = raw[raw["mkt"].isin(keep_mkts)].copy()
    rm["inv"] = 1.0 / rm["odds"]
    g = rm.groupby(["event_id", "mkt"]).agg(n=("inv", "size"), s=("inv", "sum")).reset_index()
    g = g[(g["n"] >= 2) & (g["n"] <= 3)]
    marg = g.pivot(index="event_id", columns="mkt", values="s")
    marg.columns = ["margin__" + c for c in marg.columns]

    X = pd.concat([odds_w, marg, depth], axis=1).reindex(ids)
    X = X.loc[:, X.notna().mean() >= min_cov]
    X = X.loc[:, X.nunique(dropna=True) > 2]
    return X, per_event


# ── ANALYSIS ──────────────────────────────────────────────────────────────
def univariate(d, feats):
    y = d["won"].values
    rows = []
    for c in feats:
        v = d[c].values
        a = v[(y == 1) & ~np.isnan(v)]
        b = v[(y == 0) & ~np.isnan(v)]
        if len(a) < 15 or len(b) < 15:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append((c, u / (len(a) * len(b)), np.median(a), np.median(b), p))
    r = pd.DataFrame(rows, columns=["feature", "auc", "med_won", "med_lost", "p"])
    if r.empty:
        return r
    r = r.sort_values("p").reset_index(drop=True)
    m = len(r)
    q = (r["p"] * m / (np.arange(m) + 1))[::-1].cummin()[::-1].clip(upper=1.0)
    r["q"] = q.values
    return r


def holdout_rules(d, feats, k=5):
    d = d.sort_values("when").reset_index(drop=True)
    cut = int(len(d) * 0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    ranked = univariate(tr, feats).head(k)
    out = []
    base = te["won"].mean() * 100
    for _, r in ranked.iterrows():
        f = r["feature"]
        thr = tr[f].median()
        high = r["med_won"] > r["med_lost"]
        cond = (te[f] >= thr) if high else (te[f] <= thr)
        sub = te[cond.fillna(False)]
        if len(sub) == 0:
            continue
        profit = np.where(sub["won"] == 1, sub["decimal_odds"] - 1, -1.0)
        out.append((f, ">=" if high else "<=", thr, len(sub), sub["won"].mean() * 100,
                    profit.mean() * 100, r["p"]))
    return out, base, len(tr), len(te)


def ml_check(d, feats, n_perm):
    d = d.sort_values("when").reset_index(drop=True)
    cut = int(len(d) * 0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    if te["won"].nunique() < 2 or tr["won"].nunique() < 2:
        return None

    def fit_auc(ytr):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05,
                                           min_samples_leaf=20, random_state=0)
        m.fit(tr[feats], ytr)
        return roc_auc_score(te["won"], m.predict_proba(te[feats])[:, 1])

    real = fit_auc(tr["won"].values)
    rng = np.random.default_rng(0)
    null = np.array([fit_auc(rng.permutation(tr["won"].values)) for _ in range(n_perm)])
    p = (1 + (null >= real).sum()) / (1 + n_perm)
    # Baseline: how well the market's OWN price for this outcome ranks winners
    price_auc = roc_auc_score(te["won"], 1.0 / te["decimal_odds"])
    return real, null.mean(), np.percentile(null, 95), p, len(tr), len(te), price_auc


def section(title, w=72):
    return "\n" + "=" * w + "\n  " + title + "\n" + "=" * w


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Odds-pattern miner: winners vs losers")
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--min-odds", type=float, default=None, help="band lower edge (exclusive)")
    ap.add_argument("--max-odds", type=float, default=None, help="band upper edge (inclusive)")
    ap.add_argument("--source", choices=["fixtures", "picks"], default="fixtures")
    ap.add_argument("--since", type=str, default=None, help="only games on/after YYYY-MM-DD")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="drop odds columns present for fewer than this share of games")
    ap.add_argument("--perms", type=int, default=30, help="label-shuffle runs for the ML null")
    args = ap.parse_args()

    t = TARGETS[args.target]
    if not os.path.exists(DB_PATH):
        sys.exit(f"ERROR: database not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    print(f"Loading {args.target} games ({args.source}) in band "
          f"{args.min_odds}-{args.max_odds} ...")
    ev = (load_fixtures(conn, t, args.min_odds, args.max_odds, args.since)
          if args.source == "fixtures"
          else load_picks(t, args.min_odds, args.max_odds, args.since))
    if len(ev) < 60:
        sys.exit(f"Only {len(ev)} games in that band - too few to study. Widen the band "
                 f"or use --source fixtures.")

    print(f"{len(ev)} games. Pulling their full pre-match odds sheets...")
    X, per_event = build_features(conn, ev["event_id"].tolist(), args.min_coverage)
    conn.close()
    feats = list(X.columns)
    d = ev.merge(X, left_on="event_id", right_index=True, how="left")
    d = d.sort_values("when").reset_index(drop=True)

    n, w = len(d), int(d["won"].sum())
    be = (1.0 / d["decimal_odds"]).mean() * 100
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append("=" * 72)
    L.append(f"  ODDS PATTERN MINER  -  {args.target}")
    L.append(f"  Generated : {ts}")
    L.append(f"  Source    : {args.source}   Band: {args.min_odds} - {args.max_odds}")
    L.append(f"  Period    : {d['when'].min():%Y-%m-%d} to {d['when'].max():%Y-%m-%d}")
    L.append("=" * 72)
    L.append(f"  Games {n}   Won {w} ({w / n * 100:.1f}%)   Lost {n - w}   "
             f"Break-even win% at these odds: {be:.1f}%")
    L.append(f"  Odds columns studied: {len(feats)}   "
             f"(median odds rows per game: {int(per_event.median())})")
    L.append("  Check: odds rows per game should be roughly constant. If the DB overwrote odds")
    L.append("  after kickoff (live odds), the 'pre-match' sheet is contaminated.")

    # 1) univariate
    uni = univariate(d, feats)
    L.append(section("1. EACH ODDS COLUMN: WON vs LOST"))
    if uni.empty:
        L.append("  Not enough data per column.")
    else:
        m = len(uni)
        L.append(f"  Columns tested         : {m}")
        L.append(f"  p < 0.05 by raw test   : {(uni['p'] < 0.05).sum()}   "
                 f"(chance alone gives about {0.05 * m:.0f})")
        L.append(f"  Survive FDR (q < 0.10) : {(uni['q'] < 0.10).sum()}")
        if (uni["q"] < 0.10).sum() > 0.5 * m:
            L.append("  NOTE: most columns 'pass' because they all measure the same thing (how many")
            L.append("  goals the market expects) and big samples make tiny gaps significant.")
            L.append("  This count is NOT evidence of a hidden pattern.")
        L.append("")
        L.append(f"  {'Column':<52} {'AUC':>5} {'MedWon':>7} {'MedLost':>8} {'p':>8} {'q':>6}")
        L.append("  " + "-" * 90)
        for _, r in uni.head(15).iterrows():
            L.append(f"  {r['feature'][:51]:<52} {r['auc']:>5.3f} {r['med_won']:>7.2f} "
                     f"{r['med_lost']:>8.2f} {r['p']:>8.4f} {r['q']:>6.3f}")
        L.append("  AUC 0.50 = columns do not separate winners from losers. Related markets")
        L.append("  (e.g. DNB, double chance, 1X2) move together - do not count them as separate")
        L.append("  findings.")

    # 1b) profile: what the winners' odds sheets look like vs the losers'
    L.append(section("1b. PROFILE: typical pre-match odds, WINNERS vs LOSERS"))
    headline = [
        "odds__1x2____home", "odds__1x2____draw", "odds__1x2____away",
        "odds__over_under__total=0_5__over_0_5", "odds__over_under__total=1_5__over_1_5",
        "odds__over_under__total=2_5__over_2_5", "odds__over_under__total=3_5__over_3_5",
        "odds__gg_ng____yes", "odds__gg_ng____no",
        "odds__1st_half___over_under__total=0_5__over_0_5",
        "odds__1st_half___over_under__total=1_5__over_1_5",
        "odds__1st_half___1x2____home", "odds__1st_half___1x2____draw",
        "odds__1st_half___1x2____away",
    ]
    prof_cols = [c for c in headline if c in feats]
    if not uni.empty:
        for c in uni.head(10)["feature"]:
            if c not in prof_cols:
                prof_cols.append(c)
    won_d, lost_d = d[d["won"] == 1], d[d["won"] == 0]

    def fmt(s):
        s = s.dropna()
        if len(s) < 5:
            return "n/a"
        return f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]"

    L.append(f"  {len(won_d)} winners vs {len(lost_d)} losers. Format: median [25th-75th percentile]")
    L.append(f"  {'Column':<46} {'WINNERS':<24} {'LOSERS':<24}")
    L.append("  " + "-" * 96)
    for c in prof_cols:
        L.append(f"  {c[:45]:<46} {fmt(won_d[c]):<24} {fmt(lost_d[c]):<24}")
    L.append("  If the winner and loser ranges overlap almost completely, the two groups look")
    L.append("  the same on that column, so it cannot tell them apart.")

    # 2) hold-out rules
    L.append(section("2. HOLD-OUT RULE TEST (choose on earlier 70%, test on later 30%)"))
    rules, base, ntr, nte = holdout_rules(d, feats)
    L.append(f"  Train games {ntr}   Test games {nte}   Test win rate (no filter): {base:.1f}%")
    L.append("")
    L.append(f"  {'Rule (threshold = train median)':<58} {'Games':>5} {'Win%':>6} {'ROI':>8}")
    L.append("  " + "-" * 82)
    for f, op, thr, k, wr, roi, p in rules:
        L.append(f"  {(f[:40] + ' ' + op + ' ' + f'{thr:.2f}'):<58} {k:>5} {wr:>5.1f}% {roi:>+7.1f}%")
    if not rules:
        L.append("  No rules produced.")
    L.append("  A real pattern beats the unfiltered win rate on the TEST games too.")

    # 3) ML
    L.append(section("3. ML TEST (train earlier, score later, vs shuffled-label null)"))
    if len(feats) == 0 or n < 150:
        L.append("  Skipped: fewer than 150 games or no usable columns.")
        ml = None
    else:
        print("Running ML check ...")
        ml = ml_check(d, feats, args.perms)
        if ml is None:
            L.append("  Skipped: one class missing in train or test.")
        else:
            real, nmean, n95, p, ntr2, nte2, price_auc = ml
            L.append(f"  Train {ntr2} / Test {nte2} games")
            L.append(f"  Test AUC (real labels)        : {real:.3f}")
            L.append(f"  Market's own price alone AUC  : {price_auc:.3f}   <- the bar to beat")
            L.append(f"  Test AUC (shuffled, average)  : {nmean:.3f}")
            L.append(f"  Shuffled 95th percentile      : {n95:.3f}")
            L.append(f"  p-value                       : {p:.3f}")
            L.append("  Real AUC well above the shuffled 95th percentile = something transferable.")
            L.append("  AUC near 0.50 = the odds sheet does not predict who wins in this band.")

    # verdict
    L.append(section("VERDICT"))
    rules_pos = sum(1 for r in rules if r[5] > 0 and r[3] >= 100)
    ml_ok = (ml is not None and ml[3] < 0.05 and ml[0] > ml[6] + 0.01)
    if ml_ok and rules_pos > 0:
        v = ("SIGNAL: the odds sheet beats the market's own price AND a hold-out rule made "
             "money on later games. Next: full backtest of that rule on new days.")
    elif ml_ok or rules_pos > 0:
        v = "WEAK/MIXED: one test passes, the other does not. Treat as a hypothesis only."
    else:
        v = ("NO PATTERN BEYOND THE MARKET'S OWN PRICE: the odds sheet only repeats what this "
             "market's price already says, and no hold-out rule made money.")
    L.append("  " + v)
    L.append("\n" + "=" * 72)

    report = "\n".join(L)
    print("\n" + report)
    rep = f"pattern_report_{args.target}.txt"
    csv = f"pattern_dataset_{args.target}.csv"
    with open(rep, "w", encoding="utf-8") as fh:
        fh.write(report)
    d.to_csv(csv, index=False)
    print(f"\nSaved {rep} and {csv}")


if __name__ == "__main__":
    main()
"""
=============================================================
  shape_miner.py
  Looks at the WHOLE pre-match odds sheet of each game (every
  column kept by pattern_miner.py, no limit on how many markets
  take part) and asks: do winners have a common SHAPE that losers
  do not?

  It does not care about profit or beating the bookmaker.

  Input : pattern_dataset_<target>.csv  (written by pattern_miner.py)
          Run pattern_miner.py first for the market + odds band you
          want, e.g.
              python pattern_miner.py --target over_1_5 --min-odds 1.25 --max-odds 1.40
          then:
              python shape_miner.py --target over_1_5

  What it prints:
    1. DEPENDENCY MAP   which odds columns move together (families) and
                        how many independent things the sheet really holds
    2. THE PICTURE      the main directions of variation in the sheet,
                        the groups of similar-looking sheets, and a
                        side-by-side of the most winner-heavy group vs
                        the most loser-heavy group (all columns)
    3. LATER-GAMES CHECK walk-forward: shapes learned on earlier games
                        are used to score later games. Tells you whether
                        the shape really relates to winning, or was chance.
    4. VERDICT

  Nothing about results is used to build the shapes; results are only
  used afterwards to see which shapes the winners sit in.

  OPTIONS:
    --k 8            number of sheet-groups
    --folds 4        number of later-games test folds (walk-forward)
    --knn 50         neighbours for the "most similar sheets" test
    --min-cluster 100  smallest group allowed in the best-vs-worst picture
    --csv FILE       use a specific dataset file instead of --target
=============================================================
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def short(c, n=44):
    return c.replace("odds__", "").replace("margin__", "margin:")[:n]


def section(title, w=76):
    return "\n" + "=" * w + "\n  " + title + "\n" + "=" * w


# ── MATRIX PREP ───────────────────────────────────────────────────────────
def raw_matrix(df, cols):
    X = df[cols].astype(float).copy()
    for c in cols:
        if c.startswith("odds__"):
            X[c] = np.log(X[c].where(X[c] > 0))   # odds ratios matter, not absolute gaps
    return X


class Prep:
    """Median-fill, standardise, clip. Fitted on training games only."""

    def fit(self, X):
        self.med = X.median()
        Xf = X.fillna(self.med)
        self.mu = Xf.mean()
        sd = Xf.std(ddof=0).replace(0, np.nan)
        self.sd = sd
        self.keep = list(sd.dropna().index)
        return self

    def transform(self, X):
        Xf = X[self.keep].fillna(self.med[self.keep])
        Z = (Xf - self.mu[self.keep]) / self.sd[self.keep]
        return Z.clip(-5, 5).values


def fit_pca(Z, var=0.90, cap=30):
    n_comp = max(2, min(cap, Z.shape[1], Z.shape[0] - 1))
    p = PCA(n_components=n_comp, random_state=0).fit(Z)
    cum = np.cumsum(p.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var) + 1)
    k = max(2, min(k, n_comp))
    return p, k


def auc_p(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or np.std(s) == 0:
        return 0.5, 1.0
    auc = roc_auc_score(y, s)
    p = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").pvalue
    return auc, p


# ── ONE WALK-FORWARD FOLD ────────────────────────────────────────────────
def fold_scores(tr, te, cols, k, knn):
    pre = Prep().fit(raw_matrix(tr, cols))
    Ztr = pre.transform(raw_matrix(tr, cols))
    Zte = pre.transform(raw_matrix(te, cols))
    pca, kk = fit_pca(Ztr)
    Ptr = pca.transform(Ztr)[:, :kk]
    Pte = pca.transform(Zte)[:, :kk]

    ytr = tr["won"].values
    base = ytr.mean()

    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(Ptr)
    ctr, cte = km.labels_, km.predict(Pte)
    cw = {}
    for c in range(k):
        m = ctr == c
        cw[c] = (ytr[m].sum() + 5 * base) / (m.sum() + 5)     # smoothed group win rate
    s_cluster = np.array([cw[c] for c in cte])

    nn = NearestNeighbors(n_neighbors=min(knn, len(Ptr))).fit(Ptr)
    idx = nn.kneighbors(Pte, return_distance=False)
    s_knn = ytr[idx].mean(axis=1)                              # win rate of most similar past sheets
    return s_cluster, s_knn


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Whole-sheet shape miner: winners vs losers")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--knn", type=int, default=50)
    ap.add_argument("--min-cluster", type=int, default=100)
    args = ap.parse_args()

    path = args.csv or (f"pattern_dataset_{args.target}.csv" if args.target else None)
    if not path or not os.path.exists(path):
        sys.exit("ERROR: dataset not found. Run pattern_miner.py first, then use --target "
                 "(or --csv FILE).")
    tag = args.target or os.path.splitext(os.path.basename(path))[0]

    d = pd.read_csv(path)
    d["when"] = pd.to_datetime(d["when"], errors="coerce")
    d = d.dropna(subset=["won", "when"]).sort_values("when").reset_index(drop=True)
    cols = [c for c in d.columns
            if (c.startswith("odds__") or c.startswith("margin__") or c == "n_markets")
            and d[c].notna().any()]
    n = len(d)
    if n < 300 or len(cols) < 5:
        sys.exit(f"Dataset too small ({n} games, {len(cols)} columns).")
    y = d["won"].astype(int).values

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append("=" * 76)
    L.append(f"  ODDS SHAPE MINER  -  {tag}")
    L.append(f"  Generated : {ts}")
    L.append(f"  Games {n}   Won {y.sum()} ({y.mean() * 100:.1f}%)   Lost {n - y.sum()}")
    L.append(f"  Period    : {d['when'].min():%Y-%m-%d} to {d['when'].max():%Y-%m-%d}")
    L.append(f"  Odds columns used: {len(cols)} (all of them, together)")
    L.append("=" * 76)

    # shared descriptive prep (all games, no results used)
    pre = Prep().fit(raw_matrix(d, cols))
    Z = pre.transform(raw_matrix(d, cols))
    kept = pre.keep

    # 1) DEPENDENCY MAP --------------------------------------------------
    L.append(section("1. DEPENDENCY MAP - which columns move together"))
    C = np.nan_to_num(np.corrcoef(Z, rowvar=False))
    A = np.abs(C)
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0, None)
    link = linkage(squareform(D, checks=False), method="average")
    fam = fcluster(link, t=0.30, criterion="distance")      # |corr| >= ~0.70 on average
    fam_sizes = pd.Series(fam).value_counts()
    n_fam = len(fam_sizes)
    singles = int((fam_sizes == 1).sum())
    pca_all, kk_all = fit_pca(Z)
    L.append(f"  {len(kept)} columns collapse into {n_fam} families "
             f"(columns inside a family have average |correlation| >= 0.70).")
    L.append(f"  {singles} columns stand alone (move independently of the rest).")
    L.append(f"  The whole sheet is captured by about {kk_all} independent directions (90% of variation).")
    L.append("")
    L.append(f"  {'Family':<7} {'Size':>5}  Centre column  |  other members")
    L.append("  " + "-" * 74)
    for rank, (fid, size) in enumerate(fam_sizes.head(12).items(), 1):
        idx = np.where(fam == fid)[0]
        if len(idx) > 1:
            sub = A[np.ix_(idx, idx)]
            centre = idx[int(np.argmax(sub.mean(axis=1)))]
        else:
            centre = idx[0]
        others = [short(kept[i], 24) for i in idx if i != centre][:3]
        L.append(f"  {rank:<7} {size:>5}  {short(kept[centre], 30):<30} | {', '.join(others)}")
    L.append("  Families that are tightly tied together are one piece of information repeated,")
    L.append("  not many independent findings.")

    # 2) THE PICTURE -----------------------------------------------------
    L.append(section("2. THE PICTURE - shapes of the sheet, and where winners sit"))
    P = pca_all.transform(Z)[:, :kk_all]
    L.append("  (a) Main directions of variation (found WITHOUT looking at results).")
    L.append("      For each: how well it separates winners from losers, and which columns drive it.")
    L.append("      In-sample description; section 3 tests whether it holds on later games.")
    rows = []
    for j in range(min(kk_all, 15)):
        auc, _ = auc_p(y, P[:, j])
        rows.append((j, auc, abs(auc - 0.5)))
    rows.sort(key=lambda r: -r[2])
    for j, auc, _ in rows[:4]:
        comp = pca_all.components_[j]
        order = np.argsort(comp)
        hi = [short(kept[i], 30) for i in order[::-1][:5]]
        lo = [short(kept[i], 30) for i in order[:5]]
        direction = "winners score HIGHER" if auc > 0.5 else "winners score LOWER"
        L.append("")
        L.append(f"      Direction {j + 1}: AUC {max(auc, 1 - auc):.3f} ({direction}), "
                 f"explains {pca_all.explained_variance_ratio_[j] * 100:.1f}% of sheet variation")
        L.append(f"        high end: {'; '.join(hi)}")
        L.append(f"        low end : {'; '.join(lo)}")

    L.append("")
    L.append(f"  (b) {args.k} groups of similar-looking sheets (found WITHOUT looking at results):")
    km_all = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(P)
    lab = km_all.labels_
    tab = []
    for c in range(args.k):
        m = lab == c
        tab.append((c, int(m.sum()), y[m].mean() * 100))
    tab.sort(key=lambda r: -r[2])
    chi_p = chi2_contingency(pd.crosstab(lab, y))[1]
    L.append(f"      {'Group':<7} {'Games':>6} {'Win%':>7}")
    for c, cnt, wr in tab:
        L.append(f"      {c:<7} {cnt:>6} {wr:>6.1f}%")
    L.append(f"      Overall win rate {y.mean() * 100:.1f}%.  Do groups differ in win rate "
             f"(in-sample chi-square p) : {chi_p:.4f}")

    elig = [r for r in tab if r[1] >= args.min_cluster]
    if len(elig) >= 2:
        best, worst = elig[0], elig[-1]
        mb, mw = lab == best[0], lab == worst[0]
        zdiff = Z[mb].mean(axis=0) - Z[mw].mean(axis=0)
        top = np.argsort(-np.abs(zdiff))[:15]
        L.append("")
        L.append(f"  (c) MOST WINNER-HEAVY group {best[0]} ({best[1]} games, {best[2]:.1f}% won) "
                 f"vs MOST LOSER-HEAVY group {worst[0]} ({worst[1]} games, {worst[2]:.1f}% won)")
        L.append("      The 15 columns where their sheets differ most (median pre-match odds):")
        L.append(f"      {'Column':<44} {'WINNER-heavy':>13} {'LOSER-heavy':>12} {'Gap(z)':>8}")
        L.append("      " + "-" * 80)
        for i in top:
            col = kept[i]
            vb = d.loc[mb, col].median()
            vw = d.loc[mw, col].median()
            L.append(f"      {short(col):<44} {vb:>13.2f} {vw:>12.2f} {zdiff[i]:>+8.2f}")
        L.append("      Gap(z): how many standard deviations apart the two groups sit on that column.")
    pd.DataFrame({"event_id": d["event_id"], "group": lab, "won": y,
                  "decimal_odds": d["decimal_odds"]}).to_csv(
        f"shape_groups_{tag}.csv", index=False)

    # 3) LATER-GAMES CHECK ----------------------------------------------
    L.append(section("3. LATER-GAMES CHECK (walk-forward)"))
    L.append("  Shapes are learned on earlier games only (results not used to build them), then")
    L.append("  each later game is scored by how past games with the most similar sheet did.")
    chunks = np.array_split(np.arange(n), args.folds + 1)
    pooled_y, pooled_c, pooled_k = [], [], []
    L.append("")
    L.append(f"  {'Fold':<6} {'Train':>6} {'Test':>6} {'Group AUC':>10} {'Similar-sheet AUC':>18}")
    L.append("  " + "-" * 52)
    for i in range(1, args.folds + 1):
        tr_idx = np.concatenate(chunks[:i])
        te_idx = chunks[i]
        tr, te = d.iloc[tr_idx], d.iloc[te_idx]
        s_c, s_k = fold_scores(tr, te, cols, args.k, args.knn)
        yt = te["won"].astype(int).values
        a_c, _ = auc_p(yt, s_c)
        a_k, _ = auc_p(yt, s_k)
        L.append(f"  {i:<6} {len(tr):>6} {len(te):>6} {a_c:>10.3f} {a_k:>18.3f}")
        pooled_y.append(yt)
        pooled_c.append(s_c)
        pooled_k.append(s_k)
    yp = np.concatenate(pooled_y)
    sc = np.concatenate(pooled_c)
    sk = np.concatenate(pooled_k)
    a_c, p_c = auc_p(yp, sc)
    a_k, p_k = auc_p(yp, sk)
    L.append("  " + "-" * 52)
    L.append(f"  {'Pooled':<6} {'':>6} {len(yp):>6} {a_c:>10.3f} {a_k:>18.3f}")
    L.append("")
    L.append(f"  Group score      AUC {a_c:.3f}   p = {p_c:.4f}")
    L.append(f"  Similar-sheet    AUC {a_k:.3f}   p = {p_k:.4f}")
    L.append("  AUC 0.50 = shape tells nothing about who wins. 0.55+ = clear. 0.60+ = strong.")

    hi_cut, lo_cut = np.percentile(sk, 75), np.percentile(sk, 25)
    top_m, bot_m = sk >= hi_cut, sk <= lo_cut
    nt, nb = int(top_m.sum()), int(bot_m.sum())
    wt, wb = yp[top_m].mean(), yp[bot_m].mean()
    ci_t = 1.96 * np.sqrt(wt * (1 - wt) / nt)
    ci_b = 1.96 * np.sqrt(wb * (1 - wb) / nb)
    gap = (wt - wb) * 100
    ci_gap = 1.96 * np.sqrt(wt * (1 - wt) / nt + wb * (1 - wb) / nb) * 100
    L.append("")
    L.append("  In plain terms (later games only):")
    L.append(f"    Games whose sheet LOOKS LIKE past winners : {nt:>5} games, won {wt * 100:.1f}% (+/-{ci_t * 100:.1f})")
    L.append(f"    Games whose sheet LOOKS LIKE past losers  : {nb:>5} games, won {wb * 100:.1f}% (+/-{ci_b * 100:.1f})")
    L.append(f"    Gap: {gap:+.1f} points (+/-{ci_gap:.1f}).  Smaller gaps than that cannot be told from chance here.")

    # 4) VERDICT ---------------------------------------------------------
    L.append(section("VERDICT"))
    best_auc = max(a_c, a_k)
    best_p = p_k if a_k >= a_c else p_c
    if best_p < 0.01 and best_auc >= 0.55:
        v = ("SHAPE FOUND: winners' sheets share a shape that losers' do not, and it held on "
             "later games. Section 2(c) shows the picture.")
    elif best_p < 0.05 and best_auc > 0.52:
        v = ("FAINT SHAPE: a real but weak relation held on later games. Section 2(c) shows what it "
             "looks like; treat it as a hint, not a rule.")
    else:
        v = ("NO CONFIRMED SHAPE: on later games the sheet's overall shape did not separate "
             "winners from losers beyond chance. Sample size limits what can be detected "
             f"(here about {ci_gap:.0f} points).")
    L.append("  " + v)
    L.append("\n" + "=" * 76)

    report = "\n".join(L)
    print(report)
    out = f"shape_report_{tag}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nSaved {out} and shape_groups_{tag}.csv")


if __name__ == "__main__":
    main()

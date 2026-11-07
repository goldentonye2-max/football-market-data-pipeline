"""
=============================================================
  top_picks.py
  Takes the picks file written by predict_today_v2.py and gives
  you the TOP N PICKS FOR EACH MARKET, where you choose N and
  how picks are ranked. Every setting is a flag, so you can
  tweak without editing code.

  USAGE (inside ML_model - Copy):
      python top_picks.py                         # top 2 per market, shortest odds first
      python top_picks.py --top 1                 # top 1 per market
      python top_picks.py --top 3 --score model_prob
      python top_picks.py --top 4 --min-odds 1.30 --max-odds 1.70
      python top_picks.py --markets "Over 1.5" "Over 2.5" "Away Win"
      python top_picks.py --exclude-markets "Corners" "Score"
      python top_picks.py --one-per-fixture       # a fixture appears only once
      python top_picks.py --csv-in all_picks_2026-10-27.csv

  --score   short_odds (default) | model_prob | edge | long_odds
            Ties are broken with the second score in this order:
            short_odds -> model_prob; model_prob -> short_odds;
            edge -> short_odds; long_odds -> model_prob.

  Output: top_picks_<date>.txt (table) and top_picks_<date>.csv
          (same columns as your picks file, so you can also run:
           python booking_generator.py --picks-csv top_picks_<date>.csv --top 50)

  Backtest note: on your 27 graded days, the best 2 per market by
  short_odds or model_prob won ~13-16 points more often than 2
  random picks from the same pool - but the ROI was still negative
  (they are the favourites, which also pay less). Edge ranking did
  not beat random. Track results from today forward.
=============================================================
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd

SCORE_KEYS = {
    #  name        (primary col,      ascending, tie col,        tie ascending)
    "short_odds": ("decimal_odds",   True,      "model_prob_%", False),
    "long_odds":  ("decimal_odds",   False,     "model_prob_%", False),
    "model_prob": ("model_prob_%",   False,     "decimal_odds", True),
    "edge":       ("edge_%",         False,     "decimal_odds", True),
}


def latest_picks_file():
    files = sorted(glob.glob("all_picks_????-??-??.csv"))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser(description="Top N picks per market")
    ap.add_argument("--top", type=int, default=2, help="picks per market (1, 2, 3, 4 ...)")
    ap.add_argument("--score", choices=sorted(SCORE_KEYS), default="short_odds")
    ap.add_argument("--csv-in", type=str, default=None, help="picks file (default: latest all_picks_*.csv)")
    ap.add_argument("--markets", nargs="+", default=None,
                    help="only these markets (partial, case-insensitive match on the pick label)")
    ap.add_argument("--exclude-markets", nargs="+", default=None)
    ap.add_argument("--min-odds", type=float, default=None)
    ap.add_argument("--max-odds", type=float, default=None)
    ap.add_argument("--min-edge", type=float, default=None)
    ap.add_argument("--max-edge", type=float, default=None)
    ap.add_argument("--min-model-prob", type=float, default=None, help="in %%, e.g. 60")
    ap.add_argument("--min-auc", type=float, default=None)
    ap.add_argument("--min-pool", type=int, default=1,
                    help="skip a market unless it has at least this many candidate picks")
    ap.add_argument("--one-per-fixture", action="store_true",
                    help="after choosing, keep each fixture only once (its best-ranked pick)")
    ap.add_argument("--max-total", type=int, default=50, help="cap on total picks (SportyBet max is 50)")
    ap.add_argument("--sort-final", choices=["market", "kickoff", "odds"], default="market")
    args = ap.parse_args()

    path = args.csv_in or latest_picks_file()
    if not path or not os.path.exists(path):
        sys.exit("ERROR: no picks file. Run predict_today_v2.py first (it writes all_picks_<date>.csv).")
    df = pd.read_csv(path)
    need = {"event_id", "target", "pick", "decimal_odds", "model_prob_%", "edge_%"}
    if not need.issubset(df.columns):
        sys.exit(f"ERROR: picks file is missing columns: {sorted(need - set(df.columns))}")
    n0 = len(df)
    df = df.dropna(subset=["decimal_odds"]).copy()
    notes = [f"{n0} picks in {path}"]

    def keep(mask, text):
        nonlocal df
        df = df[mask]
        notes.append(f"{len(df)} after {text}")

    if args.markets:
        keep(df["pick"].apply(lambda p: any(m.lower() in str(p).lower() for m in args.markets)),
             "market filter: " + ", ".join(args.markets))
    if args.exclude_markets:
        keep(~df["pick"].apply(lambda p: any(m.lower() in str(p).lower() for m in args.exclude_markets)),
             "excluding: " + ", ".join(args.exclude_markets))
    if args.min_odds is not None:
        keep(df["decimal_odds"] >= args.min_odds, f"odds >= {args.min_odds}")
    if args.max_odds is not None:
        keep(df["decimal_odds"] <= args.max_odds, f"odds <= {args.max_odds}")
    if args.min_edge is not None:
        keep(df["edge_%"] >= args.min_edge, f"edge >= {args.min_edge}")
    if args.max_edge is not None:
        keep(df["edge_%"] <= args.max_edge, f"edge <= {args.max_edge}")
    if args.min_model_prob is not None:
        keep(df["model_prob_%"] >= args.min_model_prob, f"model prob >= {args.min_model_prob}%")
    if args.min_auc is not None and "model_auc" in df.columns:
        keep(df["model_auc"] >= args.min_auc, f"model AUC >= {args.min_auc}")
    if df.empty:
        sys.exit("No picks left after your filters. Relax them.")

    col, asc, tcol, tasc = SCORE_KEYS[args.score]
    df = df.sort_values([col, tcol], ascending=[asc, tasc], kind="mergesort")

    pools = df.groupby("target").size()
    chosen = df.groupby("target", sort=False).head(args.top)
    chosen = chosen[chosen["target"].map(pools) >= args.min_pool].copy()
    chosen["pool"] = chosen["target"].map(pools)
    chosen["rank_in_market"] = chosen.groupby("target").cumcount() + 1
    notes.append(f"{len(chosen)} picks = top {args.top} per market by {args.score} "
                 f"({chosen['target'].nunique()} markets)")

    if args.one_per_fixture:
        chosen = chosen.sort_values(["rank_in_market", col], ascending=[True, asc]) \
                       .drop_duplicates("event_id", keep="first")
        notes.append(f"{len(chosen)} after one pick per fixture")

    if len(chosen) > args.max_total:
        chosen = chosen.sort_values(["rank_in_market", col], ascending=[True, asc]).head(args.max_total)
        notes.append(f"{len(chosen)} after cap of {args.max_total}")

    if args.sort_final == "kickoff" and "kickoff" in chosen.columns:
        chosen = chosen.sort_values("kickoff")
    elif args.sort_final == "odds":
        chosen = chosen.sort_values("decimal_odds")
    else:
        chosen = chosen.sort_values(["pick", "rank_in_market"])
    chosen = chosen.reset_index(drop=True)

    today = datetime.now().strftime("%Y-%m-%d")
    L = []
    L.append("=" * 104)
    L.append(f"  TOP {args.top} PER MARKET  |  ranked by {args.score}  |  source {path}")
    L.append(f"  Generated {datetime.now():%Y-%m-%d %H:%M}")
    L.append("=" * 104)
    for n in notes:
        L.append(f"  -> {n}")
    L.append("")
    L.append(f"  {'Market':<20} {'#':>2} {'Pool':>4}  {'Match':<36} {'Odds':>5} {'Model%':>6} {'Edge%':>6}  KO")
    L.append("  " + "-" * 100)
    last = None
    for _, r in chosen.iterrows():
        mk = str(r["pick"])[:19] if r["pick"] != last else ""
        last = r["pick"]
        ko = str(r.get("kickoff", ""))[-8:-3] if "kickoff" in chosen.columns else ""
        L.append(f"  {mk:<20} {int(r['rank_in_market']):>2} {int(r['pool']):>4}  "
                 f"{str(r.get('match', '?'))[:35]:<36} {r['decimal_odds']:>5.2f} "
                 f"{r['model_prob_%']:>6.1f} {r['edge_%']:>+6.1f}  {ko}")
    tot = 1.0
    for o in chosen["decimal_odds"]:
        tot *= float(o)
    L.append("")
    L.append(f"  {len(chosen)} picks. Combined odds if booked as one slip: {tot:,.1f}")
    L.append("  Pool = how many picks that market had to choose from. Pool 2 with top 2 means no choice was made.")
    L.append("=" * 104)
    out = "\n".join(L)
    print(out)

    txt = f"top_picks_{today}.txt"
    csv = f"top_picks_{today}.csv"
    open(txt, "w", encoding="utf-8").write(out)
    chosen.drop(columns=["pool", "rank_in_market"]).to_csv(csv, index=False)
    print(f"\nSaved {txt} and {csv}")
    print(f"To book them: python booking_generator.py --picks-csv {csv} --top {min(len(chosen), 50)}")


if __name__ == "__main__":
    main()

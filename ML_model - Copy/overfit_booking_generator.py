"""
=============================================================
  OVERFIT BOOKING GENERATOR  (v2 - single combined filter)
  Same pipeline as booking_generator.py, but the pick filter
  is hard-coded to the backtest "sweet spots":

    HT Over 1.5 Goals : edge 8% to 12%
    Away Win          : edge >= 12%
    HT Away Win       : edge >= 12%

    Odds zones (all markets), matching the backtest bands:
      1.51 - 1.70   (bands 1.51-1.60 and 1.61-1.70)
      1.91 - 2.00   (band 1.91-2.00)

  One pick per fixture (highest edge), top 50 by edge.

  USAGE:
    python overfit_booking_generator.py --dry-run
    python overfit_booking_generator.py
    python overfit_booking_generator.py --top 30 --sort-by odds

  NOTE: imports helpers from booking_generator.py (keep both
  files in the same folder). Update DEVICE_ID there only.
=============================================================
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, date

import booking_generator as bg

# ── THE OVERFIT FILTER ────────────────────────────────────────
# (target column value, min_edge, max_edge)  max_edge None = no cap
MARKET_RULES = {
    "target__ht_over_1_5_goals": (8.0,  12.0),
    "target__away_win":          (12.0, None),
    "target__ht_away_win":       (12.0, None),
}

# (low exclusive, high inclusive) - same as pd.cut bins in the report
ODDS_ZONES = [(1.50, 1.70), (1.90, 2.00)]

OUT_PREFIX = "overfit_booking"


def overfit_filter(df):
    df = df.copy()
    print(f"  → {len(df)} total picks before filtering")

    # 1) per-market edge rules
    mask = pd.Series(False, index=df.index)
    for target, (lo, hi) in MARKET_RULES.items():
        m = (df["target"] == target) & (df["edge_%"] >= lo)
        if hi is not None:
            m &= (df["edge_%"] <= hi)
        mask |= m
    df = df[mask]
    print(f"  → {len(df)} after per-market edge rules")

    # 2) odds zones
    zone = pd.Series(False, index=df.index)
    for lo, hi in ODDS_ZONES:
        zone |= (df["decimal_odds"] > lo) & (df["decimal_odds"] <= hi)
    df = df[zone]
    print(f"  → {len(df)} after odds zones "
          f"{' & '.join(f'{lo+0.01:.2f}-{hi:.2f}' for lo, hi in ODDS_ZONES)}")

    if df.empty:
        print("\n  No fixtures match the overfit filter today.")
        sys.exit(0)

    # 3) one pick per fixture, highest edge
    df = df.sort_values("edge_%", ascending=False)
    df = df.drop_duplicates(subset="event_id", keep="first")
    print(f"  → {len(df)} after deduplication (one pick per fixture)")
    return df


def select_top(df, top_n, sort_col):
    cols = {"edge": "edge_%", "odds": "decimal_odds",
            "model_prob": "model_prob_%", "auc": "model_auc"}
    col = cols.get(sort_col, "edge_%")
    if col not in df.columns:
        col = "edge_%"
    df = df.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
    print(f"  → {len(df)} picks selected (top {top_n} by {col})")

    print("\n  Breakdown by market:")
    for target, n in df["target"].value_counts().items():
        print(f"    {n:>3}  {target}")
    return df


def save_output(selections, code, top_n, sort_col):
    today = date.today().isoformat()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname = os.path.join(bg.OUTPUT_DIR, f"{OUT_PREFIX}_{today}_top{top_n}.txt")

    total_odds = 1.0
    for s in selections:
        total_odds *= float(s["odds"])

    lines = [
        "=" * 70,
        f"  OVERFIT ACCUMULATOR — {today}",
        f"  Generated : {ts}",
        f"  Legs      : {len(selections)}",
        f"  Sorted by : {sort_col}",
        f"  Combined odds : {total_odds:,.2f}",
        "  Filter    : HT Over 1.5 edge 8-12% | Away Win & HT Away Win edge 12%+",
        "              odds 1.51-1.70 and 1.91-2.00",
        "=" * 70,
        "",
        f"  BOOKING CODE:  {code or 'FAILED — see terminal output'}",
        "",
        "=" * 70,
        "",
        f"  {'#':<4} {'Time':<6} {'Match':<35} {'Pick':<20} "
        f"{'Odds':>6}  {'Edge%':>6}  {'AUC':>6}",
        "  " + "─" * 85,
    ]
    for i, s in enumerate(selections, 1):
        ko = str(s["kickoff"])[-8:-3] if s["kickoff"] else "?"
        lines.append(
            f"  {i:<4} {ko:<6} {s['match'][:34]:<35} {s['pick'][:19]:<20} "
            f"{float(s['odds']):>6.2f}  +{s['edge_%']:>5.1f}%  {s['model_auc']:>6.4f}"
        )

    lines += ["", "=" * 70, "  Tournament breakdown:"]
    counts = {}
    for s in selections:
        t = s.get("tournament", "Unknown")
        counts[t] = counts.get(t, 0) + 1
    for t, c in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {c:>3}x  {t}")
    lines += ["", "=" * 70]

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  ✅ Saved to {fname}")
    return fname


def main():
    parser = argparse.ArgumentParser(
        description="Overfit booking generator - single combined filter")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of picks (max 50, default 50)")
    parser.add_argument("--sort-by", type=str, default="edge",
                        choices=["edge", "odds", "model_prob", "auc"])
    parser.add_argument("--picks-csv", type=str, default=bg.PICKS_CSV)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview slip without calling the API")
    args = parser.parse_args()

    top_n = min(args.top, bg.MAX_LEGS)
    dry_run = args.dry_run or bg.DRY_RUN

    print("\n" + "=" * 70)
    print("  OVERFIT BOOKING GENERATOR — SportyBet Accumulator")
    print("=" * 70)
    print(f"  Picks CSV  : {args.picks_csv}")
    print(f"  Top N      : {top_n}")
    print(f"  Sort by    : {args.sort_by}")
    print(f"  Dry run    : {dry_run}")

    # 1. Load + filter
    print(f"\n[1/4] Loading picks from {args.picks_csv}...")
    if not os.path.exists(args.picks_csv):
        raise FileNotFoundError(
            f"Picks file not found: {args.picks_csv}\n"
            f"Run predict_today_v2.py --csv first.")
    df = pd.read_csv(args.picks_csv)
    if df.empty:
        raise ValueError("Picks CSV is empty — no picks to book.")
    df = overfit_filter(df)
    picks_df = select_top(df, top_n, args.sort_by)

    # 2. Validate against DB
    print(f"\n[2/4] Validating selections against {bg.DB_PATH}...")
    selections = bg.build_selections(picks_df)
    if not selections:
        print("\n  ✗ No valid selections after DB lookup. Exiting.")
        return

    # 3. Slip
    print("\n[3/4] Bet slip preview:")
    bg.print_slip(selections)

    # 4. Book
    if dry_run:
        print("\n  [DRY RUN] — API call skipped.")
        code = "DRY-RUN-CODE"
    else:
        if input("\n  Book this accumulator? (y/n): ").strip().lower() != "y":
            print("  Cancelled.")
            return
        print(f"\n[4/4] Posting {len(selections)}-leg accumulator to SportyBet...")
        code, _ = bg.post_accumulator(selections)
        if code:
            print(f"\n  ✅ BOOKING CODE:  {code}")
        else:
            print("\n  ✗ Booking failed. Check the raw response above.")

    save_output(selections, code, top_n, args.sort_by)

    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  Booking code : {code}")
    print(f"  Legs         : {len(selections)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

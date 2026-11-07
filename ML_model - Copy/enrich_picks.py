"""
enrich_picks.py
==============================================================
For every graded pick, adds ALL other market options for the
same game as feature columns.

Discovers every distinct (market_id, specifier, outcome_desc)
combination in your database and keeps only those that appear
in >= MIN_COVERAGE of finished fixtures (default 50%).  That
way you get the full market profile of each game without
hundreds of near-empty columns from obscure competitions.

Outputs
-------
  picks_enriched.csv    — one row per pick, with ALL market
                          option odds as extra feature columns
  feature_coverage.csv  — every discovered market option with
                          its coverage % (inspect to tune threshold)

Usage
-----
  python enrich_picks.py
  python enrich_picks.py --min-coverage 0.40   # lower = more features
  python enrich_picks.py --min-coverage 0.60   # higher = fewer features
  python enrich_picks.py --db "../../sportybet.db"
==============================================================
"""

import argparse
import glob
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

DB_PATH = r"..\sportybet.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_date(series):
    """Convert kickoff_time (unix ms/s or datetime string) to YYYY-MM-DD."""
    s = series.astype(str).str.strip()
    if s.str.fullmatch(r"\d+").all():
        v = s.astype("int64")
        dt = pd.to_datetime(v, unit="ms" if v.max() > 1e11 else "s")
    else:
        dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d")


def make_col_name(market_id, specifier, outcome_desc):
    """Create a clean, safe column name from market metadata."""
    spec = f"_{specifier}" if specifier else ""
    name = f"m{market_id}{spec}_{outcome_desc}"
    return (name
            .replace(" ", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace(">", "gt")
            .replace("<", "lt")
            .replace("(", "")
            .replace(")", ""))


def load_graded_picks():
    """Load all graded picks files and return a combined DataFrame."""
    files = sorted(glob.glob("all_picks_????-??-??_graded.csv"))
    if not files:
        sys.exit("ERROR: No graded picks files found (all_picks_*_graded.csv).\n"
                 "Run grade_all.py first to grade your picks.")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            day = f.replace("all_picks_", "").replace("_graded.csv", "")
            df["_pick_date"] = day
            df["_source_file"] = f
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")
    if not dfs:
        sys.exit("No picks loaded.")
    picks = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(picks):,} picks from {len(files)} graded files.")
    return picks, files


def find_event_id_column(picks):
    """Find the event_id column in the picks DataFrame, trying common names."""
    for name in ["event_id", "EventId", "event", "fixture_id", "id"]:
        if name in picks.columns:
            return name
    # Try case-insensitive
    lower_map = {c.lower(): c for c in picks.columns}
    for name in ["event_id", "eventid", "event", "fixture_id", "id"]:
        if name in lower_map:
            return lower_map[name]
    return None


def normalise_event_id(series):
    """
    Handle two common event_id formats:
      sr:match:73265412  →  73265412
      73265412           →  73265412
    Returns a string Series of the numeric part only.
    """
    s = series.astype(str).str.strip()
    # Strip any 'sr:match:' or 'sr:season:' prefix
    s = s.str.replace(r"^sr:\w+:", "", regex=True)
    return s


def to_won(series):
    """
    Convert result/won column to numeric 0/1.
    Handles: 'WON'/'LOST', 'WIN'/'LOSS', 1/0, True/False, 'yes'/'no'.
    """
    s = series.astype(str).str.strip().str.upper()
    mapping = {"WON": 1, "WIN": 1, "YES": 1, "TRUE": 1, "1": 1,
               "LOST": 0, "LOSS": 0, "NO": 0, "FALSE": 0, "0": 0}
    return s.map(mapping)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Enrich graded picks with full game odds profile")
    ap.add_argument("--min-coverage", type=float, default=0.50,
                    help="Market option must appear in this fraction of finished events "
                         "(default: 0.50 = 50%%)")
    ap.add_argument("--db", default=DB_PATH, help="Path to sportybet.db")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        sys.exit(f"ERROR: database not found: {args.db}\n"
                 f"Try: python enrich_picks.py --db \"../sportybet.db\"")

    conn = sqlite3.connect(args.db)

    # -----------------------------------------------------------------------
    # 1. Total finished events (baseline for coverage calculation)
    # -----------------------------------------------------------------------
    total_events = conn.execute(
        "SELECT COUNT(DISTINCT event_id) FROM results WHERE ft_home IS NOT NULL"
    ).fetchone()[0]
    print(f"\nTotal finished events in DB: {total_events:,}")

    # -----------------------------------------------------------------------
    # 2. Discover ALL market options and compute coverage
    # -----------------------------------------------------------------------
    print(f"Scanning all market options in odds table...")
    coverage = pd.read_sql_query("""
        SELECT o.market_id,
               COALESCE(o.specifier, '') AS specifier,
               o.outcome_desc,
               COUNT(DISTINCT o.event_id) AS event_count
        FROM odds o
        JOIN results r ON r.event_id = o.event_id
        WHERE r.ft_home IS NOT NULL
          AND o.odds IS NOT NULL
          AND CAST(o.odds AS REAL) > 1.0
        GROUP BY o.market_id, COALESCE(o.specifier, ''), o.outcome_desc
        ORDER BY event_count DESC
    """, conn)

    coverage["coverage_pct"] = coverage["event_count"] / total_events * 100
    coverage["col_name"] = coverage.apply(
        lambda r: make_col_name(r["market_id"], r["specifier"], r["outcome_desc"]), axis=1
    )
    coverage.to_csv("feature_coverage.csv", index=False)

    total_options = len(coverage)
    print(f"Total distinct market options found: {total_options:,}")
    print(f"Saved full list to feature_coverage.csv\n")

    # Keep only those above coverage threshold
    keep = coverage[coverage["coverage_pct"] >= args.min_coverage * 100].copy().reset_index(drop=True)
    print(f"Market options with >= {args.min_coverage*100:.0f}% coverage: {len(keep)}")
    print("-" * 65)
    for _, r in keep.iterrows():
        print(f"  {r['col_name']:<50}  {r['coverage_pct']:>5.1f}%  ({r['event_count']:,} events)")
    print("-" * 65)

    # -----------------------------------------------------------------------
    # 3. Load graded picks and identify the event_id column
    # -----------------------------------------------------------------------
    print()
    picks, files = load_graded_picks()
    print(f"Columns in picks file: {list(picks.columns)}\n")

    eid_col = find_event_id_column(picks)
    if eid_col is None:
        print("WARNING: No event_id column found in picks files.")
        print("Columns available:", list(picks.columns))
        print("\nTo make this work reliably, your picks files need an event_id column.")
        print("Attempting to match by date + fixture name as a fallback...\n")
    else:
        print(f"Using '{eid_col}' as the event identifier.")
        picks["_event_id"] = normalise_event_id(picks[eid_col])
        sample = picks["_event_id"].dropna().iloc[0] if len(picks) else "?"
        print(f"  Sample event_id after normalisation: {sample}")

    # -----------------------------------------------------------------------
    # 4. Determine which events we need odds for
    #    (only the events that appear in our picks — much faster than
    #     fetching all 25k events)
    # -----------------------------------------------------------------------
    if eid_col is not None:
        event_ids = picks["_event_id"].dropna().unique().tolist()
        print(f"Fetching odds for {len(event_ids):,} unique events from picks...")
    else:
        # Fallback: try to match by pick_date + fixture string
        # Get all events with their kickoff date and try string matching
        event_ids = None   # will fetch all

    # -----------------------------------------------------------------------
    # 5. Build a feature key for each kept market option
    # -----------------------------------------------------------------------
    keep["key"] = (keep["market_id"].astype(str) + "|" +
                   keep["specifier"] + "|" +
                   keep["outcome_desc"])
    key_to_col = dict(zip(keep["key"], keep["col_name"]))

    # -----------------------------------------------------------------------
    # 6. Fetch all odds for kept features (filtered to our events if possible)
    # -----------------------------------------------------------------------
    # Fetch odds in chunks of 2000 (avoids SQLite parameter limits and
    # is much faster than fetching all 25k events when we only need ~8k)
    CHUNK = 2000
    if event_ids is not None:
        chunks = [event_ids[i:i+CHUNK] for i in range(0, len(event_ids), CHUNK)]
        print(f"Fetching odds in {len(chunks)} chunk(s) of up to {CHUNK} events...")
        parts = []
        for i, chunk in enumerate(chunks, 1):
            ph = ",".join("?" * len(chunk))
            df = pd.read_sql_query(f"""
                SELECT CAST(event_id AS TEXT) AS event_id,
                       CAST(market_id AS TEXT) || '|' || COALESCE(specifier,'') || '|' || outcome_desc AS key,
                       CAST(odds AS REAL) AS odds
                FROM odds
                WHERE event_id IN ({ph})
                  AND odds IS NOT NULL
                  AND CAST(odds AS REAL) > 1.0
            """, conn, params=chunk)
            parts.append(df)
            print(f"  chunk {i}/{len(chunks)}: {len(df):,} rows", flush=True)
        all_odds = pd.concat(parts, ignore_index=True)
        # If nothing matched, the DB may store IDs with sr:match: prefix — try that
        if len(all_odds) == 0:
            print("  No match on numeric IDs — retrying with sr:match: prefix...")
            sr_ids = ["sr:match:" + eid for eid in event_ids]
            chunks2 = [sr_ids[i:i+CHUNK] for i in range(0, len(sr_ids), CHUNK)]
            parts = []
            for chunk in chunks2:
                ph = ",".join("?" * len(chunk))
                df = pd.read_sql_query(f"""
                    SELECT CAST(event_id AS TEXT) AS event_id,
                           CAST(market_id AS TEXT) || '|' || COALESCE(specifier,'') || '|' || outcome_desc AS key,
                           CAST(odds AS REAL) AS odds
                    FROM odds
                    WHERE event_id IN ({ph})
                      AND odds IS NOT NULL
                      AND CAST(odds AS REAL) > 1.0
                """, conn, params=chunk)
                parts.append(df)
            all_odds = pd.concat(parts, ignore_index=True)
            if len(all_odds) > 0:
                all_odds["event_id"] = normalise_event_id(all_odds["event_id"])
                print("  Matched using sr:match: prefix format in DB.")
    else:
        # No event_ids known — fetch all (slow but complete)
        print("Fetching full odds table (this may take a moment)...")
        all_odds = pd.read_sql_query("""
            SELECT CAST(event_id AS TEXT) AS event_id,
                   CAST(market_id AS TEXT) || '|' || COALESCE(specifier,'') || '|' || outcome_desc AS key,
                   CAST(odds AS REAL) AS odds
            FROM odds
            WHERE odds IS NOT NULL AND CAST(odds AS REAL) > 1.0
        """, conn)
        all_odds["event_id"] = normalise_event_id(all_odds["event_id"])

    # Keep only the market options that passed coverage filter
    all_odds = all_odds[all_odds["key"].isin(keep["key"].values)].copy()
    all_odds["col_name"] = all_odds["key"].map(key_to_col)

    # If multiple rows per (event_id, col_name), keep the one with the lowest odds
    # (most precise / closest to market price)
    all_odds = all_odds.sort_values("odds").drop_duplicates(subset=["event_id", "col_name"])

    print(f"Odds rows fetched and filtered: {len(all_odds):,}")

    # -----------------------------------------------------------------------
    # 7. Pivot to game × feature matrix
    # -----------------------------------------------------------------------
    print("Pivoting to game-level feature matrix...")
    game_features = all_odds.pivot(index="event_id", columns="col_name", values="odds")
    game_features.columns.name = None
    game_features = game_features.reset_index()
    print(f"Feature matrix: {len(game_features):,} games × {len(game_features.columns)-1} features\n")

    # -----------------------------------------------------------------------
    # 8. Join picks with game features
    # -----------------------------------------------------------------------
    if eid_col is not None:
        # Rename game_features event_id so it doesn't collide with picks' event_id column
        game_features = game_features.rename(columns={"event_id": "_gf_event_id"})
        picks_enriched = picks.merge(game_features, left_on="_event_id",
                                     right_on="_gf_event_id", how="left")
        matched = picks_enriched["_gf_event_id"].notna().sum()
        print(f"Picks matched to game features: {matched:,} / {len(picks):,}")
    else:
        # Fallback: attempt match by fixture date
        # Fetch event dates
        evt_dates = pd.read_sql_query("""
            SELECT CAST(event_id AS TEXT) AS event_id, kickoff_time
            FROM fixtures
        """, conn)
        evt_dates["pick_date"] = to_date(evt_dates["kickoff_time"])
        game_features = game_features.merge(evt_dates[["event_id", "pick_date"]],
                                             on="event_id", how="left")

        # Try to find a date column in picks
        date_col = None
        for dc in ["date", "pick_date", "kickoff_date", "Date", "kickoff_time", "_pick_date"]:
            if dc in picks.columns:
                date_col = dc
                break
        if date_col:
            picks["_pick_date_norm"] = pd.to_datetime(
                picks[date_col], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            # We can only match by date here — will produce multiple matches
            picks_enriched = picks.merge(
                game_features, left_on="_pick_date_norm", right_on="pick_date", how="left"
            )
            print(f"WARNING: matched by date only (no event_id). "
                  f"Results may be inaccurate — multiple games per day.")
        else:
            picks_enriched = picks.copy()
            print("WARNING: Could not match picks to game features.")

    # -----------------------------------------------------------------------
    # 9. Save enriched picks
    # -----------------------------------------------------------------------
    picks_enriched.to_csv("picks_enriched.csv", index=False)
    n_feat = len(game_features.columns) - 1
    print(f"\nSaved picks_enriched.csv")
    print(f"  Rows : {len(picks_enriched):,} picks")
    print(f"  Cols : {len(picks_enriched.columns)} total  "
          f"({len(picks.columns)} original + {n_feat} market-odds features)")

    # -----------------------------------------------------------------------
    # 10. Correlation analysis: which features separate winners from losers?
    # -----------------------------------------------------------------------
    won_col = None
    for wc in ["result", "won", "Won", "Result", "correct"]:
        if wc in picks_enriched.columns:
            won_col = wc
            break

    if won_col is None:
        print("\nNo result/won column found — skipping correlation analysis.")
        print("Columns available:", list(picks_enriched.columns[:20]))
        conn.close()
        return

    # Convert to numeric 0/1 (handles WON/LOST text, 1/0, True/False)
    won = to_won(picks_enriched[won_col])
    picks_enriched["_won"] = won   # add clean numeric column to output

    n_graded = won.notna().sum()
    n_wins   = int((won == 1).sum())
    if n_graded == 0:
        print(f"\nResult column '{won_col}' found but no valid WON/LOST values recognised.")
        print(f"Sample values: {picks_enriched[won_col].value_counts().head(5).to_dict()}")
        conn.close()
        return
    print(f"\nResult column : '{won_col}'")
    print(f"Graded picks  : {n_graded:,}  |  Wins: {n_wins:,}  |  "
          f"Win rate: {n_wins/n_graded*100:.1f}%")

    # Compute correlation of each feature with won
    feature_cols = [c for c in keep["col_name"] if c in picks_enriched.columns]
    print(f"\nComputing correlations for {len(feature_cols)} features...")

    corr_rows = []
    for col in feature_cols:
        vals = pd.to_numeric(picks_enriched[col], errors="coerce")
        mask = won.notna() & vals.notna()
        n = mask.sum()
        if n < 20:
            continue
        c = vals[mask].corr(won[mask])
        # Mean odds for winners vs losers
        mean_win  = vals[won == 1].mean()
        mean_lose = vals[won == 0].mean()
        corr_rows.append(dict(feature=col, corr=c, n=n,
                               mean_odds_win=mean_win, mean_odds_lose=mean_lose,
                               diff=mean_win - mean_lose))

    if not corr_rows:
        print("Not enough data for correlation analysis (need >= 20 picks per feature).")
        conn.close()
        return

    corr_df = pd.DataFrame(corr_rows).sort_values("corr")
    corr_df.to_csv("feature_correlations.csv", index=False)
    print("Saved feature_correlations.csv\n")

    # --- Display top signals ---
    sep = "=" * 80
    print(sep)
    print(f"  FEATURES THAT SIGNAL A LOSER  (most negative correlation with 'won')")
    print(f"  When these odds are LOW, your pick is MORE likely to lose")
    print(sep)
    print(f"  {'Feature':<50} {'Corr':>6}  {'Avg odds: Win':>13}  {'Lose':>8}  {'Diff':>8}")
    print("-" * 80)
    for _, r in corr_df.head(15).iterrows():
        print(f"  {r['feature']:<50} {r['corr']:>+6.3f}  "
              f"{r['mean_odds_win']:>12.2f}  {r['mean_odds_lose']:>7.2f}  {r['diff']:>+8.2f}")

    print(f"\n{sep}")
    print(f"  FEATURES THAT SIGNAL A WINNER  (most positive correlation with 'won')")
    print(f"  When these odds are HIGH, your pick is MORE likely to win")
    print(sep)
    print(f"  {'Feature':<50} {'Corr':>6}  {'Avg odds: Win':>13}  {'Lose':>8}  {'Diff':>8}")
    print("-" * 80)
    for _, r in corr_df.tail(15).iterrows():
        print(f"  {r['feature']:<50} {r['corr']:>+6.3f}  "
              f"{r['mean_odds_win']:>12.2f}  {r['mean_odds_lose']:>7.2f}  {r['diff']:>+8.2f}")

    # --- Per-market breakdown ---
    if "target" in picks_enriched.columns or "market" in picks_enriched.columns:
        mk_col = "target" if "target" in picks_enriched.columns else "market"
        print(f"\n{sep}")
        print(f"  WINNER/LOSER SIGNAL — BROKEN DOWN BY YOUR PICKED MARKET")
        print(sep)
        for mkt, grp in picks_enriched.groupby(mk_col):
            w = grp["_won"]          # already converted to 0/1 numeric
            n_w = int((w == 1).sum())
            n_l = int((w == 0).sum())
            if n_w + n_l < 10:
                continue
            print(f"\n  Market: {mkt}  "
                  f"(wins {n_w}/{n_w+n_l} = {n_w/(n_w+n_l)*100:.0f}%)")
            mkt_corrs = []
            for col in feature_cols:
                vals = pd.to_numeric(grp[col], errors="coerce")
                mask = w.notna() & vals.notna()
                if mask.sum() < 10:
                    continue
                c = vals[mask].corr(w[mask])
                if abs(c) > 0.05:   # only show meaningful correlations
                    mkt_corrs.append((col, c,
                                      vals[w == 1].mean(),
                                      vals[w == 0].mean()))
            mkt_corrs.sort(key=lambda x: x[1])
            if not mkt_corrs:
                print("    (not enough data)")
                continue
            print(f"  {'Feature':<50} {'Corr':>6}  "
                  f"{'Avg odds: Win':>13}  {'Lose':>8}")
            # Show top 5 negative and top 5 positive
            for col, c, mw, ml in mkt_corrs[:5]:
                print(f"  {col:<50} {c:>+6.3f}  {mw:>12.2f}  {ml:>7.2f}")
            if len(mkt_corrs) > 10:
                print(f"  {'...':<50}")
            for col, c, mw, ml in mkt_corrs[-5:]:
                print(f"  {col:<50} {c:>+6.3f}  {mw:>12.2f}  {ml:>7.2f}")

    print(f"\n{sep}")
    print(f"  NEXT STEPS")
    print(sep)
    print(f"  1. Open picks_enriched.csv in Excel and sort/filter on the high-correlation features.")
    print(f"  2. Use feature_correlations.csv to build simple filter rules:")
    print(f"     e.g.  'only pick over_2.5 when m1__Draw odds > 3.5'")
    print(f"  3. Test those rules on the held-out days (the most recent 10–15 days).")
    print(f"  4. Once a rule survives testing, add it to top_picks.py as a filter flag.")
    print(sep)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
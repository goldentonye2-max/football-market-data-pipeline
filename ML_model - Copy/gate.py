"""
gate.py
===============================================================
THE BET GATE — daily filter for live picks.

WORKFLOW (every matchday):
    1. python predict_today.py          → generates all_picks_YYYY-MM-DD.csv
    2. python gate.py                   → reads that file, queries sportybet.db
                                          for signal odds, outputs BET / SKIP

HOW IT WORKS:
    - Loads today's picks from all_picks_YYYY-MM-DD.csv
    - Queries sportybet.db for the 6 signal markets per game:
        m900036  HT Home Win-to-Nil Yes / No
        m900037  HT Away Win-to-Nil Yes / No
        m900038  FT Home Win-to-Nil Yes / No
        m900052  FT Away Win-to-Nil Yes / No
        m450004  Excluded Goals Home (outcomes: 0,1,2,3,4,5+)
        m810002  Excluded Goals Away (outcomes: 0,1,2,3+)
    - Derives p75 thresholds from picks_enriched.csv (ALL historical data)
    - Applies thresholds to today's picks → BET / SKIP

VALIDATED OUT-OF-SAMPLE RESULTS (at p75):
    home_win      77.6% WR   +7.4% ROI
    away_win      58.5% WR   +0.4% ROI
    over_2_5      73.7% WR   +0.4% ROI
    over_3_5      48.8% WR   +1.7% ROI
    ht_over_1_5   57.0% WR   +2.8% ROI

USAGE:
    python gate.py
    python gate.py --date 2026-11-06
    python gate.py --percentile 75      (default: strict, confirmed +ROI)
    python gate.py --percentile 50      (broader, more picks)
    python gate.py --save               (saves gate_YYYY-MM-DD.csv)
===============================================================
"""

import argparse
import os
import sqlite3
import sys
from datetime import date as dt_date

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH      = r"..\sportybet.db"
ENRICHED_CSV = "picks_enriched.csv"

# ── Signal market IDs ─────────────────────────────────────────────────────────
# These are the 6 markets that carry the edge signals.
# Column names match what enrich_picks.py writes into picks_enriched.csv.
SIGNAL_MARKETS = ["900036", "900037", "900038", "900052", "450004", "810002", "18"]

# ── Filter definitions ────────────────────────────────────────────────────────
FILTERS = {
    "home_win": [
        ("m900036_No",  "high"),
        ("m900037_Yes", "high"),
        ("m900038_No",  "high"),
        ("m900052_Yes", "high"),
    ],
    "away_win": [
        ("m900036_Yes", "high"),
        ("m900037_No",  "high"),
        ("m900038_Yes", "high"),
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
        ("m18_total=4_5_under_4_5", "high"),
    ],
    "ht_over_1_5": [
        ("m450004_4",   "high"),
        ("m810002_3+",  "high"),
    ],
}

KNOWN_RESULTS = {
    "home_win":    {"wr": 77.6, "roi": +7.4},
    "away_win":    {"wr": 58.5, "roi": +0.4},
    "over_2_5":    {"wr": 73.7, "roi": +0.4},
    "over_3_5":    {"wr": 48.8, "roi": +1.7},
    "ht_over_1_5": {"wr": 57.0, "roi": +2.8},
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_fkey(target_str):
    for k in FILTERS:
        if k.lower() in str(target_str).lower():
            return k
    return None


def derive_thresholds(enriched_df, percentile):
    """Derive p-th percentile thresholds from the full enriched history."""
    thresholds = {}
    for fkey, fcols in FILTERS.items():
        thresholds[fkey] = {}
        mask = enriched_df["target"].apply(lambda t: fkey.lower() in str(t).lower())
        sub  = enriched_df[mask]
        # Case-insensitive column lookup so filter keys always match the CSV column,
        # regardless of how enrich_picks.py named it (e.g. Under vs under in m18).
        ci_map = {c.lower(): c for c in sub.columns}
        for col, _ in fcols:
            actual = col if col in sub.columns else ci_map.get(col.lower())
            if actual is None:
                continue
            vals = pd.to_numeric(sub[actual], errors="coerce").dropna()
            if len(vals) >= 10:
                thresholds[fkey][col] = float(np.percentile(vals, percentile))
    return thresholds


def load_signal_odds(db_path, event_ids):
    """
    Query sportybet.db for the 6 signal markets for the given event_ids.
    Returns a DataFrame with one row per event, columns like m900036_No, m900036_Yes, etc.
    """
    if not os.path.exists(db_path):
        return None, f"Database not found at {db_path}"

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        return None, str(e)

    placeholders  = ",".join(["?"] * len(event_ids))
    mkt_placeholders = ",".join(["?"] * len(SIGNAL_MARKETS))

    # Try to find the market_id column name (some DBs use 'market_id', some 'marketId')
    try:
        cols_info = pd.read_sql_query("PRAGMA table_info(odds)", conn)
        col_names = cols_info["name"].str.lower().tolist()
    except Exception:
        col_names = []

    mkt_col = "market_id"
    if "market_id" not in col_names:
        for candidate in ["marketid", "market", "mkt_id"]:
            if candidate in col_names:
                mkt_col = candidate
                break

    # Outcome column
    out_col = "outcome_desc"
    if "outcome_desc" not in col_names:
        for candidate in ["outcome", "outcome_name", "outcomedesc"]:
            if candidate in col_names:
                out_col = candidate
                break

    query = f"""
        SELECT event_id, {mkt_col} AS market_id, {out_col} AS outcome_desc, odds
        FROM odds
        WHERE event_id   IN ({placeholders})
          AND CAST({mkt_col} AS TEXT) IN ({mkt_placeholders})
          AND odds IS NOT NULL
    """
    try:
        raw = pd.read_sql_query(query, conn,
                                params=list(event_ids) + SIGNAL_MARKETS)
    except Exception as e:
        # Fallback: load all odds for these events and filter in Python
        try:
            raw = pd.read_sql_query(
                f"SELECT event_id, {mkt_col} AS market_id, {out_col} AS outcome_desc, odds "
                f"FROM odds WHERE event_id IN ({placeholders})",
                conn, params=list(event_ids)
            )
            raw["market_id"] = raw["market_id"].astype(str)
            raw = raw[raw["market_id"].isin(SIGNAL_MARKETS)]
        except Exception as e2:
            conn.close()
            return None, f"Query failed: {e2}"

    conn.close()

    if raw.empty:
        return pd.DataFrame({"event_id": list(event_ids)}), None

    # Build column name:  m{market_id}_{outcome_desc}
    # Normalise outcome_desc slightly (strip spaces, keep +/= chars)
    raw["market_id"]    = raw["market_id"].astype(str).str.strip()
    raw["outcome_desc"] = raw["outcome_desc"].astype(str).str.strip()
    raw["col_name"]     = "m" + raw["market_id"] + "_" + raw["outcome_desc"]

    # For m18, we also want the under_4.5 total market formatted as
    # m18_total=4_5_Under_4_5 — predict_today uses a different naming convention,
    # so we construct it from market_id + specifier + outcome if specifier is available
    # (handled below if specifier column exists)

    # Pivot: one column per (market_id, outcome_desc), one row per event_id
    deduped = raw.drop_duplicates(subset=["event_id", "col_name"])
    wide    = deduped.pivot(index="event_id", columns="col_name", values="odds")
    wide    = wide.reset_index()

    return wide, None


def load_m18_specifier_odds(db_path, event_ids):
    """
    Separately load m18 odds with specifier to build the
    m18_total=4_5_Under_4_5 column used in the over_3_5 filter.
    """
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        conn   = sqlite3.connect(db_path)
        cols_i = pd.read_sql_query("PRAGMA table_info(odds)", conn)
        col_names = cols_i["name"].str.lower().tolist()
    except Exception:
        return pd.DataFrame()

    mkt_col = "market_id"
    out_col = "outcome_desc"
    spec_col = None
    for c in ["specifier", "spec", "market_specifier"]:
        if c in col_names:
            spec_col = c
            break

    if spec_col is None:
        conn.close()
        return pd.DataFrame()

    placeholders = ",".join(["?"] * len(event_ids))
    try:
        raw = pd.read_sql_query(
            f"SELECT event_id, {spec_col} AS specifier, {out_col} AS outcome_desc, odds "
            f"FROM odds WHERE event_id IN ({placeholders}) AND CAST({mkt_col} AS TEXT)='18'",
            conn, params=list(event_ids)
        )
    except Exception:
        conn.close()
        return pd.DataFrame()
    conn.close()

    if raw.empty:
        return pd.DataFrame()

    # Build column name matching enrich_picks.py logic:
    # "m18_" + clean(specifier) + "_" + clean(outcome_desc)
    def clean(s):
        return (str(s).strip().lower()
                .replace(" ", "_").replace(".", "_")
                .replace("/", "_").replace("-", "_"))

    raw["col_name"] = ("m18_" + raw["specifier"].apply(clean)
                       + "_" + raw["outcome_desc"].apply(clean))
    raw = raw.drop_duplicates(subset=["event_id", "col_name"])

    wide = raw.pivot(index="event_id", columns="col_name", values="odds").reset_index()
    return wide


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date",        default=None,
                    help="date to gate YYYY-MM-DD (default: today)")
    ap.add_argument("--percentile",  type=float, default=75,
                    help="filter threshold percentile (default 75 → confirmed +ROI)")
    ap.add_argument("--min-signals", type=int,   default=None,
                    help="minimum signals that must agree (default: ALL available)")
    ap.add_argument("--save",        action="store_true",
                    help="save output to gate_YYYY-MM-DD.csv")
    ap.add_argument("--enriched",    default=ENRICHED_CSV,
                    help=f"enriched history CSV (default: {ENRICHED_CSV})")
    ap.add_argument("--db",          default=DB_PATH,
                    help=f"database path (default: {DB_PATH})")
    args = ap.parse_args()

    gate_date = dt_date.today().isoformat() if args.date is None else args.date

    SEP  = "=" * 80
    SEP2 = "-" * 80

    # ── Find today's picks file ────────────────────────────────────────────
    picks_file = f"all_picks_{gate_date}.csv"
    if not os.path.exists(picks_file):
        # Try to find the most recent all_picks file
        import glob
        candidates = sorted(glob.glob("all_picks_2*.csv"))
        candidates = [c for c in candidates if "_graded" not in c]
        if candidates:
            picks_file = candidates[-1]
            print(f"  Note: {gate_date} picks not found, using {picks_file}")
        else:
            sys.exit(f"ERROR: no picks file found for {gate_date}.\n"
                     f"Run predict_today.py first.")

    print(f"\nLoading picks from {picks_file}...")
    picks = pd.read_csv(picks_file, low_memory=False)
    print(f"  {len(picks):,} picks")

    if picks.empty:
        sys.exit("No picks in file.")

    # Identify event_id column
    id_col = next((c for c in ["event_id", "_event_id", "_gf_event_id", "fixture_id"]
                   if c in picks.columns), None)
    if id_col is None:
        sys.exit("ERROR: cannot find event_id column in picks file.")

    event_ids = picks[id_col].dropna().unique().tolist()
    print(f"  {len(event_ids)} unique fixtures")

    # ── Derive thresholds from enriched history ────────────────────────────
    if not os.path.exists(args.enriched):
        sys.exit(f"ERROR: {args.enriched} not found — run enrich_picks.py first.")

    print(f"\nLoading historical thresholds from {args.enriched}...")
    enriched = pd.read_csv(args.enriched, low_memory=False)
    print(f"  {len(enriched):,} historical picks")

    thresholds = derive_thresholds(enriched, args.percentile)

    # Quick threshold summary
    print(f"\n  P{args.percentile:.0f} thresholds derived:")
    for fkey, tdict in thresholds.items():
        if tdict:
            vals_str = "  ".join(f"{c}>{v:.2f}" for c, v in tdict.items())
            print(f"    {fkey:<15}  {vals_str}")

    # ── Load signal odds from DB ───────────────────────────────────────────
    print(f"\nQuerying {args.db} for signal odds...")
    signal_wide, err = load_signal_odds(args.db, event_ids)

    if err:
        print(f"  Warning: {err}")
        print("  Proceeding without DB odds — gate will mark picks as NO DATA.")
        signal_wide = pd.DataFrame({id_col: event_ids})
    elif signal_wide is not None:
        print(f"  {len(signal_wide)} fixtures with signal odds")
        signal_cols = [c for c in signal_wide.columns if c != "event_id"]
        print(f"  Signal columns found: {len(signal_cols)}")
        for c in sorted(signal_cols)[:20]:
            print(f"    {c}")
        if len(signal_cols) > 20:
            print(f"    ... and {len(signal_cols)-20} more")

    # Load m18 specifier odds for over_3_5 filter
    m18_wide = load_m18_specifier_odds(args.db, event_ids)
    if not m18_wide.empty:
        print(f"  m18 specifier cols: {[c for c in m18_wide.columns if c != 'event_id'][:5]}")

    # ── Merge signal odds into picks ───────────────────────────────────────
    if signal_wide is not None and not signal_wide.empty:
        # signal_wide uses 'event_id'; picks uses id_col
        signal_wide = signal_wide.rename(columns={"event_id": id_col})
        picks = picks.merge(signal_wide, on=id_col, how="left")

    if not m18_wide.empty:
        m18_wide = m18_wide.rename(columns={"event_id": id_col})
        picks = picks.merge(m18_wide, on=id_col, how="left")

    # ── Evaluate each pick ─────────────────────────────────────────────────
    rows_out = []
    for _, row in picks.iterrows():
        tgt   = str(row.get("target", ""))
        fkey  = get_fkey(tgt)

        base_info = {
            "home_team":    row.get("home_team", ""),
            "away_team":    row.get("away_team", ""),
            "kickoff":      row.get("kickoff", row.get("kickoff_time", "")),
            "target":       tgt,
            "decimal_odds": row.get("decimal_odds", ""),
        }

        if fkey is None:
            rows_out.append({**base_info,
                             "signals": "-", "max_signals": "-",
                             "decision": "NOT FILTERED",
                             "detail": "target not in gate"})
            continue

        tdict   = thresholds.get(fkey, {})
        fcols   = FILTERS[fkey]
        n_fire  = 0
        n_avail = 0
        details = []

        for col, direction in fcols:
            if col not in tdict:
                details.append(f"{col}=NO_HIST")
                continue
            thresh = tdict[col]
            val    = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            if pd.isna(val):
                details.append(f"{col}=MISSING")
                continue
            n_avail += 1
            fired = (direction == "high" and val > thresh) or \
                    (direction == "low"  and val < thresh)
            if fired:
                n_fire += 1
                details.append(f"{col}={val:.2f}✓")
            else:
                details.append(f"{col}={val:.2f}✗")

        min_req = args.min_signals if args.min_signals is not None else n_avail

        if n_avail == 0:
            decision = "NO DATA"
        elif n_fire >= n_avail and n_avail >= max(2, len(fcols) - 1):
            decision = "✅ BET"
        elif n_fire >= min_req and n_fire >= 2:
            decision = "⚡ PARTIAL"
        else:
            decision = "❌ SKIP"

        rows_out.append({**base_info,
                         "signals":      n_fire,
                         "max_signals":  n_avail,
                         "decision":     decision,
                         "detail":       " | ".join(details)})

    result = pd.DataFrame(rows_out)

    # Sort: BET → PARTIAL → NOT FILTERED → NO DATA → SKIP
    order  = {"✅ BET": 0, "⚡ PARTIAL": 1, "NOT FILTERED": 2, "NO DATA": 3, "❌ SKIP": 4}
    result["_s"] = result["decision"].map(order).fillna(5)
    result["_o"] = pd.to_numeric(result["decimal_odds"], errors="coerce").fillna(0)
    result = result.sort_values(["_s", "_o"], ascending=[True, False]).drop(columns=["_s","_o"])

    # ── Print ──────────────────────────────────────────────────────────────
    bets     = result[result["decision"] == "✅ BET"]
    partials = result[result["decision"] == "⚡ PARTIAL"]
    skips    = result[result["decision"] == "❌ SKIP"]
    no_data  = result[result["decision"] == "NO DATA"]
    unfilt   = result[result["decision"] == "NOT FILTERED"]

    print(f"\n{SEP}")
    print(f"  GATE RESULTS — {gate_date}   (p{args.percentile:.0f} thresholds)")
    print(SEP)
    print(f"  ✅ BET         : {len(bets):>4} picks  (all signals agree)")
    print(f"  ⚡ PARTIAL     : {len(partials):>4} picks  (most signals agree)")
    print(f"  ❌ SKIP        : {len(skips):>4} picks  (signals disagree)")
    print(f"  NO DATA        : {len(no_data):>4} picks  (signal odds missing for this game)")
    print(f"  NOT FILTERED   : {len(unfilt):>4} picks  (target not in gate scope)")

    # ── BET picks ──────────────────────────────────────────────────────────
    if len(bets) > 0:
        print(f"\n{SEP}")
        print(f"  ✅ BET THESE  ({len(bets)} picks — all signals agree)")
        print(SEP)
        for fk in FILTERS:
            fk_bets = bets[bets["target"].apply(lambda t: fk.lower() in str(t).lower())]
            if len(fk_bets) == 0:
                continue
            known = KNOWN_RESULTS.get(fk, {})
            wr_s  = f"  [out-of-sample: WR {known['wr']:.0f}%  ROI {known['roi']:+.1f}%]" if known else ""
            print(f"\n  ── {fk.upper()}{wr_s}")
            for _, r in fk_bets.iterrows():
                print(f"  {r['home_team']} vs {r['away_team']}")
                print(f"    kick: {str(r['kickoff'])[:16]}   odds: {r['decimal_odds']}   "
                      f"{r['signals']}/{r['max_signals']} signals")
                print(f"    {r['detail']}")
    else:
        print(f"\n  No picks pass the full combo filter today at p{args.percentile:.0f}.")
        if len(partials) > 0:
            print(f"  Run with --min-signals 2 to include PARTIAL picks.")
        if args.percentile == 75:
            print(f"  Run with --percentile 50 for the broader filter.")

    # ── PARTIAL picks ──────────────────────────────────────────────────────
    if len(partials) > 0:
        print(f"\n{SEP}")
        print(f"  ⚡ PARTIAL  ({len(partials)} picks — most signals agree, one missing/failed)")
        print(SEP)
        for _, r in partials.iterrows():
            print(f"  {r['home_team']} vs {r['away_team']:<30}  "
                  f"odds {r['decimal_odds']}  "
                  f"{r['signals']}/{r['max_signals']} signals  "
                  f"kick {str(r['kickoff'])[:16]}")

    # ── Target summary ─────────────────────────────────────────────────────
    if len(bets) > 0:
        print(f"\n{SEP}")
        print(f"  BET PICKS BY TARGET")
        print(SEP)
        for fk in FILTERS:
            fk_bets = bets[bets["target"].apply(lambda t: fk.lower() in str(t).lower())]
            if len(fk_bets) == 0:
                continue
            known = KNOWN_RESULTS.get(fk, {})
            wr_s = f"  expected WR {known['wr']:.0f}%  ROI {known['roi']:+.1f}%" if known else ""
            print(f"  {fk:<20}  {len(fk_bets):>3} picks{wr_s}")

    # ── Save ───────────────────────────────────────────────────────────────
    if args.save:
        out_name = f"gate_{gate_date}.csv"
        save_df  = result[["home_team","away_team","kickoff","target",
                            "decimal_odds","signals","max_signals","decision","detail"]]
        save_df.to_csv(out_name, index=False)
        print(f"\n  Saved → {out_name}")

    print(f"\n{SEP}")
    print(f"  USAGE TIPS")
    print(SEP)
    print(f"  python gate.py                     ← today's BET picks (p75, confirmed +ROI)")
    print(f"  python gate.py --percentile 50     ← more picks, still above baseline")
    print(f"  python gate.py --min-signals 2     ← include PARTIAL picks (2+ signals agree)")
    print(f"  python gate.py --save              ← save to gate_{gate_date}.csv")
    print(SEP)


if __name__ == "__main__":
    main()
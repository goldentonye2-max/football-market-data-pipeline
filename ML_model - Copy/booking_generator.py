"""
=============================================================
  BOOKING GENERATOR
  Reads picks_today_v2.csv → looks up outcomeId from DB →
  POSTs to SportyBet /orders/share → saves booking code
  + full pick list to a text file.
=============================================================
  USAGE:
    python booking_generator.py --top 20
    python booking_generator.py --top 10 --sort-by odds
    python booking_generator.py --top 50 --dry-run

  WORKFLOW:
    1. Read picks_today_v2.csv (output of predict_today_v2.py)
    2. Filter to top N picks by chosen sort column
    3. Look up market_id + specifier + outcome_desc + outcomeId
       from your sportybet.db for each pick
    4. POST to SportyBet share API
    5. Save booking code + full pick list to text file

  NOTES:
    - Max 50 legs per accumulator (SportyBet hard limit)
    - Update DEVICE_ID whenever your session expires
    - Set DRY_RUN = True to test without calling the API
=============================================================
"""

import os
import sys
import json
import sqlite3
import argparse
import requests
import pandas as pd
from datetime import datetime, date

# ── CONFIG ────────────────────────────────────────────────────
import glob

DB_PATH    = r"..\sportybet.db"
OUTPUT_DIR = "."

# Always use the most recently saved picks file
_files = sorted(glob.glob("all_picks_????-??-??.csv"))
if _files:
    PICKS_CSV = _files[-1]   # latest file by date in filename
else:
    PICKS_CSV = f"all_picks_{datetime.date.today()}.csv"  # fallback

# Update this from your browser incognito session cookie
DEVICE_ID    = "bf4fcdba-6e0d-418b-97f9-f9579b67bf39"

DRY_RUN      = False                 # set True to test without calling API
MAX_LEGS     = 50                    # SportyBet hard limit per accumulator

SHARE_URL    = "https://www.sportybet.com/api/ng/orders/share"
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# TARGET → (market_id, specifier, outcome_desc) MAPPING
# Built from your actual DB output.
# This translates predict_today_v2.py target column names
# into the exact values stored in your odds table.
# ══════════════════════════════════════════════════════════════

TARGET_TO_MARKET = {
    # ── Full match result ──────────────────────────────────────
    "target__home_win"          : ("1",  "",          "Home"),
    "target__draw"              : ("1",  "",          "Draw"),
    "target__away_win"          : ("1",  "",          "Away"),

    # ── Double chance ──────────────────────────────────────────
    "target__home_or_draw"      : ("10", "",          "Home or Draw"),
    "target__draw_or_away"      : ("10", "",          "Draw or Away"),
    "target__home_or_away"      : ("10", "",          "Home or Away"),

    # ── Draw no bet ────────────────────────────────────────────
    "target__dnb_home"          : ("11", "",          "Home"),
    "target__dnb_away"          : ("11", "",          "Away"),

    # ── Over/Under goals (full match) ─────────────────────────
    "target__over_0_5_goals"    : ("18", "total=0.5", "Over 0.5"),
    "target__under_0_5_goals"   : ("18", "total=0.5", "Under 0.5"),
    "target__over_1_5_goals"    : ("18", "total=1.5", "Over 1.5"),
    "target__under_1_5_goals"   : ("18", "total=1.5", "Under 1.5"),
    "target__over_2_5_goals"    : ("18", "total=2.5", "Over 2.5"),
    "target__under_2_5_goals"   : ("18", "total=2.5", "Under 2.5"),
    "target__over_3_5_goals"    : ("18", "total=3.5", "Over 3.5"),
    "target__under_3_5_goals"   : ("18", "total=3.5", "Under 3.5"),
    "target__over_4_5_goals"    : ("18", "total=4.5", "Over 4.5"),
    "target__under_4_5_goals"   : ("18", "total=4.5", "Under 4.5"),
    "target__over_5_5_goals"    : ("18", "total=5.5", "Over 5.5"),
    "target__under_5_5_goals"   : ("18", "total=5.5", "Under 5.5"),

    # ── BTTS / GG-NG ──────────────────────────────────────────
    "target__btts_yes"          : ("29", "",          "Yes"),
    "target__btts_no"           : ("29", "",          "No"),

    # ── Clean sheets ──────────────────────────────────────────
    "target__home_clean_sheet"  : ("31", "",          "Yes"),
    "target__away_clean_sheet"  : ("32", "",          "Yes"),

    # ── HT result ─────────────────────────────────────────────
    "target__ht_home_win"       : ("60", "",          "Home"),
    "target__ht_draw"           : ("60", "",          "Draw"),
    "target__ht_away_win"       : ("60", "",          "Away"),

    # ── HT over/under ─────────────────────────────────────────
    "target__ht_over_0_5_goals" : ("68", "total=0.5", "Over 0.5"),
    "target__ht_under_0_5_goals": ("68", "total=0.5", "Under 0.5"),
    "target__ht_over_1_5_goals" : ("68", "total=1.5", "Over 1.5"),
    "target__ht_under_1_5_goals": ("68", "total=1.5", "Under 1.5"),

    # ── Corners over/under (market 18 with high totals) ───────
    "target__over_6_5_corners"  : ("18", "total=6.5", "Over 6.5"),
    "target__over_7_5_corners"  : ("18", "total=7.5", "Over 7.5"),
    "target__over_8_5_corners"  : ("18", "total=8.5", "Over 8.5"),
    "target__over_9_5_corners"  : ("18", "total=9.5", "Over 9.5"),
    "target__over_10_5_corners" : ("18", "total=10", "Over 10"),
    "target__over_11_5_corners" : ("18", "total=10", "Over 10"),  # fallback
}


def get_outcome_id(market_id: str, specifier: str, outcome_desc: str) -> str:
    """
    Derive Betradar outcomeId from market_id + outcome_desc.
    Confirmed from your DB (market_id_probe output).
    """
    # 1X2
    if market_id == "1":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(outcome_desc, "1")

    # Double Chance
    if market_id == "10":
        return {
            "Home or Draw": "9",
            "Home or Away": "10",
            "Draw or Away": "11",
        }.get(outcome_desc, "9")

    # Draw No Bet
    if market_id == "11":
        return {"Home": "4", "Away": "5"}.get(outcome_desc, "4")

    # Over/Under (goals, corners, cards — all use same market 18)
    if market_id == "18":
        return "12" if outcome_desc.startswith("Over") else "13"

    # GG/NG (BTTS)
    if market_id == "29":
        return "74" if outcome_desc.lower() == "yes" else "76"

    # Home Team Clean Sheet
    if market_id == "31":
        return "74" if outcome_desc.lower() == "yes" else "76"

    # Away Team Clean Sheet
    if market_id == "32":
        return "74" if outcome_desc.lower() == "yes" else "76"

    # Halftime/Fulltime
    if market_id == "47":
        return {
            "Home/Home": "418", "Home/Draw": "420", "Home/Away": "422",
            "Draw/Home": "424", "Draw/Draw": "426", "Draw/Away": "428",
            "Away/Home": "430", "Away/Draw": "432", "Away/Away": "434",
        }.get(outcome_desc, "426")

    # 1st Half 1X2
    if market_id == "60":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(outcome_desc, "1")

    # 1st Half Over/Under
    if market_id == "68":
        return "12" if outcome_desc.startswith("Over") else "13"

    print(f"  ⚠  Unknown market_id={market_id!r} — defaulting outcomeId to '1'")
    return "1"


def lookup_selection(conn, event_id, market_id, specifier, outcome_desc):
    """
    Confirm this exact market exists in the DB for this fixture.
    Returns (odds, probability) or None if not found.
    """
    row = conn.execute("""
        SELECT odds, probability
        FROM odds
        WHERE event_id    = ?
          AND market_id   = ?
          AND COALESCE(specifier, '') = ?
          AND outcome_desc = ?
        LIMIT 1
    """, (event_id, market_id, specifier, outcome_desc)).fetchone()
    return row


def load_and_filter_picks(csv_path, top_n, sort_col, min_edge=None,
                           min_auc=None, min_odds=None, max_odds=None,
                           markets=None, min_hit_rate=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Picks file not found: {csv_path}\n"
            f"Run predict_today_v2.py --csv first."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Picks CSV is empty — no picks to book.")

    print(f"  → {len(df)} total picks in CSV before filtering")

    # ── Apply filters ─────────────────────────────────────────
    if min_edge is not None:
        df = df[df["edge_%"] >= min_edge]
        print(f"  → {len(df)} after edge ≥ {min_edge}%")

    if min_auc is not None:
        df = df[df["model_auc"] >= min_auc]
        print(f"  → {len(df)} after AUC ≥ {min_auc}")

    if min_odds is not None:
        df = df[df["decimal_odds"] >= min_odds]
        print(f"  → {len(df)} after odds ≥ {min_odds}")

    if max_odds is not None:
        df = df[df["decimal_odds"] <= max_odds]
        print(f"  → {len(df)} after odds ≤ {max_odds}")

    if markets:
        def market_match(pick_label):
            return any(m.lower() in str(pick_label).lower() for m in markets)
        df = df[df["pick"].apply(market_match)]
        print(f"  → {len(df)} after market filter: {', '.join(markets)}")

    if min_hit_rate is not None:
        df = df[df["hit_rate_%"] >= min_hit_rate]
        print(f"  → {len(df)} after hit rate ≥ {min_hit_rate}%")

    if df.empty:
        print("\n  No fixtures found matching your filters.")
        sys.exit(0)

    # One pick per fixture only — keep the highest edge one
    df = df.sort_values("edge_%", ascending=False)
    df = df.drop_duplicates(subset="event_id", keep="first")
    print(f"  → {len(df)} after deduplication (one pick per fixture)")

    df = df.head(top_n).reset_index(drop=True)

    # ── Sort and take top N ───────────────────────────────────
    SORT_COLS = {
        "edge"       : "edge_%",
        "odds"       : "decimal_odds",
        "model_prob" : "model_prob_%",
        "auc"        : "model_auc",
    }
    col = SORT_COLS.get(sort_col, "edge_%")
    if col not in df.columns:
        print(f"  ⚠  Sort column '{col}' not in CSV — defaulting to edge_%")
        col = "edge_%"

    df = df.sort_values(col, ascending=False).head(top_n).reset_index(drop=True)
    print(f"  → {len(df)} picks selected (top {top_n} by {col})")
    return df


def build_selections(picks_df):
    """
    For each pick, look up the market mapping and confirm in DB.
    Returns list of validated selection dicts ready for the API.
    """
    conn = sqlite3.connect(DB_PATH)
    selections = []
    skipped    = []

    for _, row in picks_df.iterrows():
        target   = row.get("target", "")
        event_id = row.get("event_id", "")
        match    = row.get("match", "?")
        pick     = row.get("pick", "?")

        # Look up market mapping
        mapping = TARGET_TO_MARKET.get(target)
        if mapping is None:
            skipped.append((match, pick, "no market mapping for target"))
            continue

        market_id, specifier, outcome_desc = mapping

        # Confirm exists in DB
        db_row = lookup_selection(conn, event_id, market_id, specifier, outcome_desc)
        if db_row is None:
            # Try with empty specifier as fallback
            db_row = lookup_selection(conn, event_id, market_id, "", outcome_desc)
            if db_row is None:
                skipped.append((match, pick, f"not found in DB (market={market_id}, spec={specifier!r}, outcome={outcome_desc!r})"))
                continue
            specifier = ""

        odds, probability = db_row
        outcome_id = get_outcome_id(market_id, specifier, outcome_desc)

        selections.append({
            # API fields
            "eventId"     : str(event_id),
            "marketId"    : market_id,
            "specifier"   : specifier if specifier else None,
            "outcomeId"   : outcome_id,
            # Display fields
            "match"       : match,
            "pick"        : pick,
            "odds"        : odds,
            "probability" : probability,
            "edge_%"      : row.get("edge_%", 0),
            "model_prob_%": row.get("model_prob_%", 0),
            "model_auc"   : row.get("model_auc", 0),
            "kickoff"     : row.get("kickoff", ""),
            "tournament"  : row.get("tournament", ""),
        })

    conn.close()

    if skipped:
        print(f"\n  ⚠  {len(skipped)} picks skipped:")
        for match, pick, reason in skipped:
            print(f"     {match} | {pick} → {reason}")

    print(f"\n  → {len(selections)} selections validated against DB")
    return selections


def print_slip(selections):
    """Pretty-print the full bet slip."""
    print("\n" + "═" * 70)
    print(f"  BET SLIP — {len(selections)} legs")
    print("═" * 70)

    total_odds = 1.0
    for i, s in enumerate(selections, 1):
        ko = str(s["kickoff"])[-8:-3] if s["kickoff"] else "?"
        print(f"  {i:>3}. [{ko}] {s['match']}")
        print(f"       Pick     : {s['pick']}")
        print(f"       Odds     : {s['odds']:.2f}   "
              f"Model: {s['model_prob_%']:.1f}%   "
              f"Edge: +{s['edge_%']:.1f}%   "
              f"AUC: {s['model_auc']:.4f}")
        print()
        total_odds *= float(s["odds"])

    print(f"  Combined odds : {total_odds:,.2f}")
    print("═" * 70)


def post_accumulator(selections):
    """POST to SportyBet share endpoint. Returns (code, raw_response)."""
    headers = {
        "accept"          : "*/*",
        "accept-language" : "en",
        "clientid"        : "web",
        "content-type"    : "application/json;charset=UTF-8",
        "operid"          : "2",
        "origin"          : "https://www.sportybet.com",
        "platform"        : "web",
        "referer"         : "https://www.sportybet.com/ng/sport/football/",
        "sporty-referer"  : "utm_source=https://www.google.com/",
        "user-agent"      : ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/150.0.0.0 Safari/537.36"),
    }
    cookies = {
        "locale"    : "en",
        "device-id" : DEVICE_ID,
        "sb_country": "ng",
    }
    payload = {
        "selections": [
            {
                "eventId"  : s["eventId"],
                "marketId" : s["marketId"],
                "specifier": s["specifier"],
                "outcomeId": s["outcomeId"],
            }
            for s in selections
        ]
    }

    try:
        resp = requests.post(
            SHARE_URL, headers=headers, cookies=cookies,
            json=payload, timeout=20
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        print(f"  ✗ HTTP {e.response.status_code}: {e.response.text[:300]}")
        return None, {}
    except Exception as e:
        print(f"  ✗ Request error: {e}")
        return None, {}

    # Extract booking code — try all common response shapes
    d    = data.get("data", {})
    code = None
    if isinstance(d, dict):
        code = d.get("shareCode") or d.get("bookingCode") or d.get("code")
    if not code:
        code = data.get("shareCode") or data.get("bookingCode") or data.get("code")
    if not code:
        print(f"  ⚠  Raw response: {json.dumps(data)[:400]}")

    return code, data


def save_output(selections, code, top_n, sort_col):
    """Save booking code + full pick list to a dated text file."""
    today     = date.today().isoformat()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fname     = os.path.join(OUTPUT_DIR, f"booking_{today}_top{top_n}.txt")

    total_odds = 1.0
    for s in selections:
        total_odds *= float(s["odds"])

    lines = [
        "=" * 70,
        f"  SPORTYBET ACCUMULATOR — {today}",
        f"  Generated : {timestamp}",
        f"  Legs      : {len(selections)}",
        f"  Sorted by : {sort_col}",
        f"  Combined odds : {total_odds:,.2f}",
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

    lines += [
        "",
        "=" * 70,
        f"  Tournament breakdown:",
    ]

    # Tournament breakdown
    tourn_counts = {}
    for s in selections:
        t = s.get("tournament", "Unknown")
        tourn_counts[t] = tourn_counts.get(t, 0) + 1
    for t, cnt in sorted(tourn_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {cnt:>3}x  {t}")

    lines += ["", "=" * 70]

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  ✅ Saved to {fname}")
    return fname


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate SportyBet accumulator booking code from ML picks"
    )
    parser.add_argument("--top",      type=int, default=20,
                        help="Number of picks to include (max 50, default 20)")
    parser.add_argument("--sort-by",  type=str, default="edge",
                        choices=["edge", "odds", "model_prob", "auc"],
                        help="Sort picks by this column (default: edge)")
    parser.add_argument("--picks-csv", type=str, default=PICKS_CSV,
                        help=f"Path to picks CSV (default: {PICKS_CSV})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview slip without calling the API")
    parser.add_argument("--min-edge",  type=float, default=None,
                        help="Minimum edge %% e.g. --min-edge 12")
    parser.add_argument("--min-auc",   type=float, default=None,
                        help="Minimum model AUC e.g. --min-auc 0.65")
    parser.add_argument("--min-odds",  type=float, default=None,
                        help="Minimum decimal odds e.g. --min-odds 1.71")
    parser.add_argument("--max-odds",  type=float, default=None,
                        help="Maximum decimal odds e.g. --max-odds 3.00")
    parser.add_argument("--markets",   nargs="+", default=None,
                        help='Market whitelist e.g. --markets "HT Away Win" "Over 2.5"')
    parser.add_argument("--min-hit-rate", type=float, default=None,
                        help="Minimum hit rate %% e.g. --min-hit-rate 40")
    args = parser.parse_args()

    top_n    = min(args.top, MAX_LEGS)
    dry_run  = args.dry_run or DRY_RUN
    sort_col = args.sort_by

    print("\n" + "=" * 70)
    print("  BOOKING GENERATOR — SportyBet Accumulator")
    print("=" * 70)
    print(f"  Picks CSV  : {args.picks_csv}")
    print(f"  Top N      : {top_n}")
    print(f"  Sort by    : {sort_col}")
    print(f"  Dry run    : {dry_run}")
    print(f"  Device ID  : {DEVICE_ID[:8]}...")

    # 1. Load picks
    print(f"\n[1/4] Loading picks from {args.picks_csv}...")
    picks_df = load_and_filter_picks(
    args.picks_csv, top_n, sort_col,
    min_edge  = args.min_edge,
    min_auc   = args.min_auc,
    min_odds  = args.min_odds,
    max_odds  = args.max_odds,
    markets   = args.markets,
    min_hit_rate = args.min_hit_rate,
    )

    # 2. Validate against DB
    print(f"\n[2/4] Validating selections against {DB_PATH}...")
    selections = build_selections(picks_df)

    if not selections:
        print("\n  ✗ No valid selections after DB lookup. Exiting.")
        return

    # 3. Print slip
    print(f"\n[3/4] Bet slip preview:")
    print_slip(selections)

    # 4. Confirm and book
    if dry_run:
        print("\n  [DRY RUN] — API call skipped.")
        code = "DRY-RUN-CODE"
    else:
        confirm = input("\n  Book this accumulator? (y/n): ").strip().lower()
        if confirm != "y":
            print("  Cancelled.")
            return

        print(f"\n[4/4] Posting {len(selections)}-leg accumulator to SportyBet...")
        code, raw = post_accumulator(selections)

        if code:
            print(f"\n  ✅ BOOKING CODE:  {code}")
        else:
            print("\n  ✗ Booking failed. Check the raw response above.")
            print("     Common causes:")
            print("     - Device ID expired (get a new one from your browser)")
            print("     - One or more events already started")
            print("     - SportyBet API temporarily down")

    # 5. Save output file
    save_output(selections, code, top_n, sort_col)

    print("\n" + "=" * 70)
    print(f"  DONE")
    print(f"  Booking code : {code}")
    print(f"  Legs         : {len(selections)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

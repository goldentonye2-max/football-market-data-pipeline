"""
SportyBet Accumulator Auto-Booker
==================================
Reads picks from your SQLite database, builds the accumulator payload,
posts to SportyBet's /orders/share endpoint, and returns the booking code.

Usage:
    1. Edit PICKS list with your chosen selections (one per fixture)
    2. Run: python3 sportybet_booker.py
    3. Copy the booking code and redeem it on SportyBet

Requirements:
    pip install requests
"""

import requests
import json
import sqlite3
import sys
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG — update device-id from your browser
# ─────────────────────────────────────────────
DEVICE_ID   = "2cc688e1-b52d-4382-8597-cc551efdd0fc"   # from your incognito session cookie
DB_PATH     = "sportybet.db"                            # path to your SQLite file

# ─────────────────────────────────────────────
#  OUTCOME ID MAPPING TABLE
#  Sportradar (Betradar) standard numeric IDs
#  market_id → { outcome_desc → outcomeId }
# ─────────────────────────────────────────────
OUTCOME_ID_MAP = {
    # 1X2
    "1": {
        "Home": "1",
        "Draw": "2",
        "Away": "3",
    },
    # Double Chance
    "10": {
        "Home or Draw": "9",
        "Draw or Away": "10",
        "Home or Away": "11",
    },
    # Draw No Bet
    "11": {
        "Home": "4",
        "Away": "5",
    },
    # Over/Under (market 18) — line comes from specifier, ID just says Over/Under
    "18": {
        "Over 0.5":  "13",  "Under 0.5":  "12",
        "Over 1":    "13",  "Under 1":    "12",
        "Over 1.5":  "13",  "Under 1.5":  "12",
        "Over 2":    "13",  "Under 2":    "12",
        "Over 2.5":  "13",  "Under 2.5":  "12",
        "Over 3":    "13",  "Under 3":    "12",
        "Over 3.5":  "13",  "Under 3.5":  "12",
        "Over 4":    "13",  "Under 4":    "12",
        "Over 4.5":  "13",  "Under 4.5":  "12",
        "Over 5":    "13",  "Under 5":    "12",
        "Over 5.5":  "13",  "Under 5.5":  "12",
    },
    # Asian Handicap (2-way, market 16)
    "16": {
        # Home outcomes (handicap favoring home)
        "Home (-0.5)": "14", "Away (+0.5)": "15",
        "Home (-1.0)": "14", "Away (+1.0)": "15",
        "Home (-1.5)": "14", "Away (+1.5)": "15",
        "Home (-2.0)": "14", "Away (+2.0)": "15",
        "Home (-2.5)": "14", "Away (+2.5)": "15",
        "Home (-3.0)": "14", "Away (+3.0)": "15",
        "Home (-3.5)": "14", "Away (+3.5)": "15",
        "Home (0)":    "14", "Away (0)":    "15",
        # Home outcomes (handicap favoring away)
        "Home (+0.5)": "14", "Away (-0.5)": "15",
        "Home (+1.0)": "14", "Away (-1.0)": "15",
        "Home (+1.5)": "14", "Away (-1.5)": "15",
        "Home (+2.0)": "14", "Away (-2.0)": "15",
        "Home (+2.5)": "14", "Away (-2.5)": "15",
    },
    # BTTS / GG-NG
    "29": {
        "Yes": "74",
        "No":  "76",
    },
    # Home O/U (market 19)
    "19": {
        "Over 0.5": "13", "Under 0.5": "12",
        "Over 1.5": "13", "Under 1.5": "12",
        "Over 2.5": "13", "Under 2.5": "12",
        "Over 3.5": "13", "Under 3.5": "12",
        "Over 4.5": "13", "Under 4.5": "12",
    },
    # Away O/U (market 20)
    "20": {
        "Over 0.5": "13", "Under 0.5": "12",
        "Over 1.5": "13", "Under 1.5": "12",
        "Over 2.5": "13", "Under 2.5": "12",
        "Over 3.5": "13", "Under 3.5": "12",
        "Over 4.5": "13", "Under 4.5": "12",
    },
}


def get_outcome_id(market_id, outcome_desc):
    """Look up the Sportradar outcomeId for a given market + outcome."""
    mid = str(market_id)
    if mid not in OUTCOME_ID_MAP:
        return None
    return OUTCOME_ID_MAP[mid].get(outcome_desc)


def load_picks_from_db(picks_spec):
    """
    Load and validate picks from the database.

    picks_spec is a list of dicts:
        [
          {"event_id": "sr:match:12345", "market_id": "1",  "outcome_desc": "Home"},
          {"event_id": "sr:match:67890", "market_id": "18", "outcome_desc": "Under 2.5"},
          ...
        ]

    Returns a list of validated selections ready for the API.
    """
    conn = sqlite3.connect(DB_PATH)
    selections = []

    for pick in picks_spec:
        eid  = pick["event_id"]
        mid  = str(pick["market_id"])
        odsc = pick["outcome_desc"]

        # Pull from database to get odds, probability, specifier, fixture info
        row = conn.execute("""
            SELECT o.odds, o.probability, o.specifier,
                   f.home_team, f.away_team, f.kickoff_time
            FROM odds o
            JOIN fixtures f ON o.event_id = f.event_id
            WHERE o.event_id    = ?
              AND o.market_id   = ?
              AND o.outcome_desc = ?
            LIMIT 1
        """, (eid, mid, odsc)).fetchone()

        if row is None:
            print(f"  ⚠  NOT FOUND in DB: {eid} | market={mid} | {odsc}")
            continue

        odds, prob, specifier, home, away, kickoff = row

        # Look up outcomeId
        oid = get_outcome_id(mid, odsc)
        if oid is None:
            print(f"  ⚠  No outcomeId mapping for market={mid}, outcome='{odsc}' — SKIPPING")
            print(f"      ({home} vs {away})")
            continue

        # Format specifier: empty string → null
        spec_val = specifier if (specifier and specifier.strip()) else None

        selections.append({
            "event_id":    eid,
            "market_id":   mid,
            "outcome_desc": odsc,
            "outcome_id":  oid,
            "specifier":   spec_val,
            "odds":        odds,
            "probability": prob,
            "home":        home,
            "away":        away,
            "kickoff":     kickoff,
        })

    conn.close()
    return selections


def book_accumulator(selections):
    """
    POST to SportyBet /orders/share and return the booking code.
    """
    url = "https://www.sportybet.com/api/ng/orders/share"

    headers = {
        "accept":           "*/*",
        "accept-language":  "en",
        "clientid":         "web",
        "content-type":     "application/json;charset=UTF-8",
        "operid":           "2",
        "origin":           "https://www.sportybet.com",
        "platform":         "web",
        "referer":          "https://www.sportybet.com/ng/sport/football/",
        "sporty-referer":   "utm_source=https://www.google.com/",
        "user-agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
    }

    cookies = {
        "locale":     "en",
        "device-id":  DEVICE_ID,
        "sb_country": "ng",
    }

    payload = {
        "selections": [
            {
                "eventId":   s["event_id"],
                "marketId":  s["market_id"],
                "specifier": s["specifier"],
                "outcomeId": s["outcome_id"],
            }
            for s in selections
        ]
    }

    print(f"\nPosting {len(selections)}-leg accumulator to SportyBet...")
    resp = requests.post(url, headers=headers, cookies=cookies, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def print_slip(selections):
    """Pretty-print the bet slip before booking."""
    print("\n" + "═" * 65)
    print(f"  BET SLIP — {len(selections)} selections")
    print("═" * 65)
    total_odds = 1.0
    for i, s in enumerate(selections, 1):
        spec_str = f" ({s['specifier']})" if s['specifier'] else ""
        print(f"  {i}. {s['home']} vs {s['away']}")
        print(f"     Market : {s['market_id']}{spec_str}  |  Pick: {s['outcome_desc']}")
        print(f"     Odds   : {s['odds']}  |  Fair prob: {s['probability']*100:.1f}%")
        print(f"     outcomeId → {s['outcome_id']}")
        print()
        total_odds *= s['odds']
    print(f"  Combined odds: {total_odds:.4f}")
    print("═" * 65)


# ─────────────────────────────────────────────
#  PICKS — edit this for each day's selections
#  One entry per fixture you want in the acca.
#  event_id  → from your fixtures table
#  market_id → from your odds table
#  outcome_desc → exactly as it appears in your odds table
# ─────────────────────────────────────────────
PICKS = [
    {
        "event_id":    "sr:match:72221154",
        "market_id":   "1",
        "outcome_desc": "Home",
    },
    {
        "event_id":    "sr:match:72221156",
        "market_id":   "1",
        "outcome_desc": "Draw",
    },
    # Add more picks here — one per fixture
    # {
    #     "event_id":    "sr:match:XXXXXXXX",
    #     "market_id":   "18",
    #     "outcome_desc": "Under 2.5",
    # },
]


if __name__ == "__main__":
    print(f"\nSportyBet Auto-Booker  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"DB: {DB_PATH}  |  Device: {DEVICE_ID[:8]}...")

    # 1. Load and validate picks
    selections = load_picks_from_db(PICKS)

    if not selections:
        print("\n✗ No valid selections. Check your PICKS list and the mapping table.")
        sys.exit(1)

    if len(selections) < len(PICKS):
        print(f"\n⚠  Only {len(selections)}/{len(PICKS)} picks resolved — review warnings above.")

    # 2. Print the slip
    print_slip(selections)

    # 3. Confirm before posting
    confirm = input("  Book this bet? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        sys.exit(0)

    # 4. Post and get booking code
    result = book_accumulator(selections)
    print(f"\n  RAW RESPONSE: {json.dumps(result, indent=2)}")

    # 5. Extract booking code — adjust key based on actual response structure
    code = (result.get("data", {}).get("shareCode")
            or result.get("shareCode")
            or result.get("code")
            or result.get("data", {}).get("code")
            or "⚠ Check raw response above for the code")

    print(f"\n{'═'*65}")
    print(f"  BOOKING CODE:  {code}")
    print(f"{'═'*65}\n")
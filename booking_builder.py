#!/usr/bin/env python3
"""
booking_builder.py
==================
Queries the database for today's (or any date's) qualifying picks,
distributes them into accumulator slips, and generates SportyBet
booking codes via POST /api/ng/orders/share.

USAGE:
  python booking_builder.py                  ← today
  python booking_builder.py 2026-07-18       ← specific date

OUTPUT:
  booking_codes_YYYY-MM-DD.txt  (booking codes + full pick lists)

NO LOGIN REQUIRED — runs in the same incognito cookie context.

CHANGES FROM ORIGINAL:
  - Added book_slip_with_fallback(): if a slip is rejected, it binary-
    searches for the specific bad picks, removes them, then re-submits
    with only the valid selections. A single invalid pick can no longer
    kill the entire slip.
  - Added _KNOWN_BAD cache: bad (event, market, outcome) combos found
    while pruning one slip are automatically skipped in later slips.
  - Added MIN_PICKS_TO_BOOK threshold: if too few picks survive pruning
    the slip is skipped rather than submitting a tiny accumulator.
"""

import os
import sys
import json
import time
import sqlite3
from datetime import date, datetime
from typing import Optional

import requests

# ══════════════════════════════════════════════════════════════════════════════
# USER CONFIG — edit this section before running
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "sportybet.db")

# ── Paste your device-id cookie value here ────────────────────────────────────
DEVICE_ID = "bf4fcdba-6e0d-418b-97f9-f9579b67bf39"

# ── Slip settings ─────────────────────────────────────────────────────────────
NUM_SLIPS      = 5     # how many separate accumulator slips to generate
PICKS_PER_SLIP = 50    # max legs per slip (SportyBet hard limit = 50)

# How picks are spread across slips:
#   "round_robin" → pick 1→slip1, pick 2→slip2, …, pick 6→slip1 …  (all slips equal quality)
#   "sequential"  → best 50 into slip 1, next 50 into slip 2 …     (slip 1 is the best)
DISTRIBUTION = "round_robin"

# Set True to preview picks without calling the API
DRY_RUN = True

# ── Fallback / pruning settings ───────────────────────────────────────────────
# If a slip fails, bad picks are removed and the slip is re-submitted.
# If fewer than this many valid picks survive, the slip is skipped entirely.
MIN_PICKS_TO_BOOK = 5

# ── Signals — which bets to look for and their probability/odds windows ────────
# Each tuple: (label, market_id, specifier, outcome_desc, min_prob, max_prob, min_odds, max_odds)
#   min_prob / max_prob : Betradar implied probability range (0–1)
#   min_odds / max_odds : decimal odds range  (0.0 / 99.0 = no filter)
#
# Priority matters: a fixture that qualifies for multiple signals is assigned
# to the FIRST signal it matches, avoiding duplicate legs in the same slip.

SIGNALS = [
    # Signal D — Over 1.5 Goals (91.7% actual hit rate when prob ≥ 0.80)
    ("D-Over1.5", "18", "total=1.5", "Over 1.5",  0.80, 1.00, 0.0, 99.0),

    # Signal B — Over 2.5 Goals (proven +12% edge in 0.45–0.65 probability window)
    ("B-Over2.5", "18", "total=2.5", "Over 2.5",  0.45, 0.65, 0.0, 99.0),

    # Signal C — Both Teams to Score Yes (market consistently underprices 2+ goals)
    ("C-BTTS",    "29", "",           "Yes",        0.50, 1.00, 0.0, 99.0),
]


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME ID DERIVATION
#
# SportyBet's share API needs a numeric outcomeId for each selection.
# Our database stores outcome_desc (e.g. "Over 1.5", "Yes") but not the ID.
# Below are the Betradar-standard IDs that SportyBet NG uses.
#
# HOW TO VERIFY / ADD NEW MARKETS:
#   1. Open sportybet.com/ng in a browser (incognito is fine)
#   2. Click the odds for the market you want → open DevTools → Network tab
#   3. Find the POST to /api/ng/factsCenter/Outcomes
#   4. The outcomeId in that payload is what you need here
# ══════════════════════════════════════════════════════════════════════════════

def get_outcome_id(market_id: str, specifier: str, outcome_desc: str) -> str:
    """
    Derive outcomeId from market_id + outcome_desc.

    All IDs below are confirmed from market_id_probe.py output
    (Tigres FC vs Atletico Nacional, Copa DIMAYOR, 2026-07-29).
    Status markers: ✓ confirmed | ✗ was wrong, now fixed | + newly added
    """

    # ── 1X2  ✓ ───────────────────────────────────────────────────────────────
    # marketId=1 | outcomes: Home=1, Draw=2, Away=3
    if market_id == "1":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(outcome_desc, "1")

    # ── Double Chance  ✓ ─────────────────────────────────────────────────────
    # marketId=10 | outcomes: Home or Draw=9, Home or Away=10, Draw or Away=11
    if market_id == "10":
        return {
            "Home or Draw": "9",
            "Home or Away": "10",
            "Draw or Away": "11",
        }.get(outcome_desc, "9")

    # ── Draw No Bet  ✗ FIXED (was Home=5/Away=6, correct is Home=4/Away=5) ──
    # marketId=11 | outcomes: Home=4, Away=5
    if market_id == "11":
        return {"Home": "4", "Away": "5"}.get(outcome_desc, "4")

    # ── Home No Bet  + ───────────────────────────────────────────────────────
    # marketId=12 | outcomes: Draw=776, Away=778
    if market_id == "12":
        return {"Draw": "776", "Away": "778"}.get(outcome_desc, "776")

    # ── Away No Bet  + ───────────────────────────────────────────────────────
    # marketId=13 | outcomes: Home=780, Draw=782
    if market_id == "13":
        return {"Home": "780", "Draw": "782"}.get(outcome_desc, "780")

    # ── Handicap 3-way  ✗ FIXED (was Home=1/Away=2, correct is 1711/1712/1713)
    # marketId=14 | specifier e.g. hcp=1:0
    # outcomes: Home=1711, Draw=1712, Away=1713
    if market_id == "14":
        if "Draw" in outcome_desc:
            return "1712"
        return "1711" if "Home" in outcome_desc else "1713"

    # ── Asian Handicap 2-way (Winning Margin)  + ─────────────────────────────
    # marketId=15 | specifier: variant=sr:winning_margin:...
    # outcomes vary by variant — fall through to fallback (too fixture-specific)

    # ── Asian Handicap 2-way  + ──────────────────────────────────────────────
    # marketId=16 | specifier e.g. hcp=0.5
    # outcomes: Home=1714, Away=1715
    if market_id == "16":
        return "1714" if "Home" in outcome_desc else "1715"

    # ── Over/Under Goals  ✓ ──────────────────────────────────────────────────
    # marketId=18 | specifier e.g. total=2.5
    # outcomes: Over=12, Under=13
    if market_id == "18":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── Home Team O/U  + ─────────────────────────────────────────────────────
    # marketId=19 | same outcome pattern as market 18
    if market_id == "19":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── Away Team O/U  + ─────────────────────────────────────────────────────
    # marketId=20 | same outcome pattern as market 18
    if market_id == "20":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── Home Team Exact Goals  + ─────────────────────────────────────────────
    # marketId=23 | outcomes are variant-specific strings (sr:exact_goals:...)
    # outcomeId IS the variant string — must come from the database, not derived here

    # ── Away Team Exact Goals  + ─────────────────────────────────────────────
    # marketId=24 | same as market 23

    # ── Odd/Even  + ──────────────────────────────────────────────────────────
    # marketId=26 | outcomes: Odd=70, Even=72
    if market_id == "26":
        return "70" if outcome_desc.lower() == "odd" else "72"

    # ── Home Team Odd/Even  + ────────────────────────────────────────────────
    # marketId=27 | outcomes: Odd=70, Even=72
    if market_id == "27":
        return "70" if outcome_desc.lower() == "odd" else "72"

    # ── Away Team Odd/Even  + ────────────────────────────────────────────────
    # marketId=28 | outcomes: Odd=70, Even=72
    if market_id == "28":
        return "70" if outcome_desc.lower() == "odd" else "72"

    # ── GG/NG (Both Teams to Score)  ✓ ──────────────────────────────────────
    # marketId=29 | outcomes: Yes=74, No=76
    if market_id == "29":
        return "74" if outcome_desc.lower() in ("yes", "gg") else "76"

    # ── Which Team To Score  + ───────────────────────────────────────────────
    # marketId=30 | outcomes: None=784, Only Home=788, Only Away=790, Both=792
    if market_id == "30":
        return {
            "None":       "784",
            "Only Home":  "788",
            "Only Away":  "790",
            "Both teams": "792",
        }.get(outcome_desc, "792")

    # ── Home / Away Clean Sheet  + ───────────────────────────────────────────
    # marketId=31 (Home), 32 (Away) | outcomes: Yes=74, No=76
    if market_id in ("31", "32"):
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── Home / Away Win To Nil  + ────────────────────────────────────────────
    # marketId=33 (Home), 34 (Away) | outcomes: Yes=74, No=76
    if market_id in ("33", "34"):
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── 1X2 & GG/NG combo  + ─────────────────────────────────────────────────
    # marketId=35 | outcomes: H&Y=78, H&N=80, D&Y=82, D&N=84, A&Y=86, A&N=88
    if market_id == "35":
        return {
            "Home & yes": "78", "Home & no": "80",
            "Draw & yes": "82", "Draw & no": "84",
            "Away & yes": "86", "Away & no": "88",
        }.get(outcome_desc, "86")

    # ── O/U & GG/NG combo  + ─────────────────────────────────────────────────
    # marketId=36 | specifier total=2.5
    # outcomes: Over&Yes=90, Under&Yes=92, Over&No=94, Under&No=96
    if market_id == "36":
        return {
            "Over 2.5 & Yes":  "90", "Under 2.5 & Yes": "92",
            "Over 2.5 & No":   "94", "Under 2.5 & No":  "96",
        }.get(outcome_desc, "90")

    # ── 1X2 & O/U combo  + ───────────────────────────────────────────────────
    # marketId=37 | outcomes: H&U=794, H&O=796, D&U=798, D&O=800, A&U=802, A&O=804
    if market_id == "37":
        return {
            "Home & Under": "794", "Home & Over": "796",
            "Draw & Under": "798", "Draw & Over": "800",
            "Away & Under": "802", "Away & Over": "804",
        }.get(outcome_desc, "804")

    # ── Correct Score  + ─────────────────────────────────────────────────────
    # marketId=45 | too many outcomes to enumerate — must come from database
    # (outcome IDs: 274=0:0, 276=1:0, 278=2:0 … 324=Other — see probe output)

    # ── Halftime/Fulltime  + ─────────────────────────────────────────────────
    # marketId=47 | outcomes: H/H=418, H/D=420, H/A=422, D/H=424, D/D=426,
    #                          D/A=428, A/H=430, A/D=432, A/A=434
    if market_id == "47":
        return {
            "Home/Home": "418", "Home/Draw": "420", "Home/Away": "422",
            "Draw/Home": "424", "Draw/Draw": "426", "Draw/Away": "428",
            "Away/Home": "430", "Away/Draw": "432", "Away/Away": "434",
        }.get(outcome_desc, "426")

    # ── Home / Away To Win Both Halves  + ────────────────────────────────────
    # marketId=48 (Home), 49 (Away) | outcomes: Yes=74, No=76
    if market_id in ("48", "49", "50", "51"):
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── Highest Scoring Half  + ───────────────────────────────────────────────
    # marketId=52 | outcomes: 1st half=436, 2nd half=438, Equal=440
    if market_id in ("52", "53", "54"):
        return {
            "1st half": "436", "2nd half": "438", "Equal": "440",
        }.get(outcome_desc, "440")

    # ── Both Halves Over X.5  + ──────────────────────────────────────────────
    # marketId=58 | outcomes: Yes=74, No=76
    if market_id == "58":
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── Both Halves Under X.5  + ─────────────────────────────────────────────
    # marketId=59 | outcomes: Yes=74, No=76
    if market_id == "59":
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── 1st Half 1X2  + ──────────────────────────────────────────────────────
    # marketId=60 | outcomes: Home=1, Draw=2, Away=3
    if market_id == "60":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(outcome_desc, "1")

    # ── 1st Half Double Chance  + ────────────────────────────────────────────
    # marketId=63 | outcomes: H/D=9, H/A=10, D/A=11
    if market_id == "63":
        return {
            "Home or Draw": "9", "Home or Away": "10", "Draw or Away": "11",
        }.get(outcome_desc, "9")

    # ── 1st Half Draw No Bet  + ──────────────────────────────────────────────
    # marketId=64 | outcomes: Home=4, Away=5
    if market_id == "64":
        return {"Home": "4", "Away": "5"}.get(outcome_desc, "4")

    # ── 1st Half Handicap 3-way  + ───────────────────────────────────────────
    # marketId=65 | outcomes: Home=1711, Draw=1712, Away=1713
    if market_id == "65":
        if "Draw" in outcome_desc:
            return "1712"
        return "1711" if "Home" in outcome_desc else "1713"

    # ── 1st Half Asian Handicap 2-way  + ─────────────────────────────────────
    # marketId=66 | outcomes: Home=1714, Away=1715
    if market_id == "66":
        return "1714" if "Home" in outcome_desc else "1715"

    # ── 1st Half Over/Under  + ───────────────────────────────────────────────
    # marketId=68 | outcomes: Over=12, Under=13
    if market_id == "68":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── 1st Half Home O/U  + ─────────────────────────────────────────────────
    # marketId=69 | outcomes: Over=12, Under=13
    if market_id == "69":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── 1st Half Away O/U  + ─────────────────────────────────────────────────
    # marketId=70 | outcomes: Over=12, Under=13
    if market_id == "70":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── 1st Half Exact Goals  + ──────────────────────────────────────────────
    # marketId=71 | variant-based outcomeIds — must come from database

    # ── 1st Half Odd/Even  + ─────────────────────────────────────────────────
    # marketId=74 | outcomes: Odd=70, Even=72
    if market_id == "74":
        return "70" if outcome_desc.lower() == "odd" else "72"

    # ── 1st Half GG/NG  + ────────────────────────────────────────────────────
    # marketId=75 | outcomes: Yes=74, No=76
    if market_id == "75":
        return "74" if outcome_desc.lower() in ("yes", "gg") else "76"

    # ── 1st Half Home/Away Clean Sheet  + ────────────────────────────────────
    # marketId=76 (Home), 77 (Away) | outcomes: Yes=74, No=76
    if market_id in ("76", "77"):
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── 1st Half 1X2 & GG/NG  + ──────────────────────────────────────────────
    # marketId=78 | outcomes: H&Y=78, H&N=80, D&Y=82, D&N=84, A&Y=86, A&N=88
    if market_id == "78":
        return {
            "Home & yes": "78", "Home & no": "80",
            "Draw & yes": "82", "Draw & no": "84",
            "Away & yes": "86", "Away & no": "88",
        }.get(outcome_desc, "86")

    # ── 1st Half 1X2 & Total  + ──────────────────────────────────────────────
    # marketId=79 | outcomes: H&U=794, H&O=796, D&U=798, D&O=800, A&U=802, A&O=804
    if market_id == "79":
        return {
            "Home & Under": "794", "Home & Over": "796",
            "Draw & Under": "798", "Draw & Over": "800",
            "Away & Under": "802", "Away & Over": "804",
        }.get(outcome_desc, "804")

    # ── 1st Half Correct Score  + ────────────────────────────────────────────
    # marketId=81 | too many outcomes — must come from database

    # ── 2nd Half 1X2  + ──────────────────────────────────────────────────────
    # marketId=83 | outcomes: Home=1, Draw=2, Away=3
    if market_id == "83":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(outcome_desc, "1")

    # ── 2nd Half Double Chance  + ────────────────────────────────────────────
    # marketId=85 | outcomes: H/D=9, H/A=10, D/A=11
    if market_id == "85":
        return {
            "Home or Draw": "9", "Home or Away": "10", "Draw or Away": "11",
        }.get(outcome_desc, "9")

    # ── 2nd Half Draw No Bet  + ──────────────────────────────────────────────
    # marketId=86 | outcomes: Home=4, Away=5
    if market_id == "86":
        return {"Home": "4", "Away": "5"}.get(outcome_desc, "4")

    # ── 2nd Half Handicap 3-way  + ───────────────────────────────────────────
    # marketId=87 | outcomes: Home=1711, Draw=1712, Away=1713
    if market_id == "87":
        if "Draw" in outcome_desc:
            return "1712"
        return "1711" if "Home" in outcome_desc else "1713"

    # ── 2nd Half Asian Handicap  + ───────────────────────────────────────────
    # marketId=88 | outcomes: Home=1714, Away=1715
    if market_id == "88":
        return "1714" if "Home" in outcome_desc else "1715"

    # ── 2nd Half Total  + ────────────────────────────────────────────────────
    # marketId=90 | outcomes: Over=12, Under=13
    if market_id == "90":
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── 2nd Half Home/Away Team Total  + ─────────────────────────────────────
    # marketId=91 (Home), 92 (Away) | outcomes: Over=12, Under=13
    if market_id in ("91", "92"):
        return "12" if outcome_desc.startswith("Over") else "13"

    # ── 2nd Half Exact Goals  + ──────────────────────────────────────────────
    # marketId=93 | variant-based — must come from database

    # ── 2nd Half Odd/Even  + ─────────────────────────────────────────────────
    # marketId=94 | outcomes: Odd=70, Even=72
    if market_id == "94":
        return "70" if outcome_desc.lower() == "odd" else "72"

    # ── 2nd Half GG/NG  + ────────────────────────────────────────────────────
    # marketId=95 | outcomes: Yes=74, No=76
    if market_id == "95":
        return "74" if outcome_desc.lower() in ("yes", "gg") else "76"

    # ── 2nd Half Home/Away Clean Sheet  + ────────────────────────────────────
    # marketId=96 (Home), 97 (Away) | outcomes: Yes=74, No=76
    if market_id in ("96", "97"):
        return "74" if outcome_desc.lower() == "yes" else "76"

    # ── 2nd Half Correct Score  + ────────────────────────────────────────────
    # marketId=98 | too many outcomes — must come from database

    # ── 2nd Half 1X2 & GG/NG  + ──────────────────────────────────────────────
    # marketId=543 | outcomes: same pattern as market 78
    if market_id == "543":
        return {
            "Home & yes": "78", "Home & no": "80",
            "Draw & yes": "82", "Draw & no": "84",
            "Away & yes": "86", "Away & no": "88",
        }.get(outcome_desc, "86")

    # ── 2nd Half 1X2 & Total  + ──────────────────────────────────────────────
    # marketId=544 | outcomes: same pattern as market 37/79
    if market_id == "544":
        return {
            "Home & Under": "794", "Home & Over": "796",
            "Draw & Under": "798", "Draw & Over": "800",
            "Away & Under": "802", "Away & Over": "804",
        }.get(outcome_desc, "804")

    # ── Corners Over/Under  ✗ FIXED (was 38/62/63/64 → 2/4, correct below) ──
    # Home corners: marketId=900300  |  Over=30, Under=31
    # Away corners: marketId=900301  |  Over=30, Under=31
    # 1st Half Home corners: marketId=900302  |  Over=30, Under=31
    # 1st Half Away corners: marketId=900303  |  Over=30, Under=31
    if market_id in ("900300", "900301", "900302", "900303"):
        return "30" if outcome_desc.startswith("Over") else "31"

    # ── Fallback ──────────────────────────────────────────────────────────────
    print(f"  ⚠  Unknown market_id={market_id!r} outcome={outcome_desc!r} "
          f"— defaulting outcomeId to '1'. Add this market to get_outcome_id().")
    return "1"


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE QUERY
# ══════════════════════════════════════════════════════════════════════════════

def collect_picks(target_date: str) -> list[dict]:
    """
    Query the database for all qualifying picks on target_date.
    Returns a list sorted by EV descending.
    Each unique event_id appears at most once (highest-priority signal wins).
    """
    conn = sqlite3.connect(DB_PATH)

    # Optional: print odds table columns so user can see if outcome_id exists
    cols = [r[1] for r in conn.execute("PRAGMA table_info(odds)").fetchall()]
    has_outcome_id_col = "outcome_id" in cols

    all_picks  = []
    seen_eids  = set()

    for (signal, market_id, spec, outcome, min_prob, max_prob, min_odds, max_odds) in SIGNALS:

        if has_outcome_id_col:
            # Use the stored outcome_id if the scraper saved it
            extra_col = "o.outcome_id,"
            extra_idx  = True
        else:
            extra_col = "'__none__',"
            extra_idx  = False

        rows = conn.execute(f"""
            SELECT
                o.event_id,
                o.market_id,
                COALESCE(o.specifier, '')              AS specifier,
                o.outcome_desc,
                o.odds,
                o.probability,
                ROUND(o.probability * o.odds - 1, 5)  AS ev,
                f.home_team,
                f.away_team,
                f.kickoff_time,
                f.tournament_name,
                f.category_name,
                {extra_col}
                f.game_id
            FROM  odds     o
            JOIN  fixtures f ON f.event_id = o.event_id
            WHERE DATE(f.kickoff_time)       = ?
              AND o.market_id                = ?
              AND COALESCE(o.specifier, '')  = ?
              AND o.outcome_desc             = ?
              AND o.probability             >= ?
              AND o.probability             <= ?
              AND o.odds                    >= ?
              AND o.odds                    <= ?
            ORDER BY ev DESC
        """, (target_date, market_id, spec, outcome,
              min_prob, max_prob, min_odds, max_odds)).fetchall()

        for r in rows:
            eid = r[0]
            if eid in seen_eids:
                continue
            seen_eids.add(eid)

            spec_val       = r[2] if r[2] else None
            db_outcome_id  = str(r[12]) if extra_idx and r[12] else None
            outcome_id     = db_outcome_id or get_outcome_id(r[1], r[2], r[3])

            all_picks.append({
                # API fields
                "eventId":      eid,
                "marketId":     r[1],
                "specifier":    spec_val,
                "outcomeId":    outcome_id,
                # Display fields
                "outcome_desc": r[3],
                "odds":         r[4],
                "probability":  r[5],
                "ev":           r[6],
                "signal":       signal,
                "home_team":    r[7],
                "away_team":    r[8],
                "kickoff_time": r[9],
                "tournament":   r[10],
                "category":     r[11],
                "game_id":      r[13],
            })

    conn.close()

    all_picks.sort(key=lambda x: x["ev"], reverse=True)
    return all_picks


# ══════════════════════════════════════════════════════════════════════════════
# SLIP DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def build_slips(picks: list[dict]) -> list[list[dict]]:
    """
    Distribute picks across NUM_SLIPS slips using the configured DISTRIBUTION
    strategy, respecting the PICKS_PER_SLIP limit.
    """
    slips: list[list[dict]] = [[] for _ in range(NUM_SLIPS)]

    if DISTRIBUTION == "round_robin":
        slot = 0
        for pick in picks:
            # Find the next slip that still has room, cycling round-robin
            for _ in range(NUM_SLIPS):
                idx = slot % NUM_SLIPS
                slot += 1
                if len(slips[idx]) < PICKS_PER_SLIP:
                    slips[idx].append(pick)
                    break
    else:  # sequential
        for i, pick in enumerate(picks):
            idx = i // PICKS_PER_SLIP
            if idx >= NUM_SLIPS:
                break
            if len(slips[idx]) < PICKS_PER_SLIP:
                slips[idx].append(pick)

    return [s for s in slips if s]   # drop empty slips


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

SHARE_URL = "https://www.sportybet.com/api/ng/orders/share"

SESSION_HEADERS = {
    "accept":          "*/*",
    "accept-language": "en",
    "clientid":        "web",
    "content-type":    "application/json;charset=UTF-8",
    "operid":          "2",
    "origin":          "https://www.sportybet.com",
    "platform":        "web",
    "referer":         "https://www.sportybet.com/ng/sport/football/",
    "sporty-referer":  "utm_source=https://www.google.com/",
    "user-agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/150.0.0.0 Safari/537.36"),
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(SESSION_HEADERS)
    s.cookies.set("locale",     "en",      domain="www.sportybet.com")
    s.cookies.set("device-id",  DEVICE_ID, domain="www.sportybet.com")
    s.cookies.set("sb_country", "ng",      domain="www.sportybet.com")
    return s


def book_slip(session: requests.Session,
              picks: list[dict]) -> tuple[Optional[str], dict]:
    """
    POST the selections to the share endpoint.
    Returns (booking_code, raw_response_dict).
    booking_code is None on failure.
    """
    selections = [
        {
            "eventId":   p["eventId"],
            "marketId":  p["marketId"],
            "specifier": p["specifier"],    # None → JSON null (correct)
            "outcomeId": p["outcomeId"],
        }
        for p in picks
    ]

    payload = {"selections": selections}

    try:
        r = session.post(SHARE_URL, json=payload, timeout=20)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        print(f"    ✗  HTTP {e.response.status_code}: {e.response.text[:300]}")
        return None, {}
    except Exception as e:
        print(f"    ✗  Request error: {e}")
        return None, {}

    # SportyBet may return the code in several shapes — try all common ones
    d = data.get("data", {})
    code = (
        (d.get("shareCode") or d.get("bookingCode") or d.get("code"))
        if isinstance(d, dict) else
        (str(d) if isinstance(d, str) and d else None)
    )
    if not code:
        code = data.get("shareCode") or data.get("bookingCode")

    if not code:
        print(f"    ⚠  Could not extract code from response: {json.dumps(data)[:300]}")

    return code, data


# ══════════════════════════════════════════════════════════════════════════════
# RETRY WITH PRUNING
#
# When a slip is rejected by the API, we can't know from the response which
# specific pick(s) caused it. The strategy here is binary search:
#
#   1. Split the slip in half and test each half.
#   2. Any half that passes is clean — all its picks are valid.
#   3. Any half that fails is recursed into, splitting again.
#   4. When we reach a single pick that still fails, that pick is definitively
#      bad. We record it in _KNOWN_BAD so it won't be retried in later slips.
#   5. After all bad picks are identified, we combine the survivors and do
#      one final submission to produce the real booking code.
#
# Cost: O(k × log n) extra API calls, where k is the number of bad picks and
# n is the slip size. For 3 bad picks in a 50-pick slip: ~18 extra calls.
# No artificial delays are added during pruning (only before the final submit).
# ══════════════════════════════════════════════════════════════════════════════

# Cache of picks confirmed invalid during this run.
# Keyed by (eventId, marketId, specifier, outcomeId).
_KNOWN_BAD: set[tuple] = set()


def _pick_key(p: dict) -> tuple:
    return (p["eventId"], p["marketId"], p.get("specifier"), p["outcomeId"])


def _find_good_picks(session: requests.Session,
                     picks: list[dict],
                     depth: int = 0) -> list[dict]:
    """
    Recursively binary-search picks to return only the valid subset.
    Picks that cause API rejection are logged and added to _KNOWN_BAD.
    Note: successful half-submissions create throwaway booking codes as a
    side effect — that is expected and harmless.
    """
    if not picks:
        return []

    # Skip anything we already know is bad from an earlier slip
    picks = [p for p in picks if _pick_key(p) not in _KNOWN_BAD]
    if not picks:
        return []

    code, raw = book_slip(session, picks)

    if code:
        # Every pick in this subset is accepted by the API
        return picks

    if len(picks) == 1:
        # Leaf node — this single pick is definitively invalid
        p = picks[0]
        _KNOWN_BAD.add(_pick_key(p))
        err = ""
        if isinstance(raw, dict):
            err = (raw.get("msg") or raw.get("message") or
                   str(raw.get("bizCode", ""))).strip()
        indent = "      " + "  " * depth
        print(f"{indent}✗ Rejected: {p.get('home_team','?')} vs "
              f"{p.get('away_team','?')}  [{p.get('signal','')}]  "
              f"market={p['marketId']} outcome={p['outcomeId']}"
              f"  → {err[:80] or 'no detail returned'}")
        return []

    # Split and recurse into each half
    mid = len(picks) // 2
    good = (_find_good_picks(session, picks[:mid], depth + 1) +
            _find_good_picks(session, picks[mid:], depth + 1))
    return good


def book_slip_with_fallback(
        session:   requests.Session,
        picks:     list[dict],
        slip_num:  int = 0,
) -> tuple[Optional[str], list[dict], dict]:
    """
    Submit a slip. If the API rejects it, automatically prune invalid picks
    via binary search and re-submit with whatever valid picks remain.

    Returns (booking_code, final_picks_used, raw_response).
    booking_code is None only when fewer than MIN_PICKS_TO_BOOK valid picks
    survive pruning (configurable at top of file).
    """
    if not picks:
        return None, [], {}

    # ── First attempt: submit everything ──────────────────────────────────────
    code, raw = book_slip(session, picks)
    if code:
        return code, picks, raw   # All picks were valid — nothing to do

    # ── Failure: isolate bad picks via binary search ───────────────────────────
    print(f"\n    ↺  Slip {slip_num} rejected ({len(picks)} picks) "
          f"— searching for invalid selections...")

    good_picks = _find_good_picks(session, picks)

    n_removed = len(picks) - len(good_picks)
    if n_removed:
        print(f"    ✂  Pruned {n_removed} invalid pick(s). "
              f"{len(good_picks)} valid picks remain.")
    else:
        # Can happen if the failure was a transient API / network error
        # rather than a bad selection. Don't discard picks in that case.
        print(f"    ⚠  No individual bad picks isolated "
              f"(possible transient error — retrying the original slip once).")
        time.sleep(2)
        code, raw = book_slip(session, picks)
        return code, (picks if code else []), raw

    if len(good_picks) < MIN_PICKS_TO_BOOK:
        print(f"    ✗  Only {len(good_picks)} valid pick(s) — "
              f"below minimum threshold ({MIN_PICKS_TO_BOOK}). "
              f"Skipping slip {slip_num}.")
        return None, good_picks, raw

    # ── Final re-submission with only the clean picks ─────────────────────────
    print(f"    → Re-submitting slip {slip_num} with {len(good_picks)} "
          f"valid picks...")
    code, raw = book_slip(session, good_picks)

    if not code:
        print(f"    ✗  Re-submission also failed. "
              f"Giving up on slip {slip_num}.")

    return code, good_picks, raw


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    # ── Date argument ─────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        target_date = sys.argv[1].strip()
    else:
        target_date = date.today().isoformat()

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Booking Builder — SportyBet Accumulator Generator")
    print(f"  Date         : {target_date}")
    print(f"  Slips        : {NUM_SLIPS}  ×  up to {PICKS_PER_SLIP} picks each")
    print(f"  Distribution : {DISTRIBUTION}")
    print(f"  Min picks    : {MIN_PICKS_TO_BOOK}  (per slip after pruning)")
    print(f"  Dry run      : {DRY_RUN}")
    print(bar)

    # ── Collect and rank picks ────────────────────────────────────────────────
    print("\n[1] Querying database...")
    picks = collect_picks(target_date)

    if not picks:
        print("\n  No qualifying picks found for this date.")
        print("  Check that the scraper has run and the date is correct.")
        return

    # Signal breakdown
    sig_cnt: dict[str, int] = {}
    for p in picks:
        sig_cnt[p["signal"]] = sig_cnt.get(p["signal"], 0) + 1

    print(f"\n  Total qualifying picks : {len(picks)}")
    for sig, cnt in sig_cnt.items():
        top3_ev = [p["ev"] for p in picks if p["signal"] == sig][:3]
        ev_str  = "  ".join(f"{e:+.4f}" for e in top3_ev)
        print(f"    {sig:<15} : {cnt:>4} picks   top-3 EV: {ev_str}")

    # ── Build slips ───────────────────────────────────────────────────────────
    print("\n[2] Building slips...")
    slips = build_slips(picks)

    for i, slip in enumerate(slips, 1):
        sc: dict[str, int] = {}
        for p in slip:
            sc[p["signal"]] = sc.get(p["signal"], 0) + 1
        avg_ev  = sum(p["ev"] for p in slip) / len(slip)
        avg_odd = sum(p["odds"] for p in slip) / len(slip)
        print(f"  Slip {i}: {len(slip):>3} picks  "
              f"avg-odds={avg_odd:.3f}  avg-EV={avg_ev:+.5f}  "
              + "  ".join(f"{k}:{v}" for k, v in sc.items()))

    # ── Generate booking codes ────────────────────────────────────────────────
    session     = make_session()
    results: list[tuple[int, str, int, int]] = []   # (slip_num, code, n_orig, n_final)

    output: list[str] = [
        f"SportyBet Booking Codes — {target_date}",
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Slips : {len(slips)}  |  Max picks each : {PICKS_PER_SLIP}  |  Mode : {DISTRIBUTION}",
        "=" * 70,
        "",
    ]

    print(f"\n[3] {'[DRY RUN] ' if DRY_RUN else ''}Calling share API for each slip...")

    for i, slip in enumerate(slips, 1):
        print(f"\n  ── Slip {i}  ({len(slip)} picks) {'─'*(52-len(str(i)))}")

        # Preview first 5 picks
        for p in slip[:5]:
            t = (p["kickoff_time"] or "")[-8:-3]   # HH:MM from kickoff_time
            print(f"    {t}  {p['home_team'][:22]:<22} vs {p['away_team'][:22]:<22}"
                  f"  [{p['signal']}] {p['outcome_desc']:<14} @{p['odds']:.2f}"
                  f"  EV={p['ev']:+.4f}")
        if len(slip) > 5:
            print(f"    ... and {len(slip) - 5} more picks")

        # ── API call (or dry-run mock) ─────────────────────────────────────────
        if DRY_RUN:
            code        = f"DRYRUN-SLIP{i}"
            final_picks = slip
            raw         = {}
        else:
            print(f"    → posting to {SHARE_URL} ...")
            code, final_picks, raw = book_slip_with_fallback(
                session, slip, slip_num=i
            )
            if i < len(slips):
                time.sleep(1.5)   # polite delay between final submissions

        # ── Record result ──────────────────────────────────────────────────────
        n_pruned = len(slip) - len(final_picks)

        if code:
            print(f"    ✓  BOOKING CODE:  {code}"
                  + (f"  ({n_pruned} pick(s) removed)" if n_pruned else ""))
            results.append((i, code, len(slip), len(final_picks)))

            pruned_note = (f" — {n_pruned} invalid pick(s) removed automatically"
                           if n_pruned else "")
            output += [
                f"┌─ Slip {i}  ({len(final_picks)} picks{pruned_note})",
                f"│  Booking code : {code}",
                "│",
                f"│  {'#':<4} {'Time':<6} {'Home':<24} {'Away':<24}  "
                f"{'Bet':<16} {'Odds':<7} {'EV':>8}",
                "│  " + "─" * 95,
            ]
            for j, p in enumerate(final_picks, 1):
                t = (p["kickoff_time"] or "?")[-8:-3]
                output.append(
                    f"│  {j:<4} {t:<6} {p['home_team'][:23]:<24} "
                    f"{p['away_team'][:23]:<24}  "
                    f"{p['outcome_desc']:<16} "
                    f"{p['odds']:<7.2f} {p['ev']:>+8.4f}"
                )
            output += ["└" + "─" * 95, ""]

        else:
            print(f"    ✗  Slip {i} FAILED — no code returned")
            if raw:
                print(f"    Raw: {json.dumps(raw)[:300]}")
            output += [f"Slip {i} — FAILED", ""]

    # ── Write output file ──────────────────────────────────────────────────────
    out_path = os.path.join(BASE_DIR, f"booking_codes_{target_date}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{bar}")
    print(f"  COMPLETE — {len(results)}/{len(slips)} booking codes generated")
    if _KNOWN_BAD:
        print(f"  Invalid picks found and removed : {len(_KNOWN_BAD)}")
    print()
    for slip_num, code, n_orig, n_final in results:
        pruned_note = f"  (pruned {n_orig - n_final})" if n_orig != n_final else ""
        print(f"  Slip {slip_num}  ({n_final:>2} picks) :  {code}{pruned_note}")
    print()
    print(f"  Full pick lists → {out_path}")
    print(f"{bar}\n")


if __name__ == "__main__":
    run()
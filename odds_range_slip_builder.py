#!/usr/bin/env python3
"""
odds_range_slip_builder.py
===========================
Builds TWO accumulator slips that prioritize odds in the 1.15-1.30 window,
extending upward numerically (never below 1.15) when that window alone
can't fill a slip to 50 legs. Generates real SportyBet booking codes via
the same /api/ng/orders/share endpoint used by booking_builder.py.

SELECTION RULES (as specified):
  - Up to 50 legs per slip.
  - Priority: odds in [1.15, 1.30] first (ascending), then odds > 1.30
    (ascending) if the slip still isn't full. Odds below 1.15 are never used.
  - A fixture never appears twice within the SAME slip.
  - A fixture MAY appear in both slips, but never on the same market twice.
    e.g. if Slip 1 has "Over 1.5" on Match X, Slip 2 can only carry Match X
    on a different market ("Over 2.5" or "BTTS Yes").
  - Draws from the same three confirmed signals as booking_builder.py
    (D-Over1.5, B-Over2.5, C-BTTS), but unlike booking_builder.py, a fixture
    is NOT capped at one market — every market that individually clears its
    signal's probability threshold is kept as its own selectable leg, which
    is what allows the same match to contribute to both slips.

USAGE:
  python odds_range_slip_builder.py                  # today
  python odds_range_slip_builder.py 2026-07-18       # specific date

REQUIRES: the same sportybet.db this repo's other scripts use, and the same
network access to sportybet.com (run it where booking_builder.py already
works for you).
"""

import os
import sys
import json
import time
import sqlite3
from datetime import date, datetime
from typing import Optional

import requests

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "sportybet.db")

DEVICE_ID = "bf4fcdba-6e0d-418b-97f9-f9579b67bf39"   # same device-id cookie as booking_builder.py

ODDS_LOW      = 1.15
ODDS_HIGH     = 1.30
LEGS_PER_SLIP = 50
DRY_RUN       = False   # set True to preview without calling the API

# Same three confirmed signals used by booking_builder.py.
# (label, market_id, specifier, outcome_desc, min_prob, max_prob)
SIGNALS = [
    ("D-Over1.5", "18", "total=1.5", "Over 1.5", 0.80, 1.00),
    ("B-Over2.5", "18", "total=2.5", "Over 2.5", 0.45, 0.65),
    ("C-BTTS",    "29", "",          "Yes",      0.50, 1.00),
]


# ══════════════════════════════════════════════════════════════════════════
# OUTCOME ID DERIVATION  (copied from booking_builder.py)
# ══════════════════════════════════════════════════════════════════════════

def get_outcome_id(market_id: str, specifier: str, outcome_desc: str) -> str:
    if market_id == "1":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(outcome_desc, "1")
    if market_id == "18":
        return "12" if outcome_desc.startswith("Over") else "13"
    if market_id == "29":
        return "74" if outcome_desc.lower() in ("yes", "gg") else "76"
    if market_id == "11":
        return {"Home": "5", "Away": "6"}.get(outcome_desc, "5")
    if market_id == "10":
        return {
            "Home or Draw": "9",
            "Home or Away": "10",
            "Draw or Away": "11",
        }.get(outcome_desc, "9")
    if market_id == "14":
        return "1" if "Home" in outcome_desc else "2"
    if market_id in ("38", "62", "63", "64"):
        return "2" if outcome_desc.startswith("Over") else "4"
    print(f"  ⚠  Unknown market_id={market_id!r} outcome={outcome_desc!r} "
          f"— defaulting outcomeId to '1'. Verify in network tab.")
    return "1"


# ══════════════════════════════════════════════════════════════════════════
# DATABASE QUERY — every qualifying (fixture, market) leg, not deduped
# ══════════════════════════════════════════════════════════════════════════

def collect_all_legs(target_date: str) -> list[dict]:
    """
    Unlike booking_builder.collect_picks(), a fixture is NOT limited to a
    single market here. Every market that clears its own signal threshold
    is kept as an independent leg, so the same fixture can be offered to
    both slips under two different markets.
    """
    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(odds)").fetchall()]
    has_outcome_id_col = "outcome_id" in cols
    extra_col = "o.outcome_id," if has_outcome_id_col else "'__none__',"

    legs = []
    for (signal, market_id, spec, outcome, min_p, max_p) in SIGNALS:
        rows = conn.execute(f"""
            SELECT o.event_id, o.market_id, COALESCE(o.specifier,''), o.outcome_desc,
                   o.odds, o.probability, f.home_team, f.away_team, f.kickoff_time,
                   f.tournament_name, f.category_name, {extra_col} f.game_id
            FROM   odds o
            JOIN   fixtures f ON f.event_id = o.event_id
            WHERE  DATE(f.kickoff_time)      = ?
              AND  o.market_id               = ?
              AND  COALESCE(o.specifier, '') = ?
              AND  o.outcome_desc            = ?
              AND  o.probability            >= ?
              AND  o.probability            <= ?
        """, (target_date, market_id, spec, outcome, min_p, max_p)).fetchall()

        for r in rows:
            (eid, mkt_id, spec_val, out_desc, odds, prob,
             home, away, kickoff, tourn, cat, oid, game_id) = r
            outcome_id = (str(oid) if (has_outcome_id_col and oid)
                          else get_outcome_id(mkt_id, spec_val, out_desc))
            legs.append({
                "eventId":      eid,
                "marketId":     mkt_id,
                "specifier":    spec_val or None,
                "outcomeId":    outcome_id,
                "outcome_desc": out_desc,
                "odds":         odds,
                "probability":  prob,
                "signal":       signal,
                "home_team":    home,
                "away_team":    away,
                "kickoff_time": kickoff,
                "tournament":   tourn,
                "category":     cat,
                "game_id":      game_id,
                "fixture_key":  eid,   # one event_id = one real-world match
            })
    conn.close()
    return legs


# ══════════════════════════════════════════════════════════════════════════
# SLIP BUILDING
# ══════════════════════════════════════════════════════════════════════════

def build_two_slips(legs: list[dict]) -> list[list[dict]]:
    in_range = sorted([l for l in legs if ODDS_LOW <= l["odds"] <= ODDS_HIGH],
                       key=lambda l: l["odds"])
    above    = sorted([l for l in legs if l["odds"] > ODDS_HIGH],
                       key=lambda l: l["odds"])
    ordered  = in_range + above   # in-range first, then climb upward

    def fill(exclude_pairs: set) -> list[dict]:
        slip, used_fixtures = [], set()
        for leg in ordered:
            pair = (leg["fixture_key"], leg["outcome_desc"])
            if pair in exclude_pairs:
                continue
            if leg["fixture_key"] in used_fixtures:      # no dup fixture within one slip
                continue
            slip.append(leg)
            used_fixtures.add(leg["fixture_key"])
            if len(slip) >= LEGS_PER_SLIP:
                break
        return slip

    slip1 = fill(exclude_pairs=set())
    used_pairs_1 = {(l["fixture_key"], l["outcome_desc"]) for l in slip1}
    slip2 = fill(exclude_pairs=used_pairs_1)   # same fixture OK, same market not OK

    return [slip1, slip2]


# ══════════════════════════════════════════════════════════════════════════
# API  (identical pattern to booking_builder.py)
# ══════════════════════════════════════════════════════════════════════════

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


def book_slip(session: requests.Session, picks: list[dict]) -> tuple[Optional[str], dict]:
    selections = [
        {
            "eventId":   p["eventId"],
            "marketId":  p["marketId"],
            "specifier": p["specifier"],
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


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run():
    target_date = sys.argv[1].strip() if len(sys.argv) > 1 else date.today().isoformat()

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Odds-Range Slip Builder  [{ODDS_LOW}\u2013{ODDS_HIGH}, climbing upward]")
    print(f"  Date         : {target_date}")
    print(f"  Slips        : 2  ×  up to {LEGS_PER_SLIP} legs each")
    print(f"  Dry run      : {DRY_RUN}")
    print(bar)

    print("\n[1] Querying database for every qualifying (fixture, market) leg...")
    legs = collect_all_legs(target_date)
    if not legs:
        print("\n  No qualifying legs found for this date. Check the scraper ran"
              " and the date is correct.")
        return
    print(f"  Total qualifying legs : {len(legs)}")
    print(f"  In [{ODDS_LOW},{ODDS_HIGH}] : {sum(1 for l in legs if ODDS_LOW <= l['odds'] <= ODDS_HIGH)}")
    print(f"  Above {ODDS_HIGH}        : {sum(1 for l in legs if l['odds'] > ODDS_HIGH)}")

    print("\n[2] Building 2 slips...")
    slips = build_two_slips(legs)
    for i, slip in enumerate(slips, 1):
        if not slip:
            continue
        odds_vals = [l["odds"] for l in slip]
        in_range_n = sum(1 for o in odds_vals if o <= ODDS_HIGH)
        print(f"  Slip {i}: {len(slip):>2} legs   odds {min(odds_vals):.2f}-{max(odds_vals):.2f}"
              f"   ({in_range_n} in [{ODDS_LOW},{ODDS_HIGH}], {len(slip)-in_range_n} above)")

    session  = make_session()
    results  = []
    output   = [
        f"Odds-Range Booking Codes — {target_date}",
        f"Generated : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Window : [{ODDS_LOW}, {ODDS_HIGH}] first, then climbing upward | Max legs : {LEGS_PER_SLIP} | Slips : 2",
        "=" * 70,
        "",
    ]

    print(f"\n[3] {'[DRY RUN] ' if DRY_RUN else ''}Calling share API for each slip...")
    for i, slip in enumerate(slips, 1):
        if not slip:
            continue
        print(f"\n  ── Slip {i}  ({len(slip)} legs) {'─'*(52-len(str(i)))}")
        slip_sorted = sorted(slip, key=lambda l: l["kickoff_time"] or "")
        for p in slip_sorted[:5]:
            t = (p["kickoff_time"] or "")[-8:-3]
            print(f"    {t}  {p['home_team'][:22]:<22} vs {p['away_team'][:22]:<22}"
                  f"  {p['outcome_desc']:<10} @{p['odds']:.2f}  [{p['signal']}]")
        if len(slip) > 5:
            print(f"    ... and {len(slip) - 5} more legs")

        if DRY_RUN:
            code, raw = f"DRYRUN-SLIP{i}", {}
        else:
            print(f"    → posting to {SHARE_URL} ...")
            code, raw = book_slip(session, slip)
            if i < len(slips):
                time.sleep(1.5)

        if code:
            print(f"    ✓  BOOKING CODE:  {code}")
            results.append((i, code, len(slip)))
            output += [
                f"┌─ Slip {i}  ({len(slip)} legs)",
                f"│  Booking code : {code}",
                "│",
                f"│  {'#':<4} {'Time':<6} {'Home':<24} {'Away':<24}  "
                f"{'Bet':<10} {'Odds':<7}",
                "│  " + "─" * 80,
            ]
            for j, p in enumerate(slip_sorted, 1):
                t = (p["kickoff_time"] or "?")[-8:-3]
                output.append(
                    f"│  {j:<4} {t:<6} {p['home_team'][:23]:<24} "
                    f"{p['away_team'][:23]:<24}  "
                    f"{p['outcome_desc']:<10} {p['odds']:<7.2f}"
                )
            output += ["└" + "─" * 80, ""]
        else:
            print(f"    ✗  Slip {i} FAILED — no code returned")
            output += [f"Slip {i} — FAILED", ""]

    out_path = os.path.join(BASE_DIR, f"odds_range_codes_{target_date}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))

    print(f"\n{bar}")
    print(f"  COMPLETE — {len(results)}/{len([s for s in slips if s])} booking codes generated")
    for slip_num, code, n in results:
        print(f"  Slip {slip_num}  ({n:>2} legs) :  {code}")
    print(f"\n  Full pick lists → {out_path}")
    print(f"{bar}\n")


if __name__ == "__main__":
    run()

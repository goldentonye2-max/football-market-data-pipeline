#!/usr/bin/env python3
"""
dual_accumulator_builder.py
============================
Builds TWO independent accumulator slips from today's (or any date's)
full fixture list. No market is excluded — every market for every
fixture competes on EV, and whichever number wins, wins, whether
that's 1X2, Over/Under, BTTS, a corner line, a handicap, or anything
else sitting in the odds table.

  SLIP A — 35 picks
      The single best-EV outcome, one per fixture, for the 35
      highest-EV fixtures of the day.

  SLIP B — 40 picks
      Built from what's left over:
        1) every fixture NOT already used in Slip A, using ITS
           best-EV outcome
        2) if that isn't enough to reach 40, the NEXT-best-EV
           outcome from fixtures that already appear in Slip A —
           never the same market/outcome twice, and never two
           picks from the same fixture inside one slip.

CHANGES FROM PREVIOUS VERSION
------------------------------
The old version was failing whole slips because a single bad
outcomeId anywhere in 35-40 picks caused the entire POST to be
rejected, with no recovery. Two things fixed that:

  1. MERGED OUTCOME MAP — the market → outcomeId table from
     booking_builder.py (confirmed against a real market_id_probe.py
     capture, covers ~50 market types) has been folded in here,
     replacing the smaller/partially-unconfirmed set this script had.

  2. RESILIENT SUBMISSION — book_slip_with_fallback() (ported from
     booking_builder.py) now backs both Slip A and Slip B. If a slip
     is rejected, it binary-searches for the specific bad picks,
     drops them, and re-submits with only what's valid. A shared
     _KNOWN_BAD cache means a bad combo found in Slip A is
     automatically skipped if it would've shown up in Slip B too.

WHAT THIS DOESN'T DO
---------------------
Some markets (Correct Score, Exact/Team Exact Goals, Asian Handicap
Winning Margin) don't have a fixed outcomeId — the ID is a
per-fixture variant string that only exists in SportyBet's own
Outcomes response. There's no table to hardcode for these; guessing
is exactly what causes silent-wrong-bet rejections. So:

  - If your scraper captured the real outcome_id for a row (an
    `outcome_id` column on `odds`), that value is used and trusted.
  - If it didn't, rows from those variant-only markets are EXCLUDED
    from consideration rather than given a fabricated ID. You'll see
    a count of how many were skipped in the run output — if that
    number is large, it means your scraper should start saving
    outcome_id, not that this script should start guessing.
  - Everything else falls back to book_slip_with_fallback(), which
    catches whatever the static map still gets wrong (SportyBet
    tweaks an ID, a market_id you haven't seen before shows up, etc.)
    and prunes it instead of failing the whole slip.

READ THIS BEFORE YOU TRUST THE "EV" NUMBER
--------------------------------------------
EV here = probability * odds - 1, using the `probability` column in
your `odds` table. That number is only meaningful if `probability`
comes from somewhere independent of the odds you're betting into —
a model, a sharper reference book, closing-line comparison, etc.

If `probability` was itself derived FROM these same odds (straight
1/odds, or a no-vig normalization computed only within this same
market), EV here mostly reflects each market's overround, not real
edge. Worth confirming where that column comes from.

One more thing worth having in view: Slip A and Slip B are 35 and 40
legs. Win probability on an accumulator is the PRODUCT of every leg's
probability, not an average — even at a genuinely sharp 85% per-leg
hit rate, 0.85^35 ≈ 0.3%. Long slips are a real-money way to test a
model's calibration, not a strategy that "usually comes in."

USAGE:
  python dual_accumulator_builder.py              # today
  python dual_accumulator_builder.py 2026-07-20   # specific date

CONFIG:
  DRY_RUN = True   -> preview both slips, skip both API calls
  DRY_RUN = False  -> generate two real booking codes
"""

import os
import sys
import json
import time
import sqlite3
import requests
from datetime import date, datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "sportybet.db")

# ══════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════

DEVICE_ID  = "bf4fcdba-6e0d-418b-97f9-f9579b67bf39"   # update if session changes
TARGET_A   = 35
TARGET_B   = 40
DRY_RUN    = True     # flip to True to preview both slips without booking

MIN_ODDS = 1.05
MAX_ODDS = 20.0

# If a slip is rejected and pruning removes bad picks, skip the slip
# entirely rather than booking a tiny accumulator with whatever's left.
MIN_PICKS_TO_BOOK = 20

# If True (default): only picks with a CONFIRMED outcomeId are eligible —
# either from the static map's explicit branches, or a real outcome_id
# scraped into the DB. The "loose pattern" guesses (Over/Under, Yes/No,
# Home/Draw/Away matched by text on an market_id we haven't explicitly
# mapped) are excluded from both slips entirely, same as variant-only
# markets with no DB id.
# Set False to let those unconfirmed guesses back into the candidate pool
# (they'll still be caught by book_slip_with_fallback if wrong, but they
# can silently book successfully with the WRONG selection if the guess
# happens to be a valid outcomeId for a different outcome on that market).
ONLY_CONFIRMED_IDS = True


# ══════════════════════════════════════════════════════════════════════════
# OUTCOME ID MAPPING
#
# Merged from booking_builder.py's get_outcome_id(), which was built and
# checked against a real market_id_probe.py capture (Tigres FC vs Atletico
# Nacional, Copa DIMAYOR, 2026-07-29). That covers far more markets than
# this script previously did, so it replaces the smaller table here.
#
# Markets marked "variant-only" below don't have a fixed outcomeId at all —
# the ID is per-fixture (e.g. exact correct-score strings). Those return
# None unless a real scraped outcome_id is available from the database.
# ══════════════════════════════════════════════════════════════════════════

# Market IDs whose outcomeId is fixture-specific and CANNOT be derived from
# market_id + outcome_desc alone. Never guess these — skip unless the DB
# has a real outcome_id for the row.
VARIANT_ONLY_MARKETS = {
    "15",  # Asian Handicap 2-way (Winning Margin) — variant=sr:winning_margin:...
    "23",  # Home Team Exact Goals
    "24",  # Away Team Exact Goals
    "45",  # Correct Score
    "71",  # 1st Half Exact Goals
    "81",  # 1st Half Correct Score
    "93",  # 2nd Half Exact Goals
    "98",  # 2nd Half Correct Score
}


def get_outcome_id(
    market_id: str,
    specifier: str,
    outcome_desc: str,
    db_outcome_id: Optional[str] = None,
) -> tuple[Optional[str], bool]:
    """
    Returns (outcome_id, confirmed).

    outcome_id is None when this row should NOT be booked — either it's a
    variant-only market with no real scraped ID, or it's a market/outcome
    combo we've never seen and can't safely pattern-match. Callers must
    drop rows where outcome_id is None rather than substituting a guess.
    """
    # ── 1. Trust the scraper's own captured ID above everything else ───────
    if db_outcome_id:
        return str(db_outcome_id), True

    # ── 2. Variant-only markets — no static ID exists, don't guess ─────────
    if market_id in VARIANT_ONLY_MARKETS:
        return None, False

    desc = outcome_desc.strip()

    # ── 1X2 ──────────────────────────────────────────────────────────────
    # marketId=1 | Home=1, Draw=2, Away=3
    if market_id == "1":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(desc, "1"), True

    # ── Double Chance ────────────────────────────────────────────────────
    # marketId=10 | Home or Draw=9, Home or Away=10, Draw or Away=11
    if market_id == "10":
        return {
            "Home or Draw": "9", "Home or Away": "10", "Draw or Away": "11",
        }.get(desc, "9"), True

    # ── Draw No Bet ──────────────────────────────────────────────────────
    # marketId=11 | Home=4, Away=5
    if market_id == "11":
        return {"Home": "4", "Away": "5"}.get(desc, "4"), True

    # ── Home No Bet ──────────────────────────────────────────────────────
    # marketId=12 | Draw=776, Away=778
    if market_id == "12":
        return {"Draw": "776", "Away": "778"}.get(desc, "776"), True

    # ── Away No Bet ──────────────────────────────────────────────────────
    # marketId=13 | Home=780, Draw=782
    if market_id == "13":
        return {"Home": "780", "Draw": "782"}.get(desc, "780"), True

    # ── Handicap 3-way ───────────────────────────────────────────────────
    # marketId=14 | Home=1711, Draw=1712, Away=1713
    if market_id == "14":
        if "Draw" in desc:
            return "1712", True
        return ("1711", True) if "Home" in desc else ("1713", True)

    # ── Asian Handicap 2-way ─────────────────────────────────────────────
    # marketId=16 | Home=1714, Away=1715
    if market_id == "16":
        return ("1714", True) if "Home" in desc else ("1715", True)

    # ── Over/Under Goals family ──────────────────────────────────────────
    # marketId=18 (full), 19 (home), 20 (away), 68/69/70 (1H variants),
    # 90/91/92 (2H variants) | Over=12, Under=13
    if market_id in ("18", "19", "20", "68", "69", "70", "90", "91", "92"):
        return ("12", True) if desc.startswith("Over") else ("13", True)

    # ── Odd/Even family ──────────────────────────────────────────────────
    # marketId=26/27/28 (full/home/away), 74/94 (1H/2H) | Odd=70, Even=72
    if market_id in ("26", "27", "28", "74", "94"):
        return ("70", True) if desc.lower() == "odd" else ("72", True)

    # ── Yes/No family (BTTS, clean sheet, win-to-nil, both-halves O/U,
    #    win-both-halves, etc.) ─────────────────────────────────────────
    # marketId=29,31,32,33,34,48,49,50,51,58,59,75,76,77,95,96,97 | Yes=74, No=76
    if market_id in (
        "29", "31", "32", "33", "34", "48", "49", "50", "51",
        "58", "59", "75", "76", "77", "95", "96", "97",
    ):
        return ("74", True) if desc.lower() in ("yes", "gg") else ("76", True)

    # ── Which Team To Score ──────────────────────────────────────────────
    # marketId=30 | None=784, Only Home=788, Only Away=790, Both=792
    if market_id == "30":
        return {
            "None": "784", "Only Home": "788",
            "Only Away": "790", "Both teams": "792",
        }.get(desc, "792"), True

    # ── 1X2 & GG/NG combo (full time and 1H/2H) ──────────────────────────
    # marketId=35 (full), 78 (1H), 543 (2H)
    # H&Y=78, H&N=80, D&Y=82, D&N=84, A&Y=86, A&N=88
    if market_id in ("35", "78", "543"):
        return {
            "Home & yes": "78", "Home & no": "80",
            "Draw & yes": "82", "Draw & no": "84",
            "Away & yes": "86", "Away & no": "88",
        }.get(desc, "86"), True

    # ── O/U & GG/NG combo ─────────────────────────────────────────────────
    # marketId=36 | Over&Yes=90, Under&Yes=92, Over&No=94, Under&No=96
    if market_id == "36":
        return {
            "Over 2.5 & Yes": "90", "Under 2.5 & Yes": "92",
            "Over 2.5 & No": "94", "Under 2.5 & No": "96",
        }.get(desc, "90"), True

    # ── 1X2 & O/U combo (full time and 1H/2H) ────────────────────────────
    # marketId=37 (full), 79 (1H), 544 (2H)
    # H&U=794, H&O=796, D&U=798, D&O=800, A&U=802, A&O=804
    if market_id in ("37", "79", "544"):
        return {
            "Home & Under": "794", "Home & Over": "796",
            "Draw & Under": "798", "Draw & Over": "800",
            "Away & Under": "802", "Away & Over": "804",
        }.get(desc, "804"), True

    # ── Halftime/Fulltime ─────────────────────────────────────────────────
    # marketId=47
    if market_id == "47":
        return {
            "Home/Home": "418", "Home/Draw": "420", "Home/Away": "422",
            "Draw/Home": "424", "Draw/Draw": "426", "Draw/Away": "428",
            "Away/Home": "430", "Away/Draw": "432", "Away/Away": "434",
        }.get(desc, "426"), True

    # ── Highest Scoring Half ─────────────────────────────────────────────
    # marketId=52/53/54 | 1st half=436, 2nd half=438, Equal=440
    if market_id in ("52", "53", "54"):
        return {
            "1st half": "436", "2nd half": "438", "Equal": "440",
        }.get(desc, "440"), True

    # ── 1st Half 1X2 ──────────────────────────────────────────────────────
    # marketId=60 | Home=1, Draw=2, Away=3
    if market_id == "60":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(desc, "1"), True

    # ── 1st Half Double Chance ────────────────────────────────────────────
    # marketId=63 | H/D=9, H/A=10, D/A=11
    if market_id == "63":
        return {
            "Home or Draw": "9", "Home or Away": "10", "Draw or Away": "11",
        }.get(desc, "9"), True

    # ── 1st Half Draw No Bet ──────────────────────────────────────────────
    # marketId=64 | Home=4, Away=5
    if market_id == "64":
        return {"Home": "4", "Away": "5"}.get(desc, "4"), True

    # ── 1st Half Handicap 3-way ───────────────────────────────────────────
    # marketId=65 | Home=1711, Draw=1712, Away=1713
    if market_id == "65":
        if "Draw" in desc:
            return "1712", True
        return ("1711", True) if "Home" in desc else ("1713", True)

    # ── 1st Half Asian Handicap 2-way ─────────────────────────────────────
    # marketId=66 | Home=1714, Away=1715
    if market_id == "66":
        return ("1714", True) if "Home" in desc else ("1715", True)

    # ── 2nd Half 1X2 ──────────────────────────────────────────────────────
    # marketId=83 | Home=1, Draw=2, Away=3
    if market_id == "83":
        return {"Home": "1", "Draw": "2", "Away": "3"}.get(desc, "1"), True

    # ── 2nd Half Double Chance ────────────────────────────────────────────
    # marketId=85 | H/D=9, H/A=10, D/A=11
    if market_id == "85":
        return {
            "Home or Draw": "9", "Home or Away": "10", "Draw or Away": "11",
        }.get(desc, "9"), True

    # ── 2nd Half Draw No Bet ──────────────────────────────────────────────
    # marketId=86 | Home=4, Away=5
    if market_id == "86":
        return {"Home": "4", "Away": "5"}.get(desc, "4"), True

    # ── 2nd Half Handicap 3-way ───────────────────────────────────────────
    # marketId=87 | Home=1711, Draw=1712, Away=1713
    if market_id == "87":
        if "Draw" in desc:
            return "1712", True
        return ("1711", True) if "Home" in desc else ("1713", True)

    # ── 2nd Half Asian Handicap ───────────────────────────────────────────
    # marketId=88 | Home=1714, Away=1715
    if market_id == "88":
        return ("1714", True) if "Home" in desc else ("1715", True)

    # ── Corners Over/Under (full + 1H, home + away) ──────────────────────
    # marketId=900300/900301/900302/900303 | Over=30, Under=31
    if market_id in ("900300", "900301", "900302", "900303"):
        return ("30", True) if desc.startswith("Over") else ("31", True)

    # ── 3. Loose pattern fallback for markets not explicitly enumerated ───
    # These patterns (Over/Under=12/13, Yes/No=74/76, Home/Draw/Away=1/2/3)
    # hold across the ~20+ market IDs confirmed above, so a new/unseen
    # market of the same shape is LIKELY correct — but it's still a guess,
    # so it's marked unconfirmed and will go through the pruning fallback.
    dl = desc.lower()
    if dl.startswith("over"):  return "12", False
    if dl.startswith("under"): return "13", False
    if dl in ("yes", "gg"):    return "74", False
    if dl in ("no", "ng"):     return "76", False
    if dl == "home":           return "1", False
    if dl == "draw":           return "2", False
    if dl == "away":           return "3", False

    # ── 4. Truly unknown — do not guess, do not include this pick ─────────
    return None, False


# ══════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════

def rank_fixture_outcomes(target_date: str) -> tuple[dict[str, list[dict]], dict]:
    """
    { event_id: [outcome_dict, ...] } — every bookable (market, specifier,
    outcome) row for that fixture, sorted best-EV first. Rows we can't get
    a trustworthy outcomeId for are excluded (see stats["skipped_no_id"]).
    """
    conn = sqlite3.connect(DB_PATH)

    cols = [r[1] for r in conn.execute("PRAGMA table_info(odds)").fetchall()]
    has_outcome_id_col = "outcome_id" in cols
    extra_col = "o.outcome_id," if has_outcome_id_col else "NULL,"

    rows = conn.execute(f"""
        SELECT
            f.event_id, f.home_team, f.away_team, f.tournament_name,
            f.category_name, f.kickoff_time,
            o.market_id, o.market_name, COALESCE(o.specifier, '') AS specifier,
            o.outcome_desc, o.odds, o.probability,
            ROUND(o.probability * o.odds - 1, 5) AS ev,
            {extra_col}
            o.event_id
        FROM  odds o
        JOIN  fixtures f ON f.event_id = o.event_id
        WHERE DATE(f.kickoff_time) = ?
          AND o.odds       >= ?
          AND o.odds       <= ?
          AND o.probability > 0
        ORDER BY f.event_id, ev DESC
    """, (target_date, MIN_ODDS, MAX_ODDS)).fetchall()
    conn.close()

    by_fixture: dict[str, list[dict]] = {}
    skipped_no_id = 0
    skipped_variant = 0
    skipped_unconfirmed = 0

    for r in rows:
        eid  = r[0]
        spec = r[8] if r[8] else None
        db_oid = str(r[13]) if has_outcome_id_col and r[13] else None

        oid, confirmed = get_outcome_id(r[6], r[8], r[9], db_outcome_id=db_oid)

        if oid is None:
            skipped_no_id += 1
            if r[6] in VARIANT_ONLY_MARKETS:
                skipped_variant += 1
            continue

        if ONLY_CONFIRMED_IDS and not confirmed:
            skipped_unconfirmed += 1
            continue

        by_fixture.setdefault(eid, []).append({
            "eventId": eid, "home_team": r[1], "away_team": r[2],
            "tournament": r[3], "category": r[4], "kickoff_time": r[5],
            "marketId": r[6], "market_name": r[7], "specifier": spec,
            "outcome_desc": r[9], "odds": r[10], "probability": r[11],
            "ev": r[12], "outcomeId": oid, "confirmed": confirmed,
        })

    stats = {
        "rows_seen": len(rows),
        "skipped_no_id": skipped_no_id,
        "skipped_variant": skipped_variant,
        "skipped_unconfirmed": skipped_unconfirmed,
        "has_outcome_id_col": has_outcome_id_col,
    }
    return by_fixture, stats


# ══════════════════════════════════════════════════════════════════════════
# SLIP CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════

def build_slips(target_date: str):
    by_fixture, dq_stats = rank_fixture_outcomes(target_date)
    total_fixtures = len(by_fixture)

    if not by_fixture:
        return [], [], {"total_fixtures": 0, **dq_stats}

    # Best (rank-0) pick per fixture, sorted by EV — this decides Slip A.
    best_per_fixture = sorted(
        ((eid, picks[0]) for eid, picks in by_fixture.items()),
        key=lambda x: x[1]["ev"], reverse=True,
    )

    slip_a_pairs = best_per_fixture[:TARGET_A]
    slip_a       = [p for _, p in slip_a_pairs]
    slip_a_ids   = {eid for eid, _ in slip_a_pairs}

    # Slip B, step 1: best pick from every fixture NOT used in Slip A.
    remaining = [
        (eid, picks[0]) for eid, picks in by_fixture.items()
        if eid not in slip_a_ids
    ]
    remaining.sort(key=lambda x: x[1]["ev"], reverse=True)
    slip_b = [p for _, p in remaining]

    # Slip B, step 2: if still short of TARGET_B, pull the next-best pick
    # from fixtures already used in Slip A — one pick per fixture, highest
    # EV first, never repeating a fixture inside Slip B.
    used_fillers = False
    if len(slip_b) < TARGET_B:
        used_fillers = True
        need = TARGET_B - len(slip_b)
        filler_pool = []
        for eid in slip_a_ids:
            for cand in by_fixture[eid][1:]:      # everything but rank-0
                filler_pool.append((eid, cand))
        filler_pool.sort(key=lambda x: x[1]["ev"], reverse=True)

        chosen = set()
        for eid, cand in filler_pool:
            if len(chosen) >= need:
                break
            if eid in chosen:
                continue
            chosen.add(eid)
            slip_b.append(cand)

    slip_b.sort(key=lambda x: x["ev"], reverse=True)
    slip_b = slip_b[:TARGET_B]

    stats = {
        "total_fixtures": total_fixtures,
        "slip_a_count": len(slip_a),
        "slip_b_count": len(slip_b),
        "used_fillers": used_fillers,
        **dq_stats,
    }
    return slip_a, slip_b, stats


# ══════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════

SHARE_URL = "https://www.sportybet.com/api/ng/orders/share"


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
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
    })
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
    try:
        r = session.post(SHARE_URL, json={"selections": selections}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as e:
        print(f"  ✗  HTTP {e.response.status_code}: {e.response.text[:400]}")
        return None, {}
    except Exception as e:
        print(f"  ✗  Exception: {type(e).__name__}: {e}")
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
        print(f"  ⚠  No code extracted. Full response:\n     {json.dumps(data)[:500]}")
    return code, data


# ══════════════════════════════════════════════════════════════════════════
# RETRY WITH PRUNING  (ported from booking_builder.py)
#
# When a slip is rejected, we can't know from the response which pick(s)
# caused it. Binary search finds them: split the slip, test each half,
# recurse into whichever half(s) still fail, until each bad pick is
# isolated. Bad picks are cached in _KNOWN_BAD so they're skipped in the
# OTHER slip too, without needing to re-discover them.
# ══════════════════════════════════════════════════════════════════════════

_KNOWN_BAD: set[tuple] = set()


def _pick_key(p: dict) -> tuple:
    return (p["eventId"], p["marketId"], p.get("specifier"), p["outcomeId"])


def _find_good_picks(session: requests.Session,
                     picks: list[dict],
                     depth: int = 0) -> list[dict]:
    if not picks:
        return []

    picks = [p for p in picks if _pick_key(p) not in _KNOWN_BAD]
    if not picks:
        return []

    code, raw = book_slip(session, picks)
    if code:
        return picks

    if len(picks) == 1:
        p = picks[0]
        _KNOWN_BAD.add(_pick_key(p))
        err = ""
        if isinstance(raw, dict):
            err = (raw.get("msg") or raw.get("message") or
                   str(raw.get("bizCode", ""))).strip()
        indent = "      " + "  " * depth
        print(f"{indent}✗ Rejected: {p.get('home_team','?')} vs "
              f"{p.get('away_team','?')}  market={p['marketId']} "
              f"outcome={p['outcomeId']}  → {err[:80] or 'no detail returned'}")
        return []

    mid = len(picks) // 2
    good = (_find_good_picks(session, picks[:mid], depth + 1) +
            _find_good_picks(session, picks[mid:], depth + 1))
    return good


def book_slip_with_fallback(
        session:    requests.Session,
        picks:      list[dict],
        slip_label: str = "",
) -> tuple[Optional[str], list[dict], dict]:
    """
    Submit a slip. If rejected, prune invalid picks via binary search and
    re-submit with whatever valid picks remain. Returns
    (booking_code, final_picks_used, raw_response).
    """
    if not picks:
        return None, [], {}

    code, raw = book_slip(session, picks)
    if code:
        return code, picks, raw

    print(f"\n  ↺  Slip {slip_label} rejected ({len(picks)} picks) "
          f"— searching for invalid selections...")

    good_picks = _find_good_picks(session, picks)
    n_removed = len(picks) - len(good_picks)

    if n_removed:
        print(f"  ✂  Pruned {n_removed} invalid pick(s). "
              f"{len(good_picks)} valid picks remain.")
    else:
        print(f"  ⚠  No individual bad picks isolated "
              f"(possible transient error — retrying once).")
        time.sleep(2)
        code, raw = book_slip(session, picks)
        return code, (picks if code else []), raw

    if len(good_picks) < MIN_PICKS_TO_BOOK:
        print(f"  ✗  Only {len(good_picks)} valid pick(s) — below minimum "
              f"threshold ({MIN_PICKS_TO_BOOK}). Skipping slip {slip_label}.")
        return None, good_picks, raw

    print(f"  → Re-submitting slip {slip_label} with {len(good_picks)} "
          f"valid picks...")
    code, raw = book_slip(session, good_picks)
    if not code:
        print(f"  ✗  Re-submission also failed. Giving up on slip {slip_label}.")

    return code, good_picks, raw


# ══════════════════════════════════════════════════════════════════════════
# FORMATTING
# ══════════════════════════════════════════════════════════════════════════

def fmt_row(i: int, p: dict) -> str:
    t   = (p["kickoff_time"] or "")[-8:-3]
    ok  = "✓" if p["confirmed"] else "?"
    mkt = p["market_name"] or p["marketId"]
    if p.get("specifier"):
        mkt = f"{mkt} ({p['specifier']})"
    return (
        f"  {i:<4} {t:<6} {p['home_team'][:20]:<21} {p['away_team'][:20]:<21}"
        f"  {mkt[:22]:<23} {p['outcome_desc']:<16}"
        f"  {p['odds']:<7.2f} {p['ev']:>+8.4f}  {ok}"
    )


HEADER_ROW = (
    f"  {'#':<4} {'Time':<6} {'Home':<21} {'Away':<21}"
    f"  {'Market':<23} {'Outcome':<16}  {'Odds':<7} {'EV':>8}  OK"
)
DIVIDER = "  " + "─" * 115


def print_and_collect(label: str, picks: list[dict]) -> list[str]:
    if not picks:
        lines = [f"{label} — 0 picks (nothing survived)"]
        print("\n".join(lines))
        return lines

    avg_ev   = sum(p["ev"]   for p in picks) / len(picks)
    avg_odds = sum(p["odds"] for p in picks) / len(picks)
    unconfirmed = [p for p in picks if not p["confirmed"]]

    lines = [
        f"{label} — {len(picks)} picks",
        f"Average EV: {avg_ev:+.4f}   Average odds: {avg_odds:.3f}",
    ]
    if unconfirmed:
        lines.append(
            f"⚠  {len(unconfirmed)} pick(s) are pattern-matched, not "
            f"explicitly confirmed (marked ?). The pruning fallback will "
            f"catch any of these that turn out wrong."
        )
    lines += ["", HEADER_ROW, DIVIDER]
    for i, p in enumerate(picks, 1):
        lines.append(fmt_row(i, p))
    lines += [DIVIDER, ""]

    print("\n".join(lines))
    return lines


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def run():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    bar = "=" * 72
    print(f"\n{bar}")
    print(f"  Dual Accumulator Builder — {target_date}")
    print(f"  Slip A : {TARGET_A} picks (best EV, one per fixture)")
    print(f"  Slip B : {TARGET_B} picks (leftover fixtures + 2nd-best from Slip A)")
    print(f"  Min picks to book (after pruning): {MIN_PICKS_TO_BOOK}")
    print(f"  Dry run: {DRY_RUN}")
    print(bar)

    print("\n[1] Ranking every market for every fixture...")
    slip_a, slip_b, stats = build_slips(target_date)

    if not slip_a:
        print("  No fixtures found for that date. Run the scraper first.")
        return

    print(f"  {stats['total_fixtures']} fixtures found today.")
    print(f"  {stats['rows_seen']} candidate odds rows scanned.")
    if stats["skipped_no_id"]:
        detail = ""
        if stats["skipped_variant"]:
            detail = (f" ({stats['skipped_variant']} were variant-only "
                       f"markets — correct score / exact goals / winning "
                       f"margin — with no scraped outcome_id)")
        print(f"  ⚠  {stats['skipped_no_id']} row(s) excluded, no trustworthy "
              f"outcomeId available{detail}.")
    if ONLY_CONFIRMED_IDS and stats["skipped_unconfirmed"]:
        print(f"  ⚠  {stats['skipped_unconfirmed']} additional row(s) excluded "
              f"— pattern-matched but not explicitly confirmed "
              f"(ONLY_CONFIRMED_IDS=True). Set it to False to include them.")
    if not stats["has_outcome_id_col"]:
        print(f"  ℹ  Your odds table has no outcome_id column — everything "
              f"is derived from the static map. Consider having the "
              f"scraper save the real outcomeId per row; it removes the "
              f"guessing entirely for variant-only markets.")
    if stats["total_fixtures"] < TARGET_A:
        print(f"  ⚠  Fewer than {TARGET_A} fixtures available — Slip A only has {len(slip_a)} picks.")
    if stats["used_fillers"]:
        print(f"  Slip B needed fill-ins from Slip A fixtures (2nd-best pick) to reach {TARGET_B}.")

    print()
    lines_a = print_and_collect(f"SLIP A ({TARGET_A})", slip_a)
    print()
    lines_b = print_and_collect(f"SLIP B ({TARGET_B})", slip_b)

    picks_dir = os.path.join(BASE_DIR, "picks")
    os.makedirs(picks_dir, exist_ok=True)

    print(f"\n[2] {'[DRY RUN] ' if DRY_RUN else ''}Generating booking codes...")

    results: dict[str, str] = {}
    final_picks_map: dict[str, list[dict]] = {"A": slip_a, "B": slip_b}

    if DRY_RUN:
        results["A"] = "DRY-RUN"
        results["B"] = "DRY-RUN"
        print("  Skipped both API calls (DRY_RUN=True).")
        print("  Set DRY_RUN = False only after checking both slips above.")
    else:
        session = make_session()

        code_a, final_a, _raw_a = book_slip_with_fallback(session, slip_a, "A")
        time.sleep(1.5)
        code_b, final_b, _raw_b = book_slip_with_fallback(session, slip_b, "B")

        results["A"] = code_a or "FAILED"
        results["B"] = code_b or "FAILED"
        final_picks_map["A"] = final_a
        final_picks_map["B"] = final_b

        print(f"\n  Slip A code: {results['A']}"
              + (f"  ({len(slip_a) - len(final_a)} pruned)" if len(final_a) != len(slip_a) else ""))
        print(f"  Slip B code: {results['B']}"
              + (f"  ({len(slip_b) - len(final_b)} pruned)" if len(final_b) != len(slip_b) else ""))
        if _KNOWN_BAD:
            print(f"\n  Invalid (event, market, specifier, outcome) combos found this run: {len(_KNOWN_BAD)}")

    # Re-render the "final" tables if pruning changed what was actually booked
    if not DRY_RUN:
        if final_picks_map["A"] != slip_a:
            print()
            lines_a = print_and_collect(f"SLIP A — AS BOOKED ({len(final_picks_map['A'])})", final_picks_map["A"])
        if final_picks_map["B"] != slip_b:
            print()
            lines_b = print_and_collect(f"SLIP B — AS BOOKED ({len(final_picks_map['B'])})", final_picks_map["B"])

    for label, lns in (("A", lines_a), ("B", lines_b)):
        path = os.path.join(picks_dir, f"slip{label}_{target_date}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("\n".join(lns))
            f.write(f"\nBooking code: {results[label]}\n")
        print(f"  Slip {label} saved -> {path}")

    print(f"{bar}\n")


if __name__ == "__main__":
    run()
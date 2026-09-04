#!/usr/bin/env python3
"""
market_id_probe.py
==================
Probes ONE fixture and prints every marketId + outcomeId SportyBet
returns — exactly what you need to hardcode into dual.py.

Auto-picks the fixture from the highest-tier league available today
(most markets). Or pass an event_id directly.

Output:
  console   — clean table, grouped by market
  market_ids.txt  — same table saved to disk
  market_raw.json — full raw API response for deep inspection

Usage:
  python market_id_probe.py                   # auto-pick best fixture
  python market_id_probe.py sr:match:XXXXXX   # specific event_id
"""

import requests, json, time, os, sys
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DEVICE_ID = "bf4fcdba-6e0d-418b-97f9-f9579b67bf39"

FIXTURE_URL = "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents"
EVENT_URL   = "https://www.sportybet.com/api/ng/factsCenter/event"

# Leagues most likely to carry the widest market range
TOP_LEAGUES = [
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "europa league", "eredivisie", "primeira liga",
    "brasileirao", "super lig", "mls", "nations league", "copa",
    "friendly", "euro", "world cup",
]

HEADERS = {
    "accept":          "*/*",
    "accept-language": "en",
    "clientid":        "web",
    "operid":          "2",
    "platform":        "web",
    "content-type":    "application/x-www-form-urlencoded; charset=UTF-8",
    "referer":         "https://www.sportybet.com/ng/sport/football/",
    "sporty-referer":  "utm_source=https://www.google.com/",
    "user-agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/149.0.0.0 Safari/537.36"),
}

# ─────────────────────────────────────────────────────────────────────────────

def now_ms():
    return int(time.time() * 1000)


def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.set("device-id",  DEVICE_ID, domain="www.sportybet.com")
    s.cookies.set("sb_country", "ng",       domain="www.sportybet.com")
    s.cookies.set("locale",     "en",       domain="www.sportybet.com")
    return s

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — pick best fixture
# ─────────────────────────────────────────────────────────────────────────────

def fetch_fixture_list(session):
    params = {
        "sportId":  "sr:sport:1",
        "marketId": "1,18,10,29,11,26,36,14,60100",
        "pageSize": 100,
        "pageNum":  1,
        "timeline": 24,
        "_t":       now_ms(),
    }
    r = session.get(FIXTURE_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def pick_best_fixture(raw):
    """
    Returns the fixture from the highest-tier league, earliest kickoff.
    Falls back to absolute first real fixture if no top-league match found.
    """
    tournaments = raw.get("data", {}).get("tournaments", [])
    candidates  = []

    for t in tournaments:
        cat  = t.get("categoryName", "")
        name = t.get("name", "")
        if cat == "Simulated Reality League":
            continue
        is_top = any(k in name.lower() or k in cat.lower() for k in TOP_LEAGUES)
        for ev in t.get("events", []):
            candidates.append({
                "eventId":    ev.get("eventId", ""),
                "home":       ev.get("homeTeamName", ""),
                "away":       ev.get("awayTeamName", ""),
                "tournament": name,
                "category":   cat,
                "kickoff_ms": ev.get("estimateStartTime", 0),
                "is_top":     is_top,
            })

    if not candidates:
        return None

    # top-league first, then earliest kickoff within that group
    candidates.sort(key=lambda x: (not x["is_top"], x["kickoff_ms"]))
    return candidates[0]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — fetch markets
# ─────────────────────────────────────────────────────────────────────────────

def fetch_markets(session, event_id):
    params = {
        "eventId":   event_id,
        "productId": "3",
        "_t":        now_ms(),
    }
    r = session.get(EVENT_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — build flat table
# ─────────────────────────────────────────────────────────────────────────────

def build_rows(markets):
    rows = []
    for mkt in markets:
        mid    = str(mkt.get("id", ""))
        mname  = mkt.get("name") or mkt.get("desc", "")
        group  = mkt.get("group", "")
        spec   = mkt.get("specifier") or ""

        for oc in mkt.get("outcomes", []):
            # include ALL — active and inactive — so you see every possible ID
            rows.append({
                "group":        group,
                "market_id":    mid,
                "market_name":  mname,
                "specifier":    spec,
                "outcome_id":   str(oc.get("id") or oc.get("outcomeId") or ""),
                "outcome_desc": oc.get("desc", ""),
                "odds":         oc.get("odds", "—"),
                "active":       "✓" if oc.get("isActive") == 1 else "✗",
            })
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — display + save
# ─────────────────────────────────────────────────────────────────────────────

W = 118  # table width

def render(fixture, rows):
    unique_markets = len(set(r["market_id"] for r in rows))

    header_lines = [
        "=" * W,
        f"  SportyBet Market ID Probe  —  {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  Fixture   : {fixture['home']} vs {fixture['away']}",
        f"  League    : {fixture['tournament']}  ({fixture['category']})",
        f"  Event ID  : {fixture['eventId']}",
        f"  Markets   : {unique_markets}   |   Outcomes (rows): {len(rows)}",
        "=" * W,
        "",
        (f"  {'Group':<14} {'MarketID':<10} {'Market Name':<30} {'Specifier':<18}"
         f" {'OutcomeID':<12} {'Outcome Desc':<24} {'Odds':<8} Active"),
        "  " + "─" * (W - 2),
    ]

    body_lines = []
    prev_mid   = None

    for r in rows:
        if r["market_id"] != prev_mid:
            if prev_mid is not None:
                body_lines.append("")          # blank line between markets
            prev_mid = r["market_id"]

        body_lines.append(
            f"  {r['group']:<14} {r['market_id']:<10} {r['market_name'][:29]:<30} "
            f"{r['specifier'][:17]:<18} {r['outcome_id']:<12} "
            f"{r['outcome_desc'][:23]:<24} {str(r['odds']):<8} {r['active']}"
        )

    footer = [
        "",
        "  " + "─" * (W - 2),
        (f"  {unique_markets} markets / {len(rows)} outcome rows  "
         f"— copy marketId + outcomeId pairs into dual.py CONFIRMED_IDS"),
        "=" * W,
    ]

    return "\n".join(header_lines + body_lines + footer)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run():
    session = get_session()

    # ── Pick fixture ───────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        event_id = sys.argv[1]
        fixture  = {
            "eventId": event_id, "home": "?", "away": "?",
            "tournament": "?", "category": "?",
        }
        print(f"Using provided event_id: {event_id}")
    else:
        print("Fetching fixture list to find best candidate...")
        raw_list = fetch_fixture_list(session)

        if raw_list.get("bizCode") != 10000:
            print(f"Fixture list API error: {raw_list}")
            return

        fixture = pick_best_fixture(raw_list)
        if not fixture:
            print("No real fixtures found today.")
            return

        print(f"\n  Selected : {fixture['home']} vs {fixture['away']}")
        print(f"  League   : {fixture['tournament']}  ({fixture['category']})")
        print(f"  Event ID : {fixture['eventId']}\n")

    event_id = fixture["eventId"]

    # ── Fetch markets ──────────────────────────────────────────────────────────
    print("Fetching all markets for this fixture...")
    raw_event = fetch_markets(session, event_id)

    if not raw_event or raw_event.get("bizCode") != 10000:
        print(f"Market API error: {json.dumps(raw_event)[:300]}")
        return

    markets = raw_event.get("data", {}).get("markets", [])
    if not markets:
        print("No markets returned. Try a bigger fixture (pass its event_id as arg).")
        return

    print(f"  {len(markets)} markets received.\n")

    # ── Build table ────────────────────────────────────────────────────────────
    rows   = build_rows(markets)
    output = render(fixture, rows)

    print(output)

    # ── Save table ─────────────────────────────────────────────────────────────
    txt_path = os.path.join(BASE_DIR, "market_ids.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nTable saved  →  {txt_path}")

    # ── Save raw JSON ──────────────────────────────────────────────────────────
    raw_path = os.path.join(BASE_DIR, "market_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_event, f, indent=2, ensure_ascii=False)
    print(f"Raw JSON     →  {raw_path}")


if __name__ == "__main__":
    run()
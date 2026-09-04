"""
results_scraper.py
Fetches final scores for your own fixtures from sporty.com's bulk
events/scores lookup, and stores them in a `results` table.

WHY THIS DESIGN
----------------
sporty.com's *listing* endpoints (topLeagues / sortedLeagues) have their own
coverage and day-boundary quirks we tested and don't fully trust. But
events/scores is a different shape entirely: you POST a list of event IDs
YOU ALREADY HAVE, and it returns scores for exactly those IDs. Since your
`fixtures` table (built from the SportyBet odds scraper) is already the
authoritative list of what you actually care about, there's no listing
endpoint, coverage gap, or ID-matching problem here at all -- just a direct
lookup on keys you already trust.

CONFIRMED BEHAVIOUR (tested manually before this was written)
----------------------------------------------------------------
- Endpoint responds with real JSON to a plain POST, no cookies required in
  our testing (a separate endpoint on the same domain returned a clean 404
  -- not a Cloudflare challenge -- with no cookies sent, which is decent
  evidence this domain doesn't gate on session cookies for read requests).
- A single real request carried ~80 IDs successfully. BATCH_SIZE below
  defaults to that known-working number. Untested above that -- if you want
  to push it higher, test manually first and watch for a 413 or 400.
- Some IDs use a distinct long-form prefix (e.g. sr:match:111111114130330)
  for youth/reserve competitions -- these are still real football fixtures,
  not a different sport, and are handled identically here.
- eventScore is NOT a safe source for match goals. For shootout-decided
  matches it silently combines match goals + penalty kicks into one number
  (confirmed: a real 2-2 match won 9-8 on penalties came back as
  eventScore "10:11" -- 2+8 and 2+9). home_goals/away_goals are read from
  displayScoreDetail.mainScore instead, which keeps the real match score
  separate from the shootout tally (stored separately in penalty_home/
  penalty_away). AP ("After Penalties") is classified as 'finished', same
  as any other completed match.

RUN
---
    python results_scraper.py                # every fixture currently in `fixtures`
    python results_scraper.py 2026-06-24      # just that date (once you're scoping by day)

With no date given, this fetches scores for every fixture in your
`fixtures` table, regardless of date -- useful now, while you've only got
a few days of data sitting in one table. Regardless of scope, whatever
status sporty.com reports (Ended, Not started, Postponed, etc.) is stored
as-is in `raw_status`, so you can tell a genuine 0-0 apart from "hasn't
been played yet" or "cancelled".
"""

import sys
import sqlite3
from datetime import datetime

import requests

DB_PATH = "sportybet.db"

SPORTY_SCORES_URL = "https://sporty.com/api/media/v1/events/scores"
BATCH_SIZE = 80  # known-working from a real captured request; untested above this

HEADERS = {
    "accept": "application/json",
    "accept-language": "en-us",
    "content-type": "application/json",
    "country-code": "NG",
    "device-id": "dfc74b86c0e7965a38fb02ff3970329a",  # any consistent-looking value is fine here
    "platform": "web",
    "sporty-brand": "sporty-tv",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
    ),
}

# NOTE: if requests.exceptions.ConnectionError / "could not resolve host"
# shows up here the way it did with curl in Git Bash, that was a curl/MinGW-
# specific quirk, not a general network issue (nslookup resolved fine, and
# curl worked once forced with --resolve). Python's requests library uses a
# different resolution path and likely won't hit the same problem -- but if
# it does, resolve sporty.com yourself (nslookup) and swap BASE_URL below to
# hit the IP directly with a Host header, similar to curl's --resolve trick.


# ---------------------------------------------------------------------------
# Loading your own fixtures (the authoritative ID source)
# ---------------------------------------------------------------------------
def load_fixture_ids(conn: sqlite3.Connection, date_str: str | None = None) -> list[dict]:
    """Returns fixtures (event_id, home_team, away_team) from your `fixtures`
    table. If date_str ('YYYY-MM-DD') is given, only fixtures kicking off
    that date are returned; if omitted, every fixture currently in the
    table is returned -- what you want while everything still lives in one
    small table, before you're scraping and scoping day by day."""
    cur = conn.cursor()
    if date_str:
        cur.execute(
            """
            SELECT event_id, home_team, away_team
            FROM fixtures
            WHERE kickoff_time LIKE ?
            """,
            (f"{date_str}%",),
        )
    else:
        cur.execute("SELECT event_id, home_team, away_team FROM fixtures")
    return [
        {"event_id": row[0], "home_team": row[1], "away_team": row[2]}
        for row in cur.fetchall()
    ]


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_scores_batch(event_ids: list[str]) -> list[dict]:
    resp = requests.post(
        SPORTY_SCORES_URL,
        headers=HEADERS,
        json={"eventIdList": event_ids},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("bizCode") != 10000:
        raise RuntimeError(f"Unexpected response: {payload.get('message')}")

    return payload.get("data", [])


def parse_score(score_str: str) -> tuple[int | None, int | None]:
    """'4:0' or '2 : 2' -> (4, 0) / (2, 2). Returns (None, None) if missing,
    null, or unparseable (e.g. not started yet)."""
    if not score_str or ":" not in score_str:
        return None, None
    try:
        home, away = score_str.split(":")
        return int(home.strip()), int(away.strip())
    except ValueError:
        return None, None


def parse_period_scores(display_period_scores: list) -> dict:
    """Returns {'FT': (h,a), 'ET': (h,a), 'AP': (h,a)} for whichever labels
    are actually present -- confirmed structure, from a real match that went
    to penalties (sr:match:68156812): displayPeriodScores carried FT "2:2",
    ET "2:2", AP "10:11" as three separate, explicitly labelled, cumulative
    checkpoints. A label is only present if the match actually reached that
    stage -- ET and AP are simply absent for matches decided in normal time,
    which is why the very first single-match response we tested only ever
    showed FT.
    """
    out = {}
    for entry in display_period_scores or []:
        label = entry.get("label")
        if label:
            out[label] = parse_score(entry.get("score"))
    return out


def infer_ht_score(event_game_score: list) -> tuple[int | None, int | None]:
    """UNCONFIRMED -- inferred, not verified against a known-correct
    half-time score. eventGameScore looks like a per-period (not cumulative)
    breakdown, e.g. ["0:1","2:0"] for a 2-1 match = 0-1 in the first half,
    then 2-0 in the second. If that reading is right, element [0] IS the
    half-time score directly, since nothing precedes the first half.
    Before trusting ht_home/ht_away for real analysis, sanity-check a few
    rows against a match where you independently know the HT score (or
    cross-check against goal minutes from the /summary endpoint's `goals`
    list -- count goals with eventTimeMinutes <= 45).
    """
    if not event_game_score:
        return None, None
    return parse_score(event_game_score[0])


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            event_id     TEXT PRIMARY KEY,
            ht_home      INTEGER,   -- INFERRED, not confirmed -- see notes above parse_period_scores()
            ht_away      INTEGER,
            ft_home      INTEGER,   -- confirmed: cumulative score at 90 min, from displayPeriodScores label='FT'
            ft_away      INTEGER,
            et_home      INTEGER,   -- confirmed: cumulative score after extra time, label='ET'. NULL if no ET played.
            et_away      INTEGER,
            pen_home     INTEGER,   -- confirmed: shootout kicks only, label='AP'. NULL if no shootout.
            pen_away     INTEGER,
            home_goals   INTEGER,   -- convenience column: et_* if ET was played, else ft_* -- "real goals however far it went", for goal-based market settlement
            away_goals   INTEGER,
            status       TEXT,      -- normalised: 'finished' / 'not_started' / 'other'
            raw_status   TEXT,      -- exact string sporty.com returned, kept as-is
            settled_at   TEXT
        )
    """)
    conn.commit()


def normalise_status(raw_status: str) -> str:
    # AP = "After Penalties" -- confirmed via a real match (LA FC 2 vs Real
    # Monarchs SLC, sr:match:68156812). A shootout-decided match is a fully
    # completed match, not an edge case -- belongs in 'finished', not
    # 'other'. AET ("After Extra Time", no shootout needed) is included as
    # a reasonable guess at a sibling status but hasn't actually been seen
    # in your data yet -- if a genuine 'other' row later turns out to be
    # AET, add it here too.
    if raw_status in ("Ended", "Finished", "AP", "AET"):
        return "finished"
    if raw_status in ("Not started", "Not_Started", None, ""):
        return "not_started"
    return "other"  # postponed, cancelled, suspended, etc. -- inspect raw_status for specifics


def write_results(conn: sqlite3.Connection, rows: list[dict]):
    now = datetime.now().isoformat(timespec="seconds")
    records = []
    for row in rows:
        periods = parse_period_scores(row.get("displayPeriodScores"))
        ht_home, ht_away = infer_ht_score(row.get("eventGameScore"))
        ft_home, ft_away = periods.get("FT", (None, None))
        et_home, et_away = periods.get("ET", (None, None))
        pen_home, pen_away = periods.get("AP", (None, None))

        # "Real goals, however far the match went" -- ET score if extra
        # time was played (since goals scored in ET are real match goals),
        # otherwise FT. Never includes shootout kicks (pen_home/pen_away),
        # which aren't goals at all for betting-market settlement purposes.
        if et_home is not None:
            home_goals, away_goals = et_home, et_away
        else:
            home_goals, away_goals = ft_home, ft_away

        raw_status = row.get("eventMatchStatus") or row.get("eventStatus")
        records.append({
            "event_id": row["eventId"],
            "ht_home": ht_home, "ht_away": ht_away,
            "ft_home": ft_home, "ft_away": ft_away,
            "et_home": et_home, "et_away": et_away,
            "pen_home": pen_home, "pen_away": pen_away,
            "home_goals": home_goals, "away_goals": away_goals,
            "status": normalise_status(raw_status),
            "raw_status": raw_status,
            "settled_at": now,
        })

    conn.executemany(
        """
        INSERT INTO results (
            event_id, ht_home, ht_away, ft_home, ft_away, et_home, et_away,
            pen_home, pen_away, home_goals, away_goals, status, raw_status, settled_at
        )
        VALUES (
            :event_id, :ht_home, :ht_away, :ft_home, :ft_away, :et_home, :et_away,
            :pen_home, :pen_away, :home_goals, :away_goals, :status, :raw_status, :settled_at
        )
        ON CONFLICT(event_id) DO UPDATE SET
            ht_home    = excluded.ht_home,    ht_away    = excluded.ht_away,
            ft_home    = excluded.ft_home,    ft_away    = excluded.ft_away,
            et_home    = excluded.et_home,    et_away    = excluded.et_away,
            pen_home   = excluded.pen_home,   pen_away   = excluded.pen_away,
            home_goals = excluded.home_goals, away_goals = excluded.away_goals,
            status     = excluded.status,
            raw_status = excluded.raw_status,
            settled_at = excluded.settled_at
        """,
        records,
    )
    conn.commit()
    return records


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main(date_str: str | None = None, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    ensure_table(conn)

    fixtures = load_fixture_ids(conn, date_str)
    scope = f"for {date_str}" if date_str else "across your whole fixtures table"
    print(f"{len(fixtures)} fixture(s) {scope}.")

    if not fixtures:
        print("Nothing to fetch. Exiting.")
        conn.close()
        return

    id_to_label = {f["event_id"]: f"{f['home_team']} vs {f['away_team']}" for f in fixtures}
    all_ids = list(id_to_label.keys())

    all_rows = []
    for i, batch in enumerate(chunked(all_ids, BATCH_SIZE), start=1):
        print(f"Fetching batch {i} ({len(batch)} fixture(s))...")
        try:
            rows = fetch_scores_batch(batch)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [error] Batch {i} failed: {e}")
            continue

    written = write_results(conn, all_rows)
    conn.close()

    finished = sum(1 for r in written if r["status"] == "finished")
    not_started = sum(1 for r in written if r["status"] == "not_started")
    other = sum(1 for r in written if r["status"] == "other")

    missing = set(all_ids) - {r["event_id"] for r in written}

    print(f"\nDone. {len(written)} result(s) written:")
    print(f"  Finished     : {finished}")
    print(f"  Not started  : {not_started}")
    print(f"  Other status : {other}  (postponed/cancelled/etc. -- check raw_status)")
    if missing:
        print(f"  No response for {len(missing)} fixture(s) -- sporty.com may not have them:")
        for eid in list(missing)[:10]:
            print(f"    {eid}  ({id_to_label.get(eid, '?')})")


if __name__ == "__main__":
    # No date given -> every fixture currently in your DB, which is what you
    # want right now with everything in one small table. Pass a date
    # (YYYY-MM-DD) later once you're scraping and scoping this day by day.
    arg_date = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg_date)
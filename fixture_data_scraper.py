#!/usr/bin/env python3
DATE_PERIOD = 24
f"""
SportyBet Odds Scraper — SQLite Version
========================================
Fetches all upcoming football fixtures (next {DATE_PERIOD} hours) from SportyBet Nigeria,
pulls every active market (excluding Players group) for each real fixture,
and stores everything permanently in a SQLite database.

Filters applied before anything touches the database:
  - SRL / Simulated Reality League fixtures → blocked entirely
  - Players market group → blocked entirely
  - Inactive outcomes (isActive != 1) → blocked entirely

Safe to re-run: fixtures already in the database are detected and skipped.

Database : sportybet.db  (same folder as this script)
Log      : scraper.log   (same folder as this script)

Usage    : python sportybet_scraper_db.py
"""

import requests
import sqlite3
import time
import os
import logging
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, "sportybet.db")

FIXTURE_URL = "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents"
EVENT_URL   = "https://www.sportybet.com/api/ng/factsCenter/event"


HEADERS = {
    "accept":          "*/*",
    "accept-language": "en",
    "clientid":        "web",
    "operid":          "2",
    "platform":        "web",
    "content-type":    "application/x-www-form-urlencoded; charset=UTF-8",
    "referer":         f"https://www.sportybet.com/ng/sport/football/upcoming?time={DATE_PERIOD}",
    "sporty-referer":  "utm_source=https://www.google.com/",
    "user-agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
}

# Any fixture whose category_name matches this set is simulated — skip entirely
SIMULATED_CATEGORIES = {"Simulated Reality League"}

# Market groups to exclude from storage
EXCLUDE_GROUPS = {"Players"}

PAGE_SIZE   = 100   # max fixtures per page
EVENT_DELAY = 2     # seconds between per-match API calls
PAGE_DELAY  = 1     # seconds between fixture-list page calls
RETRY_COUNT = 3     # retries on a failed event call
RETRY_WAIT  = 5     # seconds before each retry


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    log_path = os.path.join(BASE_DIR, "scraper.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE — schema + setup
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA = """
    -- One row per real football fixture, inserted once when first scraped.
    CREATE TABLE IF NOT EXISTS fixtures (
        event_id        TEXT PRIMARY KEY,   -- sr:match:XXXXXXXX  (Betradar ID)
        game_id         TEXT,               -- SportyBet internal ID
        tournament_id   TEXT,               -- sr:tournament:XX
        tournament_name TEXT,               -- e.g. "Premier League"
        category_name   TEXT,               -- e.g. "England"
        home_team       TEXT,
        away_team       TEXT,
        kickoff_time    TEXT,               -- YYYY-MM-DD HH:MM
        kickoff_ms      INTEGER,            -- Unix ms — use for sorting/filtering
        scraped_at      TEXT                -- YYYY-MM-DD HH:MM
    );

    -- One row per active outcome per market per fixture.
    -- Never overwritten — grows permanently with every scrape run.
    CREATE TABLE IF NOT EXISTS odds (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id     TEXT    NOT NULL,
        market_group TEXT,                  -- Main / Goals / Half / Combo etc.
        market_id    TEXT,                  -- Betradar market type ID
        market_name  TEXT,                  -- "1X2", "Over/Under", "Handicap"
        specifier    TEXT,                  -- "total=2.5", "hcp=1:0" (blank if none)
        outcome_desc TEXT,                  -- "Home", "Away", "Over 2.5"
        outcome_id   TEXT,                  -- Betradar outcome ID (e.g. "12", "74")
        odds         REAL,
        probability  REAL,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    -- One row per scraper execution — bookkeeping only.
    CREATE TABLE IF NOT EXISTS scrape_log (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        run_date           TEXT,
        run_started_at     TEXT,
        run_completed_at   TEXT,
        fixtures_scraped   INTEGER DEFAULT 0,
        odds_rows_inserted INTEGER DEFAULT 0,
        fixtures_failed    INTEGER DEFAULT 0,
        fixtures_skipped   INTEGER DEFAULT 0
    );

    -- Indexes — make queries fast as the database grows over months
    CREATE INDEX IF NOT EXISTS idx_odds_event_id
        ON odds(event_id);

    CREATE INDEX IF NOT EXISTS idx_odds_market_name
        ON odds(market_name);

    CREATE INDEX IF NOT EXISTS idx_odds_market_group
        ON odds(market_group);

    CREATE INDEX IF NOT EXISTS idx_fixtures_tournament
        ON fixtures(tournament_name);

    CREATE INDEX IF NOT EXISTS idx_fixtures_kickoff
        ON fixtures(kickoff_ms);

    CREATE INDEX IF NOT EXISTS idx_fixtures_scraped_at
        ON fixtures(scraped_at);
"""


def init_db(conn):
    """Create all tables and indexes if they do not exist yet."""
    conn.executescript(SCHEMA)

    # Migrate existing databases — adds outcome_id if not already there
    existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(odds)").fetchall()]
    if "outcome_id" not in existing_cols:
        conn.execute("ALTER TABLE odds ADD COLUMN outcome_id TEXT")
        log.info("  Migration: added outcome_id column to existing odds table")

    conn.commit()


def already_scraped(conn, event_id):
    """
    Return True if this event already has odds rows in the database.
    Prevents duplicate inserts if the scraper is run more than once in a day.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM odds WHERE event_id = ?", (event_id,)
    ).fetchone()
    return row[0] > 0


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def now_ms():
    """Current Unix timestamp in milliseconds — used as cache buster."""
    return int(time.time() * 1000)


def fmt_kickoff(ms):
    """Convert estimateStartTime (Unix ms) to a readable datetime string."""
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def safe_float(value, default=0.0):
    """
    Convert a value to float safely.
    Handles scientific notation strings like '0E-10'.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════════════
# API CALLS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fixture_page(page_num):
    f"""Fetch one page of upcoming football fixtures (next {DATE_PERIOD} hours)."""
    params = {
        "sportId":  "sr:sport:1",
        "marketId": "1,18,10,29,11,26,36,14,60100",
        "pageSize": PAGE_SIZE,
        "pageNum":  page_num,
        "timeline": DATE_PERIOD,
        "_t":       now_ms(),
    }
    resp = requests.get(FIXTURE_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_event_markets(event_id):
    """
    Fetch the full market board for one match.
    Retries up to RETRY_COUNT times before giving up.
    """
    params = {
        "eventId":   event_id,
        "productId": "3",
        "_t":        now_ms(),
    }
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = requests.get(EVENT_URL, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning(f"    Attempt {attempt}/{RETRY_COUNT} failed: {exc}")
            if attempt < RETRY_COUNT:
                log.info(f"    Waiting {RETRY_WAIT}s before retry...")
                time.sleep(RETRY_WAIT)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Collect all real fixtures (filters out SRL)
# ══════════════════════════════════════════════════════════════════════════════

def collect_all_fixtures():
    """
    Pages through the fixture list endpoint until all fixtures are collected.
    SRL / simulated fixtures are filtered out here — they never reach the DB.
    Returns a flat list of real fixture dicts with tournament metadata attached.
    """
    all_fixtures = []
    total_num    = None
    page         = 1
    srl_count    = 0

    while True:
        log.info(f"  Fixture list — page {page}...")
        raw = fetch_fixture_page(page)

        if raw.get("bizCode") != 10000:
            log.error(f"  Bad response on page {page}: bizCode={raw.get('bizCode')}")
            break

        data        = raw.get("data", {})
        tournaments = data.get("tournaments", [])

        if total_num is None:
            total_num   = data.get("totalNum", 0)
            pages_total = -(-total_num // PAGE_SIZE)
            log.info(
                f"  Total fixtures reported by API: {total_num}  "
                f"({pages_total} page(s) — includes SRL)"
            )

        page_real = 0
        page_srl  = 0

        for tournament in tournaments:
            cat_name = tournament.get("categoryName", "")

            # Build tournament metadata to attach to each event
            t_meta = {
                "tournament_id":   tournament.get("id", ""),
                "tournament_name": tournament.get("name", ""),
                "category_name":   cat_name,
            }

            for event in tournament.get("events", []):
                # SRL check — filter before anything else
                if cat_name in SIMULATED_CATEGORIES:
                    page_srl  += 1
                    srl_count += 1
                    continue

                all_fixtures.append({**event, **t_meta})
                page_real += 1

        log.info(
            f"  Page {page}: "
            f"{page_real} real  |  {page_srl} SRL filtered  |  "
            f"running total: {len(all_fixtures)} real fixtures"
        )

        # Stop when we have accounted for everything the API reported
        if (len(all_fixtures) + srl_count) >= total_num:
            break

        page += 1
        time.sleep(PAGE_DELAY)

    log.info(f"  SRL fixtures filtered out this run: {srl_count}")
    return all_fixtures


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Fetch markets per fixture and write to database
# ══════════════════════════════════════════════════════════════════════════════

def scrape_and_store(conn, fixtures):
    """
    For each fixture: fetch full markets, filter, insert into DB.
    Returns (success_count, fail_count, skip_count, total_odds_rows).
    """
    success_count   = 0
    fail_count      = 0
    skip_count      = 0
    total_odds_rows = 0
    scraped_at      = datetime.now().strftime("%Y-%m-%d %H:%M")

    for i, fixture in enumerate(fixtures, 1):
        event_id = fixture.get("eventId", "")
        game_id  = fixture.get("gameId", "")
        home     = fixture.get("homeTeamName", "")
        away     = fixture.get("awayTeamName", "")
        kick_ms  = fixture.get("estimateStartTime")
        kickoff  = fmt_kickoff(kick_ms)
        t_id     = fixture.get("tournament_id", "")
        t_name   = fixture.get("tournament_name", "")
        cat_name = fixture.get("category_name", "")

        log.info(f"  [{i:>3}/{len(fixtures)}]  {home} vs {away}  ({event_id})")

        # ── Skip if already in database ────────────────────────────────────────
        if already_scraped(conn, event_id):
            log.info(f"    Already in database — skipping")
            skip_count += 1
            continue

        # ── Fetch full market board ────────────────────────────────────────────
        raw = fetch_event_markets(event_id)

        if not raw or raw.get("bizCode") != 10000:
            log.warning(f"    FAILED after {RETRY_COUNT} attempts — skipping")
            fail_count += 1
            time.sleep(EVENT_DELAY)
            continue

        markets = raw.get("data", {}).get("markets", [])

        # ── Insert fixture row ─────────────────────────────────────────────────
        conn.execute("""
            INSERT OR IGNORE INTO fixtures
                (event_id, game_id, tournament_id, tournament_name, category_name,
                 home_team, away_team, kickoff_time, kickoff_ms, scraped_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            event_id, game_id, t_id, t_name, cat_name,
            home, away, kickoff, kick_ms, scraped_at
        ))

        # ── Build odds rows ────────────────────────────────────────────────────
        odds_rows = []

        for market in markets:
            group = market.get("group", "")

            # Skip excluded groups
            if group in EXCLUDE_GROUPS:
                continue

            market_id   = market.get("id", "")
            market_name = market.get("name") or market.get("desc", "")
            specifier   = market.get("specifier", "")

            for outcome in market.get("outcomes", []):
                if outcome.get("isActive") != 1:
                    continue

                outcome_id = str(outcome.get("id") or outcome.get("outcomeId") or "")

                odds_rows.append((
                    event_id,
                    group,
                    market_id,
                    market_name,
                    specifier,
                    outcome.get("desc", ""),
                    outcome_id,                        # ← new
                    safe_float(outcome.get("odds")),
                    round(safe_float(outcome.get("probability")), 6),
                ))

        # ── Batch insert all odds for this fixture ─────────────────────────────
        conn.executemany("""
            INSERT INTO odds
                (event_id, market_group, market_id, market_name,
                 specifier, outcome_desc, outcome_id, odds, probability)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, odds_rows)

        conn.commit()

        log.info(
            f"    {len(markets)} raw markets → "
            f"{len(odds_rows)} odds rows stored"
        )

        total_odds_rows += len(odds_rows)
        success_count   += 1

        time.sleep(EVENT_DELAY)

    return success_count, fail_count, skip_count, total_odds_rows


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    setup_logging()

    today      = datetime.now().strftime("%Y-%m-%d")
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log.info("=" * 65)
    log.info(f"  SportyBet DB Scraper  —  {today}")
    log.info(f"  Database : {DB_PATH}")
    log.info("=" * 65)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # ── STEP 1 ─────────────────────────────────────────────────────────────────
    log.info("\nSTEP 1: Collecting fixture list (next {} hours)...")
    fixtures = collect_all_fixtures()
    log.info(f"  Done. {len(fixtures)} real fixtures to process.\n")

    if not fixtures:
        log.error("  No real fixtures found — exiting.")
        conn.close()
        return

    # ── STEP 2 ─────────────────────────────────────────────────────────────────
    log.info("STEP 2: Fetching markets and storing in database...")
    est_mins = (len(fixtures) * EVENT_DELAY) // 60
    est_secs = (len(fixtures) * EVENT_DELAY) % 60
    log.info(f"  Estimated time: ~{est_mins} min {est_secs} sec\n")

    success, failed, skipped, odds_rows = scrape_and_store(conn, fixtures)

    # ── LOG THIS RUN ───────────────────────────────────────────────────────────
    completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO scrape_log
            (run_date, run_started_at, run_completed_at,
             fixtures_scraped, odds_rows_inserted,
             fixtures_failed, fixtures_skipped)
        VALUES (?,?,?,?,?,?,?)
    """, (today, started_at, completed_at, success, odds_rows, failed, skipped))
    conn.commit()
    conn.close()

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  SCRAPER COMPLETE")
    log.info(f"  Real fixtures scraped  : {success}")
    log.info(f"  Already in DB (skipped): {skipped}")
    log.info(f"  Failed                 : {failed}")
    log.info(f"  Odds rows inserted     : {odds_rows:,}")
    log.info(f"  Database               : {DB_PATH}")
    log.info("=" * 65)


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""
FlashScore Linker & Stats Scraper
===================================
Two steps, one script:

STEP 1 — LINKER
  Fetches FlashScore's daily match list for a given date, fuzzy-matches
  each FlashScore match against your fixtures table (by team name + kickoff
  proximity), and stores the confirmed FlashScore match ID plus all
  available scores (FT, HT, ET, penalties) in match_links.

STEP 2 — STATS SCRAPER
  For every row in match_links that has no stats yet, fetches:
    df_st_1_{fs_id}   → full match statistics (shots, corners, possession…)
    df_sui_1_{fs_id}  → match incidents (goals, cards, substitutions, VAR)
  Parses both and stores into flashscore_stats and flashscore_incidents.

SAFE ON EXISTING DATABASES
  New score columns are added automatically via ALTER TABLE if they do not
  exist yet. All existing rows are untouched — new columns default to NULL.

DATABASE TABLES CREATED / UPDATED:
  match_links          — event_id ↔ fs_match_id bridge + all scorelines
  flashscore_stats     — stats per match per section (EAV shape)
  flashscore_incidents — one row per incident per match

USAGE:
  python flashscore_pipeline.py                  # yesterday (most common)
  python flashscore_pipeline.py --days-ago 2     # two days ago
  python flashscore_pipeline.py --date 2026-07-13
  python flashscore_pipeline.py --link-only      # link without pulling stats
  python flashscore_pipeline.py --stats-only     # pull stats for already-linked
  python flashscore_pipeline.py --force-stats    # re-fetch stats already stored
"""

import os
import re
import time
import sqlite3
import logging
import argparse
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher

from curl_cffi import requests


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "sportybet.db")

FS_NINJA   = "https://global.flashscore.ninja/44/x/feed"
FS_HEADERS = {
    "Referer":            "https://www.flashscore.com.ng/",
    "User-Agent":         ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/150.0.0.0 Safari/537.36"),
    "sec-ch-ua":          '"Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
    "x-fsign":            "SW9D1eZo",
}

MATCH_THRESHOLD = 0.70   # minimum combined team-name similarity to confirm a link
KICKOFF_WINDOW  = 7200   # ±seconds allowed between kickoff times (2 hours)
REQUEST_DELAY   = 1.5    # seconds between per-match stat API calls


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(BASE_DIR, "flashscore.log"),
                                encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

CREATE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS match_links (
        event_id        TEXT PRIMARY KEY,
        fs_match_id     TEXT NOT NULL,
        match_date      TEXT,
        home_team_sb    TEXT,
        home_team_fs    TEXT,
        away_team_sb    TEXT,
        away_team_fs    TEXT,
        match_score     REAL,
        -- Scorelines populated from FlashScore list feed during linking
        home_score_ft   INTEGER,
        away_score_ft   INTEGER,
        home_score_ht   INTEGER,
        away_score_ht   INTEGER,
        home_score_et   INTEGER,      -- NULL if no extra time
        away_score_et   INTEGER,
        home_pens       INTEGER,      -- NULL if no penalty shootout
        away_pens       INTEGER,
        red_cards_home  INTEGER DEFAULT 0,
        red_cards_away  INTEGER DEFAULT 0,
        match_status    TEXT,         -- 'FT' | 'AET' | 'AP'
        stats_fetched   INTEGER DEFAULT 0,
        linked_at       TEXT,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    CREATE TABLE IF NOT EXISTS flashscore_stats (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        fs_match_id TEXT    NOT NULL,
        event_id    TEXT,
        section     TEXT,
        stat_name   TEXT,
        home_value  TEXT,
        away_value  TEXT,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    CREATE TABLE IF NOT EXISTS flashscore_incidents (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        fs_match_id    TEXT    NOT NULL,
        event_id       TEXT,
        period         TEXT,
        minute         TEXT,
        incident_type  TEXT,
        competitor     TEXT,
        player_name    TEXT,
        assist_name    TEXT,
        description    TEXT,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    CREATE INDEX IF NOT EXISTS idx_match_links_fs_id
        ON match_links(fs_match_id);
    CREATE INDEX IF NOT EXISTS idx_match_links_date
        ON match_links(match_date);
    CREATE INDEX IF NOT EXISTS idx_fs_stats_match
        ON flashscore_stats(fs_match_id);
    CREATE INDEX IF NOT EXISTS idx_fs_incidents_match
        ON flashscore_incidents(fs_match_id);
"""

# Columns added in this version — safe to add to existing databases
NEW_COLUMNS = [
    ("home_score_ft",  "INTEGER"),
    ("away_score_ft",  "INTEGER"),
    ("home_score_ht",  "INTEGER"),
    ("away_score_ht",  "INTEGER"),
    ("home_score_et",  "INTEGER"),
    ("away_score_et",  "INTEGER"),
    ("home_pens",      "INTEGER"),
    ("away_pens",      "INTEGER"),
    ("red_cards_home", "INTEGER DEFAULT 0"),
    ("red_cards_away", "INTEGER DEFAULT 0"),
    ("match_status",   "TEXT"),
]


def init_db(conn):
    """Create tables and add any new columns to existing tables."""
    conn.executescript(CREATE_SCHEMA)

    # Add new columns to match_links if they do not exist yet
    # (handles databases created before this version)
    existing_cols = {
        row[1] for row in
        conn.execute("PRAGMA table_info(match_links)").fetchall()
    }
    for col_name, col_def in NEW_COLUMNS:
        if col_name not in existing_cols:
            conn.execute(
                f"ALTER TABLE match_links ADD COLUMN {col_name} {col_def}"
            )
            log.info(f"  Added column match_links.{col_name}")

    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — date offset
# ══════════════════════════════════════════════════════════════════════════════

def date_to_offset(target_date: date) -> int:
    """Convert a date to FlashScore's relative day offset from today."""
    return (target_date - date.today()).days


def offset_url(offset: int) -> str:
    return f"{FS_NINJA}/f_1_{offset}_1_en-ng_1"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — FlashScore delimited format parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_fs_text(text: str) -> list:
    records = []
    for raw in text.split("~"):
        if not raw.strip():
            continue
        fields = {}
        for part in raw.split("¬"):
            if "÷" in part:
                code, _, value = part.partition("÷")
                fields[code.strip()] = value.strip()
        if fields:
            records.append(fields)
    return records


def safe_int(value) -> int | None:
    """Convert a FlashScore field value to int, returning None if absent."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def resolve_status(r: dict) -> str:
    """
    Determine match status string from FlashScore fields.
    RPA/RPB present → penalty shootout (AP).
    AJ/AK present   → extra time (AET).
    Otherwise       → full time (FT).
    """
    if r.get("RPA") or r.get("RPB"):
        return "AP"
    if r.get("AJ") or r.get("AK"):
        return "AET"
    return "FT"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — team name normalisation
# ══════════════════════════════════════════════════════════════════════════════

def normalise(name: str) -> str:
    name = name.lower()
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\b(fc|sc|ac|bk|sk|fk|nk|sv|if|ik|afc|rfc)\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def combined_sim(home_sb, away_sb, home_fs, away_fs) -> float:
    return (similarity(home_sb, home_fs) + similarity(away_sb, away_fs)) / 2


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LINKER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fs_matches(day_offset: int) -> list:
    """
    Fetch and parse FlashScore's daily match list.
    Returns list of match dicts including all available scoreline fields.
    """
    url  = offset_url(day_offset)
    resp = requests.get(url, headers=FS_HEADERS, impersonate="chrome110",
                        timeout=15)
    resp.raise_for_status()

    records = parse_fs_text(resp.text)
    matches = []

    for r in records:
        if "AA" not in r:
            continue

        matches.append({
            "fs_id":       r["AA"],
            "home_team":   r.get("CX", ""),
            "away_team":   r.get("AF", ""),
            "kickoff_ts":  int(r["AD"]) if r.get("AD", "").isdigit() else 0,
            "ab_status":   r.get("AB", ""),
            # Scorelines — present only for finished matches
            "home_ft":     safe_int(r.get("AG")),
            "away_ft":     safe_int(r.get("AH")),
            "home_ht":     safe_int(r.get("AT")),
            "away_ht":     safe_int(r.get("AU")),
            "home_et":     safe_int(r.get("AJ")),   # None if no ET
            "away_et":     safe_int(r.get("AK")),
            "home_pens":   safe_int(r.get("RPA")),  # None if no pens
            "away_pens":   safe_int(r.get("RPB")),
            "red_home":    safe_int(r.get("GRA")) or 0,
            "red_away":    safe_int(r.get("GRB")) or 0,
            "status":      resolve_status(r),
        })

    return matches


def link_date(target_date: date, conn) -> tuple:
    """
    Link FlashScore matches to SportyBet fixtures for a given date.
    Stores match IDs AND all available scorelines.
    Returns (linked_count, already_linked_count).
    """
    date_str   = target_date.isoformat()
    day_offset = date_to_offset(target_date)

    log.info(f"  Fetching FlashScore list for {date_str} (offset {day_offset})...")
    fs_matches = fetch_fs_matches(day_offset)
    log.info(f"  FlashScore: {len(fs_matches)} matches")

    sb_fixtures = conn.execute("""
        SELECT event_id, home_team, away_team, kickoff_ms
        FROM   fixtures
        WHERE  DATE(kickoff_time) = ?
    """, (date_str,)).fetchall()

    log.info(f"  SportyBet fixtures for {date_str}: {len(sb_fixtures)}")

    used_fs_ids  = set()
    linked       = 0
    already_done = 0

    for row in sb_fixtures:
        event_id, home_sb, away_sb, kickoff_ms_sb = row
        kickoff_s_sb = (kickoff_ms_sb or 0) / 1000

        existing = conn.execute(
            "SELECT fs_match_id FROM match_links WHERE event_id = ?",
            (event_id,)
        ).fetchone()
        if existing:
            already_done += 1
            continue

        best_score = 0.0
        best_fs    = None

        for fs in fs_matches:
            if fs["fs_id"] in used_fs_ids:
                continue
            if kickoff_s_sb and fs["kickoff_ts"]:
                if abs(fs["kickoff_ts"] - kickoff_s_sb) > KICKOFF_WINDOW:
                    continue
            score = combined_sim(home_sb, away_sb,
                                 fs["home_team"], fs["away_team"])
            if score > best_score:
                best_score = score
                best_fs    = fs

        if best_fs and best_score >= MATCH_THRESHOLD:
            conn.execute("""
                INSERT OR REPLACE INTO match_links (
                    event_id, fs_match_id, match_date,
                    home_team_sb, home_team_fs,
                    away_team_sb, away_team_fs,
                    match_score,
                    home_score_ft, away_score_ft,
                    home_score_ht, away_score_ht,
                    home_score_et, away_score_et,
                    home_pens,     away_pens,
                    red_cards_home, red_cards_away,
                    match_status,
                    stats_fetched, linked_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)
            """, (
                event_id,       best_fs["fs_id"],  date_str,
                home_sb,        best_fs["home_team"],
                away_sb,        best_fs["away_team"],
                round(best_score, 4),
                best_fs["home_ft"],   best_fs["away_ft"],
                best_fs["home_ht"],   best_fs["away_ht"],
                best_fs["home_et"],   best_fs["away_et"],
                best_fs["home_pens"], best_fs["away_pens"],
                best_fs["red_home"],  best_fs["red_away"],
                best_fs["status"],
                datetime.now().isoformat(),
            ))
            used_fs_ids.add(best_fs["fs_id"])
            linked += 1

            # Build score display for log
            ft  = (f"{best_fs['home_ft']}-{best_fs['away_ft']}"
                   if best_fs["home_ft"] is not None else "?-?")
            ht  = (f"HT:{best_fs['home_ht']}-{best_fs['away_ht']}"
                   if best_fs["home_ht"] is not None else "")
            ext = (f" AET:{best_fs['home_et']}-{best_fs['away_et']}"
                   if best_fs["home_et"] is not None else "")
            pen = (f" AP:{best_fs['home_pens']}-{best_fs['away_pens']}"
                   if best_fs["home_pens"] is not None else "")

            log.info(
                f"    ✅ [{best_score:.2f}] {home_sb} vs {away_sb}"
                f"  →  {ft} {ht}{ext}{pen}  ({best_fs['fs_id']})"
            )
        else:
            sc = f"{best_score:.2f}" if best_fs else "no candidate"
            log.warning(f"    ✗  {home_sb} vs {away_sb}  (best score: {sc})")

    conn.commit()
    return linked, already_done


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STATS SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_raw(endpoint: str) -> str:
    url  = f"{FS_NINJA}/{endpoint}"
    resp = requests.get(url, headers=FS_HEADERS, impersonate="chrome110",
                        timeout=15)
    resp.raise_for_status()
    return resp.text


SECTION_NAMES = {"1": "match", "2": "1st_half", "3": "2nd_half"}

def parse_stats(text: str) -> list:
    stats   = []
    section = "match"
    for raw in text.split("~"):
        if not raw.strip():
            continue
        fields = {}
        for part in raw.split("¬"):
            if "÷" in part:
                code, _, value = part.partition("÷")
                fields[code.strip()] = value.strip()
        if "SE" in fields and "SH" not in fields:
            section = SECTION_NAMES.get(fields.get("SE", "1"), "match")
            continue
        if "SH" in fields and ("SJ" in fields or "SK" in fields):
            stats.append({
                "section":    section,
                "stat_name":  fields.get("SH", ""),
                "home_value": fields.get("SJ", ""),
                "away_value": fields.get("SK", ""),
            })
    return stats


INCIDENT_TYPE_MAP = {
    "goal":          "goal",
    "yellow":        "yellow_card",
    "red":           "red_card",
    "yellowred":     "red_card_2nd_yellow",
    "substitution":  "substitution",
    "sub":           "substitution",
    "var":           "var",
    "owngoal":       "own_goal",
    "penalty":       "penalty",
    "missedpenalty": "missed_penalty",
}

PERIOD_MAP = {
    "1": "1st_half",
    "2": "2nd_half",
    "3": "extra_time",
    "4": "penalties",
}

def parse_incidents(text: str) -> list:
    incidents = []
    for r in parse_fs_text(text):
        if "IK" not in r and "IT" not in r:
            continue
        raw_type = r.get("IT", "").lower()
        stoppage = r.get("IL", "")
        minute   = r.get("IK", "")
        incidents.append({
            "period":        PERIOD_MAP.get(r.get("IH", ""), r.get("IH", "")),
            "minute":        f"{minute}+{stoppage}" if stoppage else minute,
            "incident_type": INCIDENT_TYPE_MAP.get(raw_type, raw_type),
            "competitor":    r.get("IU", "").lower(),
            "player_name":   r.get("IV", ""),
            "assist_name":   r.get("IW", ""),
            "description":   r.get("IM", ""),
        })
    return incidents


def fetch_and_store_stats(conn, event_id: str, fs_id: str):
    """Fetch df_st_1_ and df_sui_1_ for one match and store results."""

    # Stats
    try:
        stats = parse_stats(fetch_raw(f"df_st_1_{fs_id}"))
        if stats:
            conn.execute(
                "DELETE FROM flashscore_stats WHERE fs_match_id = ?", (fs_id,)
            )
            conn.executemany("""
                INSERT INTO flashscore_stats
                    (fs_match_id, event_id, section, stat_name,
                     home_value, away_value)
                VALUES (?,?,?,?,?,?)
            """, [(fs_id, event_id, s["section"], s["stat_name"],
                   s["home_value"], s["away_value"]) for s in stats])
            log.info(f"    Stats    : {len(stats)} rows")
        else:
            log.info(f"    Stats    : empty")
    except Exception as exc:
        log.warning(f"    Stats fetch failed: {exc}")

    time.sleep(REQUEST_DELAY / 2)

    # Incidents
    try:
        incidents = parse_incidents(fetch_raw(f"df_sui_1_{fs_id}"))
        if incidents:
            conn.execute(
                "DELETE FROM flashscore_incidents WHERE fs_match_id = ?",
                (fs_id,)
            )
            conn.executemany("""
                INSERT INTO flashscore_incidents
                    (fs_match_id, event_id, period, minute,
                     incident_type, competitor, player_name,
                     assist_name, description)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, [(fs_id, event_id, i["period"], i["minute"],
                   i["incident_type"], i["competitor"], i["player_name"],
                   i["assist_name"], i["description"]) for i in incidents])
            log.info(f"    Incidents: {len(incidents)} rows")
        else:
            log.info(f"    Incidents: empty")
    except Exception as exc:
        log.warning(f"    Incidents fetch failed: {exc}")

    conn.execute(
        "UPDATE match_links SET stats_fetched = 1 WHERE fs_match_id = ?",
        (fs_id,)
    )
    conn.commit()


def scrape_stats_for_date(target_date: date, conn, force: bool = False):
    date_str = target_date.isoformat()
    where    = "match_date = ?" if force else \
               "match_date = ? AND stats_fetched = 0"

    pending = conn.execute(f"""
        SELECT event_id, fs_match_id, home_team_sb, away_team_sb
        FROM   match_links
        WHERE  {where}
    """, (date_str,)).fetchall()

    log.info(f"  Matches needing stats for {date_str}: {len(pending)}")

    for i, (event_id, fs_id, home, away) in enumerate(pending, 1):
        log.info(f"  [{i:>3}/{len(pending)}]  {home} vs {away}  ({fs_id})")
        fetch_and_store_stats(conn, event_id, fs_id)
        time.sleep(REQUEST_DELAY)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FlashScore Linker & Stats Scraper"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--date",     type=str,
                       help="Target date YYYY-MM-DD")
    group.add_argument("--days-ago", type=int, default=1,
                       help="Days ago (default: 1 = yesterday)")
    p.add_argument("--link-only",   action="store_true",
                   help="Only run the linker step, skip stats")
    p.add_argument("--stats-only",  action="store_true",
                   help="Only run stats for already-linked matches")
    p.add_argument("--force-stats", action="store_true",
                   help="Re-fetch stats even if already stored")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    setup_logging()
    args = parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today() - timedelta(days=args.days_ago)

    log.info("=" * 65)
    log.info(f"  FlashScore Pipeline  —  {target_date.isoformat()}")
    log.info(f"  Database : {DB_PATH}")
    log.info("=" * 65)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    if not args.stats_only:
        log.info("\n[1] LINKER...")
        linked, skipped = link_date(target_date, conn)
        log.info(f"    Newly linked : {linked}")
        log.info(f"    Already done : {skipped}")
    else:
        log.info("\n[1] LINKER — skipped (--stats-only)")

    if not args.link_only:
        log.info("\n[2] STATS...")
        scrape_stats_for_date(target_date, conn,
                              force=args.force_stats)
    else:
        log.info("\n[2] STATS — skipped (--link-only)")

    total_links = conn.execute(
        "SELECT COUNT(*) FROM match_links WHERE match_date = ?",
        (target_date.isoformat(),)
    ).fetchone()[0]

    stats_done = conn.execute(
        "SELECT COUNT(*) FROM match_links "
        "WHERE match_date = ? AND stats_fetched = 1",
        (target_date.isoformat(),)
    ).fetchone()[0]

    conn.close()

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Date             : {target_date.isoformat()}")
    log.info(f"  Matches linked   : {total_links}")
    log.info(f"  Stats fetched    : {stats_done}")
    log.info("=" * 65)


if __name__ == "__main__":
    run()


#!/usr/bin/env python3
"""
FlashScore Linker & Stats Scraper
===================================
Two steps, one script:

STEP 1 — LINKER
  Fetches FlashScore's daily match list for a given date, fuzzy-matches
  each FlashScore match against your fixtures table (by team name + kickoff
  proximity), and stores the confirmed FlashScore match ID in match_links.

STEP 2 — STATS SCRAPER
  For every row in match_links that has no stats yet, fetches:
    df_st_1_{fs_id}   → full match statistics (shots, corners, possession…)
    df_sui_1_{fs_id}  → match incidents (goals, cards, substitutions, VAR)
  Parses both and stores into flashscore_stats and flashscore_incidents.

DATABASE TABLES CREATED:
  match_links          — event_id ↔ fs_match_id bridge
  flashscore_stats     — stats per match per section (EAV shape)
  flashscore_incidents — one row per incident per match

USAGE:
  python flashscore_pipeline.py                  # yesterday (most common)
  python flashscore_pipeline.py --days-ago 2     # two days ago
  python flashscore_pipeline.py --date 2026-07-13
  python flashscore_pipeline.py --link-only      # link without pulling stats
  python flashscore_pipeline.py --stats-only     # pull stats for already-linked
"""

import os
import re
import sys
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

FS_NINJA     = "https://global.flashscore.ninja/44/x/feed"
FS_HEADERS   = {
    "Referer":            "https://www.flashscore.com.ng/",
    "User-Agent":         ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/150.0.0.0 Safari/537.36"),
    "sec-ch-ua":          '"Not;A=Brand";v="8", "Chromium";v="150", "Google Chrome";v="150"',
    "sec-ch-ua-mobile":   "?0",
    "sec-ch-ua-platform": '"Windows"',
    "x-fsign":            "SW9D1eZo",
}

MATCH_THRESHOLD  = 0.70   # minimum combined team-name similarity to confirm a link
KICKOFF_WINDOW   = 7200   # ± seconds allowed between kickoff times (2 hours)
REQUEST_DELAY    = 1.5    # seconds between per-match stat API calls


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

SCHEMA = """
    -- Bridge table: SportyBet/Sportradar event_id ↔ FlashScore match ID
    CREATE TABLE IF NOT EXISTS match_links (
        event_id        TEXT PRIMARY KEY,   -- sr:match:XXXXXXXX
        fs_match_id     TEXT NOT NULL,      -- FlashScore 8-char ID e.g. pU0PQ9nR
        match_date      TEXT,               -- YYYY-MM-DD
        home_team_sb    TEXT,
        home_team_fs    TEXT,
        away_team_sb    TEXT,
        away_team_fs    TEXT,
        match_score     REAL,               -- fuzzy match confidence 0-1
        stats_fetched   INTEGER DEFAULT 0,  -- 1 once stats are stored
        linked_at       TEXT,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    -- Match statistics — one row per stat per section per match
    -- EAV shape keeps schema clean regardless of how many stats FlashScore adds
    CREATE TABLE IF NOT EXISTS flashscore_stats (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        fs_match_id TEXT    NOT NULL,
        event_id    TEXT,
        section     TEXT,           -- 'match' | '1st_half' | '2nd_half'
        stat_name   TEXT,
        home_value  TEXT,
        away_value  TEXT,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    -- Match incidents — one row per event (goal, card, sub, VAR)
    CREATE TABLE IF NOT EXISTS flashscore_incidents (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        fs_match_id    TEXT    NOT NULL,
        event_id       TEXT,
        period         TEXT,       -- '1st_half' | '2nd_half' | 'extra_time'
        minute         TEXT,       -- e.g. '72' or '90+3'
        incident_type  TEXT,       -- 'goal' | 'yellow_card' | 'red_card' | 'substitution' | 'var'
        competitor     TEXT,       -- 'home' | 'away'
        player_name    TEXT,
        assist_name    TEXT,
        description    TEXT,
        FOREIGN KEY (event_id) REFERENCES fixtures(event_id)
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_match_links_fs_id
        ON match_links(fs_match_id);
    CREATE INDEX IF NOT EXISTS idx_match_links_date
        ON match_links(match_date);
    CREATE INDEX IF NOT EXISTS idx_fs_stats_match
        ON flashscore_stats(fs_match_id);
    CREATE INDEX IF NOT EXISTS idx_fs_incidents_match
        ON flashscore_incidents(fs_match_id);
"""

def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — date offset
# ══════════════════════════════════════════════════════════════════════════════

def date_to_offset(target_date: date) -> int:
    """
    Convert a date to FlashScore's relative day offset from today.
    Today = 0, yesterday = -1, two days ago = -2, etc.
    """
    return (target_date - date.today()).days


def offset_url(offset: int) -> str:
    """Build the FlashScore daily list feed URL for a given day offset."""
    return f"{FS_NINJA}/f_1_{offset}_1_en-ng_1"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — FlashScore delimited format parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_fs_text(text: str) -> list[dict]:
    """
    Parse FlashScore's ¬÷ delimited format into a list of field dicts.
    Each '~' separated block becomes one dict.
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — team name normalisation for fuzzy matching
# ══════════════════════════════════════════════════════════════════════════════

def normalise(name: str) -> str:
    """
    Strip common noise from team names before similarity comparison.
    Removes country suffixes like (Eng), (Ger), accents, punctuation.
    """
    name = name.lower()
    name = re.sub(r"\([^)]*\)", "", name)     # remove (Eng), (Ger) etc.
    name = re.sub(r"\b(fc|sc|ac|bk|sk|fk|nk|sv|if|ik|afc|rfc)\b", "", name)
    name = re.sub(r"[^a-z0-9\s]", " ", name)  # strip punctuation
    name = re.sub(r"\s+", " ", name).strip()
    return name


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalise(a), normalise(b)).ratio()


def combined_sim(home_sb, away_sb, home_fs, away_fs) -> float:
    """Average of home + away team similarity."""
    return (similarity(home_sb, home_fs) + similarity(away_sb, away_fs)) / 2


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LINKER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fs_matches(day_offset: int) -> list[dict]:
    """
    Fetch and parse FlashScore's daily match list for the given day offset.
    Returns a list of match dicts with keys:
        fs_id, home_team, away_team, kickoff_ts, status
    """
    url  = offset_url(day_offset)
    resp = requests.get(url, headers=FS_HEADERS, impersonate="chrome110", timeout=15)
    resp.raise_for_status()

    records = parse_fs_text(resp.text)

    matches = []
    for r in records:
        if "AA" not in r:       # AA = FlashScore match ID
            continue
        matches.append({
            "fs_id":      r["AA"],
            "home_team":  r.get("CX", ""),
            "away_team":  r.get("AF", ""),
            "kickoff_ts": int(r["AD"]) if r.get("AD", "").isdigit() else 0,
            "status":     r.get("AB", ""),   # 3 = finished
        })

    return matches


def link_date(target_date: date, conn) -> tuple[int, int]:
    """
    Link FlashScore matches to SportyBet fixtures for a given date.
    Returns (linked_count, already_linked_count).
    """
    date_str   = target_date.isoformat()
    day_offset = date_to_offset(target_date)

    log.info(f"  Fetching FlashScore list for {date_str} (offset {day_offset})...")
    fs_matches = fetch_fs_matches(day_offset)
    log.info(f"  FlashScore returned {len(fs_matches)} match records for {date_str}")

    # Pull SportyBet fixtures for this date
    sb_fixtures = conn.execute("""
        SELECT event_id, home_team, away_team, kickoff_ms
        FROM   fixtures
        WHERE  DATE(kickoff_time) = ?
    """, (date_str,)).fetchall()

    log.info(f"  SportyBet fixtures for {date_str}: {len(sb_fixtures)}")

    # Track which FS matches have already been claimed (prevent double-linking)
    used_fs_ids = set()

    linked        = 0
    already_done  = 0

    for row in sb_fixtures:
        event_id, home_sb, away_sb, kickoff_ms_sb = row
        kickoff_s_sb = (kickoff_ms_sb or 0) / 1000

        # Skip if already linked
        existing = conn.execute(
            "SELECT fs_match_id FROM match_links WHERE event_id = ?", (event_id,)
        ).fetchone()
        if existing:
            already_done += 1
            continue

        best_score = 0.0
        best_fs    = None

        for fs in fs_matches:
            if fs["fs_id"] in used_fs_ids:
                continue

            # Kickoff proximity guard
            if kickoff_s_sb and fs["kickoff_ts"]:
                diff = abs(fs["kickoff_ts"] - kickoff_s_sb)
                if diff > KICKOFF_WINDOW:
                    continue

            score = combined_sim(home_sb, away_sb, fs["home_team"], fs["away_team"])
            if score > best_score:
                best_score = score
                best_fs    = fs

        if best_fs and best_score >= MATCH_THRESHOLD:
            conn.execute("""
                INSERT OR REPLACE INTO match_links
                    (event_id, fs_match_id, match_date,
                     home_team_sb, home_team_fs,
                     away_team_sb, away_team_fs,
                     match_score, stats_fetched, linked_at)
                VALUES (?,?,?,?,?,?,?,?,0,?)
            """, (
                event_id, best_fs["fs_id"], date_str,
                home_sb,       best_fs["home_team"],
                away_sb,       best_fs["away_team"],
                round(best_score, 4),
                datetime.now().isoformat(),
            ))
            used_fs_ids.add(best_fs["fs_id"])
            linked += 1
            log.info(
                f"    ✅  [{best_score:.2f}]  {home_sb} vs {away_sb}"
                f"  →  {best_fs['home_team']} vs {best_fs['away_team']}"
                f"  ({best_fs['fs_id']})"
            )
        else:
            score_str = f"{best_score:.2f}" if best_fs else "no candidate"
            log.warning(f"    ✗   {home_sb} vs {away_sb}  (best score: {score_str})")

    conn.commit()
    return linked, already_done


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — STATS SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_raw(endpoint: str) -> str:
    url  = f"{FS_NINJA}/{endpoint}"
    resp = requests.get(url, headers=FS_HEADERS, impersonate="chrome110", timeout=15)
    resp.raise_for_status()
    return resp.text


# ── Stats parser ──────────────────────────────────────────────────────────────


def _normalise_section(se_value: str) -> str:
    v = se_value.lower()
    if "1st" in v:
        return "1st_half"
    if "2nd" in v:
        return "2nd_half"
    return "match"


def parse_stats(text: str) -> list[dict]:
    """
    Confirmed field codes from live data:
      SE = section label  ('Match' / '1st Half' / '2nd Half')
      SG = stat name      ('Ball possession', 'Total shots', ...)
      SH = home value     ('44%', '6', ...)
      SI = away value     ('56%', '10', ...)

    Previous bug: was looking for SH as stat name and SJ/SK as home/away --
    neither SJ nor SK exist in real data, so the condition was never true
    and stats always returned empty.
    """
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

        if "SE" in fields and "SG" not in fields:
            section = _normalise_section(fields["SE"])
            continue

        if "SG" in fields:
            stats.append({
                "section":    section,
                "stat_name":  fields["SG"],
                "home_value": fields.get("SH", ""),
                "away_value": fields.get("SI", ""),
            })

    return stats


# ── Incidents parser ──────────────────────────────────────────────────────────

INCIDENT_TYPE_MAP = {
    "goal":         "goal",
    "yellow":       "yellow_card",
    "red":          "red_card",
    "yellowred":    "red_card_2nd_yellow",
    "substitution": "substitution",
    "sub":          "substitution",
    "var":          "var",
    "owngoal":      "own_goal",
    "penalty":      "penalty",
    "missedpenalty":"missed_penalty",
}

PERIOD_MAP = {
    "1": "1st_half",
    "2": "2nd_half",
    "3": "extra_time",
    "4": "penalties",
}

def parse_incidents(text: str) -> list[dict]:
    """
    Parse df_sui_1_ response into a list of incident dicts.
    """
    incidents = []
    records   = parse_fs_text(text)

    for r in records:
        # Incidents have IK (minute) or IT (type)
        if "IK" not in r and "IT" not in r:
            continue

        raw_type  = r.get("IT", "").lower()
        inc_type  = INCIDENT_TYPE_MAP.get(raw_type, raw_type)
        period    = PERIOD_MAP.get(r.get("IH", ""), r.get("IH", ""))
        minute    = r.get("IK", "")
        stoppage  = r.get("IL", "")
        full_min  = f"{minute}+{stoppage}" if stoppage else minute

        competitor = r.get("IU", "").lower()   # home / away
        player     = r.get("IV", "")           # player name
        assist     = r.get("IW", "")           # assist / player coming on
        desc       = r.get("IM", "")           # description

        incidents.append({
            "period":        period,
            "minute":        full_min,
            "incident_type": inc_type,
            "competitor":    competitor,
            "player_name":   player,
            "assist_name":   assist,
            "description":   desc,
        })

    return incidents


# ── Main stats fetcher ────────────────────────────────────────────────────────

def fetch_and_store_stats(conn, event_id: str, fs_id: str):
    """
    Fetch df_st_1_ and df_sui_1_ for one match and store results.
    """
    # ── Stats ──────────────────────────────────────────────────────────────────
    try:
        stats_text = fetch_raw(f"df_st_1_{fs_id}")
        stats      = parse_stats(stats_text)

        if stats:
            # Clear any previous rows for this match
            conn.execute(
                "DELETE FROM flashscore_stats WHERE fs_match_id = ?", (fs_id,)
            )
            conn.executemany("""
                INSERT INTO flashscore_stats
                    (fs_match_id, event_id, section, stat_name, home_value, away_value)
                VALUES (?,?,?,?,?,?)
            """, [
                (fs_id, event_id, s["section"], s["stat_name"],
                 s["home_value"], s["away_value"])
                for s in stats
            ])
            log.info(f"    Stats   : {len(stats)} rows stored")
        else:
            log.info(f"    Stats   : empty response")

    except Exception as exc:
        log.warning(f"    Stats fetch failed: {exc}")

    time.sleep(REQUEST_DELAY / 2)

    # ── Incidents ──────────────────────────────────────────────────────────────
    try:
        inc_text  = fetch_raw(f"df_sui_1_{fs_id}")
        incidents = parse_incidents(inc_text)

        if incidents:
            conn.execute(
                "DELETE FROM flashscore_incidents WHERE fs_match_id = ?", (fs_id,)
            )
            conn.executemany("""
                INSERT INTO flashscore_incidents
                    (fs_match_id, event_id, period, minute,
                     incident_type, competitor, player_name, assist_name, description)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, [
                (fs_id, event_id, i["period"], i["minute"],
                 i["incident_type"], i["competitor"], i["player_name"],
                 i["assist_name"], i["description"])
                for i in incidents
            ])
            log.info(f"    Incidents: {len(incidents)} rows stored")
        else:
            log.info(f"    Incidents: empty response")

    except Exception as exc:
        log.warning(f"    Incidents fetch failed: {exc}")

    # Mark as done
    conn.execute(
        "UPDATE match_links SET stats_fetched = 1 WHERE fs_match_id = ?",
        (fs_id,)
    )
    conn.commit()


def scrape_stats_for_date(target_date: date, conn, force: bool = False):
    """
    Fetch stats for all linked matches on a given date that haven't
    been fetched yet (or all of them if force=True).
    """
    date_str = target_date.isoformat()
    where    = "match_date = ?" if force else "match_date = ? AND stats_fetched = 0"

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
    p = argparse.ArgumentParser(description="FlashScore Linker & Stats Scraper")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--date",      type=str, help="Target date YYYY-MM-DD")
    group.add_argument("--days-ago",  type=int, default=1,
                       help="Days ago (default: 1 = yesterday)")
    p.add_argument("--link-only",  action="store_true",
                   help="Only run the linker step, skip stats")
    p.add_argument("--stats-only", action="store_true",
                   help="Only run the stats step for already-linked matches")
    p.add_argument("--force-stats", action="store_true",
                   help="Re-fetch stats even if already stored")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run():
    setup_logging()
    args = parse_args()

    # Resolve target date
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

    # ── STEP 1: Link ───────────────────────────────────────────────────────────
    if not args.stats_only:
        log.info("\n[1] LINKER — matching FlashScore matches to fixtures...")
        linked, skipped = link_date(target_date, conn)
        log.info(f"    Newly linked : {linked}")
        log.info(f"    Already done : {skipped}")
    else:
        log.info("\n[1] LINKER — skipped (--stats-only)")

    # ── STEP 2: Stats ──────────────────────────────────────────────────────────
    if not args.link_only:
        log.info("\n[2] STATS — fetching match statistics and incidents...")
        scrape_stats_for_date(target_date, conn, force=args.force_stats)
    else:
        log.info("\n[2] STATS — skipped (--link-only)")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_links  = conn.execute(
        "SELECT COUNT(*) FROM match_links WHERE match_date = ?",
        (target_date.isoformat(),)
    ).fetchone()[0]

    stats_done = conn.execute(
        "SELECT COUNT(*) FROM match_links WHERE match_date = ? AND stats_fetched = 1",
        (target_date.isoformat(),)
    ).fetchone()[0]

    total_stat_rows = conn.execute(
        "SELECT COUNT(*) FROM flashscore_stats fs "
        "JOIN match_links ml ON ml.fs_match_id = fs.fs_match_id "
        "WHERE ml.match_date = ?",
        (target_date.isoformat(),)
    ).fetchone()[0]

    total_inc_rows = conn.execute(
        "SELECT COUNT(*) FROM flashscore_incidents fi "
        "JOIN match_links ml ON ml.fs_match_id = fi.fs_match_id "
        "WHERE ml.match_date = ?",
        (target_date.isoformat(),)
    ).fetchone()[0]

    conn.close()

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Date               : {target_date.isoformat()}")
    log.info(f"  Matches linked     : {total_links}")
    log.info(f"  Stats fetched      : {stats_done}")
    log.info(f"  flashscore_stats   : {total_stat_rows:,} rows")
    log.info(f"  flashscore_incidents: {total_inc_rows:,} rows")
    log.info("=" * 65)


if __name__ == "__main__":
    run()

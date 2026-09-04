# Football Market Data Pipeline

A personal football betting research system that scrapes pre-match odds from SportyBet, collects full match results and statistics from sporty.com and FlashScore, links them together on a shared Betradar match ID, and runs a pick engine against the accumulated database.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
│                                                                  │
│   SportyBet (Nigeria)          sporty.com          FlashScore    │
│   All prematch markets         Results + scores    Full stats    │
│   ~400 markets per fixture     Same Betradar IDs   + incidents   │
└────────────┬───────────────────────┬──────────────────┬──────────┘
             │                       │                  │
             ▼                       ▼                  ▼
┌────────────────────┐  ┌────────────────────┐  ┌─────────────────┐
│ fixture_data_      │  │ results_scraper.py  │  │ flashscore_     │
│ scraper.py         │  │                     │  │ pipeline.py     │
│                    │  │ Pulls HT / FT / AET │  │                 │
│ Scrapes fixtures   │  │ AP scores using the │  │ Step 1: Fuzzy-  │
│ + ALL market odds  │  │ same event_id from  │  │ matches FS      │
│ for the next 24h   │  │ SportyBet — no      │  │ matches to your │
│ Stores into SQLite │  │ linking needed      │  │ fixtures table  │
│                    │  │                     │  │                 │
└────────┬───────────┘  └──────────┬──────────┘  │ Step 2: Pulls   │
         │                         │              │ 30+ stats per   │
         ▼                         ▼              │ match + all     │
┌───────────────────────────────────────────────┐ │ incidents       │
│                   sportybet.db                │ └────────┬────────┘
│                                               │          │
│  fixtures        — one row per real match     │          │
│  odds            — all active outcomes/odds   │◄─────────┘
│  results         — scores from sporty.com     │
│  match_links     — event_id ↔ FlashScore ID   │
│  flashscore_stats     — stats EAV table       │
│  flashscore_incidents — goals, cards, subs    │
│  scrape_log      — run history                │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  daily_picks.py │
              │                 │
              │  Reads odds +   │
              │  probabilities  │
              │  Ranks picks by │
              │  market window  │
              │  + EV           │
              └─────────────────┘
```

---

## Files

| File | Purpose | Run order |
|---|---|---|
| `fixture_data_scraper.py` | Scrapes SportyBet fixtures + all market odds into the database | 1 — run daily, morning |
| `results_scraper.py` | Pulls HT/FT/AET/AP scores from sporty.com for past-kickoff fixtures | 2 — run after matches finish |
| `flashscore_pipeline.py` | Links fixtures to FlashScore matches, pulls full stats + incidents | 3 — run after results are in |
| `daily_picks.py` | Generates ranked pick list from today's odds database | Any time after step 1 |
| `picks.txt` | Personal betting history export — feeds the pick engine's win filter | Static reference file |

---

## Database Schema

### `fixtures`
One row per real football fixture. Populated by `fixture_data_scraper.py`.

| Column | Type | Description |
|---|---|---|
| `event_id` | TEXT PK | Betradar match ID — `sr:match:XXXXXXXX` |
| `game_id` | TEXT | SportyBet internal ID |
| `tournament_id` | TEXT | Betradar tournament ID |
| `tournament_name` | TEXT | e.g. `Premier League` |
| `category_name` | TEXT | e.g. `England` |
| `home_team` | TEXT | |
| `away_team` | TEXT | |
| `kickoff_time` | TEXT | `YYYY-MM-DD HH:MM` |
| `kickoff_ms` | INTEGER | Unix milliseconds — use for sorting |
| `scraped_at` | TEXT | When this fixture was first captured |

### `odds`
One row per active outcome per market per fixture. Never overwritten — grows permanently.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | |
| `event_id` | TEXT FK | Links to `fixtures` |
| `market_group` | TEXT | `Main`, `Goals`, `Half`, `Combo`, `Minutes`, etc. |
| `market_id` | TEXT | Betradar market type ID |
| `market_name` | TEXT | e.g. `1X2`, `Over/Under`, `Handicap` |
| `specifier` | TEXT | e.g. `total=2.5`, `hcp=1:0` |
| `outcome_desc` | TEXT | e.g. `Home`, `Over 2.5` |
| `odds` | REAL | Decimal odds |
| `probability` | REAL | Betradar implied probability (0–1) |

### `results`
Scores from sporty.com. Keyed directly on `event_id` — no linking required.

### `match_links`
Bridge table between your Betradar IDs and FlashScore's proprietary IDs.

| Column | Type | Description |
|---|---|---|
| `event_id` | TEXT PK | From `fixtures` |
| `fs_match_id` | TEXT | FlashScore 8-character ID |
| `match_date` | TEXT | `YYYY-MM-DD` |
| `home_team_sb` | TEXT | Team name from SportyBet |
| `home_team_fs` | TEXT | Team name from FlashScore |
| `match_score` | REAL | Fuzzy match confidence (0–1) |
| `stats_fetched` | INTEGER | 0 / 1 flag |

### `flashscore_stats`
EAV shape — one row per stat per section per match. Keeps schema clean regardless of how many stat categories FlashScore returns.

| Column | Type | Description |
|---|---|---|
| `fs_match_id` | TEXT | FlashScore ID |
| `event_id` | TEXT | Betradar ID |
| `section` | TEXT | `match`, `1st_half`, `2nd_half` |
| `stat_name` | TEXT | e.g. `Expected Goals`, `Corners`, `Shots On Target` |
| `home_value` | TEXT | |
| `away_value` | TEXT | |

### `flashscore_incidents`
One row per match event.

| Column | Type | Description |
|---|---|---|
| `fs_match_id` | TEXT | |
| `event_id` | TEXT | |
| `period` | TEXT | `1st_half`, `2nd_half`, `extra_time` |
| `minute` | TEXT | e.g. `72` or `90+3` |
| `incident_type` | TEXT | `goal`, `yellow_card`, `red_card`, `substitution`, `var` |
| `competitor` | TEXT | `home` or `away` |
| `player_name` | TEXT | |
| `assist_name` | TEXT | Player coming on (substitutions) or assist (goals) |

### `scrape_log`
One row per scraper execution. Bookkeeping only.

---

## Setup

```bash
pip install requests curl_cffi playwright
playwright install chromium
```

No API keys required. All data is fetched from publicly accessible endpoints
using the same internal API calls that each site's own frontend makes.

---

## Usage

### 1. Scrape today's fixtures and odds

```bash
python fixture_data_scraper.py
```

Fetches the next 24 hours of football fixtures from SportyBet Nigeria.
Safe to re-run — fixtures already in the database are skipped automatically.
Filters applied before anything touches the database:
- SRL (Simulated Reality League) fixtures — blocked entirely
- Players market group — blocked entirely
- Inactive outcomes — blocked entirely

### 2. Pull results after matches finish

```bash
python results_scraper.py
```

Queries your own `fixtures` table for past-kickoff matches with no result yet,
then fetches scores from sporty.com. The `event_id` is shared between
SportyBet and sporty.com (both use Betradar) so no fuzzy matching is needed.

### 3. Link to FlashScore and pull full stats

```bash
python flashscore_pipeline.py                # yesterday (default)
python flashscore_pipeline.py --days-ago 2  # two days ago
python flashscore_pipeline.py --date 2026-07-13
python flashscore_pipeline.py --link-only   # link without pulling stats
python flashscore_pipeline.py --stats-only  # stats for already-linked matches
```

Step 1 fuzzy-matches FlashScore's match list against your `fixtures` table
by team name + kickoff proximity and stores the FlashScore match ID.
Step 2 fetches 30+ statistics per match (xG, possession, shots, corners,
fouls, passes, tackles, etc.) split by full match / 1st half / 2nd half,
plus every incident (goals, cards, substitutions, VAR decisions).

### 4. Generate picks

```bash
python daily_picks.py
```

Reads today's odds from the database and produces a ranked pick list
based on market windows and expected value calculated from Betradar's
own probability field.

---

## Data Sources

| Source | What it provides | Link method |
|---|---|---|
| **SportyBet Nigeria** | Fixtures + all prematch markets and odds | Native — `event_id` is primary key |
| **sporty.com** | Match results (HT / FT / AET / AP scores) | Direct — shares Betradar `event_id` |
| **FlashScore** | Full match stats + all match incidents | Fuzzy match — separate ID system |

SportyBet and sporty.com both use **Betradar / SportRadar** as their data
provider, which is why the `sr:match:XXXXXXXX` IDs match directly between
them with no fuzzy matching needed.

FlashScore uses its own proprietary 8-character IDs (e.g. `pU0PQ9nR`) so
the `match_links` table is needed as a bridge.

---

## Why this exists

The goal is to build a growing personal database of pre-match odds against
actual outcomes, enabling calibration analysis — measuring whether implied
probabilities from Betradar's model match real-world outcome frequencies over
time, and identifying markets where systematic mispricing can be exploited for
better-informed predictions.

---

## Notes

- **No data is ever overwritten.** Every scrape run adds new rows permanently.
  The database is designed to grow over time — the longer it runs, the more
  useful the analysis.
- **SRL fixtures are filtered at scrape time**, not query time. Only real
  football ever enters the database.
- The `sportybet.db` file and all `*.log` files should not be committed to
  version control. Add them to `.gitignore`.
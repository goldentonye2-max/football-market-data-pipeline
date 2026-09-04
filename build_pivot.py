"""
build_pivot.py
Builds one wide row per fixture combining:
  - pre-match implied probabilities (from odds table)
  - final stats (from flashscore_stats)
  - final result (from results table)

Output: pivot.csv — the foundation for all analysis.
"""

import sqlite3
import csv
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sportybet.db")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pivot.csv")

conn = sqlite3.connect(DB_PATH)

# ── Step 1: pull key pre-match probabilities ──────────────────────────────────
# We identify each market by (market_id, specifier, outcome_desc)
# confirmed from your real data in earlier sessions.

PROB_QUERIES = {
    "prob_home":     ("1",  "",            "Home"),
    "prob_draw":     ("1",  "",            "Draw"),
    "prob_away":     ("1",  "",            "Away"),
    "prob_ou15_ov":  ("18", "total=1.5",   "Over 1.5"),
    "prob_ou25_ov":  ("18", "total=2.5",   "Over 2.5"),
    "prob_ou35_ov":  ("18", "total=3.5",   "Over 3.5"),
    "prob_btts_yes": ("29", "",            "Yes"),
    "prob_dnb_home": ("11", "",            "Home"),
    "prob_dnb_away": ("11", "",            "Away"),
}

# Build a dict: event_id -> {prob_home: x, prob_draw: x, ...}
probs = {}
for col_name, (mid, spec, outcome) in PROB_QUERIES.items():
    rows = conn.execute("""
        SELECT event_id, probability
        FROM   odds
        WHERE  market_id   = ?
          AND  COALESCE(specifier, '') = ?
          AND  outcome_desc = ?
    """, (mid, spec, outcome)).fetchall()
    for event_id, prob in rows:
        probs.setdefault(event_id, {})[col_name] = prob

# ── Step 2: pull final stats per fixture (match section only) ─────────────────
# Convert percentage strings like "44%" → 44.0, leave numbers as-is.

def to_float(val):
    if val is None:
        return None
    v = str(val).strip().replace("%", "")
    # Handle "84% (395/472)" style — take the numeric prefix only
    v = v.split()[0].replace("%", "")
    try:
        return float(v)
    except ValueError:
        return None

STAT_COLS = [
    "Ball possession",
    "Total shots",
    "Shots on target",
    "Shots off target",
    "Corner kicks",
    "Yellow cards",
    "Red cards",
    "Fouls",
    "Free kicks",
    "Throw ins",
    "Offsides",
    "Big chances",
    "Expected goals (xG)",
    "Blocked shots",
]

stats = {}  # event_id -> {col: {home: x, away: x}}
stat_rows = conn.execute("""
    SELECT ml.event_id, fs.stat_name, fs.home_value, fs.away_value
    FROM   flashscore_stats fs
    JOIN   match_links ml ON ml.fs_match_id = fs.fs_match_id
    WHERE  fs.section = 'match'
      AND  fs.stat_name IN ({})
""".format(",".join("?" * len(STAT_COLS))), STAT_COLS).fetchall()

for event_id, stat_name, home_val, away_val in stat_rows:
    stats.setdefault(event_id, {})[stat_name] = {
        "home": to_float(home_val),
        "away": to_float(away_val),
    }

# ── Step 3: pull results ──────────────────────────────────────────────────────
results = {}
for event_id, ft_home, ft_away, status in conn.execute("""
    SELECT event_id, ft_home, ft_away, status FROM results
""").fetchall():
    if ft_home is not None and ft_away is not None:
        if ft_home > ft_away:
            outcome = "H"
        elif ft_home == ft_away:
            outcome = "D"
        else:
            outcome = "A"
        results[event_id] = {
            "ft_home": ft_home, "ft_away": ft_away,
            "total_goals": ft_home + ft_away,
            "result": outcome,
        }

# ── Step 4: pull fixture metadata ─────────────────────────────────────────────
fixtures = {}
for row in conn.execute("""
    SELECT event_id, home_team, away_team, tournament_name, category_name, kickoff_time
    FROM fixtures
""").fetchall():
    fixtures[row[0]] = {
        "home_team": row[1], "away_team": row[2],
        "tournament": row[3], "category": row[4],
        "kickoff": row[5],
    }

# ── Step 5: assemble pivot rows ───────────────────────────────────────────────
def get_stat(event_id, stat_name, side):
    return stats.get(event_id, {}).get(stat_name, {}).get(side)

def get_stat_total(event_id, stat_name):
    h = get_stat(event_id, stat_name, "home")
    a = get_stat(event_id, stat_name, "away")
    if h is not None and a is not None:
        return h + a
    return None

pivot_rows = []
for event_id in probs:
    if event_id not in results:
        continue
    if event_id not in stats:
        continue

    p = probs[event_id]
    r = results[event_id]
    f = fixtures.get(event_id, {})
    s = stats.get(event_id, {})

    # Symmetry signal: how evenly matched are the teams?
    dnb_h = p.get("prob_dnb_home")
    dnb_a = p.get("prob_dnb_away")
    symmetry = 1 - abs(dnb_h - dnb_a) if (dnb_h and dnb_a) else None

    row = {
        "event_id":         event_id,
        "home_team":        f.get("home_team", ""),
        "away_team":        f.get("away_team", ""),
        "tournament":       f.get("tournament", ""),
        "category":         f.get("category", ""),
        "kickoff":          f.get("kickoff", ""),

        # Pre-match probabilities
        **{k: p.get(k) for k in PROB_QUERIES},
        "symmetry":         symmetry,

        # Result
        "ft_home":          r["ft_home"],
        "ft_away":          r["ft_away"],
        "total_goals":      r["total_goals"],
        "result":           r["result"],

        # Stats — totals
        "corners_total":    get_stat_total(event_id, "Corner kicks"),
        "corners_home":     get_stat(event_id, "Corner kicks", "home"),
        "corners_away":     get_stat(event_id, "Corner kicks", "away"),
        "shots_total":      get_stat_total(event_id, "Total shots"),
        "shots_ot_total":   get_stat_total(event_id, "Shots on target"),
        "shots_off_total":  get_stat_total(event_id, "Shots off target"),
        "possession_home":  get_stat(event_id, "Ball possession", "home"),
        "possession_away":  get_stat(event_id, "Ball possession", "away"),
        "yellow_total":     get_stat_total(event_id, "Yellow cards"),
        "red_total":        get_stat_total(event_id, "Red cards"),
        "fouls_total":      get_stat_total(event_id, "Fouls"),
        "throw_ins_total":  get_stat_total(event_id, "Throw ins"),
        "offsides_total":   get_stat_total(event_id, "Offsides"),
        "big_chances_total":get_stat_total(event_id, "Big chances"),
        "xg_total":         get_stat_total(event_id, "Expected goals (xG)"),
        "xg_home":          get_stat(event_id, "Expected goals (xG)", "home"),
        "xg_away":          get_stat(event_id, "Expected goals (xG)", "away"),
    }

    pivot_rows.append(row)

conn.close()

# ── Step 6: write CSV ─────────────────────────────────────────────────────────
if pivot_rows:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pivot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pivot_rows)

print(f"Pivot table: {len(pivot_rows)} rows → {OUT_CSV}")
print(f"\nSample coverage (non-null counts across {len(pivot_rows)} rows):")
sample_cols = [
    "prob_home", "prob_ou25_ov", "prob_btts_yes",
    "corners_total", "shots_total", "possession_home",
    "yellow_total", "fouls_total", "xg_total",
]
for col in sample_cols:
    count = sum(1 for r in pivot_rows if r.get(col) is not None)
    pct   = count / len(pivot_rows) * 100
    print(f"  {col:<22} {count:>4} / {len(pivot_rows)}  ({pct:.0f}%)")
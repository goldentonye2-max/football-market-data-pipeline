"""
=============================================================
  MASTER TRAINING TABLE BUILDER
  Reads sportybet.db → outputs one flat ML-ready CSV
=============================================================
  INPUTS  (from DB):
    - fixtures         → match identity & kickoff info
    - odds             → all pre-match market odds (true prob + displayed odds)
    - results          → post-match scorelines
    - flashscore_stats → post-match stats (corners, possession, shots...)
    - flashscore_incidents → goals, cards, minute-by-minute events
    - match_links      → SportyBet ↔ FlashScore ID bridge

  OUTPUTS:
    - master_training_table.csv
        Each ROW = one unique fixture
        Columns split into:
          true_prob__*     → bookmaker backend probabilities (features)
          implied_prob__*  → frontend odds converted to probability (features)
          vig_delta__*     → implied - true (bookie margin signal, features)
          cross__*         → cross-market engineered features
          home_stat__*     → post-match home team stats (for analysis/verification)
          away_stat__*     → post-match away team stats (for analysis/verification)
          target__*        → what actually happened (ML targets)
=============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import os

# ── CONFIG ──────────────────────────────────────────────────
DB_PATH     = "sportybet.db"   # adjust path if needed
OUTPUT_CSV  = "master_training_table.csv"
# ────────────────────────────────────────────────────────────


def clean_col_name(text):
    """Turn any string into a safe, consistent column name."""
    if pd.isna(text) or text is None:
        return "none"
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace("'", "")
        .replace(">", "over")
        .replace("<", "under")
    )


def clean_numeric(val):
    """Parse stat values safely (handles percentages, fractions, etc.)."""
    if val is None or val == "":
        return np.nan
    val = str(val).replace("%", "").strip()
    # Handle "X:Y" format (e.g. possession "55:45" → ignore, use home_val directly)
    if ":" in val:
        try:
            return float(val.split(":")[0])
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan


def build_master_table(db_path=DB_PATH):
    print("\n" + "="*60)
    print("  MASTER TRAINING TABLE BUILDER")
    print("="*60)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}")

    conn = sqlite3.connect(db_path)

    # ══════════════════════════════════════════════════════════
    # STEP 1 — Load base fixtures
    # ══════════════════════════════════════════════════════════
    print("\n[1/8] Loading fixtures...")
    fixtures = pd.read_sql_query("""
        SELECT
            f.event_id,
            f.tournament_name,
            f.category_name,
            f.home_team,
            f.away_team,
            f.kickoff_time,
            ml.match_date,
            ml.fs_match_id,
            ml.red_cards_home,
            ml.red_cards_away
        FROM fixtures f
        LEFT JOIN match_links ml ON f.event_id = ml.event_id
    """, conn)
    print(f"    → {len(fixtures)} fixtures loaded")

    # ══════════════════════════════════════════════════════════
    # STEP 2 — Load & pivot odds table (long → wide)
    # ══════════════════════════════════════════════════════════
    print("\n[2/8] Loading and pivoting odds...")
    odds_raw = pd.read_sql_query("""
        SELECT event_id, market_name, specifier, outcome_desc, odds, probability
        FROM odds
        WHERE odds IS NOT NULL AND probability IS NOT NULL
    """, conn)

    n_fixtures = fixtures["event_id"].nunique()

    # Build a unique column key per market/specifier/outcome combo
    odds_raw["col_key"] = (
        odds_raw["market_name"].apply(clean_col_name) + "__" +
        odds_raw["specifier"].fillna("").apply(clean_col_name) + "__" +
        odds_raw["outcome_desc"].apply(clean_col_name)
    )

    # Remove duplicate event_id + col_key (keep first occurrence)
    odds_raw = odds_raw.drop_duplicates(subset=["event_id", "col_key"])

    # ── MEMORY FIX: drop col_keys present in < MIN_COL_COVERAGE of fixtures ──
    # This is the root cause of the 13.5 GB crash.
    # 172,669 unique col_keys × 10,466 fixtures × 3 pivot tables × float64 = OOM.
    # Keeping only columns present in ≥ 50% of fixtures reduces this to ~dozens.
    MIN_COL_COVERAGE = 0.50   # keep col_keys seen in ≥ 50% of fixtures
                               # lower to 0.30 if you want more columns (uses more RAM)

    col_key_counts   = odds_raw.groupby("col_key")["event_id"].nunique()
    col_key_coverage = col_key_counts / n_fixtures
    keep_col_keys    = col_key_coverage[col_key_coverage >= MIN_COL_COVERAGE].index

    n_before = odds_raw["col_key"].nunique()
    odds_raw  = odds_raw[odds_raw["col_key"].isin(keep_col_keys)]
    n_after   = odds_raw["col_key"].nunique()

    print(f"    → col_key coverage filter ({MIN_COL_COVERAGE*100:.0f}%): "
          f"{n_before:,} → {n_after:,} unique market columns kept")
    print(f"    → Kept markets:")
    for ck in sorted(keep_col_keys):
        pct = col_key_coverage[ck] * 100
        print(f"       {ck:<70} {pct:>5.1f}%")

    # Implied probability from displayed odds
    odds_raw["implied_prob"] = 1.0 / odds_raw["odds"]

    # Vig delta = implied - true (how much margin the bookie added)
    odds_raw["vig_delta"] = odds_raw["implied_prob"] - odds_raw["probability"]

    # Pivot: true probability (backend)
    pivot_true = odds_raw.pivot(
        index="event_id", columns="col_key", values="probability"
    )
    pivot_true.columns = ["true_prob__" + c for c in pivot_true.columns]

    # Pivot: implied probability (frontend)
    pivot_implied = odds_raw.pivot(
        index="event_id", columns="col_key", values="implied_prob"
    )
    pivot_implied.columns = ["implied_prob__" + c for c in pivot_implied.columns]

    # Pivot: vig delta
    pivot_vig = odds_raw.pivot(
        index="event_id", columns="col_key", values="vig_delta"
    )
    pivot_vig.columns = ["vig_delta__" + c for c in pivot_vig.columns]

    odds_wide = pd.concat([pivot_true, pivot_implied, pivot_vig], axis=1).reset_index()
    print(f"    → {len(odds_wide)} fixtures × {len(odds_wide.columns)-1} odds feature columns")

    # ══════════════════════════════════════════════════════════
    # STEP 3 — Engineer cross-market interaction features
    # ══════════════════════════════════════════════════════════
    print("\n[3/8] Engineering cross-market features...")

    def find_col(df, keywords, prefix="true_prob__"):
        """Find a column whose name contains ALL given keywords."""
        for col in df.columns:
            if col.startswith(prefix) and all(k.lower() in col.lower() for k in keywords):
                return col
        return None

    cross_features = pd.DataFrame(index=odds_wide.index)
    cross_features["event_id"] = odds_wide["event_id"].values

    # --- Home advantage gap (how much bookie favours home team) ---
    col_hw = find_col(odds_wide, ["1x2", "home"])
    col_aw = find_col(odds_wide, ["1x2", "away"])
    if col_hw and col_aw:
        cross_features["cross__home_advantage_gap"] = (
            odds_wide[col_hw].values - odds_wide[col_aw].values
        )
        print(f"    ✓ home_advantage_gap")

    # --- Draw attraction (how strongly the draw is priced) ---
    col_draw = find_col(odds_wide, ["1x2", "draw"])
    if col_draw:
        cross_features["cross__draw_prob"] = odds_wide[col_draw].values
        print(f"    ✓ draw_prob")

    # --- Over 2.5 goal probability ---
    col_over25 = find_col(odds_wide, ["2_5", "over"])
    if col_over25:
        cross_features["cross__over25_true"] = odds_wide[col_over25].values
        print(f"    ✓ over25_true ({col_over25})")

    # --- Corner market vig vs goal market vig (market confidence gap) ---
    col_corner_vig = find_col(odds_wide, ["corner", "over"], prefix="vig_delta__")
    col_goal_vig   = find_col(odds_wide, ["2_5", "over"],   prefix="vig_delta__")
    if col_corner_vig and col_goal_vig:
        cross_features["cross__corner_vs_goal_vig_gap"] = (
            odds_wide[col_corner_vig].values - odds_wide[col_goal_vig].values
        )
        print(f"    ✓ corner_vs_goal_vig_gap")

    # --- BTTS vs Over 2.5 alignment ---
    col_btts = find_col(odds_wide, ["btts", "yes"])
    if col_btts and col_over25:
        cross_features["cross__btts_vs_over25_gap"] = (
            odds_wide[col_btts].values - odds_wide[col_over25].values
        )
        print(f"    ✓ btts_vs_over25_gap")

    # --- Total implied overround per market group ---
    # (measures how much the bookie is shading the market overall)
    mkt_overround = (
        odds_raw.groupby(["event_id"])["implied_prob"]
        .sum()
        .reset_index()
        .rename(columns={"implied_prob": "cross__total_market_overround"})
    )
    cross_features = cross_features.merge(mkt_overround, on="event_id", how="left")
    print(f"    ✓ total_market_overround (sum of all implied probs)")

    print(f"    → {len(cross_features.columns)-1} cross-market features created")

    # ══════════════════════════════════════════════════════════
    # STEP 4 — Load & pivot flashscore stats (long → wide)
    # ══════════════════════════════════════════════════════════
    print("\n[4/8] Loading and pivoting FlashScore stats...")
    fs_stats_raw = pd.read_sql_query("""
        SELECT event_id, stat_name, home_value, away_value
        FROM flashscore_stats
        WHERE event_id IS NOT NULL
    """, conn)

    if not fs_stats_raw.empty:
        fs_stats_raw["stat_key"] = fs_stats_raw["stat_name"].apply(clean_col_name)
        fs_stats_raw["home_val_num"] = fs_stats_raw["home_value"].apply(clean_numeric)
        fs_stats_raw["away_val_num"] = fs_stats_raw["away_value"].apply(clean_numeric)

        # Remove duplicates
        fs_stats_raw = fs_stats_raw.drop_duplicates(subset=["event_id", "stat_key"])

        pivot_home_stats = fs_stats_raw.pivot(
            index="event_id", columns="stat_key", values="home_val_num"
        )
        pivot_home_stats.columns = ["home_stat__" + c for c in pivot_home_stats.columns]

        pivot_away_stats = fs_stats_raw.pivot(
            index="event_id", columns="stat_key", values="away_val_num"
        )
        pivot_away_stats.columns = ["away_stat__" + c for c in pivot_away_stats.columns]

        stats_wide = pd.concat([pivot_home_stats, pivot_away_stats], axis=1).reset_index()
        print(f"    → {len(stats_wide.columns)-1} stat columns")
    else:
        print("    ! No FlashScore stats found in DB yet")
        stats_wide = pd.DataFrame(columns=["event_id"])

    # ══════════════════════════════════════════════════════════
    # STEP 5 — Load results and build TARGET variables
    # ══════════════════════════════════════════════════════════
    print("\n[5/8] Engineering target variables from results...")
    results = pd.read_sql_query("""
        SELECT event_id, ht_home, ht_away, ft_home, ft_away,
               home_goals, away_goals, status
        FROM results
        WHERE ft_home IS NOT NULL AND ft_away IS NOT NULL
    """, conn)

    results["total_goals"] = results["ft_home"] + results["ft_away"]
    results["ht_goals"]    = results["ht_home"]  + results["ht_away"]

    # Goal line targets
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        col = f"target__over_{str(line).replace('.', '_')}_goals"
        results[col] = (results["total_goals"] > line).astype(int)

    # HT goal targets
    for line in [0.5, 1.5]:
        col = f"target__ht_over_{str(line).replace('.', '_')}_goals"
        results[col] = (results["ht_goals"] > line).astype(int)

    # Match result
    results["target__home_win"]  = (results["ft_home"] > results["ft_away"]).astype(int)
    results["target__draw"]      = (results["ft_home"] == results["ft_away"]).astype(int)
    results["target__away_win"]  = (results["ft_home"] < results["ft_away"]).astype(int)

    # HT result
    results["target__ht_home_win"] = (results["ht_home"] > results["ht_away"]).astype(int)
    results["target__ht_draw"]     = (results["ht_home"] == results["ht_away"]).astype(int)
    results["target__ht_away_win"] = (results["ht_home"] < results["ht_away"]).astype(int)

    # BTTS
    results["target__btts_yes"] = (
        (results["ft_home"] > 0) & (results["ft_away"] > 0)
    ).astype(int)
    results["target__btts_no"] = 1 - results["target__btts_yes"]

    # Clean sheet
    results["target__home_clean_sheet"] = (results["ft_away"] == 0).astype(int)
    results["target__away_clean_sheet"] = (results["ft_home"] == 0).astype(int)

    # Exact score targets (most common)
    for hs, as_ in [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(1,2),(2,2),(3,0),(3,1),(3,2)]:
        col = f"target__score_{hs}_{as_}"
        results[col] = ((results["ft_home"] == hs) & (results["ft_away"] == as_)).astype(int)

    target_result_cols = ["event_id"] + [c for c in results.columns if c.startswith("target__")]
    results_targets = results[target_result_cols]
    print(f"    → {len(target_result_cols)-1} result-based target columns")

    # ══════════════════════════════════════════════════════════
    # STEP 6 — Card targets from incidents
    # ══════════════════════════════════════════════════════════
    print("\n[6/8] Engineering card targets from incidents...")
    incidents = pd.read_sql_query("""
        SELECT event_id, incident_type, competitor
        FROM flashscore_incidents
        WHERE event_id IS NOT NULL
    """, conn)

    if not incidents.empty:
        incidents["incident_lower"] = incidents["incident_type"].str.lower().str.strip()

        yellow = incidents[incidents["incident_lower"].str.contains("yellow", na=False)]
        red    = incidents[incidents["incident_lower"].str.contains("red",    na=False)]

        yc = yellow.groupby("event_id").size().reset_index(name="total_yellow_cards")
        rc = red.groupby("event_id").size().reset_index(name="total_red_cards")
        cards = yc.merge(rc, on="event_id", how="outer").fillna(0)
        cards["total_cards"] = cards["total_yellow_cards"] + cards["total_red_cards"]

        for line in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
            col = f"target__over_{str(line).replace('.', '_')}_cards"
            cards[col] = (cards["total_cards"] > line).astype(int)

        card_target_cols = ["event_id"] + [c for c in cards.columns if c.startswith("target__")]
        cards_targets = cards[card_target_cols]
        print(f"    → {len(card_target_cols)-1} card-based target columns")
    else:
        print("    ! No incident data found yet")
        cards_targets = pd.DataFrame(columns=["event_id"])

    # ══════════════════════════════════════════════════════════
    # STEP 7 — Corner targets from flashscore_stats
    # ══════════════════════════════════════════════════════════
    print("\n[7/8] Engineering corner targets...")
    if not fs_stats_raw.empty:
        corners_raw = fs_stats_raw[
            fs_stats_raw["stat_key"].str.contains("corner", na=False)
        ].copy()

        if not corners_raw.empty:
            corners_agg = corners_raw.groupby("event_id").agg(
                home_corners=("home_val_num", "sum"),
                away_corners=("away_val_num", "sum")
            ).reset_index()
            corners_agg["total_corners"] = (
                corners_agg["home_corners"] + corners_agg["away_corners"]
            )

            for line in [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
                col = f"target__over_{str(line).replace('.', '_')}_corners"
                corners_agg[col] = (corners_agg["total_corners"] > line).astype(int)

            corner_target_cols = ["event_id"] + [c for c in corners_agg.columns if c.startswith("target__")]
            corners_targets = corners_agg[corner_target_cols]
            print(f"    → {len(corner_target_cols)-1} corner-based target columns")
        else:
            print("    ! No corner stats found in flashscore_stats yet")
            corners_targets = pd.DataFrame(columns=["event_id"])
    else:
        corners_targets = pd.DataFrame(columns=["event_id"])

    # ══════════════════════════════════════════════════════════
    # STEP 8 — Merge everything into one master flat table
    # ══════════════════════════════════════════════════════════
    print("\n[8/8] Merging all components into master table...")

    master = fixtures.copy()

    # Features (pre-match)
    master = master.merge(odds_wide,      on="event_id", how="inner")
    master = master.merge(cross_features, on="event_id", how="left")

    # Targets (post-match) — inner join ensures only completed matches
    master = master.merge(results_targets, on="event_id", how="inner")

    # Optional targets
    if not cards_targets.empty and len(cards_targets.columns) > 1:
        master = master.merge(cards_targets, on="event_id", how="left")
    if not corners_targets.empty and len(corners_targets.columns) > 1:
        master = master.merge(corners_targets, on="event_id", how="left")

    # Post-match stats (for analysis/verification — NOT to be used as ML features)
    if not stats_wide.empty and len(stats_wide.columns) > 1:
        master = master.merge(stats_wide, on="event_id", how="left")

    # ══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════
    feature_cols    = [c for c in master.columns if c.startswith(("true_prob__", "implied_prob__", "vig_delta__", "cross__"))]
    target_cols     = [c for c in master.columns if c.startswith("target__")]
    stat_cols       = [c for c in master.columns if c.startswith(("home_stat__", "away_stat__"))]

    print("\n" + "="*60)
    print("  MASTER TRAINING TABLE — SUMMARY")
    print("="*60)
    print(f"  Total completed fixtures : {len(master):,}")
    print(f"  Total columns            : {len(master.columns):,}")
    print(f"  Feature columns          : {len(feature_cols):,}")
    print(f"    - true_prob__*         : {len([c for c in feature_cols if c.startswith('true_prob__')]):,}")
    print(f"    - implied_prob__*      : {len([c for c in feature_cols if c.startswith('implied_prob__')]):,}")
    print(f"    - vig_delta__*         : {len([c for c in feature_cols if c.startswith('vig_delta__')]):,}")
    print(f"    - cross__*             : {len([c for c in feature_cols if c.startswith('cross__')]):,}")
    print(f"  Target columns           : {len(target_cols):,}")
    print(f"  Stat columns (reference) : {len(stat_cols):,}")

    print(f"\n  TARGET HIT RATES:")
    print(f"  {'TARGET':<40} {'MATCHES':>8} {'HIT%':>7}")
    print(f"  {'-'*57}")
    for t in sorted(target_cols):
        valid  = master[t].notna().sum()
        if valid == 0:
            continue
        hits   = int(master[t].sum())
        pct    = hits / valid * 100
        print(f"  {t:<40} {valid:>8,} {pct:>6.1f}%")

    # Save
    master.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✅ Saved to: {OUTPUT_CSV}")
    print("="*60 + "\n")

    conn.close()
    return master


# ── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    df = build_master_table(DB_PATH)
"""
=============================================================
  PREDICT TODAY — ONE BEST PICK PER FIXTURE
  Loads all trained .pkl models → scores fixtures →
  selects the single highest-edge pick per fixture →
  outputs a ranked shortlist you can filter and play.
=============================================================
  WORKFLOW:
    1. Load today's fixtures from sportybet.db (or CSV)
    2. Build feature columns matching lean_training_table
    3. Run every model on every fixture
    4. For each fixture: pick the ONE outcome where
       model_prob - implied_prob is highest (best value)
    5. Output ranked picks table (best edge at top)

  USAGE:
    python predict_today.py
    python predict_today.py --min-edge 0.03   (only edges > 3%)
    python predict_today.py --top 20          (top 20 picks only)
    python predict_today.py --min-auc 0.60    (only good models)
    python predict_today.py --csv             (save to CSV too)
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import sqlite3
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
DB_PATH      = r"..\sportybet.db"
MODELS_DIR     = "models"
OUTPUT_CSV     = "picks_today_v2.csv"

# Default filters (override with CLI args)
MIN_EDGE       = 0.03    # minimum model_prob - implied_prob to consider
MIN_AUC        = 0.54    # skip picks from models weaker than this
MIN_MODEL_PROB = 0.20    # don't back something our model says < 20% likely
MAX_MODEL_PROB = 0.90    # don't back something so likely it has no value

# Implied probability column prefix (from your odds feature columns)
IMPLIED_PREFIX = "implied_prob__"
TRUE_PREFIX    = "true_prob__"

# Map from target column name → the col_key used in odds columns
# This lets us look up the bookmaker's actual implied probability
# for each model's prediction to compute the edge.
TARGET_TO_ODDS_KEY = {
    # Main match result
    "target__home_win"              : "1x2____home",
    "target__draw"                  : "1x2____draw",
    "target__away_win"              : "1x2____away",

    # BTTS
    "target__btts_yes"              : "gg_ng____yes",
    "target__btts_no"               : "gg_ng____no",

    # Over/under goals (full match)
    "target__over_0_5_goals"        : "over_under__total=0_5__over_0_5",
    "target__over_1_5_goals"        : "over_under__total=1_5__over_1_5",
    "target__over_2_5_goals"        : "over_under__total=2_5__over_2_5",
    "target__over_3_5_goals"        : "over_under__total=3_5__over_3_5",
    "target__over_4_5_goals"        : "over_under__total=4_5__over_4_5",
    "target__over_5_5_goals"        : "over_under__total=5_5__over_5_5",

    # HT goals
    "target__ht_over_0_5_goals"     : "1st_half___over_under__total=0_5__over_0_5",
    "target__ht_over_1_5_goals"     : "1st_half___over_under__total=1_5__over_1_5",

    # HT result
    "target__ht_home_win"           : "1st_half___1x2____home",
    "target__ht_draw"               : "1st_half___1x2____draw",
    "target__ht_away_win"           : "1st_half___1x2____away",

    # Clean sheets
    "target__home_clean_sheet"      : "home_team_clean_sheet____yes",
    "target__away_clean_sheet"      : "away_team_clean_sheet____yes",

    # Corners (full match)
    "target__over_6_5_corners"      : "over_under__total=6_5__over_6_5",
    "target__over_7_5_corners"      : "over_under__total=7_5__over_7_5",
    "target__over_8_5_corners"      : "over_under__total=8_5__over_8_5",
    "target__over_9_5_corners"      : "over_under__total=9_5__over_9_5",
    "target__over_10_5_corners"     : "over_under__total=10_5__over_10_5",
    "target__over_11_5_corners"     : "over_under__total=11_5__over_11_5",
    "target__over_12_5_corners"     : "over_under__total=12_5__over_12_5",

    # Exact scores
    "target__score_1_0"             : "correct_score____10",
    "target__score_0_1"             : "correct_score____01",
    "target__score_1_1"             : "correct_score____11",
    "target__score_2_0"             : "correct_score____20",
    "target__score_0_2"             : "correct_score____02",
    "target__score_2_1"             : "correct_score____21",
    "target__score_1_2"             : "correct_score____12",
    "target__score_2_2"             : "correct_score____22",
    "target__score_0_0"             : "correct_score____00",
    "target__score_3_0"             : "correct_score____30",
    "target__score_3_1"             : "correct_score____31",
    "target__score_3_2"             : "correct_score____32",

    # Stat-based (new targets — these don't have direct bookmaker odds columns
    # so we use the model_prob only and set implied_prob to the base hit_rate)
    # They're included but flagged as "no odds match" in the output.
    "target__new__corners__match__over_7_5"          : None,
    "target__new__corners__match__over_8_5"          : None,
    "target__new__corners__match__over_9_5"          : None,
    "target__new__corners__match__over_10_5"         : None,
    "target__new__corners__match__over_11_5"         : None,
    "target__new__corners__h1__over_3_5"             : None,
    "target__new__corners__h1__over_4_5"             : None,
    "target__new__corners__h1__over_5_5"             : None,
    "target__new__corners__h2__over_3_5"             : None,
    "target__new__corners__h2__over_4_5"             : None,
    "target__new__corners__h2__over_5_5"             : None,
    "target__new__shots_on_target__match__over_5_5"  : None,
    "target__new__shots_on_target__match__over_6_5"  : None,
    "target__new__shots_on_target__match__over_7_5"  : None,
    "target__new__shots_on_target__match__over_8_5"  : None,
    "target__new__total_shots__match__over_18_5"     : None,
    "target__new__total_shots__match__over_20_5"     : None,
    "target__new__total_shots__match__over_22_5"     : None,
    "target__new__total_shots__match__over_24_5"     : None,
    "target__new__yellow_cards__match__over_2_5"     : None,
    "target__new__yellow_cards__match__over_3_5"     : None,
    "target__new__yellow_cards__match__over_4_5"     : None,
    "target__new__yellow_cards__h1__over_1_5"        : None,
    "target__new__fouls__match__over_18_5"           : None,
    "target__new__fouls__match__over_20_5"           : None,
    "target__new__fouls__match__over_22_5"           : None,
    "target__new__xg__match__over_2_0"               : None,
    "target__new__xg__match__over_2_5"               : None,
    "target__new__xg__match__over_3_0"               : None,
    "target__new__big_chances__match__over_2_5"      : None,
    "target__new__big_chances__match__over_3_5"      : None,
    "target__new__big_chances__match__over_4_5"      : None,
    "target__new__offsides__match__over_1_5"         : None,
    "target__new__offsides__match__over_2_5"         : None,
    "target__new__offsides__match__over_3_5"         : None,
    "target__new__shots_in_box__match__over_10_5"    : None,
    "target__new__shots_in_box__match__over_12_5"    : None,
    "target__new__shots_in_box__match__over_14_5"    : None,
    "target__new__gk_saves__match__over_3_5"         : None,
    "target__new__gk_saves__match__over_4_5"         : None,
    "target__new__gk_saves__match__over_5_5"         : None,
    "target__new__home_possession__match__over_50"   : None,
}

# Human-readable labels for each target
TARGET_LABELS = {
    "target__home_win"              : "Home Win",
    "target__draw"                  : "Draw",
    "target__away_win"              : "Away Win",
    "target__btts_yes"              : "BTTS Yes",
    "target__btts_no"               : "BTTS No",
    "target__over_0_5_goals"        : "Over 0.5 Goals",
    "target__over_1_5_goals"        : "Over 1.5 Goals",
    "target__over_2_5_goals"        : "Over 2.5 Goals",
    "target__over_3_5_goals"        : "Over 3.5 Goals",
    "target__over_4_5_goals"        : "Over 4.5 Goals",
    "target__over_5_5_goals"        : "Over 5.5 Goals",
    "target__ht_over_0_5_goals"     : "HT Over 0.5",
    "target__ht_over_1_5_goals"     : "HT Over 1.5",
    "target__ht_home_win"           : "HT Home Win",
    "target__ht_draw"               : "HT Draw",
    "target__ht_away_win"           : "HT Away Win",
    "target__home_clean_sheet"      : "Home Clean Sheet",
    "target__away_clean_sheet"      : "Away Clean Sheet",
    "target__over_6_5_corners"      : "Corners Over 6.5",
    "target__over_7_5_corners"      : "Corners Over 7.5",
    "target__over_8_5_corners"      : "Corners Over 8.5",
    "target__over_9_5_corners"      : "Corners Over 9.5",
    "target__over_10_5_corners"     : "Corners Over 10.5",
    "target__over_11_5_corners"     : "Corners Over 11.5",
    "target__over_12_5_corners"     : "Corners Over 12.5",
    "target__score_1_0"             : "Score 1-0",
    "target__score_0_1"             : "Score 0-1",
    "target__score_1_1"             : "Score 1-1",
    "target__score_2_0"             : "Score 2-0",
    "target__score_0_2"             : "Score 0-2",
    "target__score_2_1"             : "Score 2-1",
    "target__score_1_2"             : "Score 1-2",
    "target__score_2_2"             : "Score 2-2",
    "target__score_0_0"             : "Score 0-0",
    "target__score_3_0"             : "Score 3-0",
    "target__score_3_1"             : "Score 3-1",
    "target__score_3_2"             : "Score 3-2",
}

def label(target):
    if target in TARGET_LABELS:
        return TARGET_LABELS[target]
    # Auto-label stat targets
    return (target
        .replace("target__new__", "")
        .replace("target__", "")
        .replace("__", " ")
        .replace("_", " ")
        .title())


def clean_col_name(text):
    if pd.isna(text) or text is None:
        return "none"
    return (
        str(text).strip().lower()
        .replace(" ", "_").replace("/", "_").replace("-", "_")
        .replace(".", "_").replace("(", "").replace(")", "")
        .replace(":", "").replace("'", "")
        .replace(">", "over").replace("<", "under")
    )


def prob_to_decimal_odds(p):
    """Convert probability to decimal odds (e.g. 0.5 → 2.00)."""
    if pd.isna(p) or p <= 0:
        return np.nan
    return round(1.0 / p, 2)


def load_all_models(models_dir):
    """Load every .pkl file from the models directory."""
    models = {}
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}\nRun train_models.py first.")

    pkl_files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {models_dir}\nRun train_models.py first.")

    print(f"  Loading {len(pkl_files)} models...")
    for fname in sorted(pkl_files):
        path = os.path.join(models_dir, fname)
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            target = data["target"]
            models[target] = data
        except Exception as e:
            print(f"    ⚠ Could not load {fname}: {e}")

    print(f"  → {len(models)} models loaded successfully")
    return models


def load_fixtures_from_db(db_path, window_hours=36):
    """
    Load upcoming fixtures from sportybet.db.
    window_hours: how many hours ahead to look (default 36 = today + tomorrow AM)
    Returns fixtures with their pre-match odds as a wide feature DataFrame.
    """
    conn = sqlite3.connect(db_path)

    # ── Detect kickoff_time format in DB ──────────────────────
    # Sportybet stores kickoff_time as either:
    #   "2025-08-29 15:00:00"  (SQLite datetime string)
    #   "1724940000"           (Unix timestamp integer)
    # We detect which and build the right WHERE clause.
    sample = conn.execute(
        "SELECT kickoff_time FROM fixtures WHERE kickoff_time IS NOT NULL LIMIT 5"
    ).fetchall()

    sample_vals = [str(r[0]).strip() for r in sample if r[0] is not None]
    is_unix = all(v.isdigit() for v in sample_vals)

    window_secs = window_hours * 3600

    if is_unix:
        # Unix timestamp: compare against current epoch
        time_filter = f"""
            CAST(f.kickoff_time AS INTEGER) >= (strftime('%s','now') - 7200)
            AND CAST(f.kickoff_time AS INTEGER) <= (strftime('%s','now') + {window_secs})
        """
        print(f"  → Kickoff format: Unix timestamp")
    else:
        # SQLite datetime string
        time_filter = f"""
            f.kickoff_time >= datetime('now', '-2 hours')
            AND f.kickoff_time <= datetime('now', '+{window_hours} hours')
        """
        print(f"  → Kickoff format: datetime string")

    # Load fixtures kicking off in the next 36 hours (today + tomorrow morning)
    # The -2 hour buffer catches matches that just kicked off but aren't scored yet
    fixtures = pd.read_sql_query(f"""
        SELECT
            f.event_id,
            f.tournament_name,
            f.category_name,
            f.home_team,
            f.away_team,
            f.kickoff_time
        FROM fixtures f
        LEFT JOIN results r ON f.event_id = r.event_id
        WHERE ({time_filter})
          AND (r.ft_home IS NULL OR r.event_id IS NULL)
        ORDER BY f.kickoff_time
    """, conn)

    # Fallback: if time filter returns nothing (bad timestamps, timezone mismatch),
    # show what the next 500 unresulted fixtures look like so user can diagnose
    if fixtures.empty:
        print("  ⚠ No fixtures found in the 36-hour window.")
        print("    Checking your kickoff_time values to diagnose...")
        sample_rows = pd.read_sql_query("""
            SELECT event_id, home_team, away_team, kickoff_time
            FROM fixtures
            WHERE kickoff_time IS NOT NULL
            ORDER BY kickoff_time DESC
            LIMIT 10
        """, conn)
        print("\n  Most recent kickoff_time values in your DB:")
        print(sample_rows.to_string(index=False))
        print("\n  → Run with --window 72 to widen the search window, or check")
        print("    that your scraper is writing kickoff_time in a standard format.")
        conn.close()
        return pd.DataFrame(), pd.DataFrame()

    print(f"  → {len(fixtures)} upcoming fixtures found")

    if fixtures.empty:
        conn.close()
        return pd.DataFrame(), pd.DataFrame()

    # Load odds for these fixtures
    event_ids = fixtures["event_id"].tolist()
    placeholders = ",".join(["?"] * len(event_ids))

    odds_raw = pd.read_sql_query(f"""
        SELECT event_id, market_name, specifier, outcome_desc, odds, probability
        FROM odds
        WHERE event_id IN ({placeholders})
          AND odds IS NOT NULL
          AND probability IS NOT NULL
    """, conn, params=event_ids)

    conn.close()

    if odds_raw.empty:
        print("  ⚠ No odds found for upcoming fixtures.")
        return fixtures, pd.DataFrame()

    # Build col_key (same logic as build_training_table.py)
    odds_raw["col_key"] = (
        odds_raw["market_name"].apply(clean_col_name) + "__" +
        odds_raw["specifier"].fillna("").apply(clean_col_name) + "__" +
        odds_raw["outcome_desc"].apply(clean_col_name)
    )
    odds_raw = odds_raw.drop_duplicates(subset=["event_id", "col_key"])

    odds_raw["implied_prob"] = 1.0 / odds_raw["odds"]
    odds_raw["vig_delta"]    = odds_raw["implied_prob"] - odds_raw["probability"]

    # Pivot all three
    pivot_true = odds_raw.pivot(index="event_id", columns="col_key", values="probability")
    pivot_true.columns = [f"true_prob__{c}" for c in pivot_true.columns]

    pivot_implied = odds_raw.pivot(index="event_id", columns="col_key", values="implied_prob")
    pivot_implied.columns = [f"implied_prob__{c}" for c in pivot_implied.columns]

    pivot_vig = odds_raw.pivot(index="event_id", columns="col_key", values="vig_delta")
    pivot_vig.columns = [f"vig_delta__{c}" for c in pivot_vig.columns]

    # Build cross features (same as build_training_table.py step 3)
    odds_wide = pd.concat([pivot_true, pivot_implied, pivot_vig], axis=1).reset_index()

    def find_col(df, keywords, prefix="true_prob__"):
        for col in df.columns:
            if col.startswith(prefix) and all(k.lower() in col.lower() for k in keywords):
                return col
        return None

    cross = pd.DataFrame({"event_id": odds_wide["event_id"].values})
    col_hw   = find_col(odds_wide, ["1x2", "home"])
    col_aw   = find_col(odds_wide, ["1x2", "away"])
    col_draw = find_col(odds_wide, ["1x2", "draw"])
    col_o25  = find_col(odds_wide, ["2_5", "over"])

    if col_hw and col_aw:
        cross["cross__home_advantage_gap"] = odds_wide[col_hw].values - odds_wide[col_aw].values
    if col_draw:
        cross["cross__draw_prob"] = odds_wide[col_draw].values
    if col_o25:
        cross["cross__over25_true"] = odds_wide[col_o25].values

    mkt_overround = (
        odds_raw.groupby("event_id")["implied_prob"].sum()
        .reset_index().rename(columns={"implied_prob": "cross__total_market_overround"})
    )
    cross = cross.merge(mkt_overround, on="event_id", how="left")

    odds_wide = odds_wide.merge(cross, on="event_id", how="left")

    return fixtures, odds_wide


def get_implied_prob_for_target(target, odds_wide_row):
    """
    Look up the bookmaker's implied probability for a given target.
    Returns (implied_prob, decimal_odds) or (None, None) if not found.
    """
    odds_key = TARGET_TO_ODDS_KEY.get(target)
    if odds_key is None:
        return None, None  # stat-based target — no direct market

    col = f"{IMPLIED_PREFIX}{odds_key}"
    val = odds_wide_row.get(col, np.nan)

    if pd.isna(val) or val <= 0:
        return None, None

    return float(val), prob_to_decimal_odds(val)


def predict_all_fixtures(fixtures, odds_wide, models, min_edge, min_auc, min_model_prob, max_model_prob, odds_only=True):
    """
    Vectorised prediction — runs each model across ALL fixtures at once.
    One batch predict_proba call per model instead of one per fixture.
    Typical speedup: 50-200x over the row-by-row approach.
    """
    if fixtures.empty or odds_wide.empty:
        return pd.DataFrame(), pd.DataFrame()

    data = fixtures.merge(odds_wide, on="event_id", how="inner")

    if data.empty:
        print("  \u26a0 No fixtures could be matched with odds data.")
        return pd.DataFrame(), pd.DataFrame()

    n_fixtures = len(data)
    print(f"  \u2192 Scoring {n_fixtures} fixtures across {len(models)} models (vectorised)...")

    records = []

    for target, model_data in models.items():
        model        = model_data["model"]
        feature_cols = model_data["feature_cols"]
        model_auc    = model_data.get("auc", 0)
        hit_rate     = model_data.get("hit_rate", 0.5)

        if model_auc < min_auc:
            continue

        # Build full feature matrix for all fixtures at once
        X = data.reindex(columns=feature_cols)

        # Only keep fixtures where <=50% of features are missing
        missing_frac = X.isnull().mean(axis=1)
        valid_mask   = (missing_frac <= 0.50).values

        if valid_mask.sum() == 0:
            continue

        X_valid = X[valid_mask]

        try:
            probs = model.predict_proba(X_valid)[:, 1]
        except Exception:
            continue

        # Implied probability lookup
        odds_key       = TARGET_TO_ODDS_KEY.get(target)
        has_odds_match = odds_key is not None
        implied_col    = f"{IMPLIED_PREFIX}{odds_key}" if odds_key else None

        if has_odds_match and implied_col in data.columns:
            implied_probs    = data.loc[valid_mask, implied_col].values.astype(float)
            decimal_odds_arr = np.where(implied_probs > 0,
                                        np.round(1.0 / implied_probs, 2), np.nan)
        else:
            has_odds_match   = False
            implied_probs    = np.full(valid_mask.sum(), hit_rate)
            decimal_odds_arr = np.full(valid_mask.sum(), np.nan)

        edges = probs - implied_probs

        # In odds_only mode, skip targets with no real bookmaker market
        if odds_only and not has_odds_match:
            continue

        keep = (
            (probs >= min_model_prob) &
            (probs <= max_model_prob) &
            (edges >= min_edge) &
            (~np.isnan(implied_probs))
        )

        if keep.sum() == 0:
            continue

        valid_data = data[valid_mask].reset_index(drop=True)

        for i in np.where(keep)[0]:
            row = valid_data.iloc[i]
            dec_odds = float(decimal_odds_arr[i]) if not np.isnan(decimal_odds_arr[i]) else None
            records.append({
                "event_id"       : row["event_id"],
                "home_team"      : row.get("home_team", "?"),
                "away_team"      : row.get("away_team", "?"),
                "match"          : f"{row.get('home_team','?') } vs {row.get('away_team','?')}",
                "tournament"     : row.get("tournament_name", "?"),
                "kickoff"        : row.get("kickoff_time", "?"),
                "target"         : target,
                "pick"           : label(target),
                "model_prob_%"   : round(float(probs[i]) * 100, 1),
                "implied_prob_%"  : round(float(implied_probs[i]) * 100, 1),
                "edge_%"         : round(float(edges[i]) * 100, 1),
                "decimal_odds"   : dec_odds,
                "model_auc"      : round(model_auc, 4),
                "has_odds_match" : has_odds_match,
                "hit_rate_%"     : round(hit_rate * 100, 1),
            })

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    all_picks_df = pd.DataFrame(records)

    # One best pick per fixture = highest edge
    best_picks_df = (
        all_picks_df
        .sort_values("edge_%", ascending=False)
        .drop_duplicates(subset="event_id", keep="first")
        .reset_index(drop=True)
    )

    return best_picks_df, all_picks_df


def print_picks_table(picks_df, title="PICKS", max_rows=None):
    if picks_df.empty:
        print(f"\n  No picks found.")
        return

    df = picks_df.copy()
    if max_rows:
        df = df.head(max_rows)

    print(f"\n  {title} ({len(df)} picks):")
    print(f"  {'MATCH':<35} {'TOURNAMENT':<22} {'PICK':<25} "
          f"{'MODEL%':>7} {'ODDS%':>7} {'EDGE%':>6} {'ODDS':>6} {'AUC':>6}")
    print(f"  {'-'*120}")

    for _, row in df.iterrows():
        match      = str(row["match"])[:34]
        tourn      = str(row["tournament"])[:21]
        pick       = str(row["pick"])[:24]
        model_p    = f"{row['model_prob_%']:.1f}%"
        implied_p  = f"{row['implied_prob_%']:.1f}%"
        edge       = f"+{row['edge_%']:.1f}%"
        odds_str   = f"{row['decimal_odds']:.2f}" if row.get("decimal_odds") else "N/A"
        auc_str    = f"{row['model_auc']:.4f}"
        odds_flag  = "" if row.get("has_odds_match") else " *"

        print(f"  {match:<35} {tourn:<22} {pick:<25} "
              f"{model_p:>7} {implied_p:>7} {edge:>6} {odds_str:>6} {auc_str:>6}{odds_flag}")

    if any(not r.get("has_odds_match", True) for _, r in df.iterrows()):
        print(f"\n  * = stat-based target (no direct market odds; edge vs historical base rate)")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Predict today's fixtures")
    parser.add_argument("--min-edge",       type=float, default=MIN_EDGE,
                        help=f"Min edge threshold (default: {MIN_EDGE})")
    parser.add_argument("--min-auc",        type=float, default=MIN_AUC,
                        help=f"Min model AUC (default: {MIN_AUC})")
    parser.add_argument("--min-model-prob", type=float, default=MIN_MODEL_PROB,
                        help=f"Min model probability (default: {MIN_MODEL_PROB})")
    parser.add_argument("--top",            type=int,   default=None,
                        help="Show only top N picks (by edge)")
    parser.add_argument("--window",         type=int,   default=36,
                        help="Hours ahead to include fixtures (default: 36)")
    parser.add_argument("--csv",            action="store_true",
                        help="Save picks to CSV")
    parser.add_argument("--include-stats",  action="store_true",
                        help="Include stat-based picks (no real market odds, edge vs base rate)")
    parser.add_argument("--sort-by",        type=str, default="edge",
                    choices=["edge", "odds", "hit_rate", "model_prob", "auc"],
                    help="Sort picks by: edge (default), odds, hit_rate, model_prob, auc")
    parser.add_argument("--markets",        nargs="+", default=None,
                    help='Only include these markets. Partial match, case-insensitive. '
                         'e.g. --markets "HT Away Win" "Over 2.5"')
    parser.add_argument("--min-odds",       type=float, default=None,
                    help="Minimum decimal odds e.g. --min-odds 1.71")
    parser.add_argument("--max-odds",       type=float, default=None,
                    help="Maximum decimal odds e.g. --max-odds 3.00")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  PREDICT TODAY — ONE BEST PICK PER FIXTURE")
    print("="*65)
    print(f"\n  Settings:")
    print(f"    Min edge        : {args.min_edge*100:.1f}%")
    print(f"    Min model AUC   : {args.min_auc:.2f}")
    print(f"    Min model prob  : {args.min_model_prob*100:.0f}%")
    print(f"    Fixture window  : next {args.window} hours")
    print(f"    Odds-only mode  : {not args.include_stats} (use --include-stats to see stat picks)")
    if args.markets:
        print(f"    Market filter   : {', '.join(args.markets)}")
    if args.min_odds is not None:
        print(f"    Min odds        : {args.min_odds}")
    if args.max_odds is not None:
        print(f"    Max odds        : {args.max_odds}")

    # 1. Load models
    print(f"\n[1/4] Loading trained models from ./{MODELS_DIR}/...")
    models = load_all_models(MODELS_DIR)

    if not models:
        print("  No models found. Run train_models.py first.")
        return

    # Show model summary
    print(f"\n  Model inventory ({len(models)} models):")
    print(f"  {'TARGET':<50} {'AUC':>6}  {'HIT%':>5}")
    print(f"  {'-'*65}")
    for target, md in sorted(models.items(), key=lambda x: x[1].get("auc", 0), reverse=True):
        print(f"  {target:<50} {md.get('auc', 0):.4f}  {md.get('hit_rate', 0)*100:.1f}%")

    # 2. Load fixtures
    print(f"\n[2/4] Loading upcoming fixtures from {DB_PATH}...")
    if not os.path.exists(DB_PATH):
        print(f"  ⚠ DB not found at {DB_PATH}")
        print(f"  Place your sportybet.db in the same folder as this script.")
        return

    fixtures, odds_wide = load_fixtures_from_db(DB_PATH, window_hours=args.window)

    if fixtures.empty:
        print("  No fixtures to score.")
        return

    # 3. Predict
    print(f"\n[3/4] Running predictions...")
    odds_only = not args.include_stats
    best_picks_df, all_picks_df = predict_all_fixtures(
        fixtures, odds_wide, models,
        min_edge       = args.min_edge,
        min_auc        = args.min_auc,
        min_model_prob = args.min_model_prob,
        max_model_prob = MAX_MODEL_PROB,
        odds_only      = odds_only,
    )

    # 4. Display
    print(f"\n[4/4] Results")
    print("="*65)

    if best_picks_df.empty:
        print(f"\n  ⚠ No picks found above the edge threshold ({args.min_edge*100:.1f}%).")
        print(f"  Try lowering --min-edge (e.g. --min-edge 0.01)")
        return

    # Sort picks by chosen column
    SORT_MAP = {
        "edge"       : "edge_%",
        "odds"       : "decimal_odds",
        "hit_rate"   : "hit_rate_%",
        "model_prob" : "model_prob_%",
        "auc"        : "model_auc",
    }
        # ── Apply combination filters ────────────────────────────
    if args.markets:
        def market_match(pick_label):
            return any(m.lower() in pick_label.lower() for m in args.markets)
        best_picks_df = best_picks_df[best_picks_df["pick"].apply(market_match)]
        all_picks_df  = all_picks_df[all_picks_df["pick"].apply(market_match)]

    if args.min_odds is not None:
        best_picks_df = best_picks_df[best_picks_df["decimal_odds"] >= args.min_odds]
        all_picks_df  = all_picks_df[all_picks_df["decimal_odds"] >= args.min_odds]

    if args.max_odds is not None:
        best_picks_df = best_picks_df[best_picks_df["decimal_odds"] <= args.max_odds]
        all_picks_df  = all_picks_df[all_picks_df["decimal_odds"] <= args.max_odds]

    if best_picks_df.empty:
        print("\n  No picks found after applying your filters.")
        print("  Try relaxing --min-odds, --max-odds, or --markets.")
        return
    # ────────────────────────────────────────────────────────

    sort_col = SORT_MAP.get(args.sort_by, "edge_%")
    best_picks_df = best_picks_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    print(f"  Sorted by       : {args.sort_by} ({sort_col})")

    # ── ALL PICKS (one per fixture) ─────────────────────────
    print_picks_table(best_picks_df, title="ALL PICKS — ONE PER FIXTURE (sorted by edge)")

    # ── TOP N FILTER ────────────────────────────────────────
    if args.top:
        print()
        top_df = best_picks_df.head(args.top)
        print_picks_table(top_df, title=f"TOP {args.top} PICKS")

    # ── SUMMARY STATS ───────────────────────────────────────
    print(f"\n  SUMMARY:")
    print(f"    Total fixtures scored    : {len(fixtures)}")
    print(f"    Fixtures with a pick     : {len(best_picks_df)}")
    print(f"    Fixtures with NO pick    : {len(fixtures) - len(best_picks_df)}")
    print(f"    Avg edge (picks only)    : +{best_picks_df['edge_%'].mean():.1f}%")
    print(f"    Avg model prob           : {best_picks_df['model_prob_%'].mean():.1f}%")
    print(f"    Avg decimal odds         : "
          f"{best_picks_df['decimal_odds'].dropna().mean():.2f}")

    # Pick type breakdown
    print(f"\n  PICK TYPE BREAKDOWN:")
    pick_counts = best_picks_df["pick"].value_counts().head(15)
    for pick_type, count in pick_counts.items():
        bar = "█" * count
        print(f"    {pick_type:<30} {count:>3}  {bar}")

    # High-confidence filter (edge > 5%)
    high_conf = best_picks_df[best_picks_df["edge_%"] >= 5.0]
    if not high_conf.empty:
        print(f"\n  HIGH CONFIDENCE PICKS (edge ≥ 5%):")
        print_picks_table(high_conf, title=f"HIGH CONFIDENCE — {len(high_conf)} picks")

    # ── SAVE TO CSV ─────────────────────────────────────────
        # Always save the dated all-picks file — weekly_summary.py needs this
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    dated_path = f"all_picks_{today}.csv"
    all_picks_df.to_csv(dated_path, index=False)
    print(f"\n  ✅ All picks saved to {dated_path}")

    if args.csv:
        save_df = best_picks_df.head(args.top) if args.top else best_picks_df
        save_df.to_csv(OUTPUT_CSV, index=False)
        top_label = f"top {args.top}" if args.top else "all"
        print(f"  ✅ Filtered picks saved to {OUTPUT_CSV}")

    print("\n" + "="*65)
    print("  DONE. Filter tips:")
    print("    --top 20           → take only the best 20 by edge")
    print("    --min-edge 0.05    → only picks with ≥5% edge")
    print("    --min-auc 0.62     → only picks from GOOD+ models")
    print("    --csv              → save everything to CSV")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
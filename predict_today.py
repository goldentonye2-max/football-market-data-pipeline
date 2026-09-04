"""
=============================================================
  DAILY PREDICTION ENGINE
  Scores today's upcoming fixtures through all trained models
  Finds +EV picks and builds the daily accumulator

  Run this every day after your scraper has collected
  today's fixture odds.

  Output:
    - daily_predictions.csv  → full model scores per fixture
    - daily_accumulator.txt  → today's recommended picks
=============================================================
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime, date
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────
# DB is in the root pipeline folder
DB_PATH          = "sportybet.db"

# All ML outputs are inside the ML_model subfolder
ML_DIR           = "ML_model"
LEAN_CSV         = os.path.join(ML_DIR, "lean_training_table.csv")
MODELS_DIR       = os.path.join(ML_DIR, "models")
OUTPUT_CSV       = os.path.join(ML_DIR, "daily_predictions.csv")
ACCUMULATOR_TXT  = os.path.join(ML_DIR, "daily_accumulator.txt")

# Minimum model probability to consider a pick
MIN_MODEL_PROB   = 0.55

# Maximum model probability — 100% means out-of-distribution, reject
MAX_MODEL_PROB   = 0.92

# Minimum fraction of feature columns that must be non-null for a
# prediction to be trusted. Below this = model is guessing blind.
MIN_FEATURE_COVERAGE = 0.20   # at least 20% of features must have real values

# Minimum EV threshold to flag as a value pick
MIN_EV           = 0.02   # 2% edge over bookmaker

# Minimum bookmaker odds
MIN_ODDS         = 1.20

# Maximum bookmaker odds — above this is too risky for accumulator
MAX_ODDS         = 4.00

# How many legs in the accumulator
ACCUM_LEGS       = 3

# Only use models with AUC above this in picks
MIN_AUC_FOR_PICK = 0.64
# ─────────────────────────────────────────────────────────────


def load_model(model_path):
    """Load a saved model package."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_todays_fixtures(conn):
    """
    Pull today's upcoming fixtures from the database.
    Returns a DataFrame with event_id and match info.
    """
    today = date.today().isoformat()

    query = f"""
        SELECT
            f.event_id,
            f.home_team,
            f.away_team,
            f.tournament_name,
            f.category_name,
            f.kickoff_time
        FROM fixtures f
        WHERE DATE(f.kickoff_time) = '{today}'
          AND f.event_id NOT IN (
              SELECT event_id FROM results WHERE status IS NOT NULL
          )
        ORDER BY f.kickoff_time
    """
    df = pd.read_sql_query(query, conn)
    return df


def build_feature_row(conn, event_id, feature_cols):
    """
    Build one feature row for a single fixture by pivoting
    its odds from the database into the same wide format
    used during training.
    """
    odds_q = f"""
        SELECT market_name, specifier, outcome_desc, odds, probability
        FROM odds
        WHERE event_id = '{event_id}'
          AND odds IS NOT NULL
          AND probability IS NOT NULL
    """
    odds = pd.read_sql_query(odds_q, conn)

    if odds.empty:
        return None

    # Build col keys (must match training format exactly)
    def clean(text):
        if pd.isna(text) or text is None:
            return "none"
        return (
            str(text).strip().lower()
            .replace(" ", "_").replace("/", "_").replace("-", "_")
            .replace(".", "_").replace("(", "").replace(")", "")
            .replace(":", "").replace("'", "")
            .replace(">", "over").replace("<", "under")
        )

    odds["col_key"] = (
        odds["market_name"].apply(clean) + "__" +
        odds["specifier"].fillna("").apply(clean) + "__" +
        odds["outcome_desc"].apply(clean)
    )
    odds = odds.drop_duplicates(subset=["col_key"])

    odds["implied_prob"] = 1.0 / odds["odds"]
    odds["vig_delta"]    = odds["implied_prob"] - odds["probability"]

    # Build feature dict
    feature_dict = {}
    for _, row in odds.iterrows():
        key = row["col_key"]
        feature_dict[f"true_prob__{key}"]    = row["probability"]
        feature_dict[f"implied_prob__{key}"] = row["implied_prob"]
        feature_dict[f"vig_delta__{key}"]    = row["vig_delta"]

    # Cross features
    # Home advantage gap
    hw = feature_dict.get("true_prob__1x2____home")
    aw = feature_dict.get("true_prob__1x2____away")
    if hw and aw:
        feature_dict["cross__home_advantage_gap"] = hw - aw

    dw = feature_dict.get("true_prob__1x2____draw")
    if dw:
        feature_dict["cross__draw_prob"] = dw

    # Build one-row DataFrame aligned to training feature columns
    row_df = pd.DataFrame([feature_dict])

    # Align to training columns (adds NaN for missing, drops extras)
    row_aligned = row_df.reindex(columns=feature_cols)

    return row_aligned


def get_bookmaker_odds_for_pick(conn, event_id, target):
    """
    Find the bookmaker's displayed odds for a specific target outcome.
    Maps target column names back to market/outcome names.
    Returns the decimal odds if found, else None.
    """
    # Mapping from target name → (market keywords, outcome keywords)
    TARGET_TO_MARKET = {
        "target__home_win"                          : ("1x2",           "home"),
        "target__draw"                              : ("1x2",           "draw"),
        "target__away_win"                          : ("1x2",           "away"),
        "target__btts_yes"                          : ("gg_ng",         "yes"),
        "target__btts_no"                           : ("gg_ng",         "no"),
        "target__over_1_5_goals"                    : ("over_under",    "over_1_5"),
        "target__over_2_5_goals"                    : ("over_under",    "over_2_5"),
        "target__over_3_5_goals"                    : ("over_under",    "over_3_5"),
        "target__over_4_5_goals"                    : ("over_under",    "over_4_5"),
        "target__home_clean_sheet"                  : ("home_clean",    "yes"),
        "target__away_clean_sheet"                  : ("away_clean",    "yes"),
        "target__ht_home_win"                       : ("1st_half___1x2", "home"),
        "target__ht_draw"                           : ("1st_half___1x2", "draw"),
        "target__ht_away_win"                       : ("1st_half___1x2", "away"),
        "target__ht_over_0_5_goals"                 : ("1st_half", "over_0_5"),
        "target__ht_over_1_5_goals"                 : ("1st_half", "over_1_5"),
        "target__new__corners__match__over_7_5"     : ("corner", "over_7_5"),
        "target__new__corners__match__over_8_5"     : ("corner", "over_8_5"),
        "target__new__corners__match__over_9_5"     : ("corner", "over_9_5"),
        "target__new__corners__match__over_10_5"    : ("corner", "over_10_5"),
        "target__new__corners__match__over_11_5"    : ("corner", "over_11_5"),
        "target__new__yellow_cards__match__over_2_5": ("yellow_card", "over_2_5"),
        "target__new__yellow_cards__match__over_3_5": ("yellow_card", "over_3_5"),
        "target__new__yellow_cards__match__over_4_5": ("yellow_card", "over_4_5"),
        "target__new__total_shots__match__over_18_5": ("total_shot", "over_18_5"),
        "target__new__total_shots__match__over_20_5": ("total_shot", "over_20_5"),
        "target__over_9_5_corners"                  : ("corner", "over_9_5"),
        "target__over_10_5_corners"                 : ("corner", "over_10_5"),
        "target__over_11_5_corners"                 : ("corner", "over_11_5"),
    }

    if target not in TARGET_TO_MARKET:
        return None

    mkt_kw, out_kw = TARGET_TO_MARKET[target]

    odds_q = f"""
        SELECT odds
        FROM odds
        WHERE event_id = '{event_id}'
          AND LOWER(market_name) LIKE '%{mkt_kw}%'
          AND LOWER(outcome_desc) LIKE '%{out_kw}%'
        ORDER BY odds ASC
        LIMIT 1
    """
    result = pd.read_sql_query(odds_q, conn)
    if result.empty or result["odds"].iloc[0] is None:
        return None
    return float(result["odds"].iloc[0])


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print(f"  DAILY PREDICTION ENGINE — {date.today().strftime('%A %d %B %Y')}")
print("="*65)

# ── Load model results to know which models are good ──────
results_path = os.path.join(ML_DIR, "model_results_trained_only.csv")
if not os.path.exists(results_path):
    raise FileNotFoundError("Run train_models.py first.")

model_results = pd.read_csv(results_path)
good_models   = model_results[model_results["auc"] >= MIN_AUC_FOR_PICK].copy()
print(f"\n  Models loaded     : {len(model_results)}")
print(f"  Models above AUC {MIN_AUC_FOR_PICK}: {len(good_models)}")

# ── Load training feature columns ─────────────────────────
if not os.path.exists(LEAN_CSV):
    raise FileNotFoundError("lean_training_table.csv not found.")

lean_sample = pd.read_csv(LEAN_CSV, nrows=1)
feature_cols = [c for c in lean_sample.columns if c.startswith(
    ("true_prob__", "implied_prob__", "vig_delta__", "cross__"))]
print(f"  Feature cols      : {len(feature_cols):,}")

# ── Connect to database ────────────────────────────────────
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)

# ── Get today's fixtures ───────────────────────────────────
fixtures = get_todays_fixtures(conn)
print(f"  Today's fixtures  : {len(fixtures)}")

if fixtures.empty:
    print("\n  ⚠ No upcoming fixtures found for today.")
    print("  Make sure your scraper has run today.")
    conn.close()
    exit()

print(f"\n  TODAY'S FIXTURES:")
for _, row in fixtures.iterrows():
    print(f"    {row['home_team']} vs {row['away_team']} "
          f"({row['tournament_name']}) @ {row['kickoff_time']}")

# ═══════════════════════════════════════════════════════════
# SCORE EVERY FIXTURE THROUGH EVERY GOOD MODEL
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  SCORING FIXTURES")
print(f"{'='*65}\n")

all_predictions = []

for _, fixture in fixtures.iterrows():
    event_id  = fixture["event_id"]
    match_str = f"{fixture['home_team']} vs {fixture['away_team']}"

    print(f"  Scoring: {match_str}")

    # Build feature row for this fixture
    X_row = build_feature_row(conn, event_id, feature_cols)

    if X_row is None or X_row.empty:
        print(f"    ⚠ No odds data found — skipping")
        continue

    # Run through each good model
    for _, model_info in good_models.iterrows():
        target    = model_info["target"]
        model_auc = model_info["auc"]
        hit_rate  = model_info["hit_rate_%"] / 100

        # Load model file
        safe_name  = target.replace("target__", "").replace("new__", "")
        model_path = os.path.join(MODELS_DIR, f"model__{safe_name}.pkl")

        if not os.path.exists(model_path):
            continue

        try:
            pkg = load_model(model_path)
            calibrated_model = pkg["model"]
            model_feature_cols = pkg["feature_cols"]

            # Align features to what this model expects
            X_aligned = X_row.reindex(columns=model_feature_cols)

            # ── COVERAGE CHECK ──────────────────────────────
            # If the fixture has almost no feature data for this model,
            # the prediction is unreliable — skip it entirely.
            feature_coverage = X_aligned.notna().mean().mean()
            if feature_coverage < MIN_FEATURE_COVERAGE:
                continue

            # Get calibrated probability
            prob = calibrated_model.predict_proba(X_aligned)[:, 1][0]

            # Get bookmaker displayed odds for this outcome
            bookie_odds    = get_bookmaker_odds_for_pick(conn, event_id, target)
            bookie_implied = (1.0 / bookie_odds) if bookie_odds else None

            # Calculate Expected Value
            ev = None
            if bookie_odds:
                ev = (prob * bookie_odds) - 1.0

            all_predictions.append({
                "event_id"          : event_id,
                "match"             : match_str,
                "tournament"        : fixture["tournament_name"],
                "kickoff"           : fixture["kickoff_time"],
                "target"            : target,
                "model_probability" : round(prob, 4),
                "feature_coverage"  : round(feature_coverage, 3),
                "bookie_odds"       : bookie_odds,
                "bookie_implied"    : round(bookie_implied, 4) if bookie_implied else None,
                "expected_value_pct": round(ev * 100, 2) if ev is not None else None,
                "model_auc"         : model_auc,
                "historical_hit_%"  : round(hit_rate * 100, 1),
                "edge_over_bookie"  : round((prob - bookie_implied) * 100, 2) if bookie_implied else None,
            })

        except Exception as e:
            continue

print(f"\n  Total predictions generated: {len(all_predictions)}")

# ═══════════════════════════════════════════════════════════
# FILTER FOR HIGH-CONFIDENCE PICKS
# ═══════════════════════════════════════════════════════════
if not all_predictions:
    print("\n  ⚠ No predictions generated. Check that today's odds are scraped.")
    conn.close()
    exit()

preds_df = pd.DataFrame(all_predictions)
preds_df  = preds_df.sort_values("model_probability", ascending=False)

# Value picks: model probability > bookmaker implied AND EV > threshold
# Also requires real odds to exist and probability within sensible range
value_picks = preds_df[
    (preds_df["model_probability"] >= MIN_MODEL_PROB) &
    (preds_df["model_probability"] <= MAX_MODEL_PROB) &
    (preds_df["bookie_odds"].notna()) &
    (preds_df["bookie_odds"] >= MIN_ODDS) &
    (preds_df["bookie_odds"] <= MAX_ODDS) &
    (preds_df["expected_value_pct"] >= MIN_EV * 100) &
    (preds_df["model_auc"] >= MIN_AUC_FOR_PICK)
].copy() if "expected_value_pct" in preds_df.columns else pd.DataFrame()

# High-confidence picks (even without EV confirmation)
high_conf_picks = preds_df[
    (preds_df["model_probability"] >= 0.65) &
    (preds_df["model_probability"] <= MAX_MODEL_PROB) &
    (preds_df["bookie_odds"].notna()) &
    (preds_df["bookie_odds"] >= MIN_ODDS) &
    (preds_df["bookie_odds"] <= MAX_ODDS) &
    (preds_df["model_auc"] >= MIN_AUC_FOR_PICK)
].copy()

# ═══════════════════════════════════════════════════════════
# BUILD THE ACCUMULATOR
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  DAILY ACCUMULATOR — {date.today().strftime('%d %B %Y')}")
print(f"{'='*65}")

# Prefer value picks; fall back to high confidence if needed
pick_pool = value_picks if len(value_picks) >= ACCUM_LEGS else high_conf_picks

# Remove duplicate matches (one pick per match max in accumulator)
accum_picks = []
used_events = set()

for _, row in pick_pool.sort_values(
    ["model_probability", "model_auc"],
    ascending=[False, False]
).iterrows():
    if row["event_id"] not in used_events:
        accum_picks.append(row)
        used_events.add(row["event_id"])
    if len(accum_picks) == ACCUM_LEGS:
        break

# ── Print accumulator ─────────────────────────────────────
accum_lines = []
accum_lines.append(f"{'='*65}")
accum_lines.append(f"  🏆 DAILY {ACCUM_LEGS}-LEG ACCUMULATOR")
accum_lines.append(f"  {date.today().strftime('%A %d %B %Y')}")
accum_lines.append(f"{'='*65}")

combined_odds = 1.0
for i, pick in enumerate(accum_picks, 1):
    odds_str = f"@ {pick['bookie_odds']:.2f}" if pick["bookie_odds"] else "@ [check bookie]"
    ev_str   = (f"  EV: +{pick['expected_value_pct']:.1f}%"
                if pick.get("expected_value_pct") else "")
    line = (
        f"\n  LEG {i}: {pick['match']}\n"
        f"    Pick    : {pick['target'].replace('target__','').replace('new__','').replace('_',' ').upper()}\n"
        f"    Odds    : {odds_str}\n"
        f"    Model P : {pick['model_probability']*100:.1f}%{ev_str}\n"
        f"    AUC     : {pick['model_auc']:.4f}\n"
        f"    League  : {pick['tournament']}\n"
        f"    KO      : {pick['kickoff']}"
    )
    accum_lines.append(line)
    if pick["bookie_odds"]:
        combined_odds *= pick["bookie_odds"]

accum_lines.append(f"\n{'─'*65}")
if combined_odds > 1.0:
    accum_lines.append(f"  Combined Odds : {combined_odds:.2f}")
accum_lines.append(f"  Based on      : {len(preds_df)} model predictions")
accum_lines.append(f"  Picks sourced : AUC ≥ {MIN_AUC_FOR_PICK} models only")
accum_lines.append(f"{'='*65}")

if len(accum_picks) < ACCUM_LEGS:
    accum_lines.append(
        f"\n  ⚠ Only {len(accum_picks)} picks met the threshold today.\n"
        f"  Consider lowering MIN_MODEL_PROB or MIN_AUC_FOR_PICK\n"
        f"  or waiting for more fixtures to be scraped."
    )

accum_text = "\n".join(accum_lines)
print(accum_text)

# ── Print full value picks table ──────────────────────────
print(f"\n{'='*65}")
print(f"  ALL HIGH-CONFIDENCE PICKS TODAY (not just accumulator)")
print(f"{'='*65}")

if not preds_df.empty:
    top = preds_df[
        (preds_df["model_probability"] >= MIN_MODEL_PROB) &
        (preds_df["model_probability"] <= MAX_MODEL_PROB) &
        (preds_df["bookie_odds"].notna()) &
        (preds_df["bookie_odds"] >= MIN_ODDS) &
        (preds_df["bookie_odds"] <= MAX_ODDS)
    ].head(20)
    if not top.empty:
        print(f"\n  {'MATCH':<30} {'PICK':<35} {'PROB':>6}  {'ODDS':>6}  {'EV%':>6}  AUC")
        print(f"  {'-'*95}")
        for _, r in top.iterrows():
            match_short = str(r["match"])[:28]
            target_short = r["target"].replace("target__","").replace("new__","")[:33]
            odds_s = f"{r['bookie_odds']:.2f}" if r["bookie_odds"] else "N/A"
            ev_s   = f"{r['expected_value_pct']:.1f}%" if r.get("expected_value_pct") else "N/A"
            print(f"  {match_short:<30} {target_short:<35} "
                  f"{r['model_probability']*100:>5.1f}%  {odds_s:>6}  {ev_s:>6}  {r['model_auc']:.4f}")
    else:
        print(f"\n  No picks above {MIN_MODEL_PROB*100:.0f}% model confidence today.")

# ── Save outputs ──────────────────────────────────────────
preds_df.to_csv(OUTPUT_CSV, index=False)

with open(ACCUMULATOR_TXT, "w", encoding="utf-8") as f:
    f.write(accum_text)

print(f"\n  Saved: {OUTPUT_CSV}")
print(f"  Saved: {ACCUMULATOR_TXT}")

conn.close()
print(f"\n  ✅ Done. Run this every morning after your scraper completes.\n")
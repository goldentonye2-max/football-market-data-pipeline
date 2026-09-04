"""
=============================================================
  DAILY PREDICTION ENGINE — VERSION 1
  Produced: Hegelmann Away + Nups Home + Czarni Home
  Combined odds: 1.97 — ALL THREE WON
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

DB_PATH          = "sportybet.db"
ML_DIR           = "ML_model"
LEAN_CSV         = os.path.join(ML_DIR, "lean_training_table.csv")
MODELS_DIR       = os.path.join(ML_DIR, "models")
OUTPUT_CSV       = os.path.join(ML_DIR, "daily_predictions.csv")
ACCUMULATOR_TXT  = os.path.join(ML_DIR, "daily_accumulator.txt")

MIN_MODEL_PROB   = 0.55
MIN_EV           = 0.02
MIN_ODDS         = 1.40   # change to 1.20 if preferred
MAX_ODDS         = 4.00
ACCUM_LEGS       = 3
MIN_AUC_FOR_PICK = 0.64


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_todays_fixtures(conn):
    today = date.today().isoformat()
    query = f"""
        SELECT f.event_id, f.home_team, f.away_team,
               f.tournament_name, f.category_name, f.kickoff_time
        FROM fixtures f
        WHERE DATE(f.kickoff_time) = '{today}'
          AND f.event_id NOT IN (
              SELECT event_id FROM results WHERE status IS NOT NULL
          )
        ORDER BY f.kickoff_time
    """
    return pd.read_sql_query(query, conn)


def build_feature_row(conn, event_id, feature_cols):
    odds_q = f"""
        SELECT market_name, specifier, outcome_desc, odds, probability
        FROM odds
        WHERE event_id = '{event_id}'
          AND odds IS NOT NULL AND probability IS NOT NULL
    """
    odds = pd.read_sql_query(odds_q, conn)
    if odds.empty:
        return None

    def clean(text):
        if pd.isna(text) or text is None:
            return "none"
        return (str(text).strip().lower()
                .replace(" ","_").replace("/","_").replace("-","_")
                .replace(".","_").replace("(","").replace(")","")
                .replace(":","").replace("'","")
                .replace(">","over").replace("<","under"))

    odds["col_key"] = (odds["market_name"].apply(clean) + "__" +
                       odds["specifier"].fillna("").apply(clean) + "__" +
                       odds["outcome_desc"].apply(clean))
    odds = odds.drop_duplicates(subset=["col_key"])
    odds["implied_prob"] = 1.0 / odds["odds"]
    odds["vig_delta"]    = odds["implied_prob"] - odds["probability"]

    feature_dict = {}
    for _, row in odds.iterrows():
        key = row["col_key"]
        feature_dict[f"true_prob__{key}"]    = row["probability"]
        feature_dict[f"implied_prob__{key}"] = row["implied_prob"]
        feature_dict[f"vig_delta__{key}"]    = row["vig_delta"]

    hw = feature_dict.get("true_prob__1x2____home")
    aw = feature_dict.get("true_prob__1x2____away")
    if hw and aw:
        feature_dict["cross__home_advantage_gap"] = hw - aw
    dw = feature_dict.get("true_prob__1x2____draw")
    if dw:
        feature_dict["cross__draw_prob"] = dw

    row_df = pd.DataFrame([feature_dict])
    return row_df.reindex(columns=feature_cols)


def get_bookmaker_odds_for_pick(conn, event_id, target):
    TARGET_TO_MARKET = {
        "target__home_win"                          : ("1x2",            "home"),
        "target__draw"                              : ("1x2",            "draw"),
        "target__away_win"                          : ("1x2",            "away"),
        "target__btts_yes"                          : ("gg_ng",          "yes"),
        "target__btts_no"                           : ("gg_ng",          "no"),
        "target__over_1_5_goals"                    : ("over_under",     "over_1_5"),
        "target__over_2_5_goals"                    : ("over_under",     "over_2_5"),
        "target__over_3_5_goals"                    : ("over_under",     "over_3_5"),
        "target__over_4_5_goals"                    : ("over_under",     "over_4_5"),
        "target__home_clean_sheet"                  : ("home_clean",     "yes"),
        "target__away_clean_sheet"                  : ("away_clean",     "yes"),
        "target__ht_home_win"                       : ("1st_half___1x2", "home"),
        "target__ht_draw"                           : ("1st_half___1x2", "draw"),
        "target__ht_away_win"                       : ("1st_half___1x2", "away"),
        "target__ht_over_0_5_goals"                 : ("1st_half",       "over_0_5"),
        "target__ht_over_1_5_goals"                 : ("1st_half",       "over_1_5"),
        "target__new__corners__match__over_7_5"     : ("corner",         "over_7_5"),
        "target__new__corners__match__over_8_5"     : ("corner",         "over_8_5"),
        "target__new__corners__match__over_9_5"     : ("corner",         "over_9_5"),
        "target__new__corners__match__over_10_5"    : ("corner",         "over_10_5"),
        "target__new__corners__match__over_11_5"    : ("corner",         "over_11_5"),
        "target__new__yellow_cards__match__over_2_5": ("yellow_card",    "over_2_5"),
        "target__new__yellow_cards__match__over_3_5": ("yellow_card",    "over_3_5"),
        "target__new__yellow_cards__match__over_4_5": ("yellow_card",    "over_4_5"),
        "target__new__total_shots__match__over_18_5": ("total_shot",     "over_18_5"),
        "target__new__total_shots__match__over_20_5": ("total_shot",     "over_20_5"),
        "target__over_9_5_corners"                  : ("corner",         "over_9_5"),
        "target__over_10_5_corners"                 : ("corner",         "over_10_5"),
        "target__over_11_5_corners"                 : ("corner",         "over_11_5"),
    }
    if target not in TARGET_TO_MARKET:
        return None
    mkt_kw, out_kw = TARGET_TO_MARKET[target]
    odds_q = f"""
        SELECT odds FROM odds
        WHERE event_id = '{event_id}'
          AND LOWER(market_name) LIKE '%{mkt_kw}%'
          AND LOWER(outcome_desc) LIKE '%{out_kw}%'
        ORDER BY odds ASC LIMIT 1
    """
    result = pd.read_sql_query(odds_q, conn)
    if result.empty or result["odds"].iloc[0] is None:
        return None
    return float(result["odds"].iloc[0])


print("\n" + "="*65)
print(f"  DAILY PREDICTION ENGINE v1 — {date.today().strftime('%A %d %B %Y')}")
print("="*65)

results_path = os.path.join(ML_DIR, "model_results_trained_only.csv")
if not os.path.exists(results_path):
    raise FileNotFoundError("Run train_models.py first.")

model_results = pd.read_csv(results_path)
good_models   = model_results[model_results["auc"] >= MIN_AUC_FOR_PICK].copy()
print(f"\n  Models loaded     : {len(model_results)}")
print(f"  Models above AUC {MIN_AUC_FOR_PICK}: {len(good_models)}")

lean_sample  = pd.read_csv(LEAN_CSV, nrows=1)
feature_cols = [c for c in lean_sample.columns if c.startswith(
    ("true_prob__","implied_prob__","vig_delta__","cross__"))]
print(f"  Feature cols      : {len(feature_cols):,}")

conn     = sqlite3.connect(DB_PATH)
fixtures = get_todays_fixtures(conn)
print(f"  Today's fixtures  : {len(fixtures)}")

if fixtures.empty:
    print("\n  No upcoming fixtures found for today.")
    conn.close()
    exit()

print(f"\n{'='*65}\n  SCORING FIXTURES\n{'='*65}\n")

all_predictions = []
for _, fixture in fixtures.iterrows():
    event_id  = fixture["event_id"]
    match_str = f"{fixture['home_team']} vs {fixture['away_team']}"
    print(f"  Scoring: {match_str}")

    X_row = build_feature_row(conn, event_id, feature_cols)
    if X_row is None or X_row.empty:
        print(f"    ⚠ No odds data found — skipping")
        continue

    for _, model_info in good_models.iterrows():
        target    = model_info["target"]
        model_auc = model_info["auc"]
        hit_rate  = model_info["hit_rate_%"] / 100

        safe_name  = target.replace("target__","").replace("new__","")
        model_path = os.path.join(MODELS_DIR, f"model__{safe_name}.pkl")
        if not os.path.exists(model_path):
            continue

        try:
            pkg                = load_model(model_path)
            calibrated_model   = pkg["model"]
            model_feature_cols = pkg["feature_cols"]
            X_aligned          = X_row.reindex(columns=model_feature_cols)
            prob               = calibrated_model.predict_proba(X_aligned)[:, 1][0]
            bookie_odds        = get_bookmaker_odds_for_pick(conn, event_id, target)
            bookie_implied     = (1.0 / bookie_odds) if bookie_odds else None
            ev = (prob * bookie_odds) - 1.0 if bookie_odds else None

            all_predictions.append({
                "event_id"          : event_id,
                "match"             : match_str,
                "tournament"        : fixture["tournament_name"],
                "kickoff"           : fixture["kickoff_time"],
                "target"            : target,
                "model_probability" : round(prob, 4),
                "bookie_odds"       : bookie_odds,
                "bookie_implied"    : round(bookie_implied, 4) if bookie_implied else None,
                "expected_value_pct": round(ev * 100, 2) if ev is not None else None,
                "model_auc"         : model_auc,
                "historical_hit_%"  : round(hit_rate * 100, 1),
                "edge_over_bookie"  : round((prob - bookie_implied)*100, 2) if bookie_implied else None,
            })
        except Exception:
            continue

print(f"\n  Total predictions generated: {len(all_predictions)}")

preds_df = pd.DataFrame(all_predictions)
preds_df = preds_df.sort_values("model_probability", ascending=False)

# V1 filters — no MAX_MODEL_PROB cap, no bookie_odds requirement
value_picks = preds_df[
    (preds_df["model_probability"] >= MIN_MODEL_PROB) &
    (preds_df["expected_value_pct"] >= MIN_EV * 100) &
    (preds_df["model_auc"] >= MIN_AUC_FOR_PICK)
].copy() if "expected_value_pct" in preds_df.columns else pd.DataFrame()

high_conf_picks = preds_df[preds_df["model_probability"] >= 0.70].copy()

pick_pool = value_picks if len(value_picks) >= ACCUM_LEGS else high_conf_picks

accum_picks = []
used_events = set()
for _, row in pick_pool.sort_values(["model_probability","model_auc"],
                                     ascending=[False,False]).iterrows():
    if row["event_id"] not in used_events:
        accum_picks.append(row)
        used_events.add(row["event_id"])
    if len(accum_picks) == ACCUM_LEGS:
        break

print(f"\n{'='*65}")
print(f"  DAILY ACCUMULATOR — {date.today().strftime('%d %B %Y')}")
print(f"{'='*65}")

accum_lines = []
accum_lines.append(f"{'='*65}")
accum_lines.append(f"  DAILY {ACCUM_LEGS}-LEG ACCUMULATOR  (v1)")
accum_lines.append(f"  {date.today().strftime('%A %d %B %Y')}")
accum_lines.append(f"{'='*65}")

combined_odds = 1.0
for i, pick in enumerate(accum_picks, 1):
    odds_str = f"@ {pick['bookie_odds']:.2f}" if pick["bookie_odds"] else "@ [check bookie]"
    ev_str   = (f"  EV: +{pick['expected_value_pct']:.1f}%"
                if pick.get("expected_value_pct") else "")
    line = (f"\n  LEG {i}: {pick['match']}\n"
            f"    Pick    : {pick['target'].replace('target__','').replace('new__','').replace('_',' ').upper()}\n"
            f"    Odds    : {odds_str}\n"
            f"    Model P : {pick['model_probability']*100:.1f}%{ev_str}\n"
            f"    AUC     : {pick['model_auc']:.4f}\n"
            f"    League  : {pick['tournament']}\n"
            f"    KO      : {pick['kickoff']}")
    accum_lines.append(line)
    if pick["bookie_odds"]:
        combined_odds *= pick["bookie_odds"]

accum_lines.append(f"\n{'─'*65}")
if combined_odds > 1.0:
    accum_lines.append(f"  Combined Odds : {combined_odds:.2f}")
accum_lines.append(f"  Based on      : {len(preds_df)} model predictions")
accum_lines.append(f"{'='*65}")

if len(accum_picks) < ACCUM_LEGS:
    accum_lines.append(f"\n  Only {len(accum_picks)} picks met the threshold today.")

accum_text = "\n".join(accum_lines)
print(accum_text)

print(f"\n{'='*65}")
print(f"  ALL HIGH-CONFIDENCE PICKS TODAY")
print(f"{'='*65}")

if not preds_df.empty:
    top = preds_df[preds_df["model_probability"] >= MIN_MODEL_PROB].head(20)
    if not top.empty:
        print(f"\n  {'MATCH':<30} {'PICK':<35} {'PROB':>6}  {'ODDS':>6}  {'EV%':>6}  AUC")
        print(f"  {'-'*95}")
        for _, r in top.iterrows():
            odds_s = f"{r['bookie_odds']:.2f}" if r["bookie_odds"] else "N/A"
            ev_s   = f"{r['expected_value_pct']:.1f}%" if r.get("expected_value_pct") else "N/A"
            print(f"  {str(r['match'])[:28]:<30} "
                  f"{r['target'].replace('target__','').replace('new__','')[:33]:<35} "
                  f"{r['model_probability']*100:>5.1f}%  {odds_s:>6}  {ev_s:>6}  {r['model_auc']:.4f}")

preds_df.to_csv(OUTPUT_CSV, index=False)
with open(ACCUMULATOR_TXT, "w", encoding="utf-8") as f:
    f.write(accum_text)

print(f"\n  Saved: {OUTPUT_CSV}")
print(f"  Saved: {ACCUMULATOR_TXT}")
conn.close()
print(f"\n  Done.\n")

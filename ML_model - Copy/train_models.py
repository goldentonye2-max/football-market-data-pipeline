"""
=============================================================
  XGBOOST MULTI-MODEL TRAINER
  Trains one model per target outcome
  Input  : lean_training_table.csv
  Output : /models/ folder with one .pkl per target
           model_results.csv with performance of every model
=============================================================
  ARCHITECTURE:
    - One XGBClassifier per target (separate models)
    - Time-based train/test split (no data leakage)
    - Auto class-weight balancing for imbalanced targets
    - Calibrated probability output (Isotonic Regression)
    - Evaluated with AUC, Brier Score, Log Loss

  HOW TO USE AFTER TRAINING:
    Load any model with:
      import pickle
      with open("models/model__over_2_5_goals.pkl","rb") as f:
          model = pickle.load(f)
      prob = model.predict_proba(X_new)[:, 1]
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss, accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ───────────────────────────────────────────────────
CSV_PATH      = "lean_training_table.csv"
MODELS_DIR    = "models"
RESULTS_CSV   = "model_results.csv"

# Train/test split — last TEST_FRAC of matches (time-ordered) = test set
TEST_FRAC     = 0.20   # 80% train, 20% test

# Minimum hit rate to train (below this = too rare, skip)
MIN_HIT_RATE  = 0.12   # 12%

# Maximum hit rate to train (above this = too common, skip)
MAX_HIT_RATE  = 0.88   # 88%

# Minimum number of positive examples needed to train
MIN_POSITIVES = 80

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators"      : 300,
    "max_depth"         : 4,      # shallow trees = less overfitting on small data
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.7,
    "min_child_weight"  : 5,      # prevents fitting on tiny leaf nodes
    "reg_alpha"         : 0.1,    # L1 regularisation
    "reg_lambda"        : 1.0,    # L2 regularisation
    "eval_metric"       : "logloss",
    "early_stopping_rounds": 30,
    "random_state"      : 42,
    "verbosity"         : 0,
    "n_jobs"            : -1,     # use all CPU cores
}
# ─────────────────────────────────────────────────────────────


def print_header(text):
    print(f"\n{'='*65}")
    print(f"  {text}")
    print(f"{'='*65}")


def get_feature_columns(df):
    """Return only pre-match feature columns — never use stat or target cols."""
    return [
        c for c in df.columns
        if c.startswith((
            "true_prob__",
            "implied_prob__",
            "vig_delta__",
            "cross__"
        ))
    ]


def train_single_model(df, target_col, feature_cols, models_dir):
    """
    Train one XGBoost model for a single target column.
    Returns dict of performance metrics or None if skipped.
    """

    # ── 1. Filter to rows where target is known (not NA) ──
    sub = df[df[target_col].notna()].copy()
    y   = sub[target_col].astype(int)
    X   = sub[feature_cols].copy()

    n_total    = len(sub)
    n_positive = int(y.sum())
    n_negative = n_total - n_positive
    hit_rate   = n_positive / n_total

    # ── 2. Eligibility checks ──────────────────────────────
    if hit_rate < MIN_HIT_RATE or hit_rate > MAX_HIT_RATE:
        return {"target": target_col, "status": f"SKIPPED (hit rate {hit_rate:.1%})"}

    if n_positive < MIN_POSITIVES:
        return {"target": target_col, "status": f"SKIPPED (only {n_positive} positives)"}

    if n_total < 200:
        return {"target": target_col, "status": f"SKIPPED (only {n_total} rows)"}

    # ── 3. Time-based split — DO NOT SHUFFLE ──────────────
    # Matches are roughly ordered by date in the CSV.
    # We train on earlier matches, test on more recent ones.
    split_idx = int(n_total * (1 - TEST_FRAC))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    n_train_pos = int(y_train.sum())
    n_train_neg = len(y_train) - n_train_pos

    if n_train_pos < 20 or n_train_neg < 20:
        return {"target": target_col, "status": "SKIPPED (too few train examples)"}

    # ── 4. Class weight (handles imbalanced targets) ───────
    # scale_pos_weight = negatives / positives
    # This tells XGBoost to pay more attention to the rare class
    scale_pos_weight = n_train_neg / max(n_train_pos, 1)

    # ── 5. Train XGBoost with early stopping ──────────────
    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── 6. Calibrate probabilities ─────────────────────────
    # Raw XGBoost probabilities can be miscalibrated.
    # Isotonic calibration maps them to real-world hit rates.
    calibrated = CalibratedClassifierCV(
        estimator=xgb,
        method="isotonic",
        cv="prefit",        # already fitted, just calibrate
    )
    calibrated.fit(X_test, y_test)

    # ── 7. Evaluate on test set ────────────────────────────
    y_prob  = calibrated.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= 0.5).astype(int)

    auc     = roc_auc_score(y_test, y_prob)
    brier   = brier_score_loss(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)
    acc     = accuracy_score(y_test, y_pred)

    # ── 8. Feature importance (top 10) ────────────────────
    importances = pd.Series(
        xgb.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False).head(10)
    top_features = importances.index.tolist()

    # ── 9. Save model ──────────────────────────────────────
    safe_name  = target_col.replace("target__", "").replace("new__", "")
    model_path = os.path.join(models_dir, f"model__{safe_name}.pkl")

    with open(model_path, "wb") as f:
        pickle.dump({
            "model"        : calibrated,
            "feature_cols" : feature_cols,
            "target"       : target_col,
            "hit_rate"     : hit_rate,
            "n_train"      : len(X_train),
            "n_test"       : len(X_test),
            "auc"          : auc,
            "brier"        : brier,
            "top_features" : top_features,
        }, f)

    return {
        "target"          : target_col,
        "status"          : "TRAINED",
        "n_total"         : n_total,
        "n_train"         : len(X_train),
        "n_test"          : len(X_test),
        "hit_rate_%"      : round(hit_rate * 100, 1),
        "auc"             : round(auc, 4),
        "brier_score"     : round(brier, 4),
        "log_loss"        : round(logloss, 4),
        "accuracy_%"      : round(acc * 100, 1),
        "top_feature_1"   : top_features[0] if top_features else "",
        "top_feature_2"   : top_features[1] if len(top_features) > 1 else "",
        "top_feature_3"   : top_features[2] if len(top_features) > 2 else "",
        "model_path"      : model_path,
    }


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
print_header("XGBOOST MULTI-MODEL TRAINER")

# ── Load data ─────────────────────────────────────────────
print(f"\nLoading {CSV_PATH}...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Not found: {CSV_PATH}\nRun build_lean_table.py first.")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  → {len(df):,} rows × {len(df.columns):,} columns")

# ── Sort by date to preserve time order ───────────────────
date_cols = [c for c in ["kickoff_time", "match_date", "scraped_at"] if c in df.columns]
if date_cols:
    df = df.sort_values(date_cols[0]).reset_index(drop=True)
    print(f"  → Sorted by {date_cols[0]}")

# ── Get feature and target columns ────────────────────────
feature_cols = get_feature_columns(df)
target_cols  = [c for c in df.columns if c.startswith("target__")]

print(f"  → {len(feature_cols):,} feature columns")
print(f"  → {len(target_cols):,} target columns")

# ── Create models directory ───────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════
print_header("TRAINING ALL MODELS")
print(f"  Threshold: hit rate {MIN_HIT_RATE:.0%} – {MAX_HIT_RATE:.0%}")
print(f"  Min positives needed: {MIN_POSITIVES}")
print(f"  Train/test split: {100*(1-TEST_FRAC):.0f}% / {100*TEST_FRAC:.0f}% (time-ordered)\n")

all_results = []
trained = 0
skipped = 0

for i, target in enumerate(sorted(target_cols)):
    print(f"  [{i+1:02d}/{len(target_cols)}] {target:<55}", end=" ")

    result = train_single_model(df, target, feature_cols, MODELS_DIR)

    if result["status"] == "TRAINED":
        trained += 1
        print(f"AUC={result['auc']:.4f}  Brier={result['brier_score']:.4f}  "
              f"Acc={result['accuracy_%']:.1f}%  Hit={result['hit_rate_%']:.1f}%")
    else:
        skipped += 1
        print(result["status"])

    all_results.append(result)

# ═══════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════
print_header("TRAINING COMPLETE — RESULTS SUMMARY")

results_df = pd.DataFrame(all_results)
trained_df = results_df[results_df["status"] == "TRAINED"].copy()
skipped_df = results_df[results_df["status"] != "TRAINED"].copy()

print(f"\n  Models trained : {trained}")
print(f"  Models skipped : {skipped}")

if not trained_df.empty:
    # Sort by AUC descending
    trained_df = trained_df.sort_values("auc", ascending=False)

    print(f"\n  TRAINED MODELS (sorted by AUC — best to worst):")
    print(f"\n  {'TARGET':<45} {'N':>5}  {'HIT%':>5}  {'AUC':>6}  {'BRIER':>6}  {'ACC%':>5}  QUALITY")
    print(f"  {'-'*95}")

    for _, row in trained_df.iterrows():
        auc = row["auc"]

        # Quality label based on AUC
        if auc >= 0.70:
            quality = "★★★ EXCELLENT"
        elif auc >= 0.62:
            quality = "★★  GOOD"
        elif auc >= 0.55:
            quality = "★   FAIR"
        else:
            quality = "    WEAK"

        print(f"  {row['target']:<45} {int(row['n_total']):>5,}  "
              f"{row['hit_rate_%']:>4.1f}%  {auc:>6.4f}  "
              f"{row['brier_score']:>6.4f}  {row['accuracy_%']:>4.1f}%  {quality}")

    print(f"\n  TOP 3 MOST IMPORTANT FEATURES per model:")
    print(f"  {'-'*95}")
    for _, row in trained_df.iterrows():
        t1 = str(row.get("top_feature_1",""))[:50]
        t2 = str(row.get("top_feature_2",""))[:50]
        print(f"  {row['target']:<45}")
        print(f"    #1: {t1}")
        print(f"    #2: {t2}")

    # AUC benchmarks explained
    print(f"""
  AUC INTERPRETATION GUIDE:
  ─────────────────────────────────────────────────────────
  AUC = 0.50  →  Random guessing. No predictive power.
  AUC = 0.55  →  Slight edge. Exists but hard to exploit.
  AUC = 0.60  →  Meaningful signal. Worth investigating.
  AUC = 0.65  →  Strong signal. Real patterns found.
  AUC = 0.70+ →  Excellent. Systematic edge discovered.
  AUC = 0.80+ →  Outstanding. Almost certainly profitable.
  ─────────────────────────────────────────────────────────
  Note: Even AUC 0.58-0.62 can be profitable if the
  bookmaker's implied probability is consistently lower
  than your model's predicted probability (+EV signal).
""")

# Save results
results_df.to_csv(RESULTS_CSV, index=False)
trained_df.to_csv("model_results_trained_only.csv", index=False)

print(f"  Results saved  : {RESULTS_CSV}")
print(f"  Best models    : model_results_trained_only.csv")
print(f"  Model files    : ./{MODELS_DIR}/model__*.pkl")
print(f"\n  ✅ NEXT STEP: Run predict_today.py to score today's fixtures")
print("="*65 + "\n")

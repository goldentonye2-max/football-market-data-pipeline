"""
=============================================================
  FULL MARKET DISCOVERY
  Shows every distinct market available in the database
  Groups by market_name so you see the full picture
  Saves two CSVs:
    1. markets_summary.csv    — one row per market_name
    2. markets_full_detail.csv — all options, top 200 by coverage
=============================================================
"""

import sqlite3
import pandas as pd

conn = sqlite3.connect("sportybet.db")

# ── PART 1: One row per market name ───────────────────────
# Shows every distinct market that exists, sorted by coverage
print("\n" + "="*75)
print("  PART 1: ALL DISTINCT MARKET NAMES (sorted by fixture coverage)")
print("="*75)

market_summary = pd.read_sql_query("""
    SELECT
        market_name,
        COUNT(DISTINCT event_id)            AS fixture_count,
        COUNT(DISTINCT outcome_desc)        AS num_outcomes,
        COUNT(DISTINCT specifier)           AS num_specifiers,
        ROUND(MIN(probability), 4)          AS min_true_prob,
        ROUND(MAX(probability), 4)          AS max_true_prob,
        GROUP_CONCAT(DISTINCT specifier)    AS specifier_examples
    FROM odds
    WHERE odds IS NOT NULL
      AND probability IS NOT NULL
    GROUP BY market_name
    ORDER BY fixture_count DESC
""", conn)

print(f"\n  Total distinct market names: {len(market_summary)}\n")
print(f"  {'MARKET NAME':<45} {'FIXTURES':>8}  {'OUTCOMES':>8}  {'SPECIFIERS':>10}")
print(f"  {'-'*75}")

for _, row in market_summary.iterrows():
    print(f"  {str(row['market_name']):<45} "
          f"{int(row['fixture_count']):>8,}  "
          f"{int(row['num_outcomes']):>8}  "
          f"{int(row['num_specifiers']):>10}")

market_summary.to_csv("markets_summary.csv", index=False)
print(f"\n  Saved: markets_summary.csv")


# ── PART 2: Sample outcomes for each market name ──────────
# For each market, shows up to 3 example outcome options
print("\n" + "="*75)
print("  PART 2: SAMPLE OUTCOMES PER MARKET (up to 3 per market)")
print("="*75)

all_options = pd.read_sql_query("""
    SELECT
        market_name,
        specifier,
        outcome_desc,
        COUNT(DISTINCT event_id)    AS fixture_count,
        ROUND(AVG(probability), 4)  AS avg_true_prob,
        ROUND(AVG(1.0/odds), 4)     AS avg_implied_prob
    FROM odds
    WHERE odds IS NOT NULL
      AND probability IS NOT NULL
    GROUP BY market_name, specifier, outcome_desc
    ORDER BY fixture_count DESC, market_name, outcome_desc
""", conn)

# Show top 3 outcomes per market_name
sampled = (all_options
           .groupby("market_name", group_keys=False)
           .apply(lambda g: g.nlargest(3, "fixture_count"))
           .reset_index(drop=True))

print(f"\n  {'MARKET NAME':<40} {'SPECIFIER':<25} {'OUTCOME':<20} {'FIXTURES':>8}")
print(f"  {'-'*97}")

prev_market = None
for _, row in sampled.iterrows():
    market = str(row["market_name"])
    spec   = str(row["specifier"]) if pd.notna(row["specifier"]) else ""
    out    = str(row["outcome_desc"])
    cnt    = int(row["fixture_count"])

    # Print blank line between different markets for readability
    if market != prev_market and prev_market is not None:
        print()
    prev_market = market

    print(f"  {market:<40} {spec:<25} {out:<20} {cnt:>8,}")


# ── PART 3: Full detail CSV (top 200 by coverage) ─────────
top200 = all_options.head(200)
top200.to_csv("markets_full_detail.csv", index=False)

print(f"\n\n{'='*75}")
print(f"  PART 3: FILES SAVED")
print(f"{'='*75}")
print(f"  markets_summary.csv      — {len(market_summary)} distinct market names")
print(f"  markets_full_detail.csv  — top 200 market options by fixture count")
print(f"  Total unique options     — {len(all_options):,}")

# ── PART 4: Markets that appear in 500+ fixtures ──────────
# These are the most reliable candidates for ML targets
print(f"\n  MARKETS APPEARING IN 500+ FIXTURES (most trainable):")
print(f"  {'-'*50}")
reliable = market_summary[market_summary["fixture_count"] >= 500]
for _, row in reliable.iterrows():
    print(f"  {str(row['market_name']):<45} {int(row['fixture_count']):>6,} fixtures")

conn.close()
print(f"\n  Done.\n")
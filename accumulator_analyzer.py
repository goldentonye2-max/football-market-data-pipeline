#!/usr/bin/env python3
"""
accumulator_analyzer.py
========================
Local math check for a set of model picks BEFORE you place anything.

Does NOT touch the network. Does NOT touch sporty.db. Does NOT call any
bookmaker API or generate a booking code. It only re-derives EV, drops
negative-EV legs, and shows what happens to the combined probability /
odds / edge as legs get added to a slip — so you can see the shape of
the tradeoff before deciding how many legs to actually place manually.

USAGE:
  1. Paste your model's pick list into PICKS_TEXT below (same table
     format your model already prints — today's list is pre-loaded as
     the working example).
  2. python accumulator_analyzer.py

OUTPUT (console only):
  - negative-EV legs dropped, with reasons
  - concentration warnings (same probability bucket / same pick-type)
  - outlier-EV warning (a single leg's edge large enough that it's more
    likely a small-sample artifact than a real one)
  - combined probability / combined odds / fair odds / implied edge for
    ALL surviving legs together AND for top-N subsets (N=2..6), so you
    can compare a big slip against a small one side by side
"""

import re
from dataclasses import dataclass
from collections import Counter

# ══════════════════════════════════════════════════════════════════════
# 1. PASTE YOUR PICKS HERE — replace this block with a fresh list any
#    time. Same columns your model already prints: MATCH / PICK / PROB /
#    ODDS / EV% / AUC. The header row and the --- separator are ignored
#    automatically, so you can paste the whole block exactly as printed.
# ══════════════════════════════════════════════════════════════════════

PICKS_TEXT = """
MATCH                          PICK                                  PROB    ODDS     EV%  AUC
-----------------------------------------------------------------------------------------------
Melbourne Knights vs Western   home_win                             78.6%    1.23   -3.4%  0.7276
MFK Tatran Liptovsky Mikulas   home_win                             78.6%    1.34    5.3%  0.7276
Czarni Sosnowiec vs UKS SMS    home_win                             78.6%    1.39    9.2%  0.7276
Colwyn Bay vs Trefelin BGC     home_win                             78.6%    1.20   -5.7%  0.7276
Red Star FC vs Amiens          home_win                             75.0%    1.39    4.2%  0.7276
Monaco vs Cercle Brugge        home_win                             75.0%    1.34    0.5%  0.7276
Broadbeach United vs Logan L   home_win                             75.0%    1.42    6.5%  0.7276
Gamle Oslo FK vs FK Union Ca   home_win                             75.0%    1.46    9.5%  0.7276
Suwon WFC vs Sejong Sportsto   home_win                             75.0%    1.24   -7.0%  0.7276
Essendon Royals SC vs Altona   home_win                             75.0%    1.44    8.0%  0.7276
Malmo FF vs Piteaa IF DFF      home_win                             75.0%    1.35    1.2%  0.7276
FK Frydek-Mistek vs SK Hrani   home_win                             75.0%    1.38    3.5%  0.7276
Linkopings FC vs Orebro SK S   home_win                             75.0%    1.29   -3.2%  0.7276
Dundalk FC vs Sligo Rovers F   home_win                             75.0%    1.22   -8.5%  0.7276
South Hobart FC 2 vs Taroona   home_win                             75.0%    1.54   15.5%  0.7276
SV Tillmitsch vs SC Furstenf   home_win                             75.0%    1.70   27.5%  0.7276
IK Oddevold vs Norrby IF       home_win                             75.0%    1.36    2.0%  0.7276
Vietnam vs Singapore           ht_home_win                          73.7%    1.38    1.7%  0.6554
The Strongest vs Club Aurora   ht_home_win                          73.7%    1.85   36.3%  0.6554
Juventus vs Nice               ht_home_win                          73.7%    1.94   43.0%  0.6554
"""

# A single leg's EV above this triggers a sample-size sanity flag rather
# than being treated as simply "the best pick".
OUTLIER_EV_PCT = 20.0

# ══════════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Pick:
    match: str
    pick: str
    prob: float     # 0-1
    odds: float
    ev_pct: float
    auc: float


LINE_RE = re.compile(
    r'^\s*(.+?)\s{2,}(\S+)\s+(\d+(?:\.\d+)?)%\s+(\d+\.\d+)\s+'
    r'([+-]?\d+(?:\.\d+)?)%\s+(\d+\.\d+)\s*$'
)


def parse_picks(text: str) -> list[Pick]:
    picks = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith('-') or s.upper().startswith('MATCH'):
            continue
        m = LINE_RE.match(line)
        if not m:
            print(f"  (skipped — couldn't parse): {s[:70]}")
            continue
        match, pick, prob, odds, ev, auc = m.groups()
        picks.append(Pick(match.strip(), pick, float(prob) / 100,
                           float(odds), float(ev), float(auc)))
    return picks


# ══════════════════════════════════════════════════════════════════════
# COMBINED SLIP MATH
# ══════════════════════════════════════════════════════════════════════

def combined_stats(picks: list[Pick]) -> dict:
    prob = 1.0
    odds = 1.0
    for p in picks:
        prob *= p.prob
        odds *= p.odds
    fair_odds = 1 / prob if prob > 0 else float("inf")
    edge_pct = (odds / fair_odds - 1) * 100 if fair_odds not in (0, float("inf")) else 0.0
    return {
        "n": len(picks),
        "prob_pct": prob * 100,
        "odds": odds,
        "fair_odds": fair_odds,
        "edge_pct": edge_pct,
    }


def print_stats_row(label: str, s: dict):
    print(f"  {label:<10}{s['prob_pct']:>10.2f}%{s['odds']:>16,.1f}"
          f"{s['fair_odds']:>16,.1f}{s['edge_pct']:>+14.1f}%")


# ══════════════════════════════════════════════════════════════════════
# WARNINGS
# ══════════════════════════════════════════════════════════════════════

def concentration_warnings(picks: list[Pick]):
    if not picks:
        return

    buckets = Counter(round(p.prob, 2) for p in picks)
    val, cnt = buckets.most_common(1)[0]
    if cnt >= max(3, len(picks) * 0.5):
        print(f"\n  ⚠ concentration: {cnt}/{len(picks)} legs share ~{val*100:.0f}% "
              f"probability. That's almost certainly the model repeating one\n"
              f"    bucketed estimate rather than pricing each fixture on its "
              f"own — treat these legs as correlated, not independent.")

    types = Counter(p.pick for p in picks)
    val, cnt = types.most_common(1)[0]
    if cnt >= max(3, len(picks) * 0.6):
        print(f"  ⚠ concentration: {cnt}/{len(picks)} legs are the same "
              f"pick-type ('{val}'). If that model has any systematic bias,\n"
              f"    it hits all {cnt} of them at once, not independently.")

    outliers = [p for p in picks if p.ev_pct >= OUTLIER_EV_PCT]
    if outliers:
        print(f"\n  ⚠ outlier EV (≥{OUTLIER_EV_PCT:.0f}%) — verify the sample "
              f"size behind these before trusting the number;\n"
              f"    small samples are the usual cause of an edge this large:")
        for p in outliers:
            print(f"      {p.match[:35]:<35} {p.pick:<15} EV={p.ev_pct:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════

def report(picks: list[Pick]):
    print("=" * 78)
    print(f"  Parsed {len(picks)} picks")
    print("=" * 78)

    negative = [p for p in picks if p.ev_pct <= 0]
    positive = sorted([p for p in picks if p.ev_pct > 0],
                       key=lambda p: p.ev_pct, reverse=True)

    if negative:
        print(f"\n  Dropped {len(negative)} negative/zero-EV leg(s):")
        for p in negative:
            print(f"    {p.match[:35]:<35} {p.pick:<15} "
                  f"prob={p.prob*100:5.1f}%  odds={p.odds:5.2f}  EV={p.ev_pct:+6.1f}%")

    print(f"\n  {len(positive)} positive-EV leg(s) remain, sorted by EV:\n")
    for p in positive:
        print(f"    {p.match[:35]:<35} {p.pick:<15} "
              f"prob={p.prob*100:5.1f}%  odds={p.odds:5.2f}  EV={p.ev_pct:+6.1f}%")

    concentration_warnings(positive)

    print("\n" + "=" * 78)
    print("  COMBINED SLIP MATH  (assumes every leg is independent — see note)")
    print("=" * 78)
    print(f"\n  {'Legs':<10}{'Win prob':>11}{'Combined odds':>16}"
          f"{'Fair odds':>16}{'Implied edge':>15}")

    print_stats_row(f"ALL {len(positive)}", combined_stats(positive))
    print()
    for n in range(2, min(6, len(positive)) + 1):
        print_stats_row(f"top {n}", combined_stats(positive[:n]))

    print("\n  Note: combined edge multiplies (1 + ev_i) across every leg, it does\n"
          "  NOT average. Stacking legs from the same probability bucket or the\n"
          "  same pick-type — especially any flagged above as outlier-EV — means\n"
          "  you're compounding the least-trusted numbers, not diversifying away\n"
          "  from them. Fewer legs = the compounding is less punishing if any one\n"
          "  probability estimate turns out to be wrong.")


if __name__ == "__main__":
    picks = parse_picks(PICKS_TEXT)
    report(picks)

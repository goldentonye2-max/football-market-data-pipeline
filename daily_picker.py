"""
daily_picker.py
Applies the analysis findings to today's (or any date's) fixture database
and outputs the best candidates for each confirmed market signal.

CONFIRMED SIGNALS (from analysis_report.txt):
  A — Corners ≥ 9:  prob_ou25_ov ≥ 0.55 AND prob_btts_yes ≥ 0.55  → 82% hit rate
  B — Over 2.5:     prob_ou25_ov in 0.45–0.65 (true edge window)
  C — BTTS Yes:     prob_btts_yes ≥ 0.50 (market consistently underprices 2+ goals)
  D — Over 1.5:     prob_ou15_ov ≥ 0.80 (91.7% actual hit rate at this level)

USAGE:
  python daily_picker.py              # today's fixtures
  python daily_picker.py 2026-07-18   # specific date
"""

import sqlite3
import os
import sys
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sportybet.db")


def load_fixtures_with_probs(conn, target_date: str) -> list[dict]:
    """Pull today's fixtures with their key pre-match probabilities."""

    # Key markets we need
    MARKETS = {
        "prob_home":     ("1",  "",           "Home"),
        "prob_draw":     ("1",  "",           "Draw"),
        "prob_away":     ("1",  "",           "Away"),
        "prob_ou15_ov":  ("18", "total=1.5",  "Over 1.5"),
        "prob_ou25_ov":  ("18", "total=2.5",  "Over 2.5"),
        "prob_ou35_ov":  ("18", "total=3.5",  "Over 3.5"),
        "prob_btts_yes": ("29", "",           "Yes"),
        "prob_dnb_home": ("11", "",           "Home"),
        "prob_dnb_away": ("11", "",           "Away"),
    }

    ODDS = {
        "odds_home":     ("1",  "",           "Home"),
        "odds_draw":     ("1",  "",           "Draw"),
        "odds_away":     ("1",  "",           "Away"),
        "odds_ou15_ov":  ("18", "total=1.5",  "Over 1.5"),
        "odds_ou25_ov":  ("18", "total=2.5",  "Over 2.5"),
        "odds_ou35_ov":  ("18", "total=3.5",  "Over 3.5"),
        "odds_btts_yes": ("29", "",           "Yes"),
    }

    # Get fixture list for this date
    fixtures = conn.execute("""
        SELECT event_id, home_team, away_team, tournament_name,
               category_name, kickoff_time
        FROM   fixtures
        WHERE  DATE(kickoff_time) = ?
        ORDER  BY kickoff_time
    """, (target_date,)).fetchall()

    if not fixtures:
        return []

    # Build probability + odds lookup per event_id
    event_ids = [r[0] for r in fixtures]
    placeholders = ",".join("?" * len(event_ids))

    probs = {}
    odds  = {}

    for col, (mid, spec, outcome) in MARKETS.items():
        rows = conn.execute(f"""
            SELECT event_id, probability
            FROM   odds
            WHERE  event_id IN ({placeholders})
              AND  market_id = ?
              AND  COALESCE(specifier,'') = ?
              AND  outcome_desc = ?
        """, (*event_ids, mid, spec, outcome)).fetchall()
        for eid, val in rows:
            probs.setdefault(eid, {})[col] = val

    for col, (mid, spec, outcome) in ODDS.items():
        rows = conn.execute(f"""
            SELECT event_id, odds
            FROM   odds
            WHERE  event_id IN ({placeholders})
              AND  market_id = ?
              AND  COALESCE(specifier,'') = ?
              AND  outcome_desc = ?
        """, (*event_ids, mid, spec, outcome)).fetchall()
        for eid, val in rows:
            odds.setdefault(eid, {})[col] = val

    result = []
    for row in fixtures:
        eid, home, away, tournament, category, kickoff = row
        p = probs.get(eid, {})
        o = odds.get(eid, {})

        # Symmetry
        dnb_h = p.get("prob_dnb_home")
        dnb_a = p.get("prob_dnb_away")
        symmetry = round(1 - abs(dnb_h - dnb_a), 4) if (dnb_h and dnb_a) else None

        result.append({
            "event_id":   eid,
            "home_team":  home,
            "away_team":  away,
            "tournament": tournament,
            "category":   category,
            "kickoff":    kickoff,
            "symmetry":   symmetry,
            **{k: round(v, 4) if v else None for k, v in p.items()},
            **{k: round(v, 4) if v else None for k, v in o.items()},
        })

    return result


def signal_a_corners(f: dict) -> bool:
    """Both Over 2.5 AND BTTS ≥ 0.55 → corners ≥ 9 at 82%."""
    p1 = f.get("prob_ou25_ov")
    p2 = f.get("prob_btts_yes")
    return bool(p1 and p2 and p1 >= 0.55 and p2 >= 0.55)


def signal_b_over25(f: dict) -> bool:
    """Over 2.5 probability in the proven edge window 0.45–0.65."""
    p = f.get("prob_ou25_ov")
    return bool(p and 0.45 <= p <= 0.65)


def signal_c_btts(f: dict) -> bool:
    """BTTS Yes probability ≥ 0.50 (market underprices 2+ goals)."""
    p = f.get("prob_btts_yes")
    return bool(p and p >= 0.50)


def signal_d_over15(f: dict) -> bool:
    """Over 1.5 probability ≥ 0.80 (91.7% actual hit rate)."""
    p = f.get("prob_ou15_ov")
    return bool(p and p >= 0.80)


def format_fixture(f: dict, signals: list[str]) -> str:
    kickoff = f["kickoff"][11:16] if f.get("kickoff") else "?"
    ou25    = f.get("prob_ou25_ov")
    btts    = f.get("prob_btts_yes")
    ou15    = f.get("prob_ou15_ov")
    sym     = f.get("symmetry")

    odds_ou25 = f.get("odds_ou25_ov")
    odds_btts = f.get("odds_btts_yes")
    odds_ou15 = f.get("odds_ou15_ov")
    odds_home = f.get("odds_home")
    odds_draw = f.get("odds_draw")
    odds_away = f.get("odds_away")

    sig_str = " | ".join(signals)

    line = (
        f"  {kickoff}  {f['home_team']:<28} vs {f['away_team']:<28}\n"
        f"           {f['tournament']} ({f['category']})\n"
        f"           [SIGNALS: {sig_str}]\n"
    )

    detail_parts = []
    if ou25:
        detail_parts.append(f"O2.5 p={ou25:.2f} @ {odds_ou25 or '?'}")
    if btts:
        detail_parts.append(f"BTTS p={btts:.2f} @ {odds_btts or '?'}")
    if ou15:
        detail_parts.append(f"O1.5 p={ou15:.2f} @ {odds_ou15 or '?'}")
    if sym:
        detail_parts.append(f"sym={sym:.2f}")

    odds_parts = []
    if odds_home:
        odds_parts.append(f"H:{odds_home}")
    if odds_draw:
        odds_parts.append(f"D:{odds_draw}")
    if odds_away:
        odds_parts.append(f"A:{odds_away}")

    if detail_parts:
        line += f"           Probs : {' | '.join(detail_parts)}\n"
    if odds_parts:
        line += f"           1X2   : {' / '.join(odds_parts)}\n"

    return line


def main():
    target_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()

    conn = sqlite3.connect(DB_PATH)
    fixtures = load_fixtures_with_probs(conn, target_date)
    conn.close()

    lines = []

    def out(text=""):
        lines.append(text)
        print(text)

    if not fixtures:
        out(f"No fixtures found for {target_date}.")
        return

    bar = "=" * 75

    signal_a = [(f, ["A-CORNERS≥9"]) for f in fixtures if signal_a_corners(f)]
    signal_b = [(f, ["B-OVER2.5"])   for f in fixtures if signal_b_over25(f) and not signal_a_corners(f)]
    signal_c = [(f, ["C-BTTS"])      for f in fixtures if signal_c_btts(f) and not signal_a_corners(f)]
    signal_d = [(f, ["D-OVER1.5"])   for f in fixtures if signal_d_over15(f)]

    multi = {}
    for f in fixtures:
        sigs = []
        if signal_a_corners(f): sigs.append("A-CORNERS≥9")
        if signal_b_over25(f):  sigs.append("B-OVER2.5")
        if signal_c_btts(f):    sigs.append("C-BTTS")
        if signal_d_over15(f):  sigs.append("D-OVER1.5")
        if len(sigs) >= 2:
            multi[f["event_id"]] = (f, sigs)

    out(f"\n{bar}")
    out(f"  DAILY PICKS  —  {target_date}")
    out(f"  Total fixtures in database: {len(fixtures)}")
    out(bar)

    out(f"\n{'─'*75}")
    out(f"  SIGNAL A — CORNERS ≥ 9  (82% hit rate when both conditions met)")
    out(f"  Both: Over 2.5 prob ≥ 0.55 AND BTTS Yes prob ≥ 0.55")
    out(f"  {len(signal_a)} fixture(s) qualify")
    out(f"{'─'*75}")
    for f, sigs in signal_a:
        out(format_fixture(f, sigs))

    out(f"\n{'─'*75}")
    out(f"  SIGNAL B — OVER 2.5  (genuine +12% edge in 0.45–0.65 range)")
    out(f"  Over 2.5 probability between 0.45 and 0.65, not already in Signal A")
    out(f"  {len(signal_b)} fixture(s) qualify")
    out(f"{'─'*75}")
    for f, sigs in signal_b:
        out(format_fixture(f, sigs))

    out(f"\n{'─'*75}")
    out(f"  SIGNAL C — BTTS YES  (market underprices 2+ goals consistently)")
    out(f"  BTTS Yes probability ≥ 0.50, not already in Signal A")
    out(f"  {len(signal_c)} fixture(s) qualify")
    out(f"{'─'*75}")
    for f, sigs in signal_c:
        out(format_fixture(f, sigs))

    out(f"\n{'─'*75}")
    out(f"  SIGNAL D — OVER 1.5  (91.7% actual rate when prob ≥ 0.80)")
    out(f"  Over 1.5 probability ≥ 0.80")
    out(f"  {len(signal_d)} fixture(s) qualify")
    out(f"{'─'*75}")
    for f, sigs in signal_d:
        out(format_fixture(f, sigs))

    out(f"\n{'─'*75}")
    out(f"  MULTI-SIGNAL — fixtures hitting 2 or more signals simultaneously")
    out(f"  {len(multi)} fixture(s)")
    out(f"{'─'*75}")
    for eid, (f, sigs) in multi.items():
        out(format_fixture(f, sigs))

    out(f"\n{bar}")
    out(f"  Summary: A={len(signal_a)}  B={len(signal_b)}  "
        f"C={len(signal_c)}  D={len(signal_d)}  Multi={len(multi)}")
    out(bar)
    out(f"\n  Next step: run   python booking_builder.py   to generate accumulator slips.")

    # Save to file
    picks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picks")
    os.makedirs(picks_dir, exist_ok=True)
    out_path = os.path.join(picks_dir, f"picks_{target_date}.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
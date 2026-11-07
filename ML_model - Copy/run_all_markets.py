"""
=============================================================
  run_all_markets.py
  Runs pattern_miner.py for a list of (market, odds band) jobs,
  keeps each run's report and dataset under its own name (so the
  next run does not overwrite them), and prints ONE summary table.

  USAGE (inside ML_model - Copy, next to pattern_miner.py):
      python run_all_markets.py
      python run_all_markets.py --perms 10        (faster)
      python run_all_markets.py --only away_win   (one market)

  Edit JOBS below to change markets or bands. Bands are
  (min_odds exclusive, max_odds inclusive); None = no limit.
  Keep bands narrow enough that the price barely varies inside
  them, otherwise the market's own price explains the results.

  Output: batch_summary.txt  +  batch/<market>_<lo>_<hi>_report.txt
=============================================================
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

JOBS = [
    ("ht_over_1_5", 1.80, 2.00),
    ("ht_over_1_5", 2.00, 2.30),
    ("over_2_5",    1.60, 1.80),
    ("over_2_5",    1.80, 2.00),
    ("btts_yes",    1.60, 1.80),
    ("btts_yes",    1.80, 2.00),
    ("away_win",    1.50, 1.70),
    ("away_win",    1.90, 2.30),
    ("home_win",    1.50, 1.70),
    ("home_win",    1.90, 2.30),
    ("ht_away_win", 3.00, 4.00),
    ("over_1_5",    1.25, 1.40),
]

OUT_DIR = "batch"


def grab(pattern, text, cast=float, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except ValueError:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--only", type=str, default=None, help="run only this market key")
    args = ap.parse_args()

    if not os.path.exists("pattern_miner.py"):
        sys.exit("ERROR: run this from the folder that holds pattern_miner.py")
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = [j for j in JOBS if not args.only or j[0] == args.only]
    rows = []
    for i, (target, lo, hi) in enumerate(jobs, 1):
        tag = f"{target}_{lo}_{hi}"
        print(f"[{i}/{len(jobs)}] {tag} ...", flush=True)
        cmd = [sys.executable, "pattern_miner.py", "--target", target,
               "--perms", str(args.perms)]
        if lo is not None:
            cmd += ["--min-odds", str(lo)]
        if hi is not None:
            cmd += ["--max-odds", str(hi)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        rep = f"pattern_report_{target}.txt"
        dat = f"pattern_dataset_{target}.csv"
        if proc.returncode != 0 or not os.path.exists(rep):
            msg = (proc.stdout + proc.stderr).strip().splitlines()
            rows.append((tag, None, "skipped: " + (msg[-1][:70] if msg else "error")))
            continue
        text = open(rep, encoding="utf-8").read()
        shutil.copy(rep, os.path.join(OUT_DIR, f"{tag}_report.txt"))
        if os.path.exists(dat):
            shutil.copy(dat, os.path.join(OUT_DIR, f"{tag}_dataset.csv"))

        games = grab(r"Games\s+(\d+)\s+Won", text, int)
        winp = grab(r"Won \d+ \(([\d.]+)%\)", text)
        be = grab(r"Break-even win% at these odds:\s*([\d.]+)%", text)
        q = grab(r"Survive FDR \(q < 0.10\)\s*:\s*(\d+)", text, int)
        auc = grab(r"Test AUC \(real labels\)\s*:\s*([\d.]+)", text)
        pauc = grab(r"Market's own price alone AUC\s*:\s*([\d.]+)", text)
        p = grab(r"p-value\s*:\s*([\d.]+)", text)
        rules = re.findall(r"\s(\d+)\s+[\d.]+%\s+([+-][\d.]+)%\s*$", text, re.M)
        rules_pos = sum(1 for n, r in rules if int(n) >= 100 and float(r) > 0)
        verdict = "SIGNAL" if "SIGNAL:" in text else (
            "WEAK/MIXED" if "WEAK/MIXED" in text else "NO PATTERN")
        rows.append((tag, dict(games=games, winp=winp, be=be, q=q, auc=auc,
                               pauc=pauc, p=p, rp=rules_pos), verdict))

    L = []
    L.append("=" * 100)
    L.append("  BATCH SUMMARY - odds-sheet pattern miner")
    L.append("=" * 100)
    L.append(f"  {'Market_lo_hi':<24} {'Games':>6} {'Win%':>6} {'BE%':>6} {'FDRcols':>7} "
             f"{'ML AUC':>7} {'PriceAUC':>8} {'p':>6} {'RulesOK':>7}  Verdict")
    L.append("  " + "-" * 96)
    for tag, s, v in rows:
        if s is None:
            L.append(f"  {tag:<24} {v}")
            continue
        f = lambda x, fmt: (fmt % x) if x is not None else "-"
        L.append(f"  {tag:<24} {f(s['games'], '%d'):>6} {f(s['winp'], '%.1f'):>6} "
                 f"{f(s['be'], '%.1f'):>6} {f(s['q'], '%d'):>7} {f(s['auc'], '%.3f'):>7} "
                 f"{f(s['pauc'], '%.3f'):>8} {f(s['p'], '%.3f'):>6} {s['rp']:>7}  {v}")
    L.append("")
    L.append("  How to read: ML AUC must beat PriceAUC clearly (and p < 0.05) to mean anything.")
    L.append("  FDRcols = odds columns that survive the many-columns correction (large numbers in a wide")
    L.append("  band usually just mean the columns repeat the price). RulesOK = hold-out rules with")
    L.append("  100+ games and positive ROI. With 12 jobs, expect about 1 'significant' result by luck.")
    L.append("=" * 100)
    out = "\n".join(L)
    print("\n" + out)
    open("batch_summary.txt", "w", encoding="utf-8").write(out)
    print("\nSaved batch_summary.txt and the per-run reports in ./batch")


if __name__ == "__main__":
    main()
"""
grade_all.py
Grades every ungraded all_picks_YYYY-MM-DD.csv in this folder
by running evaluate_picks.py --picks-csv on each one.

Usage:
    python grade_all.py
    python grade_all.py --dry-run   (just shows what would be graded)

Place this file next to evaluate_picks.py and run from ML_model - Copy.
"""

import argparse
import glob
import os
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be graded without running anything")
    ap.add_argument("--skip-today", action="store_true",
                    help="Skip today's file (games may not be finished yet)")
    args = ap.parse_args()

    from datetime import date
    today = date.today().isoformat()

    all_files = sorted(glob.glob("all_picks_????-??-??.csv"))
    if not all_files:
        sys.exit("No all_picks_YYYY-MM-DD.csv files found here.")

    to_grade = []
    for f in all_files:
        day = f.replace("all_picks_", "").replace(".csv", "")
        graded = f"all_picks_{day}_graded.csv"
        if os.path.exists(graded):
            print(f"  SKIP  {day}  (already graded)")
            continue
        if args.skip_today and day == today:
            print(f"  SKIP  {day}  (today — games may not be finished)")
            continue
        to_grade.append((day, f, graded))

    if not to_grade:
        print("\nAll files are already graded.")
        return

    print(f"\n{len(to_grade)} file(s) to grade:")
    for day, f, _ in to_grade:
        print(f"    {f}")

    if args.dry_run:
        print("\n[DRY RUN] — nothing was run. Remove --dry-run to grade them.")
        return

    print()
    ok = fail = 0
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"   # fix for Windows cp1252 / arrow chars
    for day, f, graded in to_grade:
        print(f"  GRADING  {day} ...", flush=True)
        r = subprocess.run(
            [sys.executable, "evaluate_picks.py", "--picks-csv", f],
            capture_output=True, text=True, encoding="utf-8", env=env
        )
        if r.returncode != 0 or not os.path.exists(graded):
            print(f"    ERROR:")
            # show last few lines of output for diagnosis
            out = (r.stdout + r.stderr).strip().splitlines()
            for line in out[-5:]:
                print(f"      {line}")
            fail += 1
        else:
            # show the one-line win rate summary from the output
            for line in r.stdout.splitlines():
                if "Win rate" in line or "Settled" in line or "graded file" in line:
                    print(f"    {line.strip()}")
            print(f"    -> saved {graded}")
            ok += 1

    print(f"\n  Done. {ok} graded, {fail} failed.")
    if ok:
        print("\n  Now run the backtest:")
        print("      python top2_backtest.py --k 2")
        print("      python top2_backtest.py --k 1")

if __name__ == "__main__":
    main()
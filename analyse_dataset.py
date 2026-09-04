import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sportybet.db")

conn = sqlite3.connect(DB_PATH)

print("=" * 70)
print("WORKING DATASET — fixtures with BOTH odds AND flashscore stats")
print("=" * 70)

# How many fixtures have both odds and stats
full = conn.execute("""
    SELECT COUNT(DISTINCT f.event_id)
    FROM   fixtures f
    JOIN   odds o      ON o.event_id = f.event_id
    JOIN   match_links ml ON ml.event_id = f.event_id
    JOIN   flashscore_stats fs ON fs.fs_match_id = ml.fs_match_id
    JOIN   results r   ON r.event_id = f.event_id
    WHERE  r.status = 'finished'
""").fetchone()[0]

print(f"\nFixtures with odds + stats + finished result: {full}")

print("\n--- By tournament (top 20) ---")
rows = conn.execute("""
    SELECT f.tournament_name, f.category_name, COUNT(DISTINCT f.event_id) as n
    FROM   fixtures f
    JOIN   odds o      ON o.event_id = f.event_id
    JOIN   match_links ml ON ml.event_id = f.event_id
    JOIN   flashscore_stats fs ON fs.fs_match_id = ml.fs_match_id
    JOIN   results r   ON r.event_id = f.event_id
    WHERE  r.status = 'finished'
    GROUP  BY f.tournament_name
    ORDER  BY n DESC
    LIMIT  20
""").fetchall()

for tournament, category, n in rows:
    print(f"  {n:>4}  {category:<25}  {tournament}")

print("\n--- Stat types available ---")
stat_rows = conn.execute("""
    SELECT stat_name, section, COUNT(*) as n
    FROM   flashscore_stats
    WHERE  section = 'match'
    GROUP  BY stat_name, section
    ORDER  BY n DESC
""").fetchall()

for stat_name, section, n in stat_rows:
    print(f"  {n:>4} fixtures  [{section}]  {stat_name}")

print("\n--- Results table coverage check ---")
res = conn.execute("""
    SELECT status, COUNT(*) FROM results GROUP BY status
""").fetchall()
for status, n in res:
    print(f"  {status}: {n}")

conn.close()
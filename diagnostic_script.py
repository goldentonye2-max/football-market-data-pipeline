import sqlite3
conn = sqlite3.connect("sportybet.db")
c = conn.cursor()

# How many fixtures have ALL THREE: odds + result + flashscore stats?
c.execute("""
    SELECT COUNT(DISTINCT f.event_id)
    FROM fixtures f
    JOIN results r ON f.event_id = r.event_id
    JOIN match_links ml ON f.event_id = ml.event_id
    JOIN flashscore_stats fs ON ml.fs_match_id = fs.fs_match_id
    WHERE r.status = 'finished'
""")
print("Fixtures with odds + result + stats:", c.fetchone()[0])

# How many have odds + result only (no stats needed)?
c.execute("""
    SELECT COUNT(DISTINCT f.event_id)
    FROM fixtures f
    JOIN results r ON f.event_id = r.event_id
    WHERE r.status = 'finished'
""")
print("Fixtures with odds + result (no stats needed):", c.fetchone()[0])

# Top 20 markets by frequency
c.execute("""
    SELECT market_name, COUNT(*) as cnt,
           COUNT(DISTINCT event_id) as fixture_count
    FROM odds
    GROUP BY market_name
    ORDER BY cnt DESC
""")
print("\nAll markets by frequency:")
print(f"  {'Rows':>10}  {'Fixtures':>10}  Market")
print(f"  {'-'*10}  {'-'*10}  {'-'*40}")
for row in c.fetchall():
    print(f"  {row[1]:>10,}  {row[2]:>10,}  {row[0]}")

conn.close()
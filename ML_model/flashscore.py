import sqlite3
import pandas as pd

conn = sqlite3.connect("sportybet.db")

# See every unique stat name available
stats = pd.read_sql_query("""
    SELECT DISTINCT stat_name, section, 
           COUNT(*) as match_count
    FROM flashscore_stats
    GROUP BY stat_name, section
    ORDER BY match_count DESC
""", conn)

print(stats.to_string())
conn.close()
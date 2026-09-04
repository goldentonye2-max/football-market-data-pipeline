import sqlite3

conn = sqlite3.connect("sportybet.db")

rows = conn.execute("""
    SELECT DISTINCT market_id, market_name, specifier, outcome_desc
    FROM odds
    WHERE market_name IN (
        '1X2', 'Over/Under', 'GG/NG',
        'Home Team Clean Sheet', 'Away Team Clean Sheet',
        'Draw No Bet', 'Double Chance',
        '1st Half', 'Halftime/Fulltime'
    )
    ORDER BY market_id, specifier, outcome_desc
    LIMIT 80
""").fetchall()

for r in rows:
    print(r)

conn.close()

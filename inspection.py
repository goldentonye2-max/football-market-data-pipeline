import sqlite3
import os
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG — adjust these two paths to match your setup
# ============================================================
PROJECT_DIR = r"."          # or full path e.g. r"C:\Users\You\football-market-data-pipeline"
DB_PATH     = r"sportybet.db"  # or full path to your .db file
# ============================================================

print("=" * 65)
print("  PROJECT FILE MAP")
print("=" * 65)

for root, dirs, files in os.walk(PROJECT_DIR):
    # Skip hidden folders and common junk
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git', 'venv', '.venv')]
    
    depth = root.replace(PROJECT_DIR, '').count(os.sep)
    indent = "    " * depth
    folder_name = os.path.basename(root) or PROJECT_DIR
    print(f"{indent}📁 {folder_name}/")
    
    subindent = "    " * (depth + 1)
    for file in sorted(files):
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
        
        # Human readable size
        if size < 1024:
            size_str = f"{size}B"
        elif size < 1024**2:
            size_str = f"{size/1024:.1f}KB"
        elif size < 1024**3:
            size_str = f"{size/1024**2:.1f}MB"
        else:
            size_str = f"{size/1024**3:.1f}GB"
        
        print(f"{subindent}📄 {file:<45} {size_str:>8}   {mtime}")

print()
print("=" * 65)
print("  DATABASE SCHEMA")
print("=" * 65)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [row[0] for row in cursor.fetchall()]
print(f"\nTables found: {tables}\n")

for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    cols = cursor.fetchall()
    
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    row_count = cursor.fetchone()[0]
    
    print(f"{'─'*55}")
    print(f"  TABLE: {table}   ({row_count:,} rows)")
    print(f"{'─'*55}")
    for col in cols:
        col_id, name, col_type, notnull, default, pk = col
        pk_str   = " 🔑 PK"  if pk      else ""
        nn_str   = " NOT NULL" if notnull else ""
        def_str  = f" DEFAULT {default}" if default else ""
        print(f"  {name:<30} {col_type:<15}{pk_str}{nn_str}{def_str}")
    
    # Show one sample row
    cursor.execute(f"SELECT * FROM {table} LIMIT 1")
    sample = cursor.fetchone()
    if sample:
        print(f"\n  Sample row:")
        for col, val in zip([c[1] for c in cols], sample):
            val_str = str(val)
            if len(val_str) > 80:
                val_str = val_str[:77] + "..."
            print(f"    {col:<30} = {val_str}")
    print()

conn.close()
print("=" * 65)
print("  DONE")
print("=" * 65)
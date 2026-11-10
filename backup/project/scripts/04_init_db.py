"""Step 4 - Create the database tables.

    python scripts/04_init_db.py                  # uses SQLite by default (ids.sqlite3)
    python scripts/04_init_db.py --print-sql      # print the CREATE TABLE statements

For MySQL: set AIDS_DB=mysql (+ MYSQL_* values) in .env first; the database itself is
created automatically if the MySQL user is allowed to.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import db

ap = argparse.ArgumentParser()
ap.add_argument("--print-sql", action="store_true")
args = ap.parse_args()

if args.print_sql:
    print("\n\n".join(s + ";" for s in db.ddl_statements()))
else:
    db.init_db()
    where = (f"MySQL {config.MYSQL['host']}:{config.MYSQL['port']}/{config.MYSQL['database']}"
             if db.is_mysql() else str(config.SQLITE_PATH))
    print(f"Database ready ({config.DB_BACKEND}): {where}")

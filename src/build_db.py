"""
build_db.py — load the clean dataset into SQLite and run the committed analyst queries.

    python src/build_db.py

Builds data/trials.db (gitignored, derived) from data/crispr_trials_clean.csv, then
executes every file in sql/ and prints the result. The .sql files are the deliverable;
the .db is just a convenience so they run with no setup.
"""
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "crispr_trials_clean.csv"
DB = ROOT / "data" / "trials.db"
SQL_DIR = ROOT / "sql"


def build():
    df = pd.read_csv(CSV)
    con = sqlite3.connect(DB)
    df.to_sql("trials", con, if_exists="replace", index=False)
    con.commit()
    con.close()
    print(f"  built {DB.relative_to(ROOT)}  ({len(df):,} rows)\n")


def run_all():
    con = sqlite3.connect(DB)
    for f in sorted(SQL_DIR.glob("*.sql")):
        question = f.read_text().splitlines()[0].lstrip("- ").strip()
        print("=" * 78)
        print(f"  {f.name}  —  {question}")
        print("=" * 78)
        try:
            print(pd.read_sql_query(f.read_text(), con).to_string(index=False))
        except Exception as e:
            print(f"  ERROR: {e}")
        print()
    con.close()


if __name__ == "__main__":
    build()
    run_all()

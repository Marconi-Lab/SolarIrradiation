# warehouse/run_sql.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from google.cloud import bigquery


def run_sql_file(sql_path: Path, client: bigquery.Client) -> None:
    sql = sql_path.read_text(encoding="utf-8")
    print(f"Running: {sql_path}")
    job = client.query(sql)
    job.result()
    print(f"OK: {sql_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one or more SQL files on BigQuery.")
    parser.add_argument("project", help="GCP project ID (e.g., solar-irradiation-estimation).")
    parser.add_argument("sql_files", nargs="+", help="One or more .sql files to execute.")
    args = parser.parse_args(argv)

    client = bigquery.Client(project=args.project)

    for p_str in args.sql_files:
        p = Path(p_str)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"SQL file not found: {p}")
        run_sql_file(p, client)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

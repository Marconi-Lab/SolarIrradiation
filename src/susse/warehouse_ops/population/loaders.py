from __future__ import annotations
import pandas as pd
from ..io.bq import BigQueryClient
MERGE_KEY = ["date","geohash5","variable_id","source"]


def load_long_with_merge(bq: BigQueryClient, df: pd.DataFrame, dest_table: str) -> None:
    staging = dest_table + "_staging"
    bq.load_dataframe(df, staging, write_disposition="WRITE_TRUNCATE")
    on = " AND ".join([f"t.{k}=s.{k}" for k in MERGE_KEY])
    updates = ", ".join([f"{c}=s.{c}" for c in df.columns if c not in MERGE_KEY])
    cols = ", ".join(df.columns)
    sql = f"""
    BEGIN TRANSACTION;
    CREATE TABLE IF NOT EXISTS `{dest_table}` AS SELECT * FROM `{staging}` WHERE 1=0;
    MERGE `{dest_table}` t
    USING `{staging}` s
    ON {on}
    WHEN MATCHED THEN UPDATE SET {updates}
    WHEN NOT MATCHED THEN INSERT ({cols}) VALUES ({', '.join('s.'+c for c in df.columns)});
    DROP TABLE `{staging}`;
    COMMIT TRANSACTION;
    """
    bq.execute_ddl(sql)

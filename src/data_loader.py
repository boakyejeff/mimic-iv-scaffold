"""Single-interface data loading: BigQuery, Postgres, SQLite, or synthetic.

Real backends are thin wrappers that return one table as a DataFrame;
downstream code (:mod:`src.cohort`, :mod:`src.features`, :mod:`src.labels`)
never knows which backend is in use. The synthetic backend generates
small DataFrames matching the documented MIMIC-IV schemas so the full
pipeline (and the test suite) runs with zero credentials.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from . import config, synthetic


class DataLoader(ABC):
    """Abstract table source. Implement :meth:`get_table` for a backend."""

    @abstractmethod
    def get_table(self, module: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Return ``module.table`` as a DataFrame.

        Args:
            module: ``"hosp"`` or ``"icu"``.
            table: table name, e.g. ``"admissions"``.
            columns: optional column subset (pushed down when the backend
                supports it).
        """


class BigQueryLoader(DataLoader):
    """Read-only BigQuery access to the PhysioNet-hosted MIMIC-IV datasets."""

    def __init__(self, project: str | None = None,
                 key_path: str | None = None):
        from google.cloud import bigquery  # deferred: only needed for real runs
        self.client = bigquery.Client(
            project=project,
            credentials=None if key_path is None else _sa_credentials(key_path),
        )

    def get_table(self, module: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        dataset = (config.BIGQUERY_HOSP_DATASET if module == "hosp"
                   else config.BIGQUERY_ICU_DATASET)
        cols = "*" if not columns else ", ".join(columns)
        sql = (f"SELECT {cols} FROM "
               f"`{self.client.project}.{dataset}.{table}`")
        return self.client.query(sql).to_dataframe()


def _sa_credentials(key_path: str):
    from google.oauth2 import service_account
    return service_account.Credentials.from_service_account_file(key_path)


class PostgresLoader(DataLoader):
    """Local Postgres (e.g. built with the mimic-code loading scripts)."""

    def __init__(self, dsn: str, hosp_schema: str | None = None,
                 icu_schema: str | None = None):
        from sqlalchemy import create_engine
        self.engine = create_engine(dsn)
        self.hosp_schema = hosp_schema or config.POSTGRES_HOSP_SCHEMA
        self.icu_schema = icu_schema or config.POSTGRES_ICU_SCHEMA

    def get_table(self, module: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        schema = self.hosp_schema if module == "hosp" else self.icu_schema
        cols = "*" if not columns else ", ".join(columns)
        return pd.read_sql(f"SELECT {cols} FROM {schema}.{table}", self.engine)


class SQLiteLoader(DataLoader):
    """SQLite file — handy for a small local extract or test fixture."""

    def __init__(self, path: str):
        import sqlite3
        self._conn = sqlite3.connect(path)

    def get_table(self, module: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:  # noqa: ARG002
        cols = "*" if not columns else ", ".join(columns)
        return pd.read_sql(f"SELECT {cols} FROM {table}", self._conn)


class SyntheticLoader(DataLoader):
    """Credential-free synthetic tables matching the documented schemas.

    Used by ``scripts/run_pipeline.py --dry-run`` and the test suite.
    """

    def __init__(self, n_patients: int = 60, seed: int = 0):
        self.tables = synthetic.make_tables(n_patients=n_patients, seed=seed)

    def get_table(self, module: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.tables[(module, table)]
        return df if not columns else df[columns]

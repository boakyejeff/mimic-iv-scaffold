"""Cohort builder for MIMIC-IV v3.1 readmission prediction.

Index unit: one row per hospital admission (hadm_id) of an adult patient,
optionally linked to their first ICU stay (stay_id) within that admission.
"""

from __future__ import annotations

import pandas as pd

from . import config


def cohort_query(dialect: str = "bigquery") -> str:
    """Return the index-cohort SQL for the given SQL dialect.

    Args:
        dialect: ``"bigquery"`` or ``"postgres"`` — controls the
            project/dataset qualification of table names.

    Returns:
        SQL string producing one row per index admission with columns:
        ``subject_id, hadm_id, stay_id, admittime, dischtime, intime,
        outtime, anchor_age, gender``. ``stay_id``/``intime``/``outtime``
        are NULL for admissions with no ICU stay.
    """
    if dialect == "bigquery":
        hosp = f"`{config.BIGQUERY_PROJECT}.{config.BIGQUERY_HOSP_DATASET}`"
        icu = f"`{config.BIGQUERY_PROJECT}.{config.BIGQUERY_ICU_DATASET}`"
    elif dialect == "postgres":
        hosp = config.POSTGRES_HOSP_SCHEMA
        icu = config.POSTGRES_ICU_SCHEMA
    else:
        raise ValueError(f"Unknown dialect: {dialect!r}")

    return f"""
    WITH ranked_icu AS (
        SELECT stay_id, subject_id, hadm_id, intime, outtime,
               ROW_NUMBER() OVER (PARTITION BY hadm_id ORDER BY intime) AS rn
        FROM {icu}.icustays
    )
    SELECT a.subject_id,
           a.hadm_id,
           r.stay_id,
           a.admittime,
           a.dischtime,
           r.intime,
           r.outtime,
           p.anchor_age,
           p.gender
    FROM {hosp}.admissions AS a
    JOIN {hosp}.patients AS p USING (subject_id)
    LEFT JOIN ranked_icu AS r
           ON r.hadm_id = a.hadm_id AND r.rn = 1
    WHERE p.anchor_age >= {config.COHORT.min_age}
      AND a.dischtime IS NOT NULL
    """


def apply_exclusions(cohort: pd.DataFrame, criteria: config.CohortCriteria | None = None) -> pd.DataFrame:
    """Apply cohort exclusions to an in-memory cohort DataFrame.

    Args:
        cohort: DataFrame with at least ``subject_id, hadm_id, dischtime``
            and, when available, ``hospital_expire_flag`` and
            ``discharge_location``.
        criteria: overrides for :class:`config.CohortCriteria`.

    Returns:
        Filtered copy of ``cohort``.
    """
    criteria = criteria or config.COHORT
    df = cohort.copy()

    if criteria.exclude_in_hospital_death and "hospital_expire_flag" in df.columns:
        df = df[df["hospital_expire_flag"] == 0]
    if criteria.exclude_missing_dischtime:
        df = df[df["dischtime"].notna()]
    if criteria.exclude_discharge_against_advice and "discharge_location" in df.columns:
        df = df[df["discharge_location"] != "AGAINST ADVICE"]
    return df.reset_index(drop=True)


def build_cohort(conn, dialect: str = "bigquery", criteria: config.CohortCriteria | None = None) -> pd.DataFrame:
    """Build the index-admission cohort from a live connection.

    Args:
        conn: a DB-API / SQLAlchemy / BigQuery client exposing an
            ``execute``/``query`` path compatible with ``pandas.read_sql``.
        dialect: SQL dialect for table qualification (see :func:`cohort_query`).
        criteria: cohort inclusion/exclusion rules.

    Returns:
        Cohort DataFrame (see :func:`cohort_query` for the contract).
    """
    query = cohort_query(dialect)
    df = pd.read_sql(query, conn)
    for col in ("admittime", "dischtime", "intime", "outtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return apply_exclusions(df, criteria)

"""Outcome labels for MIMIC-IV readmission / mortality prediction.

Labels are derived *only* from ``hosp.admissions`` (plus ``hosp.patients.dod``
for mortality) — no feature table is consulted, and no future information
leaks into the feature windows by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

READMIT_WINDOW_DAYS = 30
MORTALITY_WINDOW_DAYS = 30


def add_readmission_label(cohort: pd.DataFrame, admissions: pd.DataFrame,
                          window_days: int = READMIT_WINDOW_DAYS) -> pd.DataFrame:
    """Flag index admissions followed by another admission within 30 days.

    A readmission is any *other* admission for the same ``subject_id``
    whose ``admittime`` falls in ``(dischtime, dischtime + window_days]``.
    Admissions where the patient died in hospital are excluded upstream
    (:mod:`src.cohort`), so the label is well-defined.

    Args:
        cohort: index admissions with ``subject_id, hadm_id, dischtime``.
        admissions: full ``hosp.admissions`` with
            ``subject_id, hadm_id, admittime``.
        window_days: readmission horizon (default 30).

    Returns:
        ``cohort`` copy with added ``readmit_30d`` column (0/1 int).
    """
    df = cohort.copy()
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    adm = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"])

    merged = df[["subject_id", "hadm_id", "dischtime"]].merge(adm, on="subject_id",
                                                             suffixes=("", "_next"))
    merged = merged[merged["hadm_id_next"] != merged["hadm_id"]]
    delta = (merged["admittime"] - merged["dischtime"]).dt.total_seconds() / 86400.0
    readmit_hadm = set(merged.loc[(delta > 0) & (delta <= window_days), "hadm_id"])

    df["readmit_30d"] = df["hadm_id"].isin(readmit_hadm).astype(int)
    return df


def add_mortality_label(cohort: pd.DataFrame, patients: pd.DataFrame,
                        window_days: int = MORTALITY_WINDOW_DAYS) -> pd.DataFrame:
    """Flag index admissions where the patient died within 30d of discharge.

    Uses the de-identified date of death (``hosp.patients.dod``); in MIMIC
    this is shifted consistently with all other dates, so deltas are valid.

    Args:
        cohort: index admissions with ``subject_id, hadm_id, dischtime``.
        patients: ``hosp.patients`` with ``subject_id, dod``.
        window_days: mortality horizon (default 30).

    Returns:
        ``cohort`` copy with added ``mortality_30d`` column (0/1 int).
    """
    df = cohort.copy()
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    dod = patients[["subject_id", "dod"]].copy()
    dod["dod"] = pd.to_datetime(dod["dod"])

    df = df.merge(dod, on="subject_id", how="left")
    delta = (df["dod"] - df["dischtime"]).dt.total_seconds() / 86400.0
    df["mortality_30d"] = ((delta > 0) & (delta <= window_days)).astype(int)
    return df.drop(columns=["dod"])


def label_prevalence(labeled: pd.DataFrame, label_col: str = "readmit_30d") -> float:
    """Return the positive rate of a label column (sanity check helper)."""
    return float(np.mean(labeled[label_col]))

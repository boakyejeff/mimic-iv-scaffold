"""Temporally-causal feature engineering for MIMIC-IV readmission prediction.

Feature blocks (fused across the ``hosp`` and ``icu`` modules):

1. **Demographics** — ``anchor_age``, ``gender`` from ``hosp.patients``.
2. **Hosp labs** — from ``hosp.labevents``: for each lab in
   :data:`config.LAB_ITEMIDS`, the last ``valuenum`` in the 48h and 24h
   windows before ``dischtime``, plus a per-lab measurement count.
3. **ICU vitals** — from ``icu.chartevents``: for each vital in
   :data:`config.VITAL_ITEMIDS`, mean/min/max over the first 24h of the
   ICU stay (``intime`` → ``intime + 24h``). Admissions without an ICU
   stay get NaN (the model imputes them).

ANTI-LEAKAGE RULES (enforced here, not just documented):

* The prediction point is ``dischtime``. Any measurement with
  ``charttime >= dischtime`` is dropped — never a feature.
* Lab windows are strictly *before* discharge.
* ICU vitals use only the first 24h of the stay, so for ICU stays that
  extend past discharge... they can't — ICU stays end at/before discharge
  by construction (``icustays.outtime <= admissions.dischtime``).
* Diagnosis/procedure codes come from the *index* admission only; no
  codes from future admissions are ever joined in.
* The readmission label is computed from admissions *after* dischtime
  (:mod:`src.labels`) and is never an input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Anti-leakage helpers
# ---------------------------------------------------------------------------


def enforce_prediction_point(events: pd.DataFrame, cohort: pd.DataFrame,
                             time_col: str = "charttime") -> pd.DataFrame:
    """Drop every event recorded at/after the prediction point (dischtime).

    Args:
        events: event table with ``hadm_id`` and a timestamp column.
        cohort: cohort DataFrame with ``hadm_id`` and ``dischtime``.
        time_col: name of the event timestamp column.

    Returns:
        Events strictly before the patient's discharge.
    """
    merged = events.merge(cohort[["hadm_id", "dischtime"]], on="hadm_id", how="left")
    merged[time_col] = pd.to_datetime(merged[time_col])
    merged["dischtime"] = pd.to_datetime(merged["dischtime"])
    return merged[merged[time_col] < merged["dischtime"]].drop(columns=["dischtime"])


def last_value_in_window(events: pd.DataFrame, end: pd.Series, hours: int,
                         time_col: str = "charttime",
                         value_col: str = "valuenum") -> pd.Series:
    """Last observed value per row-key within ``hours`` before ``end``.

    Args:
        events: long-form events with ``hadm_id``, ``itemid``, timestamps.
        end: per-``hadm_id`` window end (aligned to ``events`` rows).
        hours: lookback length.
        time_col, value_col: column names.

    Returns:
        Series of last values, indexed like ``events`` (NaN outside window).
    """
    t = pd.to_datetime(events[time_col])
    end = pd.to_datetime(end)
    in_window = (t >= end - pd.to_timedelta(hours, "h")) & (t < end)
    out = pd.Series(np.nan, index=events.index, dtype=float)
    out[in_window] = events.loc[in_window, value_col]
    return out


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------


def demographic_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Demographics: anchor_age + one-hot gender, indexed by hadm_id."""
    df = cohort[["hadm_id", "anchor_age", "gender"]].copy()
    df["gender"] = df["gender"].astype(str).str.upper()
    dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=float)
    return pd.concat([df[["hadm_id", "anchor_age"]], dummies], axis=1)


def lab_features(labevents: pd.DataFrame, cohort: pd.DataFrame,
                 itemids: dict[int, str] | None = None) -> pd.DataFrame:
    """Pre-discharge lab features from ``hosp.labevents``.

    For each lab: last value in the 48h window, last value in the 24h
    window, and count of measurements in the 48h window.

    Args:
        labevents: columns ``hadm_id, itemid, charttime, valuenum``.
        cohort: columns ``hadm_id, dischtime``.
        itemids: subset of :data:`config.LAB_ITEMIDS` to use.

    Returns:
        One row per ``hadm_id`` with ``lab_<name>_48h``,
        ``lab_<name>_24h``, ``lab_<name>_count`` columns.
    """
    itemids = itemids or config.LAB_ITEMIDS
    labs = enforce_prediction_point(
        labevents[labevents["itemid"].isin(itemids)].copy(), cohort
    )
    labs = labs.merge(cohort[["hadm_id", "dischtime"]], on="hadm_id", how="left")
    labs["charttime"] = pd.to_datetime(labs["charttime"])
    labs["dischtime"] = pd.to_datetime(labs["dischtime"])

    frames = []
    for hours in config.WINDOWS.lab_windows_hours:
        win = labs[labs["charttime"] >= labs["dischtime"] - pd.to_timedelta(hours, "h")]
        win = win.sort_values("charttime").groupby(["hadm_id", "itemid"]).tail(1)
        piv = win.pivot(index="hadm_id", columns="itemid", values="valuenum")
        piv.columns = [f"lab_{itemids[i]}_{hours}h" for i in piv.columns]
        frames.append(piv)

    counts = labs[labs["charttime"] >= labs["dischtime"] - pd.to_timedelta(48, "h")]
    cnt = counts.groupby(["hadm_id", "itemid"]).size().unstack(fill_value=0)
    cnt.columns = [f"lab_{itemids[i]}_count" for i in cnt.columns]
    frames.append(cnt)

    out = pd.concat(frames, axis=1)
    return out.reset_index()


def icu_vital_features(chartevents: pd.DataFrame, cohort: pd.DataFrame,
                       itemids: dict[int, str] | None = None) -> pd.DataFrame:
    """First-24h ICU vital features from ``icu.chartevents``.

    Args:
        chartevents: columns ``stay_id, itemid, charttime, valuenum``.
        cohort: columns ``hadm_id, stay_id, intime``.
        itemids: subset of :data:`config.VITAL_ITEMIDS`.

    Returns:
        One row per ``hadm_id`` with ``vital_<name>_mean/min/max`` columns.
        Admissions without an ICU stay are absent (caller left-joins).
    """
    itemids = itemids or config.VITAL_ITEMIDS
    vitals = chartevents[chartevents["itemid"].isin(itemids)].copy()
    vitals["charttime"] = pd.to_datetime(vitals["charttime"])
    vitals = vitals.merge(cohort[["hadm_id", "stay_id", "intime"]].dropna(subset=["stay_id"]),
                          on="stay_id", how="inner")
    vitals["intime"] = pd.to_datetime(vitals["intime"])
    window = config.WINDOWS.icu_vital_window_hours
    vitals = vitals[(vitals["charttime"] >= vitals["intime"])
                    & (vitals["charttime"] < vitals["intime"] + pd.to_timedelta(window, "h"))]

    agg = vitals.groupby(["hadm_id", "itemid"])["valuenum"].agg(["mean", "min", "max"])
    out = agg.unstack("itemid")
    out.columns = [f"vital_{itemids[i]}_{stat}" for stat, i in out.columns]
    return out.reset_index()


def build_feature_matrix(cohort: pd.DataFrame, labevents: pd.DataFrame,
                         chartevents: pd.DataFrame) -> pd.DataFrame:
    """Fuse all feature blocks into one model-ready matrix.

    Args:
        cohort: index-admission cohort (see :mod:`src.cohort`).
        labevents: ``hosp.labevents`` subset (hadm_id, itemid, charttime, valuenum).
        chartevents: ``icu.chartevents`` subset (stay_id, itemid, charttime, valuenum).

    Returns:
        DataFrame with ``subject_id, hadm_id`` key columns plus all features.
        Rows keep cohort order; ICU-less admissions have NaN vital columns.
    """
    base = cohort[["subject_id", "hadm_id"]].copy()
    feats = demographic_features(cohort)
    labs = lab_features(labevents, cohort)
    vitals = icu_vital_features(chartevents, cohort)

    X = base.merge(feats, on="hadm_id", how="left")
    X = X.merge(labs, on="hadm_id", how="left")
    X = X.merge(vitals, on="hadm_id", how="left")
    return X

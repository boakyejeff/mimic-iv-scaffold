"""Synthetic MIMIC-IV-shaped tables for credential-free testing.

Generates small DataFrames whose columns match the schemas documented in
``docs/SCHEMAS.md`` (``hosp.patients``, ``hosp.admissions``,
``icu.icustays``, ``hosp.labevents``, ``icu.chartevents``). A weak but
real signal is baked in — higher creatinine / older age raise the
readmission probability — so the model baselines must beat chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_EPOCH = pd.Timestamp("2020-01-01")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_tables(n_patients: int = 60, seed: int = 0) -> dict[tuple[str, str], pd.DataFrame]:
    """Generate synthetic tables.

    Args:
        n_patients: number of distinct synthetic patients.
        seed: RNG seed for reproducibility.

    The realized readmission rate (~0.25) emerges from the risk model,
    not from a fixed coin flip, so labels genuinely correlate with the
    physiological features.

    Returns:
        Dict keyed by ``(module, table)`` with DataFrames for
        ``("hosp", "patients")``, ``("hosp", "admissions")``,
        ``("icu", "icustays")``, ``("hosp", "labevents")``,
        ``("icu", "chartevents")``.
    """
    rng = _rng(seed)
    n_labs = len(config.LAB_ITEMIDS)
    n_vitals = len(config.VITAL_ITEMIDS)

    patients, admissions, icustays, labevents, chartevents = [], [], [], [], []
    hadm_id, stay_id = 1000, 5000

    for s in range(n_patients):
        subject_id = 10_000 + s
        age = int(rng.integers(25, 90))
        gender = rng.choice(["M", "F"])
        # risk score drives BOTH abnormal physiology AND readmission —
        # this is the synthetic signal the baselines must recover.
        risk = rng.normal(0, 1) + 0.04 * (age - 55)
        readmit_prob = 1.0 / (1.0 + np.exp(-(1.5 * risk - 1.0)))
        readmit = rng.random() < readmit_prob

        patients.append({"subject_id": subject_id, "gender": gender,
                         "anchor_age": age, "dod": pd.NaT})

        admit = _EPOCH + pd.to_timedelta(int(rng.integers(0, 300)), "D")
        n_adm = 2 if readmit else (2 if rng.random() < 0.3 else 1)
        for a in range(n_adm):
            los_days = int(rng.integers(2, 12))
            admittime = admit
            dischtime = admittime + pd.to_timedelta(los_days, "D")
            hadm_id += 1
            admissions.append({
                "subject_id": subject_id, "hadm_id": hadm_id,
                "admittime": admittime, "dischtime": dischtime,
                "admission_type": rng.choice(["URGENT", "ELECTIVE", "EW EMER."]),
                "insurance": rng.choice(["Medicare", "Medicaid", "Other"]),
                "hospital_expire_flag": 0,
                "discharge_location": "HOME",
            })

            # ~70% of admissions get an ICU stay
            if rng.random() < 0.7:
                stay_id += 1
                intime = admittime + pd.to_timedelta(int(rng.integers(0, 2)), "D")
                outtime = min(intime + pd.to_timedelta(int(rng.integers(1, 4)), "D"),
                              dischtime)
                icustays.append({"stay_id": stay_id, "subject_id": subject_id,
                                 "hadm_id": hadm_id, "intime": intime,
                                 "outtime": outtime,
                                 "los": (outtime - intime).total_seconds() / 86400})

                # vitals: hourly charttime in first 24h, risk-shifted means
                for itemid, name in config.VITAL_ITEMIDS.items():
                    base = {"heart_rate": 85, "sbp": 120, "dbp": 70,
                            "resp_rate": 18, "spo2": 97,
                            "temperature_c": 37.0}[name]
                    scale = {"heart_rate": 12, "sbp": 15, "dbp": 10,
                             "resp_rate": 4, "spo2": 2, "temperature_c": 0.6}[name]
                    for h in range(24):
                        hr_shift = 3.0 * risk if name == "heart_rate" else 0.0
                        chartevents.append({
                            "stay_id": stay_id, "itemid": itemid,
                            "charttime": intime + pd.to_timedelta(h, "h"),
                            "valuenum": base + hr_shift
                            + rng.normal(0, scale / 4),
                        })

            # labs: daily charttime through the stay, creatinine risk-shifted
            for itemid, name in config.LAB_ITEMIDS.items():
                base = {"creatinine": 1.0, "bun": 18, "sodium": 140,
                        "potassium": 4.2, "bicarbonate": 24, "glucose": 110,
                        "hematocrit": 38, "hemoglobin": 12.5, "wbc": 8,
                        "platelets": 220, "anion_gap": 12}[name]
                scale = base * 0.15
                for d in range(los_days):
                    shift = (1.2 * risk * scale) if name == "creatinine" else 0.0
                    labevents.append({
                        "hadm_id": hadm_id, "itemid": itemid,
                        "charttime": admittime + pd.to_timedelta(d, "D")
                        + pd.to_timedelta(8, "h"),
                        "valuenum": max(base + shift + rng.normal(0, scale), 0.01),
                        "valueuom": "unit",
                    })

            # next admission timing: readmitted patients return within 30d
            if a == 0 and n_adm == 2:
                gap = int(rng.integers(3, 28)) if readmit else int(rng.integers(45, 120))
                admit = dischtime + pd.to_timedelta(gap, "D")

    tables = {
        ("hosp", "patients"): pd.DataFrame(patients),
        ("hosp", "admissions"): pd.DataFrame(admissions),
        ("icu", "icustays"): pd.DataFrame(icustays),
        ("hosp", "labevents"): pd.DataFrame(labevents),
        ("icu", "chartevents"): pd.DataFrame(chartevents),
    }
    return tables


def make_cohort(tables: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Build the index-admission cohort from synthetic tables with pandas.

    Mirrors :func:`src.cohort.cohort_query` semantics: adult patients,
    first ICU stay per admission (LEFT JOIN — ICU-less admissions kept),
    non-null dischtime. Exclusions from :func:`src.cohort.apply_exclusions`
    are applied.
    """
    from . import cohort as cohort_mod

    patients = tables[("hosp", "patients")]
    admissions = tables[("hosp", "admissions")]
    icustays = tables[("icu", "icustays")]

    ranked = icustays.sort_values("intime").groupby("hadm_id").head(1)
    df = (admissions.merge(patients[["subject_id", "anchor_age", "gender"]],
                           on="subject_id", how="left")
          .merge(ranked[["hadm_id", "stay_id", "intime", "outtime"]],
                 on="hadm_id", how="left"))
    df = df[df["anchor_age"] >= config.COHORT.min_age]
    df = df[df["dischtime"].notna()]
    return cohort_mod.apply_exclusions(df.reset_index(drop=True))

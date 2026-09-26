"""Credential-free synthetic MIMIC-IV-shaped tables, calibrated to published marginals.

Generates DataFrames whose columns match the schemas documented in
``docs/SCHEMAS.md`` (``hosp.patients``, ``hosp.admissions``,
``icu.icustays``, ``hosp.labevents``, ``icu.chartevents``).

Calibration (see ``docs/CALIBRATION.md`` for the full table):

* 30-day readmission prevalence is tuned to ~12.5% — between the 8.7%
  30-day ICU-readmission rate reported for MIMIC-IV (Momenzadeh et al.,
  *Sci Rep* 2026) and the 17.6% rate in a critically-ill heart-failure
  MIMIC-IV v3.1 cohort (Odoeke, *Cureus* 2025).
* Age (mean 58.8, SD 19.2), female share (52.2%), insurance mix
  (Medicare 37.2% / Medicaid 9.6%), and mean hospital LOS (4.5 d) come
  from the MIMIC-IV cohort table (Johnson et al., *Sci Data* 2023).
* Comorbidity prevalences (diabetes ~19.3%, CHF ~12.3%, renal disease
  ~12.5%, chronic pulmonary disease ~14.6%) come from Charlson
  encounter-level prevalences reported for MIMIC-IV (medicalcoder paper,
  *JAMIA Open* 2026).
* Vital-sign and lab distributions are plausible clinical ranges marked
  as generator design choices (not MIMIC-measured values).

A single latent risk score per patient drives abnormal physiology,
comorbidities, length of stay, ICU admission, readmission, and
mortality — so labels genuinely correlate with the features and the
baselines must beat chance. Missingness is generated per variable at
rates documented in ``docs/CALIBRATION.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_EPOCH = pd.Timestamp("2020-01-01")

# ---------------------------------------------------------------------------
# Calibration targets (documented in docs/CALIBRATION.md)
# ---------------------------------------------------------------------------
TARGET_READMIT_PREVALENCE = 0.125   # 12.5% 30-day readmission (plausible range)
TARGET_MORTALITY_30D = 0.035        # ~3.5% 30-day mortality (plausible range)

#: Encounter-level comorbidity prevalences reported for MIMIC-IV
#: (medicalcoder, JAMIA Open 2026 — Charlson via ICD codes).
COMORBIDITY_BASE = {
    "diabetes": 0.193,   # 12.32% w/o complications + 7.02% w/ complications
    "chf": 0.123,        # congestive heart failure 12.33%
    "ckd": 0.125,        # renal disease 12.53%
    "copd": 0.146,       # chronic pulmonary disease 14.61%
}

#: First-24h ICU vital-sign distributions. Plausible clinical ranges
#: (generator design choice, not MIMIC-measured). ``risk_slope`` shifts the
#: mean per unit of standardized latent risk.
VITAL_SPECS = {
    "heart_rate":    {"mean": 84.0, "sd": 14.0, "risk_slope": 6.0,
                      "lo": 20.0, "hi": 220.0},
    "sbp":           {"mean": 122.0, "sd": 20.0, "risk_slope": -8.0,
                      "lo": 50.0, "hi": 260.0},
    "dbp":           {"mean": 66.0, "sd": 12.0, "risk_slope": -5.0,
                      "lo": 25.0, "hi": 160.0},
    "resp_rate":     {"mean": 19.5, "sd": 5.0, "risk_slope": 2.5,
                      "lo": 6.0, "hi": 60.0},
    "spo2":          {"mean": 97.5, "sd": 2.2, "risk_slope": -1.5,
                      "lo": 80.0, "hi": 100.0},
    "temperature_c": {"mean": 37.1, "sd": 0.7, "risk_slope": 0.3,
                      "lo": 34.0, "hi": 41.5},
}

#: Pre-discharge lab distributions. Plausible clinical ranges (generator
#: design choice). ``log`` = lognormal sampling; ``risk_slope`` shifts the
#: mean per unit of standardized latent risk; ``miss`` = fraction of
#: admissions where the lab is never ordered; ``mult`` multiplies the mean
#: for patients with the comorbidity; ``add`` adds to the mean.
LAB_SPECS = {
    "creatinine":  {"mean": 1.05, "sd": 0.35, "log": True, "risk_slope": 0.12,
                    "miss": 0.03, "lo": 0.2, "hi": 15.0,
                    "mult": {"ckd": 1.9, "chf": 1.25}},
    "bun":         {"mean": 19.0, "sd": 9.0, "risk_slope": 0.08,
                    "miss": 0.04, "lo": 3.0, "hi": 150.0,
                    "mult": {"ckd": 1.5}},
    "sodium":      {"mean": 139.5, "sd": 3.2, "miss": 0.02,
                    "lo": 110.0, "hi": 165.0},
    "potassium":   {"mean": 4.15, "sd": 0.45, "miss": 0.03,
                    "lo": 2.0, "hi": 7.5, "mult": {"ckd": 1.06}},
    "bicarbonate": {"mean": 23.5, "sd": 3.0, "miss": 0.05,
                    "lo": 8.0, "hi": 40.0},
    "glucose":     {"mean": 110.0, "sd": 0.30, "log": True, "risk_slope": 0.08,
                    "miss": 0.06, "lo": 40.0, "hi": 900.0,
                    "mult": {"diabetes": 1.65}},
    "hematocrit":  {"mean": 33.0, "sd": 5.5, "miss": 0.08,
                    "lo": 15.0, "hi": 60.0},
    "hemoglobin":  {"mean": 11.0, "sd": 1.9, "miss": 0.08,
                    "lo": 5.0, "hi": 20.0},
    "wbc":         {"mean": 9.8, "sd": 0.35, "log": True, "risk_slope": 0.10,
                    "miss": 0.05, "lo": 0.5, "hi": 80.0},
    "platelets":   {"mean": 215.0, "sd": 80.0, "miss": 0.07,
                    "lo": 10.0, "hi": 900.0},
    "anion_gap":   {"mean": 13.0, "sd": 3.0, "risk_slope": 0.06,
                    "miss": 0.18, "lo": 3.0, "hi": 35.0,
                    "add": {"ckd": 1.5}},
}


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _patient_level(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Draw per-patient attributes: demographics, comorbidities, risk."""
    age = np.clip(np.rint(rng.normal(58.8, 19.2, n)), 18, 91).astype(int)
    female = rng.random(n) < 0.522  # Johnson et al., Sci Data 2023

    def age_adjusted(base: float) -> np.ndarray:
        # comorbidity risk rises with age (design choice, labelled as such)
        return _expit(_logit(base) + 0.045 * (age - 58.8))

    diabetes = rng.random(n) < age_adjusted(COMORBIDITY_BASE["diabetes"])
    chf = rng.random(n) < age_adjusted(COMORBIDITY_BASE["chf"])
    ckd = rng.random(n) < age_adjusted(COMORBIDITY_BASE["ckd"])
    copd = rng.random(n) < age_adjusted(COMORBIDITY_BASE["copd"])

    # latent risk drives physiology, utilization, readmission, mortality
    base0 = (0.035 * (age - 58.8)
             + 0.50 * diabetes + 0.65 * chf + 0.65 * ckd + 0.40 * copd
             + rng.normal(0, 0.5, n))
    frequent = rng.random(n) < _expit(0.9 * base0 - 1.7)  # prior utilizer

    return pd.DataFrame({
        "subject_id": 10_000 + np.arange(n),
        "anchor_age": age,
        "gender": np.where(female, "F", "M"),
        "diabetes": diabetes, "chf": chf, "ckd": ckd, "copd": copd,
        "frequent": frequent, "base0": base0,
    })


def _admissions(rng: np.random.Generator, pats: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand patients to 1–2 admissions each with timings and risk."""
    pats = pats.copy()
    n = len(pats)
    los_days = np.clip(np.rint(rng.lognormal(1.1, 0.9, n)), 1, 60).astype(int)
    pats["los_days"] = los_days
    base = (pats["base0"].to_numpy()
            + 0.55 * pats["frequent"].to_numpy()
            + 0.35 * (los_days > 7)
            + rng.normal(0, 0.35, n))
    pats["risk"] = base

    # Calibrate the readmission intercept so the REALIZED label prevalence
    # (per admission, as computed by labels.add_readmission_label) hits the
    # target. Second admissions dilute the label, so we calibrate on the
    # label itself via a deterministic fixed-point bisection.
    u_readmit = rng.random(n)
    late2 = pats["frequent"].to_numpy() | (rng.random(n) < 0.20)

    def _label_prev(c_: float) -> float:
        r = u_readmit < _expit(base + c_)
        n_adm = n + int(np.sum(r | late2))
        return float(r.sum() / n_adm)

    lo_c, hi_c = -10.0, 10.0
    for _ in range(64):
        mid = (lo_c + hi_c) / 2.0
        if _label_prev(mid) > TARGET_READMIT_PREVALENCE:
            hi_c = mid
        else:
            lo_c = mid
    c = (lo_c + hi_c) / 2.0
    pats["readmit"] = u_readmit < _expit(base + c)

    # 30-day mortality from risk (design choice, plausible range)
    pats["die30"] = rng.random(n) < _expit(1.0 * pats["base0"].to_numpy() - 3.9)

    admit0 = (_EPOCH + pd.to_timedelta(rng.integers(0, 900, n), "D"))
    disch0 = admit0 + pd.to_timedelta(los_days, "D")

    # second admission: readmitted patients return within 30d; frequent
    # utilizers / others return later or not at all
    readmit = pats["readmit"].to_numpy()
    has_second = readmit | late2
    gap = np.where(readmit, rng.integers(3, 29, n), rng.integers(35, 301, n))
    admit1 = disch0 + pd.to_timedelta(np.where(has_second, gap, 0), "D")
    los2 = np.clip(np.rint(rng.lognormal(1.1, 0.9, n)), 1, 60).astype(int)
    disch1 = admit1 + pd.to_timedelta(np.where(has_second, los2, 0), "D")

    rows = []
    hadm_id = 1000
    insurance = rng.choice(["Medicare", "Medicaid", "Other"],
                           p=[0.372, 0.096, 0.532], size=n)  # Johnson et al.
    adm_types = rng.choice(["URGENT", "ELECTIVE", "EW EMER.", "DIRECT EMER.",
                            "OBSERVATION ADMIT"],
                           p=[0.40, 0.25, 0.15, 0.12, 0.08], size=(n, 2))
    disch_locs = rng.choice(["HOME", "SKILLED NURSING FACILITY", "REHAB",
                             "HOME HEALTH CARE"],
                            p=[0.85, 0.08, 0.04, 0.03], size=(n, 2))
    for i in range(n):
        p = pats.iloc[i]
        hadm_id += 1
        rows.append({
            "subject_id": int(p["subject_id"]), "hadm_id": hadm_id,
            "admittime": admit0[i], "dischtime": disch0[i],
            "admission_type": adm_types[i, 0], "insurance": insurance[i],
            "hospital_expire_flag": 0,
            "discharge_location": disch_locs[i, 0],
            "_risk": float(p["risk"]), "_base0": float(p["base0"]),
            "_diabetes": bool(p["diabetes"]), "_chf": bool(p["chf"]),
            "_ckd": bool(p["ckd"]), "_copd": bool(p["copd"]),
            "_los_days": int(los_days[i]),
            "_icu_prob": float(_expit(0.7 * p["base0"] - 2.05)),
        })
        if has_second[i]:
            hadm_id += 1
            rows.append({
                "subject_id": int(p["subject_id"]), "hadm_id": hadm_id,
                "admittime": admit1[i], "dischtime": disch1[i],
                "admission_type": adm_types[i, 1], "insurance": insurance[i],
                "hospital_expire_flag": 0,
                "discharge_location": disch_locs[i, 1],
                "_risk": float(p["risk"]), "_base0": float(p["base0"]),
                "_diabetes": bool(p["diabetes"]), "_chf": bool(p["chf"]),
                "_ckd": bool(p["ckd"]), "_copd": bool(p["copd"]),
                "_los_days": int(los2[i]),
                "_icu_prob": float(_expit(0.7 * p["base0"] - 2.05)),
            })
    adm = pd.DataFrame(rows)

    # date of death: within 30d of the LAST discharge for die30 patients
    died_mask = pats["die30"].to_numpy()
    last_disch = adm.groupby("subject_id")["dischtime"].max()
    dod = pd.Series(pd.NaT, index=pats.index, dtype="datetime64[ns]")
    if died_mask.any():
        dod_vals = (last_disch.loc[pats.loc[died_mask, "subject_id"]]
                    .to_numpy().astype("datetime64[ns]")
                    + rng.integers(1, 31, died_mask.sum()).astype("timedelta64[D]"))
        dod.loc[died_mask] = dod_vals
    pats["dod"] = dod.to_numpy()
    return adm, pats


def _events(rng: np.random.Generator, adm: pd.DataFrame
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Vectorized generation of icustays, chartevents (hourly first-24h
    vitals), and labevents (daily pre-discharge labs)."""
    n = len(adm)
    risk_n = np.clip(adm["_risk"].to_numpy() / 1.2, -3, 3)
    admittime = adm["admittime"].to_numpy().astype("datetime64[ns]")
    dischtime = adm["dischtime"].to_numpy().astype("datetime64[ns]")
    los_days = adm["_los_days"].to_numpy().astype(int)

    # ------------------------------------------------------------------
    # ICU stays + hourly vitals (first 24h)
    # ------------------------------------------------------------------
    has_icu = rng.random(n) < adm["_icu_prob"].to_numpy()
    idx = np.flatnonzero(has_icu)
    ns = len(idx)
    stay_ids = np.arange(5001, 5001 + ns)
    intime = (admittime[idx]
              + rng.integers(0, 2, ns).astype("timedelta64[D]"))
    outtime = np.minimum(intime + rng.integers(1, 4, ns).astype("timedelta64[D]"),
                         dischtime[idx])
    icustays = pd.DataFrame({
        "stay_id": stay_ids,
        "subject_id": adm["subject_id"].to_numpy()[idx],
        "hadm_id": adm["hadm_id"].to_numpy()[idx],
        "intime": pd.to_datetime(intime),
        "outtime": pd.to_datetime(outtime),
        "los": (outtime - intime) / np.timedelta64(1, "D"),
    })

    names = list(VITAL_SPECS)
    nv, H = len(names), 24
    vmean = np.array([VITAL_SPECS[x]["mean"] for x in names])
    vsd = np.array([VITAL_SPECS[x]["sd"] for x in names])
    vslope = np.array([VITAL_SPECS[x]["risk_slope"] for x in names])
    vlo = np.array([VITAL_SPECS[x]["lo"] for x in names])
    vhi = np.array([VITAL_SPECS[x]["hi"] for x in names])
    vital_itemids = np.array([_vital_itemid(x) for x in names])

    vals = (vmean[None, :, None] + vslope[None, :, None] * risk_n[idx, None, None]
            + rng.normal(0, 1, (ns, nv, H)) * vsd[None, :, None])
    vals = np.clip(vals, vlo[None, :, None], vhi[None, :, None])
    keep = rng.random((ns, H)) > 0.03  # 3% hourly gaps
    sel = np.broadcast_to(keep[:, None, :], (ns, nv, H))
    hours = np.arange(H).astype("timedelta64[h]")
    charttime = np.broadcast_to(intime[:, None], (ns, H))[:, None, :] + hours
    chartevents = pd.DataFrame({
        "stay_id": np.broadcast_to(stay_ids[:, None, None], (ns, nv, H))[sel],
        "itemid": np.broadcast_to(vital_itemids[None, :, None], (ns, nv, H))[sel],
        "charttime": pd.to_datetime(
            np.broadcast_to(charttime, (ns, nv, H))[sel]),
        "valuenum": vals[sel],
    })

    # ------------------------------------------------------------------
    # Labs: daily values through the stay, per-lab missingness
    # ------------------------------------------------------------------
    lab_names = list(LAB_SPECS)
    nl = len(lab_names)
    lmean = np.array([LAB_SPECS[x]["mean"] for x in lab_names], dtype=float)
    lsd = np.array([LAB_SPECS[x]["sd"] for x in lab_names])
    lslope = np.array([LAB_SPECS[x].get("risk_slope", 0.0) for x in lab_names])
    lmiss = np.array([LAB_SPECS[x]["miss"] for x in lab_names])
    llo = np.array([LAB_SPECS[x]["lo"] for x in lab_names])
    lhi = np.array([LAB_SPECS[x]["hi"] for x in lab_names])
    is_log = np.array([LAB_SPECS[x].get("log", False) for x in lab_names])
    lab_itemids = np.array([_lab_itemid(x) for x in lab_names])

    present = rng.random((n, nl)) >= lmiss[None, :]
    M = np.broadcast_to(lmean[None, :], (n, nl)).copy()
    for j, x in enumerate(lab_names):
        for cond, mult in LAB_SPECS[x].get("mult", {}).items():
            M[adm["_" + cond].to_numpy(), j] *= mult
        for cond, add in LAB_SPECS[x].get("add", {}).items():
            M[adm["_" + cond].to_numpy(), j] += add

    flat = np.flatnonzero(present.ravel())
    adm_i = flat // nl
    lab_j = flat % nl
    reps = los_days[adm_i]
    total = int(reps.sum())
    starts = np.concatenate([[0], np.cumsum(reps)[:-1]])
    day_idx = np.arange(total) - np.repeat(starts, reps)

    mu = M[adm_i, lab_j]
    sd = lsd[lab_j]
    sl = lslope[lab_j]
    lg = is_log[lab_j]
    rn_r = risk_n[adm_i]
    # expand per-cell parameters to per-row (one row per admission-day)
    mu_r = np.repeat(mu, reps)
    sd_r = np.repeat(sd, reps)
    sl_r = np.repeat(sl, reps)
    lg_r = np.repeat(lg, reps)
    rn_rr = np.repeat(rn_r, reps)
    lo_r = np.repeat(llo[lab_j], reps)
    hi_r = np.repeat(lhi[lab_j], reps)
    z = rng.normal(0, 1, total)
    lab_vals = np.where(lg_r,
                        np.exp(np.log(mu_r) + sd_r * z + sl_r * rn_rr),
                        mu_r + sl_r * sd_r * rn_rr + sd_r * z)
    lab_vals = np.clip(lab_vals, lo_r, hi_r)

    adm_rep = np.repeat(adm_i, reps)
    labevents = pd.DataFrame({
        "hadm_id": adm["hadm_id"].to_numpy()[adm_rep],
        "itemid": lab_itemids[np.repeat(lab_j, reps)],
        "charttime": pd.to_datetime(
            admittime[adm_rep] + day_idx.astype("timedelta64[D]")
            + np.timedelta64(8, "h")),
        "valuenum": lab_vals,
        "valueuom": "unit",
    })
    return icustays, chartevents, labevents


def _vital_itemid(name: str) -> int:
    return {v: k for k, v in config.VITAL_ITEMIDS.items()}[name]


def _lab_itemid(name: str) -> int:
    return {v: k for k, v in config.LAB_ITEMIDS.items()}[name]


def make_tables(n_patients: int = 60, seed: int = 0,
                target_admissions: int | None = None
                ) -> dict[tuple[str, str], pd.DataFrame]:
    """Generate synthetic MIMIC-IV-shaped tables.

    Args:
        n_patients: number of distinct synthetic patients (ignored when
            ``target_admissions`` is set).
        seed: RNG seed for reproducibility — the same seed always yields
            identical tables.
        target_admissions: if set, generate patients until at least this
            many admissions exist, then drop trailing whole patients until
            the count is at or just under the target.

    Returns:
        Dict keyed by ``(module, table)`` with DataFrames for
        ``("hosp", "patients")``, ``("hosp", "admissions")``,
        ``("icu", "icustays")``, ``("hosp", "labevents")``,
        ``("icu", "chartevents")``.
    """
    rng = np.random.default_rng(seed)

    if target_admissions is None:
        pats = _patient_level(rng, n_patients)
    else:
        # ~1.15 admissions per patient on average; overshoot then trim
        n0 = int(np.ceil(target_admissions / 1.15)) + 4
        pats = _patient_level(rng, n0)

    adm_all, pats = _admissions(rng, pats)

    if target_admissions is not None:
        # drop trailing whole patients until at/under target (deterministic)
        adm_all = adm_all.sort_values("hadm_id").reset_index(drop=True)
        adm_counts = adm_all.groupby("subject_id", sort=False).size()
        keep_subjects, total = [], 0
        for sid in adm_counts.index:  # generation order (hadm_id sorted)
            if total + adm_counts.loc[sid] > target_admissions:
                break
            keep_subjects.append(sid)
            total += adm_counts.loc[sid]
        adm = adm_all[adm_all["subject_id"].isin(keep_subjects)].reset_index(drop=True)
        pats = pats[pats["subject_id"].isin(keep_subjects)].reset_index(drop=True)
    else:
        adm = adm_all

    icustays, chartevents, labevents = _events(rng, adm)

    patients = pd.DataFrame({
        "subject_id": pats["subject_id"],
        "gender": pats["gender"],
        "anchor_age": pats["anchor_age"],
        "dod": pd.to_datetime(pats["dod"]),
    })
    admissions = adm[["subject_id", "hadm_id", "admittime", "dischtime",
                      "admission_type", "insurance", "hospital_expire_flag",
                      "discharge_location"]].copy()
    return {
        ("hosp", "patients"): patients,
        ("hosp", "admissions"): admissions,
        ("icu", "icustays"): icustays,
        ("hosp", "labevents"): labevents,
        ("icu", "chartevents"): chartevents,
    }


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


def calibration_summary(tables: dict[tuple[str, str], pd.DataFrame],
                        cohort: pd.DataFrame) -> dict:
    """Realized marginals of a generated cohort vs calibration targets.

    Useful for spot-checking the generator (see ``docs/CALIBRATION.md``).
    """
    from . import labels as labels_mod

    admissions = tables[("hosp", "admissions")]
    labeled = labels_mod.add_readmission_label(cohort, admissions)
    los_days = ((cohort["dischtime"] - cohort["admittime"])
                .dt.total_seconds() / 86400)
    return {
        "n_patients": int(cohort["subject_id"].nunique()),
        "n_admissions": int(len(cohort)),
        "readmit_30d_prevalence": float(labeled["readmit_30d"].mean()),
        "mortality_30d_prevalence": float(
            labels_mod.add_mortality_label(
                cohort, tables[("hosp", "patients")])["mortality_30d"].mean()),
        "mean_age": float(cohort["anchor_age"].mean()),
        "female_share": float((cohort["gender"] == "F").mean()),
        "icu_share": float(cohort["stay_id"].notna().mean()),
        "mean_los_days": float(los_days.mean()),
        "medicare_share": float((admissions["insurance"] == "Medicare").mean()),
        "targets": {
            "readmit_30d": TARGET_READMIT_PREVALENCE,
            "mortality_30d": TARGET_MORTALITY_30D,
            "mean_age": 58.8, "female_share": 0.522,
            "icu_share": 0.170, "mean_los_days": 4.5,
            "medicare_share": 0.372,
        },
    }

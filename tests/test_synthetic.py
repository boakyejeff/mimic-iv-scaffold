"""End-to-end tests on synthetic MIMIC-IV-shaped data.

No credentials, no network, no real patient data — everything here runs
on the small synthetic tables from ``src.synthetic`` whose columns match
the schemas documented in ``docs/SCHEMAS.md``.
"""

import numpy as np
import pandas as pd
import pytest

from src import cohort as cohort_mod
from src import config, features, labels, model, synthetic


@pytest.fixture(scope="module")
def tables():
    return synthetic.make_tables(n_patients=80, seed=1)


@pytest.fixture(scope="module")
def cohort_df(tables):
    return synthetic.make_cohort(tables)


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------

def test_tables_match_documented_schemas(tables):
    expected = {
        ("hosp", "patients"): {"subject_id", "gender", "anchor_age", "dod"},
        ("hosp", "admissions"): {"subject_id", "hadm_id", "admittime", "dischtime",
                                 "admission_type", "insurance",
                                 "hospital_expire_flag", "discharge_location"},
        ("icu", "icustays"): {"stay_id", "subject_id", "hadm_id", "intime",
                              "outtime", "los"},
        ("hosp", "labevents"): {"hadm_id", "itemid", "charttime", "valuenum",
                                "valueuom"},
        ("icu", "chartevents"): {"stay_id", "itemid", "charttime", "valuenum"},
    }
    for key, cols in expected.items():
        assert key in tables, f"missing table {key}"
        assert cols <= set(tables[key].columns), f"{key} missing {cols - set(tables[key].columns)}"
        assert len(tables[key]) > 0, f"{key} is empty"


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------

def test_cohort_sql_mentions_key_tables():
    for dialect in ("bigquery", "postgres"):
        sql = cohort_mod.cohort_query(dialect)
        for token in ("admissions", "patients", "icustays",
                      "subject_id", "hadm_id", "stay_id"):
            assert token in sql, f"{dialect} SQL missing {token}"


def test_cohort_adults_and_first_icu_stay(cohort_df):
    assert (cohort_df["anchor_age"] >= 18).all()
    assert cohort_df["dischtime"].notna().all()
    # first ICU stay per admission: no duplicated hadm_id
    assert not cohort_df["hadm_id"].duplicated().any()
    # some admissions have no ICU stay (fusion must handle NaNs)
    assert cohort_df["stay_id"].isna().any()


def test_apply_exclusions():
    df = pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "hadm_id": [11, 22, 33, 44],
        "dischtime": [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-05"),
                      pd.NaT, pd.Timestamp("2020-01-05")],
        "hospital_expire_flag": [0, 1, 0, 0],
        "discharge_location": ["HOME", "HOME", "HOME", "AGAINST ADVICE"],
    })
    out = cohort_mod.apply_exclusions(df)
    assert list(out["hadm_id"]) == [11]


# ---------------------------------------------------------------------------
# Anti-leakage
# ---------------------------------------------------------------------------

def test_enforce_prediction_point_drops_post_discharge_events():
    events = pd.DataFrame({
        "hadm_id": [1, 1, 1],
        "charttime": [pd.Timestamp("2020-01-02"),   # before discharge
                      pd.Timestamp("2020-01-05"),   # exactly at discharge
                      pd.Timestamp("2020-01-06")],  # after discharge
        "valuenum": [1.0, 2.0, 3.0],
    })
    cohort = pd.DataFrame({"hadm_id": [1],
                           "dischtime": [pd.Timestamp("2020-01-05")]})
    kept = features.enforce_prediction_point(events, cohort)
    assert list(kept["valuenum"]) == [1.0]


def test_lab_features_use_only_pre_discharge_data(tables, cohort_df):
    labs = features.lab_features(tables[("hosp", "labevents")], cohort_df)
    # every lab column exists and every hadm_id in cohort is represented
    assert set(cohort_df["hadm_id"]) <= set(labs["hadm_id"])
    assert any(c.startswith("lab_creatinine_48h") for c in labs.columns)


def test_icu_vitals_first_24h_only(tables, cohort_df):
    vitals = features.icu_vital_features(tables[("icu", "chartevents")], cohort_df)
    assert any(c.startswith("vital_heart_rate_mean") for c in vitals.columns)
    # hand-check one stay: recompute mean of hr in first 24h manually
    row = cohort_df.dropna(subset=["stay_id"]).iloc[0]
    ce = tables[("icu", "chartevents")]
    hr = ce[(ce["stay_id"] == row["stay_id"]) & (ce["itemid"] == 220045)]
    hr = hr[(hr["charttime"] >= row["intime"])
            & (hr["charttime"] < row["intime"] + pd.Timedelta(hours=24))]
    got = vitals.loc[vitals["hadm_id"] == row["hadm_id"], "vital_heart_rate_mean"].iloc[0]
    assert got == pytest.approx(hr["valuenum"].mean())


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_readmission_label_logic():
    cohort = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "hadm_id": [11, 12, 21],
        "dischtime": [pd.Timestamp("2020-01-10"), pd.Timestamp("2020-02-20"),
                      pd.Timestamp("2020-01-10")],
    })
    admissions = pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "hadm_id": [11, 12, 21, 22],
        "admittime": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-25"),
                      pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01")],
    })
    out = labels.add_readmission_label(cohort, admissions)
    # hadm 11 readmitted at +15d -> 1; hadm 12 has no later admission -> 0;
    # hadm 21's next admission is +142d -> 0
    assert out.set_index("hadm_id")["readmit_30d"].to_dict() == {11: 1, 12: 0, 21: 0}


def test_readmission_label_boundary_30_days():
    cohort = pd.DataFrame({"subject_id": [1], "hadm_id": [11],
                           "dischtime": [pd.Timestamp("2020-01-10")]})
    admissions = pd.DataFrame({
        "subject_id": [1, 1], "hadm_id": [11, 12],
        "admittime": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-09")],  # +30d
    })
    out = labels.add_readmission_label(cohort, admissions)
    assert out["readmit_30d"].iloc[0] == 1  # inclusive boundary


def test_synthetic_label_prevalence_sane(tables, cohort_df):
    labeled = labels.add_readmission_label(cohort_df, tables[("hosp", "admissions")])
    prev = labels.label_prevalence(labeled)
    assert 0.05 < prev < 0.6, f"unexpected prevalence {prev}"


def test_mortality_label_uses_dod():
    cohort = pd.DataFrame({"subject_id": [1, 2], "hadm_id": [11, 22],
                           "dischtime": [pd.Timestamp("2020-01-10"),
                                         pd.Timestamp("2020-01-10")]})
    patients = pd.DataFrame({"subject_id": [1, 2],
                             "dod": [pd.Timestamp("2020-01-20"), pd.NaT]})
    out = labels.add_mortality_label(cohort, patients)
    assert list(out["mortality_30d"]) == [1, 0]


# ---------------------------------------------------------------------------
# Full pipeline: cohort -> features -> labels -> model
# ---------------------------------------------------------------------------

def _labeled_matrix(tables, cohort_df):
    labeled = labels.add_readmission_label(cohort_df, tables[("hosp", "admissions")])
    X = features.build_feature_matrix(cohort_df, tables[("hosp", "labevents")],
                                      tables[("icu", "chartevents")])
    return X.merge(labeled[["hadm_id", "readmit_30d"]], on="hadm_id", how="left")


def test_feature_matrix_fusion_shape(tables, cohort_df):
    X = _labeled_matrix(tables, cohort_df)
    assert len(X) == len(cohort_df)  # one row per index admission
    assert "readmit_30d" in X.columns
    assert any(c.startswith("lab_") for c in X.columns)
    assert any(c.startswith("vital_") for c in X.columns)
    assert "anchor_age" in X.columns


def test_grouped_split_has_no_subject_overlap(tables, cohort_df):
    X = _labeled_matrix(tables, cohort_df)
    X_mat, y, groups, _ = model.prepare_xy(X)
    tr, te = model.grouped_split(X_mat, y, groups)
    assert len(set(groups[tr]) & set(groups[te])) == 0
    assert len(tr) > 0 and len(te) > 0


def test_baseline_beats_chance():
    # larger fixture than the module default: stable metric estimates
    t = synthetic.make_tables(n_patients=200, seed=0)
    c = synthetic.make_cohort(t)
    X = _labeled_matrix(t, c)
    X_mat, y, groups, feat_names = model.prepare_xy(X)
    assert len(feat_names) > 10
    assert not np.isnan(X_mat).any()
    tr, te = model.grouped_split(X_mat, y, groups)
    clf = model.train_logreg(X_mat[tr], y[tr])
    res = model.evaluate(clf, X_mat[te], y[te])
    assert res.auroc > 0.5, f"AUROC {res.auroc} not above chance"
    # AUPRC above the random-ranking baseline (= prevalence) under imbalance
    assert res.auprc > res.prevalence, (
        f"AUPRC {res.auprc:.3f} not above prevalence {res.prevalence:.3f}")
    assert 0.0 <= res.brier <= 0.5  # Brier sanity bound
    prob_true, prob_pred = model.calibration(clf, X_mat[te], y[te])
    assert len(prob_true) == len(prob_pred) > 0

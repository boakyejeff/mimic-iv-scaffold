"""Dataset configuration for the MIMIC-IV v3.1 readmission-prediction scaffold.

All values here are *documented constants* taken from the MIMIC-IV v3.1
schema (see docs/SCHEMAS.md). Nothing in this file touches real data —
it only names where the data will live once credentialing is complete.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
DATASET_VERSION = "3.1"
PHYSIONET_PROJECT = "https://physionet.org/content/mimiciv/3.1/"

# BigQuery location (after credentialing + project whitelisting)
BIGQUERY_PROJECT = "physionet-data"  # official PhysioNet GCP project
BIGQUERY_HOSP_DATASET = "mimiciv_v3_1_hosp"
BIGQUERY_ICU_DATASET = "mimiciv_v3_1_icu"

# Local Postgres layout (mimic-code build scripts create one database)
POSTGRES_DB = "mimiciv"
POSTGRES_HOSP_SCHEMA = "mimiciv_hosp"
POSTGRES_ICU_SCHEMA = "mimiciv_icu"

# Local raw-data paths (never committed — see .gitignore and the DUA note)
DATA_DIR = "data"
HOSP_CSV_DIR = f"{DATA_DIR}/raw/hosp"
ICU_CSV_DIR = f"{DATA_DIR}/raw/icu"
PROCESSED_DIR = f"{DATA_DIR}/processed"

# Service-account key for BigQuery (never committed)
GCP_SERVICE_ACCOUNT_KEY = "config/gcp-service-account.json"


@dataclass(frozen=True)
class CohortCriteria:
    """Inclusion / exclusion rules for the index-admission cohort."""

    min_age: int = 18                      # adult admissions only
    prediction_label: str = "readmit_30d"  # or "mortality_30d"
    # First ICU stay per hospital admission; admissions without an ICU stay
    # are kept (ICU features become NaN) — the fusion model handles this.
    first_icu_stay_only: bool = True
    # Exclusions
    exclude_in_hospital_death: bool = True     # can't be readmitted
    exclude_missing_dischtime: bool = True
    exclude_discharge_against_advice: bool = True  # discharge_location 'AGAINST ADVICE'


@dataclass(frozen=True)
class FeatureWindows:
    """Temporally-causal observation windows, all ending at discharge."""

    # Hosp labs: last value in each of the 48h / 24h windows before dischtime
    lab_windows_hours: tuple = (48, 24)
    # ICU vitals: first 24h of the ICU stay (intime → intime + 24h)
    icu_vital_window_hours: int = 24
    # Anti-leakage rule: no measurement with charttime >= dischtime may be
    # used as a feature. Prediction point = dischtime.
    prediction_point: str = "dischtime"


# Canonical vital-sign itemids from icu.d_items (labels in docs/SCHEMAS.md)

@dataclass(frozen=True)
class ModelConfig:
    """Baseline modelling choices."""

    target: str = "readmit_30d"
    test_size: float = 0.2
    group_col: str = "subject_id"   # grouped split: no patient in train+test
    random_state: int = 42
    primary_metric: str = "auprc"   # readmission is imbalanced
    use_lightgbm: bool = True       # falls back to logistic regression


COHORT = CohortCriteria()
WINDOWS = FeatureWindows()
MODEL = ModelConfig()

# Vital-sign itemids (icu.d_items.itemid → label). Used by features.py.
VITAL_ITEMIDS: dict[int, str] = {
    220045: "heart_rate",
    220179: "sbp",        # non-invasive systolic BP
    220180: "dbp",        # non-invasive diastolic BP
    220210: "resp_rate",
    220277: "spo2",
    223762: "temperature_c",
}

# Lab itemids (hosp.d_labitems.itemid → label) used by the fusion features.
LAB_ITEMIDS: dict[int, str] = {
    50912: "creatinine",
    51006: "bun",
    50983: "sodium",
    50971: "potassium",
    50882: "bicarbonate",
    50931: "glucose",
    51221: "hematocrit",
    51222: "hemoglobin",
    51301: "wbc",
    51265: "platelets",
    50868: "anion_gap",
}

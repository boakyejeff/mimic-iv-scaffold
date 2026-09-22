# MIMIC-IV v3.1 — Key Table Schemas

Source: MIMIC-IV v3.1 documentation on PhysioNet
(https://physionet.org/content/mimiciv/3.1/), as summarized in the
dataset staging report. These are the tables this scaffold uses.

## Join keys

```
subject_id ──▶ hadm_id ──▶ stay_id
  (patient)     (hospital        (ICU stay)
                 admission)
itemid ──▶ d_items          (concept dictionary: itemid → label)
itemid ──▶ d_labitems       (hosp labs dictionary)
```

- `subject_id`: unique patient identifier.
- `hadm_id`: unique hospital admission identifier.
- `stay_id`: unique ICU stay identifier.
- `itemid`: concept code; resolved to a human label via `d_items`
  (ICU) or `d_labitems` (hosp).

## `hosp` module — `mimiciv_v3_1_hosp`

| Table | Key columns | Used for |
|---|---|---|
| `patients` | `subject_id`, `gender`, `anchor_age`, `anchor_year`, `dod` | demographics; mortality label (`dod` = de-identified date of death) |
| `admissions` | `hadm_id`, `subject_id`, `admittime`, `dischtime`, `admission_type`, `admission_location`, `discharge_location`, `insurance`, `hospital_expire_flag` | cohort definition; **readmission label** (next `admittime` within 30d of `dischtime`) |
| `transfers` | `hadm_id`, `transfer_id`, `eventtype`, `careunit`, `intime`, `outtime` | unit-level movement history |
| `diagnoses_icd` | `hadm_id`, `seq_num`, `icd_code`, `icd_version` | comorbidity features (index admission only) |
| `procedures_icd` | `hadm_id`, `seq_num`, `icd_code`, `icd_version`, `chartdate` | procedure features |
| `prescriptions` | `hadm_id`, `pharmacy_id`, `drug`, `dose_val_rx`, `dose_unit_rx`, `starttime`, `stoptime` | medication features |
| `pharmacy` | `hadm_id`, `pharmacy_id`, `medication`, `doses_per_24_hrs` | dispensing detail |
| `labevents` | `labevent_id`, `hadm_id`, `specimen_id`, `itemid`, `charttime`, `valuenum`, `valueuom`, `flag` | **pre-discharge lab features** (`itemid` → `d_labitems`) |
| `microbiologyevents` | `hadm_id`, `charttime`, `spec_type_desc`, `org_name`, `ab_name` | infection features |
| `emar` | `hadm_id`, `emar_id`, `charttime`, `medication`, `event_txt` | medication administration record |
| `poe` | `hadm_id`, `poe_id`, `ordertime`, `order_type` | provider order entry |
| `omr` | `subject_id`, `chartdate`, `result_name`, `result_value` | outpatient measurements (longitudinal) |
| `services` | `hadm_id`, `transfertime`, `curr_service` | care service per admission |
| `provider` | `provider_id` (+ specialty fields) | provider dimension |

## `icu` module — `mimiciv_v3_1_icu`

| Table | Key columns | Used for |
|---|---|---|
| `icustays` | `stay_id`, `subject_id`, `hadm_id`, `intime`, `outtime`, `los`, `first_careunit`, `last_careunit` | first-ICU-stay linkage per admission |
| `chartevents` | `stay_id`, `itemid`, `charttime`, `valuenum`, `valueuom` | **first-24h vital features** (`itemid` → `d_items`) |
| `inputevents` | `stay_id`, `itemid`, `starttime`, `endtime`, `amount` | fluids / drug infusions in |
| `outputevents` | `stay_id`, `itemid`, `charttime`, `value` | fluids out (urine, drains) |
| `procedureevents` | `stay_id`, `itemid`, `starttime`, `endtime` | bedside procedures |
| `datetimeevents` | `stay_id`, `itemid`, `charttime`, `value` | timestamped events |
| `ingredientevents` | `stay_id`, `itemid`, `starttime`, `endtime`, `amount` | ingredient-level input detail |
| `d_items` | `itemid`, `label`, `category`, `unitname` | **ICU concept dictionary** |
| `caregiver` | `caregiver_id` | caregiver dimension |

## Canonical concept ids used by this scaffold

ICU vitals (`icu.d_items`):

| itemid | label | scaffold name |
|---|---|---|
| 220045 | Heart Rate | `heart_rate` |
| 220179 | Non Invasive Blood Pressure systolic | `sbp` |
| 220180 | Non Invasive Blood Pressure diastolic | `dbp` |
| 220210 | Respiratory Rate | `resp_rate` |
| 220277 | O2 saturation pulseoxymetry | `spo2` |
| 223762 | Temperature Celsius | `temperature_c` |

Hosp labs (`hosp.d_labitems`):

| itemid | label | scaffold name |
|---|---|---|
| 50912 | Creatinine | `creatinine` |
| 51006 | Urea Nitrogen (BUN) | `bun` |
| 50983 | Sodium | `sodium` |
| 50971 | Potassium | `potassium` |
| 50882 | Bicarbonate | `bicarbonate` |
| 50931 | Glucose | `glucose` |
| 51221 | Hematocrit | `hematocrit` |
| 51222 | Hemoglobin | `hemoglobin` |
| 51301 | White Blood Cells | `wbc` |
| 51265 | Platelet Count | `platelets` |
| 50868 | Anion Gap | `anion_gap` |

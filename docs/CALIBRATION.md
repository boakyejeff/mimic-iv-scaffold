# Synthetic-data calibration table

The generator in `src/synthetic.py` is calibrated to **published MIMIC-IV
marginals** — univariate targets only. It reproduces *what the literature
reports about MIMIC-IV in aggregate*; it does **not** reproduce joint
distributions, correlations, clinical workflows, or any real patient's
record. All model metrics in this repo are computed on synthetic data and
will change on real MIMIC-IV.

Conventions in the table below:

- **Target** — the marginal the generator is tuned to.
- **Source** — the publication the target comes from (verified, not invented).
- **(plausible range)** — no exact published figure was available; the
  target is a clearly-labelled design choice inside a clinically plausible
  range.
- **(design choice)** — a generator-design choice, not a MIMIC-measured
  value; labelled as such everywhere it appears.

| # | Variable | Target | Source |
|---|----------|--------|--------|
| 1 | 30-day readmission prevalence | 12.5% (tuned by bisection so the *realized label* hits the target) | (plausible range) between 8.7% 30-day ICU readmission in MIMIC-IV (Momenzadeh et al., *Sci Rep* 2026, DOI 10.1038/s41598-026-70894-8) and 17.6% in a critically-ill heart-failure MIMIC-IV v3.1 cohort (Odoeke et al., *Cureus* 2026, DOI 10.7759/cureus.111435) |
| 2 | 30-day mortality prevalence | ~3.5% | (plausible range): MIMIC-IV in-hospital mortality is 2.1% (Johnson et al., *Sci Data* 2023, DOI 10.1038/s41597-022-01899-x); 30-day post-discharge mortality is set slightly higher as a design choice |
| 3 | Age | mean 58.8, SD 19.2 (capped at 91 like MIMIC `anchor_age`) | Johnson et al., *Sci Data* 2023 — hospital-admissions cohort table |
| 4 | Sex | 52.2% female | Johnson et al., *Sci Data* 2023 |
| 5 | Insurance | Medicare 37.2% / Medicaid 9.6% / Other 53.2% | Johnson et al., *Sci Data* 2023 |
| 6 | Hospital length of stay | mean 4.5 d (lognormal, clipped 1–60 d) | Johnson et al., *Sci Data* 2023 (mean 4.5, SD 6.6) |
| 7 | ICU-stay share of admissions | ~17% | Johnson et al., *Sci Data* 2023: 73,181 ICU stays / 431,231 hospital admissions; v3.1 scale 94,458 / 546,028 (Johnson et al., PhysioNet 2024, DOI 10.13026/kpb9-mt58) |
| 8 | Diabetes prevalence | ~19.3% | medicalcoder paper, *JAMIA Open* 2026, DOI 10.1093/jamiaopen/ooag182: 12.32% w/o + 7.02% w/ chronic complications (encounter-level, MIMIC-IV v3.1) |
| 9 | Congestive heart failure | ~12.3% | same as above: 12.33% |
| 10 | Chronic kidney disease (renal disease) | ~12.5% | same as above: 12.53% |
| 11 | COPD (chronic pulmonary disease) | ~14.6% | same as above: 14.61% |
| 12 | Vital signs, first 24h ICU (HR, SBP, DBP, resp rate, SpO2, temperature) | plausible adult ranges, e.g. HR 84±14, SBP 122±20, SpO2 97.5±2.2 | (design choice): clinical reference ranges shifted by latent risk; not MIMIC-measured |
| 13 | Pre-discharge labs (creatinine, BUN, Na, K, HCO3, glucose, Hct, Hb, WBC, platelets, anion gap) | plausible adult ranges, e.g. creatinine ~1.05, Na 139.5±3.2, glucose ~110 (×1.65 if diabetic) | (design choice): clinical reference ranges with comorbidity/risk shifts; not MIMIC-measured |
| 14 | Missingness per variable | vitals 3% hourly gaps; labs 2–18% never ordered (anion gap highest at 18%) | (design choice): mimics "not ordered" patterns in EHR data; ICU-less admissions have no vitals at all |

## What the generator does and does not capture

- **Does:** univariate marginals above; a single latent risk score that
  jointly drives abnormal physiology, comorbidities, LOS, ICU admission,
  readmission, and mortality (so features genuinely predict the label);
  age-correlated comorbidity rates; realistic per-variable missingness.
- **Does not:** real joint distributions or correlations between labs;
  real clinical workflows (ordering patterns, treatment effects); real
  temporal dynamics within a stay; diagnosis/procedure codes; medications;
  notes; any real patient's data.

## Honesty notes

- Subgroup gaps in synthetic data (age bands, sex) reflect the
  generator's assumptions — e.g. risk rising with age and comorbidities —
  **not real disparities**. Do not cite them as findings about MIMIC-IV.
- Model metrics (AUROC/AUPRC/ECE) measure how well models recover the
  *synthetic* signal. They validate the pipeline, not the medicine.
- The realized marginals of any generated cohort can be checked with
  `src.synthetic.calibration_summary(tables, cohort)`.

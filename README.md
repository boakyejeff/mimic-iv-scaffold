# MIMIC-IV Readmission Prediction — Pipeline Scaffold

> **⚠️ SCAFFOLD — PENDING CREDENTIALING.**
> This repo is a complete, runnable pipeline *skeleton*. It contains **no
> MIMIC-IV data** and cannot download any: access requires PhysioNet
> credentialing + CITI training + a signed Data Use Agreement
> (see [`docs/ACCESS-GUIDE.md`](docs/ACCESS-GUIDE.md)). Until then, every
> stage runs end-to-end on built-in synthetic data (`--dry-run`).

## Goal

**30-day hospital readmission prediction by fusing the `hosp` and `icu`
modules of MIMIC-IV v3.1** — pre-discharge labs (`hosp.labevents`) plus
first-24h ICU vitals (`icu.chartevents`) plus demographics. Most public
MIMIC tutorials use only the ICU module; the cross-module fusion is the
novel angle here. A 30-day mortality label is included as an alternative
target.

Prediction point is **discharge** (`dischtime`); all features are
temporally causal with explicit anti-leakage rules
(see `src/features.py`).

## Synthetic data (calibrated — no real data)

The generator in `src/synthetic.py` is calibrated to **published
MIMIC-IV marginals** (full table in [`docs/CALIBRATION.md`](docs/CALIBRATION.md)):

- 30-day readmission prevalence tuned to **~12.5%** — between the 8.7%
  30-day ICU-readmission rate in MIMIC-IV (Momenzadeh et al., *Sci Rep*
  2026) and 17.6% in a critically-ill heart-failure MIMIC-IV v3.1 cohort
  (Odoeke et al., *Cureus* 2026).
- Age (mean 58.8), female share (52.2%), insurance mix, and mean LOS
  (4.5 d) from the MIMIC-IV cohort table (Johnson et al., *Sci Data*
  2023); dataset scale 364,627 patients / 546,028 hospitalizations /
  94,458 ICU stays (Johnson et al., PhysioNet 2024, MIMIC-IV v3.1).
- Comorbidity prevalences (diabetes ~19.3%, CHF ~12.3%, renal disease
  ~12.5%, chronic pulmonary disease ~14.6%) from Charlson
  encounter-level prevalences reported for MIMIC-IV v3.1 (medicalcoder
  paper, *JAMIA Open* 2026).
- Vital-sign and lab distributions are **plausible clinical ranges**
  (design choices, not MIMIC-measured values); per-variable missingness
  mimics EHR "not ordered" patterns.

A single latent risk score drives abnormal physiology, comorbidities,
LOS, ICU admission, readmission, and mortality, so the label genuinely
correlates with the features. Check realized marginals any time with
`src.synthetic.calibration_summary(tables, cohort)`.

## Repo layout

```
mimic-iv-scaffold/
├── README.md
├── requirements.txt
├── .gitignore
├── config/                 # runtime config (service-account key goes here, never committed)
├── docs/
│   ├── ACCESS-GUIDE.md     # step-by-step credentialing + DUA + download guide
│   ├── SCHEMAS.md          # key tables, columns, join keys, canonical itemids
│   └── CALIBRATION.md      # synthetic-data calibration table + honesty notes
├── src/
│   ├── config.py           # dataset version, BigQuery/Postgres names, cohort criteria
│   ├── cohort.py           # index-admission cohort builder (adult, first ICU stay, exclusions)
│   ├── features.py         # temporally-causal lab/vital/demographic features + anti-leakage
│   ├── labels.py           # 30-day readmission + mortality labels
│   ├── model.py            # grouped split, logreg/LightGBM, calibration, ECE, subgroups
│   ├── data_loader.py      # one interface: BigQuery / Postgres / SQLite / synthetic
│   └── synthetic.py        # calibrated synthetic tables (seeded, reproducible)
├── scripts/
│   ├── run_pipeline.py     # CLI entry point (--dry-run works today)
│   └── run_model_comparison.py  # 6-model comparison on a fixed 20k-admission cohort
├── results/
│   └── metrics.json        # aggregate model-comparison metrics (synthetic only)
├── tests/
│   └── test_synthetic.py   # pipeline + calibration + reproducibility tests
└── notebooks/              # (empty — for exploration after access)
```

## Quickstart

### Today (no credentials needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full pipeline on synthetic data: cohort -> features -> labels -> model
python scripts/run_pipeline.py --dry-run

# full 6-model comparison on a fixed 20k-admission synthetic cohort
# (AUROC / AUPRC / Brier / ECE + reliability curves + age/sex subgroups)
python scripts/run_model_comparison.py --n-admissions 20000 --seed 42

# test suite (proves the scaffold works end-to-end)
pytest tests/ -q
```

### After credentialing (see `docs/ACCESS-GUIDE.md`)

```bash
# BigQuery route
python scripts/run_pipeline.py --mode bigquery \
    --gcp-key config/gcp-service-account.json

# Local Postgres route (mimic-code build)
python scripts/run_pipeline.py --mode postgres \
    --dsn postgresql://user:pass@localhost:5432/mimiciv
```

## Model comparison (synthetic data only)

`scripts/run_model_comparison.py` generates a fixed cohort
(**19,999 admissions / 13,613 patients, seed 42**, 94 features), splits it
grouped-by-patient (80/20), and compares six variants. Realized cohort
marginals vs targets: readmission 12.4% (target 12.5%), 30-day mortality
3.3% (target ~3.5%), mean age 60.0 (58.8), female 51.8% (52.2%), ICU
share 17.6% (17%), mean LOS 4.5 d (4.5), Medicare 36.7% (37.2%).

| Model | AUROC | AUPRC | Brier | ECE |
|-------|------:|------:|------:|----:|
| L2 logistic regression | **0.701** | **0.243** | **0.101** | **0.006** |
| LogReg + Platt | 0.701 | 0.243 | 0.101 | 0.008 |
| LogReg + isotonic | 0.701 | 0.245 | 0.101 | 0.007 |
| LightGBM | 0.679 | 0.217 | 0.104 | 0.022 |
| LightGBM + Platt | 0.693 | 0.241 | 0.102 | 0.014 |
| LightGBM + isotonic | 0.694 | 0.237 | 0.102 | 0.011 |

Held-out test: n = 3,985, prevalence 12.2%. Full numbers, reliability-curve
bin data, and per-model subgroups are in [`results/metrics.json`](results/metrics.json).

**Reading these honestly:** plain L2 logistic regression wins on both
discrimination and calibration here — *expected*, because the synthetic
signal is linear-in-latent-risk by construction (logistic link), which
makes logistic regression the correctly-specified model. Calibration
(Platt/isotonic) helps LightGBM's ECE (0.022 → 0.011) but barely moves
logistic regression, which is already calibrated. **On real MIMIC-IV this
ranking can easily flip** — gradient boosting routinely beats linear
models on real EHR data. These numbers validate the *pipeline* (data
flow, grouped splits, calibration machinery), not the medicine.

## Subgroup checks (synthetic data only)

AUROC / AUPRC / ECE by age band and sex, for the best model (logistic
regression). n = held-out test rows per subgroup.

| Subgroup | n | Prevalence | AUROC | AUPRC | ECE |
|----------|---:|----------:|------:|------:|----:|
| age <65 | 2,345 | 7.2% | 0.631 | 0.148 | 0.004 |
| age 65–79 | 1,002 | 16.2% | 0.612 | 0.240 | 0.011 |
| age 80+ | 638 | 24.5% | 0.592 | 0.317 | 0.017 |
| female | 2,099 | 13.0% | 0.697 | 0.248 | 0.017 |
| male | 1,886 | 11.5% | 0.711 | 0.245 | 0.014 |

**Honest note:** these gaps reflect the *generator's assumptions*, not
real disparities. Prevalence rises with age by construction, and AUROC
falls within older bands because age dominates the synthetic risk score —
once you condition on an age band, less signal remains to discriminate
with. The machinery for subgroup auditing (grouped splits, per-group
ECE) is what carries over to real data; the numbers themselves do not.

## Data Use Agreement note

MIMIC-IV is de-identified but **gated by a Data Use Agreement**. By using
this repo with real data you agree to:

- **Never commit raw or derived patient data** — `data/`, `*.csv`,
  credentials, and service-account keys are all git-ignored. Only code,
  configs, and aggregate results belong on GitHub.
- Never redistribute the dataset; keep it on secured systems.
- Use it only for the research purpose in your credentialing application.

Full terms: https://physionet.org/sign-dua/mimiciv/3.1/

## Status

- [x] Pipeline scaffold with real interfaces + synthetic testability
- [x] Synthetic end-to-end tests passing (20/20)
- [x] Calibrated synthetic generator (published MIMIC-IV marginals;
      see `docs/CALIBRATION.md`)
- [x] 6-model comparison on a fixed 20k-admission synthetic cohort
      (see "Model comparison" below)
- [ ] PhysioNet credentialing (user action — see `docs/ACCESS-GUIDE.md`)
- [ ] First real-data run

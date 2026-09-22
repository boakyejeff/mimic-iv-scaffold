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

## Repo layout

```
mimic-iv-scaffold/
├── README.md
├── requirements.txt
├── .gitignore
├── config/                 # runtime config (service-account key goes here, never committed)
├── docs/
│   ├── ACCESS-GUIDE.md     # step-by-step credentialing + DUA + download guide
│   └── SCHEMAS.md          # key tables, columns, join keys, canonical itemids
├── src/
│   ├── config.py           # dataset version, BigQuery/Postgres names, cohort criteria
│   ├── cohort.py           # index-admission cohort builder (adult, first ICU stay, exclusions)
│   ├── features.py         # temporally-causal lab/vital/demographic features + anti-leakage
│   ├── labels.py           # 30-day readmission + mortality labels
│   ├── model.py            # grouped-by-patient split, logreg/LightGBM, AUROC/AUPRC/calibration
│   ├── data_loader.py      # one interface: BigQuery / Postgres / SQLite / synthetic
│   └── synthetic.py        # credential-free synthetic tables matching the documented schemas
├── scripts/
│   └── run_pipeline.py     # CLI entry point (--dry-run works today)
├── tests/
│   └── test_synthetic.py   # full pipeline tests on synthetic data, no real data
└── notebooks/              # (empty — for exploration after access)
```

## Quickstart

### Today (no credentials needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# full pipeline on synthetic data: cohort -> features -> labels -> model
python scripts/run_pipeline.py --dry-run

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
- [x] Synthetic end-to-end tests passing
- [ ] PhysioNet credentialing (user action — see `docs/ACCESS-GUIDE.md`)
- [ ] First real-data run

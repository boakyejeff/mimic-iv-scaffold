# Build Notes — mimic-iv-scaffold

Built 2026-09-22. Repo: `~/workspace/medical-datasets/projects/mimic-iv-scaffold`

## What was built

A **code-scaffold-only** repo for 30-day readmission prediction on
MIMIC-IV v3.1, fusing `hosp` labs + `icu` vitals (cross-module fusion —
most public tutorials use only the ICU module). No real data was touched;
no download is possible until the user completes PhysioNet credentialing.

**Files:**
- `README.md` — "SCAFFOLD — pending credentialing" banner, goal,
  quickstart (synthetic today / real after access), DUA note.
- `requirements.txt` — pandas, numpy, scikit-learn, scipy, sqlalchemy,
  psycopg2-binary, google-cloud-bigquery, pyarrow, matplotlib,
  lightgbm (optional), pytest.
- `.gitignore` — excludes `data/`, `*.csv`, `*.parquet`, credentials,
  service-account keys (DUA compliance).
- `src/config.py` — v3.1 constants, BigQuery/Postgres names, cohort
  criteria, feature windows, canonical vital/lab itemids.
- `src/cohort.py` — index-admission cohort SQL (BigQuery + Postgres
  dialects): adults, first ICU stay per admission, exclusions
  (in-hospital death, missing dischtime, AMA discharge).
- `src/features.py` — temporally-causal windows (labs 48h/24h before
  discharge; ICU vitals first 24h; demographics) with **enforced**
  anti-leakage (drops any `charttime >= dischtime`).
- `src/labels.py` — 30-day readmission from `admissions` (next admittime
  within (0, 30] days of dischtime; boundary-tested), 30-day mortality
  from `patients.dod` as alternative target.
- `src/model.py` — grouped-by-`subject_id` split, scaled logistic
  regression baseline (+ optional LightGBM), AUROC/AUPRC/Brier/calibration.
- `src/data_loader.py` — one interface: BigQuery / Postgres / SQLite /
  synthetic loaders.
- `src/synthetic.py` — credential-free synthetic tables matching the
  documented schemas, with a risk-driven signal (risk → abnormal
  creatinine/HR/age → readmission probability).
- `scripts/run_pipeline.py` — CLI with `--dry-run` (synthetic, works today),
  `--mode bigquery`, `--mode postgres`.
- `tests/test_synthetic.py` — 14 tests: schema conformance, cohort rules,
  anti-leakage, label logic incl. 30-day boundary, grouped-split
  no-overlap, end-to-end baseline beats chance.
- `docs/ACCESS-GUIDE.md` — step-by-step: PhysioNet registration → CITI
  "Data or Specimens Only Research" → credentialing → DUA signature →
  BigQuery or CSV+Postgres download, with exact URLs + checklist.
- `docs/SCHEMAS.md` — hosp/icu tables, columns, join keys
  (`subject_id → hadm_id → stay_id`; `itemid → d_items`), canonical
  vital/lab itemids — copied from the staging report.

## Test results (actually observed)

```
$ python -m pytest tests/ -q
14 passed in 3.23s          # (venv: Python 3.12, sklearn 1.9.1, pandas 3.0.6)

$ python scripts/run_pipeline.py --dry-run --n-patients 200 --seed 0
[synthetic] n_admissions=307 n_features=72 prevalence=0.224
[synthetic] AUROC=0.689 AUPRC=0.392 Brier=0.170
```

Design iterations during build (kept honest): an early synthetic
generator drew readmission labels independent of features (pure noise);
fixed so risk drives both physiology and readmission. Small fixtures
(n=80, 72 features) overfit — model test uses n=200 where AUROC is
0.59–0.78 across 10 seeds. `class_weight="balanced"` was dropped from
the baseline because it wrecked calibration (Brier); the scaffold
evaluates Brier/calibration, so the default stays unweighted.

## Pending item (user action)

**PhysioNet credentialing** — the only hard blocker. Follow
`docs/ACCESS-GUIDE.md`: register → CITI course → credentialing
application → sign the MIMIC-IV v3.1 DUA. Then run
`python scripts/run_pipeline.py --mode bigquery --gcp-key config/gcp-service-account.json`
(or the Postgres route). Suggested sanity check on first real run:
cohort counts vs published v3.1 scale (~364,627 patients / ~546,028
hospitalizations / ~94,458 ICU stays).

## Local git

`git init` done; nothing committed or pushed (no auth available, and no
push was requested). Suggested first commit by the user after review.

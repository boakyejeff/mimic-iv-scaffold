# MIMIC-IV Access Guide

MIMIC-IV is **gated**: you cannot download any of it until you are a
credentialed PhysioNet user who has signed the Data Use Agreement. This
guide walks through every step. Budget **1–3 weeks** end to end (the CITI
course is the long pole; credentialing review usually takes a few days).

> Nothing in this repo downloads or contains MIMIC-IV data. The pipeline
> runs on synthetic data (`--dry-run`) until you finish the steps below.

## Step 0 — Check eligibility

You need an institutional affiliation (university, hospital, company) that
PhysioNet can verify. Independent researchers without an affiliation are
generally not approved.

## Step 1 — Register a PhysioNet account

1. Go to **https://physionet.org/register/**
2. Register with your institutional email if you have one.
3. Confirm the verification email.

## Step 2 — Complete CITI training

MIMIC-IV requires the **"Data or Specimens Only Research"** CITI course.

1. Read the instructions: **https://physionet.org/about/citi-course/**
2. Complete the course on the CITI Program site (covers human-subjects
   research basics; typically a few hours).
3. Save the completion report — you will attach it in the next step.

Tip: start this *now*; it's the slowest step and everything else waits on it.

## Step 3 — Apply for PhysioNet credentialing

1. Go to **https://physionet.org/credential-application/**
2. Fill in your profile, institutional affiliation, and research purpose.
3. Upload your CITI completion report.
4. Submit and wait for approval (a real person reviews it).

## Step 4 — Sign the MIMIC-IV Data Use Agreement (DUA)

1. Go to **https://physionet.org/sign-dua/mimiciv/3.1/**
2. Read and sign the DUA. Key obligations:
   - **Never redistribute the data** (no posting CSVs, no committing them
     to GitHub — this repo's `.gitignore` enforces this).
   - Use it only for the research purpose you stated.
   - Keep it on secured systems; don't share credentials.

## Step 5 — Get the data (two routes)

### Route A — BigQuery (recommended)

PhysioNet hosts MIMIC-IV v3.1 on Google BigQuery:

- `mimiciv_v3_1_hosp`
- `mimiciv_v3_1_icu`

under the `physionet-data` project. After credentialing:

1. Create a GCP project and a service account; download its JSON key to
   `config/gcp-service-account.json` (**never commit this file**).
2. Request access / link your credentialed identity per the instructions
   on the MIMIC-IV project page: **https://physionet.org/content/mimiciv/3.1/**
3. Run: `python scripts/run_pipeline.py --mode bigquery --gcp-key config/gcp-service-account.json`

Note: BigQuery charges for queries — the cohort/feature SQL here is
written to scan narrow column subsets to keep costs small.

### Route B — CSV download + local Postgres

1. On the project page (**https://physionet.org/content/mimiciv/3.1/**),
   download the `hosp` and `icu` CSV archives with your PhysioNet
   credentials (`wget --user ... --password ...` or the browser).
2. Build a local Postgres with the
   [mimic-code](https://github.com/MIT-LCP/mimic-code) loading scripts.
3. Run: `python scripts/run_pipeline.py --mode postgres --dsn postgresql://user:pass@localhost:5432/mimiciv`

## Step 6 — Verify your setup

```bash
python scripts/run_pipeline.py --dry-run   # synthetic smoke test, no credentials
```

Then point the pipeline at real data and compare cohort counts against
the published v3.1 scale (~364,627 patients, ~546,028 hospitalizations,
~94,458 ICU stays) as a sanity check.

## Checklist

- [ ] PhysioNet account registered (https://physionet.org/register/)
- [ ] CITI "Data or Specimens Only Research" completed (https://physionet.org/about/citi-course/)
- [ ] Credentialing approved (https://physionet.org/credential-application/)
- [ ] MIMIC-IV v3.1 DUA signed (https://physionet.org/sign-dua/mimiciv/3.1/)
- [ ] Data accessible via BigQuery or local Postgres
- [ ] `python scripts/run_pipeline.py --dry-run` passes

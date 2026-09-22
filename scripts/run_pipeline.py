#!/usr/bin/env python3
"""CLI entry point: run the readmission-prediction pipeline.

Modes:
    --dry-run            full pipeline on synthetic data (no credentials)
    --mode bigquery      real data via BigQuery (needs credentialing + key)
    --mode postgres      real data via local Postgres (needs mimic-code build)

Examples:
    python scripts/run_pipeline.py --dry-run
    python scripts/run_pipeline.py --mode bigquery --gcp-key config/gcp-service-account.json
    python scripts/run_pipeline.py --mode postgres --dsn postgresql://user:pass@localhost:5432/mimiciv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import cohort as cohort_mod  # noqa: E402
from src import config, data_loader, features, labels, model, synthetic  # noqa: E402


def run_synthetic(n_patients: int, seed: int) -> dict:
    tables = synthetic.make_tables(n_patients=n_patients, seed=seed)
    cohort_df = synthetic.make_cohort(tables)
    labeled = labels.add_readmission_label(cohort_df, tables[("hosp", "admissions")])
    X = features.build_feature_matrix(cohort_df, tables[("hosp", "labevents")],
                                      tables[("icu", "chartevents")])
    X = X.merge(labeled[["hadm_id", "readmit_30d"]], on="hadm_id", how="left")

    X_mat, y, groups, feat_names = model.prepare_xy(X)
    tr, te = model.grouped_split(X_mat, y, groups)
    clf = model.train_logreg(X_mat[tr], y[tr])
    res = model.evaluate(clf, X_mat[te], y[te])
    print(f"[synthetic] n_admissions={len(X)} n_features={len(feat_names)} "
          f"prevalence={res.prevalence:.3f}")
    print(f"[synthetic] AUROC={res.auroc:.3f} AUPRC={res.auprc:.3f} "
          f"Brier={res.brier:.3f}")
    assert res.auroc > 0.5, "synthetic signal check failed"
    return res.as_dict()


def run_bigquery(gcp_key: str | None) -> dict:
    loader = data_loader.BigQueryLoader(key_path=gcp_key)
    raise NotImplementedError(
        "BigQuery mode requires completed credentialing. "
        "See docs/ACCESS-GUIDE.md, then re-run with --gcp-key."
        if gcp_key is None else
        "BigQuery mode is scaffolded; wire cohort SQL via src.cohort.build_cohort()."
    )


def run_postgres(dsn: str) -> dict:
    loader = data_loader.PostgresLoader(dsn)
    cohort_df = cohort_mod.build_cohort(loader.engine, dialect="postgres")
    raise NotImplementedError(
        "Postgres mode is scaffolded; feature/label extraction mirrors run_synthetic()."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="MIMIC-IV readmission pipeline")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full pipeline on synthetic data")
    ap.add_argument("--mode", choices=["bigquery", "postgres"], default=None)
    ap.add_argument("--gcp-key", default=config.GCP_SERVICE_ACCOUNT_KEY)
    ap.add_argument("--dsn", default=None, help="Postgres DSN for --mode postgres")
    ap.add_argument("--n-patients", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dry_run or args.mode is None:
        run_synthetic(args.n_patients, args.seed)
    elif args.mode == "bigquery":
        run_bigquery(args.gcp_key)
    elif args.mode == "postgres":
        if not args.dsn:
            ap.error("--mode postgres requires --dsn")
        run_postgres(args.dsn)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Model comparison on the calibrated synthetic cohort (seeded, reproducible).

Generates a fixed-size synthetic cohort, builds the fusion feature matrix,
and compares six model variants on a grouped-by-patient held-out split:

    logreg, logreg+Platt, logreg+isotonic,
    lightgbm, lightgbm+Platt, lightgbm+isotonic

Reports AUROC, AUPRC, Brier score, and ECE overall, reliability-curve
bin data, and AUROC/AUPRC/ECE by age band (<65, 65–79, 80+) and sex.

ALL metrics are computed on SYNTHETIC data (see docs/CALIBRATION.md) and
will change on real MIMIC-IV. They are not clinical validation.

Usage:
    python scripts/run_model_comparison.py --n-admissions 20000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features, labels, model, synthetic  # noqa: E402


def gender_from_dummies(X) -> list[str]:
    """Recover M/F labels from the one-hot gender columns in the matrix."""
    cols = [c for c in X.columns if c.startswith("gender_")]
    if not cols:
        return ["?"] * len(X)
    return (X[cols].idxmax(axis=1).str.replace("gender_", "", regex=False)
            .tolist())


def main() -> None:
    ap = argparse.ArgumentParser(description="Model comparison on synthetic data")
    ap.add_argument("--n-admissions", type=int, default=20000,
                    help="target synthetic cohort size (admissions)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/metrics.json",
                    help="where to write the metrics JSON")
    args = ap.parse_args()

    tables = synthetic.make_tables(seed=args.seed,
                                   target_admissions=args.n_admissions)
    cohort_df = synthetic.make_cohort(tables)
    labeled = labels.add_readmission_label(cohort_df, tables[("hosp", "admissions")])
    X = features.build_feature_matrix(cohort_df, tables[("hosp", "labevents")],
                                      tables[("icu", "chartevents")])
    X = X.merge(labeled[["hadm_id", "readmit_30d"]], on="hadm_id", how="left")

    X_mat, y, groups, feat_names = model.prepare_xy(X)
    tr, te = model.grouped_split(X_mat, y, groups, random_state=args.seed)

    ages = X["anchor_age"].to_numpy()
    genders = gender_from_dummies(X)

    builders = {
        "logreg": lambda: model.make_logreg_pipeline(args.seed),
        "logreg_platt": lambda: model.train_calibrated(
            model.make_logreg_pipeline(args.seed), X_mat[tr], y[tr],
            method="sigmoid"),
        "logreg_isotonic": lambda: model.train_calibrated(
            model.make_logreg_pipeline(args.seed), X_mat[tr], y[tr],
            method="isotonic"),
        "lgbm": lambda: model.make_lgbm(args.seed),
        "lgbm_platt": lambda: model.train_calibrated(
            model.make_lgbm(args.seed), X_mat[tr], y[tr], method="sigmoid"),
        "lgbm_isotonic": lambda: model.train_calibrated(
            model.make_lgbm(args.seed), X_mat[tr], y[tr], method="isotonic"),
    }

    results_models, reliability, subgroups = {}, {}, {}
    for name, build in builders.items():
        clf = build()
        if name in ("logreg", "lgbm"):  # train_calibrated already fitted
            clf.fit(X_mat[tr], y[tr])
        proba = clf.predict_proba(X_mat[te])[:, 1]
        res = model.evaluate(clf, X_mat[te], y[te])
        results_models[name] = res.as_dict()
        reliability[name] = model.reliability_curve_data(y[te], proba)
        subgroups[name] = model.subgroup_metrics(y[te], proba, ages[te],
                                                 [genders[i] for i in te])
        r = res
        print(f"{name:>15}: AUROC={r.auroc:.4f} AUPRC={r.auprc:.4f} "
              f"Brier={r.brier:.4f} ECE={r.ece:.4f} "
              f"(n_test={r.n_test}, prev={r.prevalence:.3f})")

    payload = {
        "meta": {
            "n_admissions": int(len(X)),
            "n_patients": int(X["subject_id"].nunique()),
            "n_features": int(len(feat_names)),
            "seed": args.seed,
            "test_size": float(len(te) / len(X)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": "SYNTHETIC ONLY — calibrated generator, no MIMIC-IV data",
            "note": ("All metrics are computed on synthetic data calibrated "
                     "to published MIMIC-IV marginals (docs/CALIBRATION.md). "
                     "They will change on real MIMIC-IV and are not clinical "
                     "validation. Subgroup gaps reflect generator assumptions, "
                     "not real disparities."),
        },
        "calibration_check": synthetic.calibration_summary(tables, cohort_df),
        "models": results_models,
        "reliability": reliability,
        "subgroups": subgroups,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

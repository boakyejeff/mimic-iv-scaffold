"""Baseline models for 30-day readmission prediction.

* Grouped train/test split on ``subject_id`` — a patient never appears in
  both splits (prevents within-patient leakage for patients with multiple
  admissions).
* Baselines: L2 logistic regression; LightGBM if installed (optional);
  Platt-scaled and isotonic calibrated variants of each via
  :class:`sklearn.calibration.CalibratedClassifierCV`.
* Metrics: AUROC, AUPRC (primary — readmission is imbalanced), Brier
  score, expected calibration error (ECE), reliability-curve data, and
  subgroup metrics by age band and sex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except ImportError:  # optional dependency
    _HAS_LGBM = False


@dataclass
class EvalResult:
    """Held-out evaluation metrics for one fitted model."""

    auroc: float
    auprc: float
    brier: float
    n_test: int
    prevalence: float
    ece: float = 0.0

    def as_dict(self) -> dict:
        return {"auroc": self.auroc, "auprc": self.auprc,
                "brier": self.brier, "n_test": self.n_test,
                "prevalence": self.prevalence, "ece": self.ece}


def prepare_xy(X: pd.DataFrame, label_col: str = "readmit_30d",
               group_col: str = "subject_id"):
    """Split a labeled feature matrix into arrays for modelling.

    Non-feature columns (``subject_id``, ``hadm_id``, the label) are
    excluded from ``X``. NaNs are median-imputed; a ``<col>_was_missing``
    indicator is added for each column with missing values so the model
    can exploit missingness (common in EHR data).

    Returns:
        ``(X_mat, y, groups, feature_names)``.
    """
    drop = {group_col, "hadm_id", label_col}
    feature_names = [c for c in X.columns if c not in drop]
    Xf = X[feature_names].copy()
    for col in feature_names:
        if Xf[col].isna().any():
            Xf[f"{col}_was_missing"] = Xf[col].isna().astype(float)
    Xf = Xf.fillna(Xf.median(numeric_only=True)).fillna(0.0)
    y = X[label_col].to_numpy(dtype=int)
    groups = X[group_col].to_numpy()
    return Xf.to_numpy(dtype=float), y, groups, list(Xf.columns)


def grouped_split(X_mat, y, groups, test_size: float | None = None,
                  random_state: int | None = None):
    """GroupShuffleSplit on ``groups`` (default: subject_id).

    Returns:
        ``(X_train, X_test, y_train, y_test)`` index tuples.
    """
    cfg = config.MODEL
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=test_size or cfg.test_size,
                            random_state=random_state or cfg.random_state)
    train_idx, test_idx = next(gss.split(X_mat, y, groups))
    return train_idx, test_idx


def make_logreg_pipeline(random_state: int = 42):
    """Unfitted scaled L2 logistic-regression pipeline (factory for calibration)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )


def train_logreg(X_train, y_train, random_state: int = 42):
    """L2-regularized logistic regression baseline (with scaling).

    Unweighted: ``class_weight="balanced"`` improves recall but is known
    to distort predicted probabilities, and this scaffold evaluates
    Brier score / calibration — so the default baseline stays calibrated.
    """
    clf = make_logreg_pipeline(random_state)
    clf.fit(X_train, y_train)
    return clf


def make_lgbm(random_state: int = 42):
    """Unfitted LightGBM classifier. Raises ImportError if not installed."""
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed; install it or use train_logreg")
    return lgb.LGBMClassifier(random_state=random_state, verbose=-1)


def train_lgbm(X_train, y_train, random_state: int = 42):
    """LightGBM baseline. Raises ImportError if lightgbm is not installed."""
    clf = make_lgbm(random_state)
    clf.fit(X_train, y_train)
    return clf


def train_calibrated(base_estimator, X_train, y_train,
                     method: str = "sigmoid", cv: int = 5,
                     random_state: int = 42):
    """Fit a probability-calibrated wrapper around an unfitted estimator.

    Args:
        base_estimator: unfitted estimator (it is cloned internally).
        X_train, y_train: training data.
        method: ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
        cv: cross-validation folds used to fit the calibrator.
        random_state: passed through where the base estimator supports it.

    Returns:
        Fitted :class:`CalibratedClassifierCV`.
    """
    cal = CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal


def evaluate(model, X_test, y_test, n_bins: int = 10) -> EvalResult:
    """Compute AUROC / AUPRC / Brier / ECE on held-out data."""
    proba = model.predict_proba(X_test)[:, 1]
    return EvalResult(
        auroc=float(roc_auc_score(y_test, proba)),
        auprc=float(average_precision_score(y_test, proba)),
        brier=float(brier_score_loss(y_test, proba)),
        n_test=int(len(y_test)),
        prevalence=float(np.mean(y_test)),
        ece=float(expected_calibration_error(y_test, proba, n_bins=n_bins)),
    )


def calibration(model, X_test, y_test, n_bins: int = 10):
    """Return (prob_true, prob_pred) calibration curve points."""
    proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Expected calibration error with uniform probability bins.

    ECE = sum_b |acc(b) - conf(b)| * n_b / n. Lower is better; a perfectly
    calibrated model has ECE = 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (proba > lo) & (proba <= hi) if b > 0 else (proba <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc = y_true[mask].mean()
        conf = proba[mask].mean()
        ece += abs(acc - conf) * (n_b / len(y_true))
    return float(ece)


def reliability_curve_data(y_true, proba, n_bins: int = 10) -> dict:
    """Bin-level data for a reliability diagram (JSON-serializable).

    Returns dict with ``bin_edges``, ``bin_centers`` (mean predicted
    probability), ``observed`` (empirical positive rate), and ``counts``.
    """
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, observed, counts = [], [], []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (proba > lo) & (proba <= hi) if b > 0 else (proba <= hi)
        n_b = int(mask.sum())
        centers.append(float(proba[mask].mean()) if n_b else float((lo + hi) / 2))
        observed.append(float(y_true[mask].mean()) if n_b else float("nan"))
        counts.append(n_b)
    return {"bin_edges": [float(e) for e in edges],
            "bin_centers": centers, "observed": observed, "counts": counts}


def subgroup_metrics(y_true, proba, ages, genders) -> dict:
    """Metrics by age band (<65, 65–79, 80+) and sex.

    Args:
        y_true: binary labels. proba: predicted positive probabilities.
        ages: per-row age. genders: per-row "M"/"F" strings.

    Returns:
        Dict of subgroup name -> {n, prevalence, auroc, auprc, ece, brier}.
        AUROC/AUPRC are None when a subgroup lacks both classes.

    NOTE: on synthetic data, subgroup gaps reflect the generator's
    assumptions (age/comorbidity-driven risk), not real disparities.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    ages = np.asarray(ages, dtype=float)
    genders = np.asarray(genders, dtype=str)

    bands = np.where(ages < 65, "<65",
                     np.where(ages < 80, "65-79", "80+"))
    out = {}
    for name, mask in [("age_<65", bands == "<65"),
                       ("age_65-79", bands == "65-79"),
                       ("age_80+", bands == "80+"),
                       ("sex_F", genders == "F"),
                       ("sex_M", genders == "M")]:
        yt, pb = y_true[mask], proba[mask]
        n = int(mask.sum())
        entry = {"n": n, "prevalence": float(yt.mean()) if n else 0.0,
                 "auroc": None, "auprc": None, "ece": None, "brier": None}
        if n > 0 and yt.min() != yt.max():
            entry["auroc"] = float(roc_auc_score(yt, pb))
            entry["auprc"] = float(average_precision_score(yt, pb))
            entry["ece"] = float(expected_calibration_error(yt, pb))
            entry["brier"] = float(brier_score_loss(yt, pb))
        out[name] = entry
    return out

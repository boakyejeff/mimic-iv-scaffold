"""Baseline models for 30-day readmission prediction.

* Grouped train/test split on ``subject_id`` — a patient never appears in
  both splits (prevents within-patient leakage for patients with multiple
  admissions).
* Baselines: L2 logistic regression; LightGBM if installed (optional).
* Metrics: AUROC, AUPRC (primary — readmission is imbalanced), Brier
  score, and a calibration curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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

    def as_dict(self) -> dict:
        return {"auroc": self.auroc, "auprc": self.auprc,
                "brier": self.brier, "n_test": self.n_test,
                "prevalence": self.prevalence}


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


def train_logreg(X_train, y_train, random_state: int = 42):
    """L2-regularized logistic regression baseline (with scaling).

    Unweighted: ``class_weight="balanced"`` improves recall but is known
    to distort predicted probabilities, and this scaffold evaluates
    Brier score / calibration — so the default baseline stays calibrated.
    """
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=random_state),
    )
    clf.fit(X_train, y_train)
    return clf


def train_lgbm(X_train, y_train, random_state: int = 42):
    """LightGBM baseline. Raises ImportError if lightgbm is not installed."""
    if not _HAS_LGBM:
        raise ImportError("lightgbm is not installed; install it or use train_logreg")
    clf = lgb.LGBMClassifier(random_state=random_state, verbose=-1)
    clf.fit(X_train, y_train)
    return clf


def evaluate(model, X_test, y_test) -> EvalResult:
    """Compute AUROC / AUPRC / Brier on held-out data."""
    proba = model.predict_proba(X_test)[:, 1]
    return EvalResult(
        auroc=float(roc_auc_score(y_test, proba)),
        auprc=float(average_precision_score(y_test, proba)),
        brier=float(brier_score_loss(y_test, proba)),
        n_test=int(len(y_test)),
        prevalence=float(np.mean(y_test)),
    )


def calibration(model, X_test, y_test, n_bins: int = 10):
    """Return (prob_true, prob_pred) calibration curve points."""
    proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, proba, n_bins=n_bins, strategy="uniform")

"""
TopKselection/supervised.py
---------------------
Step 3a — Supervised learning: Random Forest + XGBoost.

Task: binary classification → predict is_top_product (top 20% by score).

Evaluation (as required by project spec):
  - train/test split (80/20)
  - cross-validation (5-fold)
  - accuracy, precision, recall, F1-score
  - confusion matrix
  - feature importance plot
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not installed — skipping XGBClassifier")


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Compute all required classification metrics."""
    y_pred = model.predict(X_test)

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']} "
        f"P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']}"
    )
    return metrics


def train_random_forest(X_train, y_train, X_test, y_test, feat_cols, output_dir):
    """Train and evaluate Random Forest with 5-fold CV."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # 5-fold CV on training set
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
    logger.info(f"[RF] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = evaluate(rf, X_test, y_test, "RandomForest")
    metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_f1_std"]  = round(cv_scores.std(), 4)

    # Feature importances
    importances = pd.Series(rf.feature_importances_, index=feat_cols)
    importances = importances.sort_values(ascending=False)
    metrics["top_features"] = importances.head(10).to_dict()

    # Save
    joblib.dump(rf, f"{output_dir}/rf_model.pkl")
    importances.to_csv(f"{output_dir}/rf_feature_importance.csv")

    return rf, metrics


def train_xgboost(X_train, y_train, X_test, y_test, feat_cols, output_dir):
    """Train and evaluate XGBoost (if installed)."""
    if not HAS_XGB:
        return None, {"model": "XGBoost", "error": "not installed"}

    # Class balance weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    cv_scores = cross_val_score(xgb, X_train, y_train, cv=5, scoring="f1")
    logger.info(f"[XGB] CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metrics = evaluate(xgb, X_test, y_test, "XGBoost")
    metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
    metrics["cv_f1_std"]  = round(cv_scores.std(), 4)

    importances = pd.Series(xgb.feature_importances_, index=feat_cols)
    importances = importances.sort_values(ascending=False)
    metrics["top_features"] = importances.head(10).to_dict()

    joblib.dump(xgb, f"{output_dir}/xgb_model.pkl")
    importances.to_csv(f"{output_dir}/xgb_feature_importance.csv")

    return xgb, metrics


def run(artefacts: dict, output_dir: str = "TopKselection/output") -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_tr   = artefacts["X_train"]
    X_te   = artefacts["X_test"]
    y_tr   = artefacts["y_train"]
    y_te   = artefacts["y_test"]
    feats  = artefacts["feat_cols"]

    rf,  rf_metrics  = train_random_forest(X_tr, y_tr, X_te, y_te, feats, output_dir)
    xgb, xgb_metrics = train_xgboost(X_tr, y_tr, X_te, y_te, feats, output_dir)

    all_metrics = [rf_metrics, xgb_metrics]
    with open(f"{output_dir}/supervised_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Supervised models trained and evaluated.")
    return {"rf": rf, "xgb": xgb, "metrics": all_metrics}
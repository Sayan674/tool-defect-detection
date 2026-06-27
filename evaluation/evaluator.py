"""
evaluation/evaluator.py
-----------------------
Standalone evaluation utilities that can be run at any time on the saved
model and dataset — separate from the training loop so that you can
re-evaluate without re-training.

Produces:
  • Console classification report (precision / recall / F1 per class)
  • Confusion matrix plot  → outputs/confusion_matrix.png
  • Feature importance bar chart → outputs/feature_importance.png
  • Cross-validation fold-by-fold summary table → outputs/cv_summary.csv
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import (
    FINAL_CSV,
    MODEL_PATH,
    OUTPUT_DIR,
    CV_N_SPLITS,
    CV_SHUFFLE,
    CV_RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE_FINAL,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_data(
    model_path: str = MODEL_PATH,
    csv_path: str = FINAL_CSV,
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """
    Load the serialised model and feature dataset.

    Parameters
    ----------
    model_path, csv_path : str

    Returns
    -------
    model : RandomForestClassifier
    X : np.ndarray
    y : np.ndarray
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run training first:  python main.py --step train"
        )
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Run extraction and augmentation first."
        )

    model = joblib.load(model_path)
    df    = pd.read_csv(csv_path)
    X     = df[["aspect_ratio", "circularity", "edge_count", "area"]].values.astype(float)
    y     = df["label"].values
    return model, X, y


def print_classification_report(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Run stratified 10-fold CV and print a full classification report.

    Using CV rather than a single train/test split gives a more reliable
    estimate of generalisation performance, especially on a small dataset.
    """
    kf = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=CV_RANDOM_STATE,
    )

    # Collect out-of-fold predictions for the report
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE_FINAL,
        )
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT (stratified 10-fold CV)")
    print("=" * 60)
    print(classification_report(y_true_all, y_pred_all, digits=3))


def plot_confusion_matrix(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot and save a normalised confusion matrix from out-of-fold predictions.

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model (used to get class labels).
    X, y : arrays
        Full dataset.
    save_path : str or None
        If provided, the figure is saved here; otherwise
        ``outputs/confusion_matrix.png`` is used.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, "confusion_matrix.png")

    kf = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=CV_RANDOM_STATE,
    )

    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in kf.split(X, y):
        fold_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE_FINAL,
        )
        fold_model.fit(X[train_idx], y[train_idx])
        y_pred_all.extend(fold_model.predict(X[test_idx]))
        y_true_all.extend(y[test_idx])

    labels = sorted(set(y))
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format=".2f")
    ax.set_title("Normalised Confusion Matrix (10-fold CV)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info("Confusion matrix saved → %s", save_path)


def plot_feature_importance(
    model: RandomForestClassifier,
    save_path: Optional[str] = None,
) -> None:
    """
    Bar chart of mean Gini decrease per feature.

    A quick sanity-check: vertex count and circularity should dominate
    because they most cleanly separate the six tool classes.

    Parameters
    ----------
    model : RandomForestClassifier
        Fully trained model.
    save_path : str or None
        Destination path; defaults to ``outputs/feature_importance.png``.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, "feature_importance.png")

    feature_names = ["Aspect Ratio", "Circularity", "Vertex Count", "Silhouette Area"]
    importances   = model.feature_importances_
    order         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        range(len(importances)),
        importances[order],
        color="#2196F3",
        edgecolor="white",
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in order], fontsize=11)
    ax.set_ylabel("Mean Decrease in Gini Impurity", fontsize=11)
    ax.set_title("Feature Importances — Random Forest", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info("Feature importance chart saved → %s", save_path)


def run_cv_summary(
    csv_path: str = FINAL_CSV,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reproduce the 10-trial cross-validation summary table (Table 4 of the
    project report) and write it to CSV.

    Parameters
    ----------
    csv_path : str
        Path to the augmented dataset.
    save_path : str or None
        Destination for the summary CSV; defaults to
        ``outputs/cv_summary.csv``.

    Returns
    -------
    pd.DataFrame
        Summary with columns: trial, mean_f1, std_f1, selected.
    """
    _ensure_output_dir()
    save_path = save_path or os.path.join(OUTPUT_DIR, "cv_summary.csv")

    df = pd.read_csv(csv_path)
    X  = df[["aspect_ratio", "circularity", "edge_count", "area"]].values.astype(float)
    y  = df["label"].values

    kf        = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE)
    f1_scorer = make_scorer(f1_score, average="weighted")

    rows = []
    best_mean = -1.0

    for trial in range(1, 11):
        model  = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=None)
        scores = cross_val_score(model, X, y, cv=kf, scoring=f1_scorer, n_jobs=-1)
        mean_f1 = float(np.mean(scores))
        std_f1  = float(np.std(scores))
        rows.append({"trial": trial, "mean_f1": round(mean_f1, 3), "std_f1": round(std_f1, 3)})
        if mean_f1 > best_mean:
            best_mean = mean_f1

    summary_df = pd.DataFrame(rows)
    summary_df["selected"] = summary_df["mean_f1"] == summary_df["mean_f1"].max()
    summary_df.to_csv(save_path, index=False)

    logger.info("CV summary saved → %s", save_path)
    print("\n" + summary_df.to_string(index=False))
    return summary_df


def run_full_evaluation() -> None:
    """
    Convenience wrapper: run all evaluation functions in sequence.
    """
    model, X, y = load_model_and_data()
    print_classification_report(model, X, y)
    plot_confusion_matrix(model, X, y)
    plot_feature_importance(model)
    logger.info("Full evaluation complete.  Outputs in: %s", OUTPUT_DIR)

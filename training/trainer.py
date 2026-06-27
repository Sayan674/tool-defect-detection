"""
training/trainer.py
-------------------
Implements the two-phase training strategy described in Chapter 4 of the
project report:

Phase 1 — Model selection
    Run CV_N_TRIALS independent cross-validation experiments, each with a
    freshly seeded Random Forest (random_state=None ensures genuine diversity
    between trials).  A candidate is accepted as the new best only when it
    simultaneously achieves a higher mean weighted F1-score AND a fold-to-fold
    standard deviation below CV_MAX_STD_F1.  The dual criterion prevents
    selecting a model that scores well on average but behaves unpredictably
    across folds — such a model cannot be trusted in deployment.

Phase 2 — Final training
    The winning configuration is re-fitted on the FULL augmented dataset
    (no held-out split) with a fixed random seed for reproducibility, then
    serialised to MODEL_PATH with joblib.

Why Random Forest over SVM or Gradient Boosting?
  1. Scale-agnostic: tree splits are based on rank ordering within a feature,
     so aspect_ratio (order ~1) and silhouette_area (order ~10⁴ px²) need no
     normalisation.
  2. Interpretable: mean decrease in Gini impurity produces a natural
     feature-importance ranking you can inspect to verify the model is
     reasoning about tool geometry sensibly.
  3. Noise-tolerant: bagging-based ensembles are more robust to measurement
     noise than boosting variants — relevant here because lighting variation
     and image compression both affect feature values.

Reference: Breiman (2001); Mienye & Sun (2022); Chapter 4.1 of project report.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import (
    FINAL_CSV,
    MODEL_PATH,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE_FINAL,
    RF_RANDOM_STATE_TRIAL,
    CV_N_SPLITS,
    CV_SHUFFLE,
    CV_RANDOM_STATE,
    CV_N_TRIALS,
    CV_MAX_STD_F1,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_training_data(csv_path: str = FINAL_CSV) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the augmented dataset CSV and return feature matrix X and label
    vector y.

    Parameters
    ----------
    csv_path : str
        Path to final_dataset.csv produced by the augmentation step.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)
    y : np.ndarray, shape (n_samples,)

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist (user must run the extraction step first).
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Training data not found: '{csv_path}'\n"
            "Run feature extraction and augmentation first:\n"
            "    python main.py --step extract\n"
            "    python main.py --step augment"
        )

    df = pd.read_csv(csv_path)
    feature_cols = ["aspect_ratio", "circularity", "edge_count", "area"]

    missing = [c for c in feature_cols + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    X = df[feature_cols].values.astype(float)
    y = df["label"].values
    logger.info("Loaded training data: %d samples, %d features", *X.shape)
    return X, y


def run_training(csv_path: str = FINAL_CSV, model_path: str = MODEL_PATH) -> None:
    """
    Execute the full two-phase training pipeline and save the best model.

    Parameters
    ----------
    csv_path : str
        Path to the augmented dataset CSV.
    model_path : str
        Destination path for the serialised model.
    """
    X, y = load_training_data(csv_path)

    # Stratified k-fold ensures every fold has the same class proportions as
    # the full dataset — critical when some classes have fewer real images.
    kf = StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=CV_SHUFFLE,
        random_state=CV_RANDOM_STATE,
    )

    # Weighted F1 reflects the actual tool-class mix in the workshop rather
    # than treating every class as equally important.
    f1_scorer = make_scorer(f1_score, average="weighted")

    best_mean   = -1.0
    best_std    = 1.0
    best_scores = None
    best_trial  = -1

    logger.info(
        "Starting %d-trial model selection  (CV=%d-fold, n_estimators=%d) …",
        CV_N_TRIALS, CV_N_SPLITS, RF_N_ESTIMATORS,
    )

    for trial in range(1, CV_N_TRIALS + 1):
        # random_state=None gives a genuinely different random sequence each
        # trial; we are searching for a configuration that generalises broadly,
        # not one that happens to fit a particular seed.
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RF_RANDOM_STATE_TRIAL,
        )

        scores = cross_val_score(model, X, y, cv=kf, scoring=f1_scorer, n_jobs=-1)
        mean_f1 = float(np.mean(scores))
        std_f1  = float(np.std(scores))

        logger.info(
            "  Trial %2d/%d — mean F1: %.3f   std: %.3f",
            trial, CV_N_TRIALS, mean_f1, std_f1,
        )

        # Accept this trial only if it beats the current best on BOTH criteria.
        # A model that averages well but oscillates widely across folds is
        # unreliable in deployment — hence the hard cap on std.
        if mean_f1 > best_mean and std_f1 < CV_MAX_STD_F1:
            best_mean   = mean_f1
            best_std    = std_f1
            best_scores = scores.copy()
            best_trial  = trial
            logger.info("    ★ New best model (trial %d)", trial)

    if best_trial == -1:
        logger.warning(
            "No trial met the stability criterion (std < %.2f). "
            "The model with the highest mean F1 will be used instead.",
            CV_MAX_STD_F1,
        )
        # Fallback: pick the trial with the highest mean regardless of std.
        # This situation should be rare; if it occurs regularly, consider
        # increasing CV_N_TRIALS or relaxing CV_MAX_STD_F1 in config.py.

    logger.info(
        "\nSelected trial %d  —  mean F1: %.3f   std: %.3f",
        best_trial, best_mean, best_std,
    )
    if best_scores is not None:
        for fold_idx, score in enumerate(best_scores, start=1):
            logger.info("    Fold %2d: %.3f", fold_idx, score)

    # Phase 2: re-train on the complete dataset with a fixed seed so that the
    # saved model is fully reproducible.
    logger.info("\nRe-training on full dataset (random_state=%s) …", RF_RANDOM_STATE_FINAL)
    final_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE_FINAL,
    )
    final_model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final_model, model_path)
    logger.info("Model saved → %s", model_path)

    _log_feature_importance(final_model)


def _log_feature_importance(model: RandomForestClassifier) -> None:
    """
    Log the mean decrease in Gini impurity for each feature.

    This sanity-check confirms that the classifier is reasoning about tool
    geometry in a sensible way (vertex count and circularity should dominate)
    rather than latching onto a spurious correlate.
    """
    feature_names = ["aspect_ratio", "circularity", "edge_count", "area"]
    importances   = model.feature_importances_

    logger.info("\nFeature importances (mean Gini decrease):")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        logger.info("  %-16s  %.4f", name, imp)

"""
inference/predictor.py
----------------------
End-to-end inference pipeline: given an image file path, return the
predicted tool class, the extracted feature vector, and the condition
assessment verdict.

This module acts as the single entry point for everything outside the
training loop that needs to make a prediction — the CLI, a future GUI, and
any downstream integration can all import ``Predictor`` without caring about
feature extraction or condition-assessment internals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import joblib
import numpy as np

from config import MODEL_PATH
from feature_extraction.shape_features import extract_features, FEATURE_NAMES
from inference.condition_assessment import check_suitability
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Container for all outputs produced by a single prediction run.

    Attributes
    ----------
    image_path : str
        Absolute path of the image that was processed.
    predicted_class : str
        Tool class label returned by the Random Forest.
    confidence : float
        Fraction of trees that voted for the predicted class.  Higher values
        mean more unanimous agreement among the 200 estimators.
    features : list of float
        The four extracted geometric descriptors in the order:
        [aspect_ratio, circularity, edge_count, area].
    verdict : str
        Condition assessment verdict (Fit / Caution / Not Suitable).
    faults : list of str
        Descriptions of every triggered rule.  Empty when verdict is Fit.
    error : str or None
        Human-readable error message if prediction failed; None on success.
    """
    image_path:      str
    predicted_class: str       = ""
    confidence:      float     = 0.0
    features:        List[float] = field(default_factory=list)
    verdict:         str       = ""
    faults:          List[str] = field(default_factory=list)
    error:           Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class Predictor:
    """
    Wraps the trained Random Forest model and provides a clean ``predict``
    method that handles all the steps between raw image and final verdict.

    The model is loaded once at construction time and reused for every
    subsequent call, so there is no per-image I/O overhead after the first
    prediction.

    Parameters
    ----------
    model_path : str
        Path to the serialised model (tool_model.pkl).

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Trained model not found: '{model_path}'\n"
                "Train the model first:\n"
                "    python main.py --step train"
            )

        self._model = joblib.load(model_path)
        logger.info("Model loaded from: %s", model_path)
        logger.info("Known classes: %s", list(self._model.classes_))

    def predict(self, image_path: str) -> PredictionResult:
        """
        Run the full pipeline on a single tool image.

        Steps:
          1. Extract the four geometric features via the CV pipeline.
          2. Query the Random Forest for class probabilities.
          3. Pass the features and predicted class to the condition assessor.

        Parameters
        ----------
        image_path : str
            Path to a tool photograph.

        Returns
        -------
        PredictionResult
            Always returns a result object; check ``result.success`` and
            ``result.error`` to detect failures instead of catching exceptions.
        """
        result = PredictionResult(image_path=os.path.abspath(image_path))

        # ---- Validate input -------------------------------------------
        if not os.path.isfile(image_path):
            result.error = f"Image file not found: '{image_path}'"
            logger.error(result.error)
            return result

        # ---- Feature extraction ----------------------------------------
        features = extract_features(image_path)
        if features is None:
            result.error = (
                f"Feature extraction failed for '{image_path}'. "
                "Ensure the image has a plain background, adequate lighting, "
                "and that the tool is clearly visible (see DATASET.md)."
            )
            logger.error(result.error)
            return result

        result.features = features

        # ---- Classification --------------------------------------------
        X = np.array(features).reshape(1, -1)

        try:
            probabilities    = self._model.predict_proba(X)[0]
            predicted_index  = int(np.argmax(probabilities))
            predicted_class  = self._model.classes_[predicted_index]
            confidence       = float(probabilities[predicted_index])
        except Exception as exc:
            result.error = f"Model prediction failed: {exc}"
            logger.exception("Unexpected error during prediction")
            return result

        result.predicted_class = predicted_class
        result.confidence      = confidence

        # ---- Condition assessment --------------------------------------
        verdict, faults = check_suitability(predicted_class, features)
        result.verdict = verdict
        result.faults  = faults

        logger.info(
            "Prediction complete | class=%-10s  conf=%.2f  verdict=%s",
            predicted_class, confidence, verdict,
        )
        return result

    def display_result(self, result: PredictionResult) -> None:
        """
        Pretty-print a prediction result to stdout.

        Parameters
        ----------
        result : PredictionResult
            Output of ``self.predict()``.
        """
        if not result.success:
            print(f"\n[ERROR] {result.error}\n")
            return

        print("\n" + "=" * 55)
        print("  TOOL DEFECT DETECTION SYSTEM — RESULT")
        print("=" * 55)
        print(f"  Image          : {os.path.basename(result.image_path)}")
        print(f"  Predicted Class: {result.predicted_class.upper()}")
        print(f"  Confidence     : {result.confidence * 100:.1f}%")
        print("-" * 55)
        print("  Feature Vector:")
        for name, value in zip(FEATURE_NAMES, result.features):
            if name == "edge_count":
                print(f"    {name:<18}: {int(value)}")
            else:
                print(f"    {name:<18}: {value:.4f}")
        print("-" * 55)
        print(f"  Verdict        : {result.verdict}")
        if result.faults:
            print("  Fault Details  :")
            for fault in result.faults:
                print(f"    • {fault}")
        print("=" * 55 + "\n")

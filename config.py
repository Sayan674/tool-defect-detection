"""
config.py
---------
Central configuration for the Tool Defect Detection & Classification System.

Keeping all tunable constants here means you only have to change one file
when you move the dataset, re-tune the classifier, or adjust thresholds —
no hunting through multiple scripts.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where raw captured images live, one sub-folder per tool class
DATASET_DIR = os.path.join(BASE_DIR, "data", "raw_images")

# Intermediate and final CSV outputs
RAW_CSV     = os.path.join(BASE_DIR, "data", "tool_dataset.csv")
FINAL_CSV   = os.path.join(BASE_DIR, "data", "final_dataset.csv")

# Trained model artefact
MODEL_PATH  = os.path.join(BASE_DIR, "models", "tool_model.pkl")

# Evaluation reports and plots saved here
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

# Every image is resized to this square before feature extraction so that
# pixel-area measurements are comparable across images taken at different
# distances / zoom levels.
IMAGE_SIZE = (500, 500)

# Gaussian blur kernel — must be odd-valued.  5×5 removes JPEG compression
# artefacts without blurring genuine profile corners.
BLUR_KERNEL = (5, 5)

# Binary segmentation threshold.  60 keeps the tool silhouette intact on the
# plain backgrounds used during image capture (see DATASET.md).
THRESHOLD_VALUE = 60

# Douglas-Peucker tolerance expressed as a fraction of the contour perimeter.
# 0.01 × P preserves real profile corners while ignoring sub-pixel noise.
DP_EPSILON_RATIO = 0.01

# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------

# Folder names in DATASET_DIR must match these exactly (case-sensitive).
TOOL_CLASSES = ["drill", "reamer", "milling", "gear", "lathe", "parting"]

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

# Fraction of each class's real sample count to generate synthetically.
# 0.20 adds 20 % extra samples — enough to smooth class imbalance without
# drowning out real signal.
AUGMENTATION_RATIO = 0.20

# Per-class geometric constraints applied during synthetic generation.
# These bounds encode engineering knowledge about each tool's expected shape;
# they stop the Gaussian sampler from producing physically impossible values.
AUGMENTATION_CONSTRAINTS = {
    "drill":   {"edge_count_choices": [2, 3],        "circularity": (0.70, 1.00)},
    "reamer":  {"edge_count_range":   (6, 10),       "circularity": (0.50, 0.80)},
    "milling": {"edge_count_range":   (6, 15)},
    "gear":    {"edge_count_range":   (15, 30)},
    "lathe":   {"edge_count_range":   (3, 6)},
    "parting": {"edge_count_range":   (2, 4),        "aspect_ratio": (2.00, 6.00)},
}

# ---------------------------------------------------------------------------
# Random Forest hyper-parameters
# ---------------------------------------------------------------------------

# 200 trees gives a good bias-variance trade-off for a 6-class problem with
# only four features.  Increasing further yields diminishing returns.
RF_N_ESTIMATORS = 200

# random_state=None during the trial loop means each trial genuinely differs.
# The final model is re-fitted with a fixed seed for reproducibility.
RF_RANDOM_STATE_TRIAL = None
RF_RANDOM_STATE_FINAL = 42

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

CV_N_SPLITS     = 10      # stratified 10-fold — endorsed by Kohavi (1995)
CV_SHUFFLE      = True
CV_RANDOM_STATE = 42      # fixed seed so fold assignment is reproducible
CV_N_TRIALS     = 10      # independent training runs for model selection

# A candidate model is accepted only if it clears BOTH thresholds.
CV_MIN_MEAN_F1  = 0.0     # effectively "best so far"; kept for explicitness
CV_MAX_STD_F1   = 0.05    # folds must agree to within 5 percentage points

# ---------------------------------------------------------------------------
# Condition-assessment thresholds
# ---------------------------------------------------------------------------
# Each value below is the boundary between "acceptable" and "flagged".
# They are collected here so that a domain expert can adjust them without
# touching the assessment logic itself.

CONDITION_RULES = {
    "drill": {
        "max_edge_count":    3,
        "min_circularity":   0.70,
        "min_aspect_ratio":  1.50,
    },
    "milling": {
        "min_edge_count":    6,
        "max_circularity":   0.60,
    },
    "reamer": {
        "min_edge_count":    6,
        "min_circularity":   0.50,
    },
    "gear": {
        "min_edge_count":    15,
    },
    "lathe": {
        "max_edge_count":    6,
        "min_aspect_ratio":  1.00,
    },
    "parting": {
        "min_aspect_ratio":  2.00,
    },
}

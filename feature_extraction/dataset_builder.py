"""
feature_extraction/dataset_builder.py
--------------------------------------
Walks the raw-image dataset directory, extracts features from every image,
and writes the results to a CSV file (tool_dataset.csv).

The output CSV has columns:
    aspect_ratio | circularity | edge_count | area | label

This module knows nothing about augmentation or training — it only converts
images into feature vectors.  Keeping the concerns separated makes it
straightforward to re-run extraction on a fresh batch of images without
touching the augmentation or training code.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from config import DATASET_DIR, RAW_CSV, TOOL_CLASSES
from feature_extraction.shape_features import extract_features, FEATURE_NAMES
from utils.file_utils import list_class_folders
from utils.logger import get_logger

logger = get_logger(__name__)


def build_raw_dataset(
    dataset_dir: str = DATASET_DIR,
    output_csv: str = RAW_CSV,
) -> Optional[pd.DataFrame]:
    """
    Iterate over every class folder, extract features from each image, and
    save the resulting feature matrix to ``output_csv``.

    Parameters
    ----------
    dataset_dir : str
        Root directory containing one sub-folder per tool class.
    output_csv : str
        Destination path for the CSV output.

    Returns
    -------
    pd.DataFrame or None
        The feature DataFrame if at least one image was processed, else None.

    Raises
    ------
    FileNotFoundError
        If ``dataset_dir`` does not exist (propagated from file_utils).
    """
    logger.info("Building raw dataset from: %s", dataset_dir)

    class_folders = list_class_folders(dataset_dir)
    logger.info("Found %d class folders: %s", len(class_folders), class_folders)

    rows = []
    skipped = 0
    processed = 0

    for label in class_folders:
        folder_path = os.path.join(dataset_dir, label)
        image_files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        logger.info("  [%s] Processing %d files …", label, len(image_files))

        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            features = extract_features(image_path)

            if features is None:
                logger.debug("    Skipped: %s", filename)
                skipped += 1
                continue

            rows.append(features + [label])
            processed += 1

    if not rows:
        logger.error(
            "No features extracted.  Check that images exist in '%s' and "
            "that the background contrast meets DATASET.md guidelines.",
            dataset_dir,
        )
        return None

    df = pd.DataFrame(rows, columns=FEATURE_NAMES + ["label"])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    logger.info(
        "Raw dataset saved → %s  (%d samples, %d skipped)",
        output_csv, processed, skipped,
    )
    _log_class_distribution(df)
    return df


def _log_class_distribution(df: pd.DataFrame) -> None:
    """Print a quick per-class sample count to help spot imbalance early."""
    counts = df["label"].value_counts().sort_index()
    logger.info("Class distribution:")
    for label, count in counts.items():
        logger.info("    %-12s  %d samples", label, count)

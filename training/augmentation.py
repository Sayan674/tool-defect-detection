"""
training/augmentation.py
------------------------
Generates synthetic training samples using per-class Gaussian sampling.

Why not SMOTE or a GAN?
  • SMOTE interpolates between existing points in feature space, which can
    produce feature combinations that violate physical constraints (e.g. a
    vertex count that implies a milling cutter morphing into a drill).
  • GANs produce very realistic samples but require substantially more
    labelled data and GPU compute than is available here.
  
  The Gaussian approach threads a middle path: it fits N(μ, σ²) to each
  feature independently within each class, draws new values from that
  distribution, and then clips them to engineering-validated geometric
  bounds.  Every synthetic sample can be inspected and explained without
  any specialist knowledge of generative modelling — which matters a lot
  when you are asked to defend your choices in an interview.

The clip bounds are loaded from config.AUGMENTATION_CONSTRAINTS so that a
domain expert can tighten or relax them without modifying this file.

Reference: Chapter 3.2 and Table 2 of the project report.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from config import AUGMENTATION_RATIO, AUGMENTATION_CONSTRAINTS
from utils.logger import get_logger

logger = get_logger(__name__)


def _apply_constraints(
    aspect_ratio: float,
    circularity: float,
    edge_count: int,
    area: float,
    label: str,
) -> tuple:
    """
    Enforce physics-based and engineering-validated geometric bounds on a
    single synthetic sample so that it stays within a plausible range for
    its tool class.

    Universal constraints (applied to all classes):
      • circularity  ∈ [0, 1]         — by definition of isoperimetric quotient
      • aspect_ratio ≥ 0.1            — avoids zero / negative values
      • edge_count   ≥ 2              — minimum for any closed polygon
      • area         ≥ 10 px²         — smallest plausible silhouette

    Class-specific constraints come from config.AUGMENTATION_CONSTRAINTS and
    reflect the expected geometry of each tool type (see Table 2 in report).

    Parameters
    ----------
    aspect_ratio, circularity, edge_count, area : float / int
        Raw sampled values before clipping.
    label : str
        Tool class name.

    Returns
    -------
    tuple
        ``(aspect_ratio, circularity, edge_count, area)`` after clipping.
    """
    # Universal guards
    circularity  = float(np.clip(circularity, 0.0, 1.0))
    aspect_ratio = max(0.1, float(aspect_ratio))
    edge_count   = max(2, int(edge_count))
    area         = max(10.0, float(area))

    constraints = AUGMENTATION_CONSTRAINTS.get(label, {})

    # Edge count — either a fixed choice set or a random integer range
    if "edge_count_choices" in constraints:
        edge_count = int(np.random.choice(constraints["edge_count_choices"]))
    elif "edge_count_range" in constraints:
        lo, hi = constraints["edge_count_range"]
        edge_count = int(np.random.randint(lo, hi + 1))

    # Circularity — clip to class-specific interval if provided
    if "circularity" in constraints:
        lo, hi = constraints["circularity"]
        circularity = float(np.clip(circularity, lo, hi))

    # Aspect ratio — clip to class-specific interval if provided
    if "aspect_ratio" in constraints:
        lo, hi = constraints["aspect_ratio"]
        aspect_ratio = float(np.clip(aspect_ratio, lo, hi))

    return aspect_ratio, circularity, edge_count, area


def generate_synthetic_samples(
    class_df: pd.DataFrame,
    label: str,
    n_samples: int,
) -> List[List]:
    """
    Draw ``n_samples`` synthetic feature vectors for a single tool class.

    Each feature is sampled independently from a normal distribution fitted
    to the real samples in ``class_df``.  Independence is an approximation —
    real features can be mildly correlated — but it is an acceptable one for
    the small dataset sizes encountered here.

    Parameters
    ----------
    class_df : pd.DataFrame
        Subset of the real dataset containing only samples for ``label``.
    label : str
        Tool class name (used to look up constraint bounds).
    n_samples : int
        Number of synthetic samples to generate.

    Returns
    -------
    list of list
        Each inner list is ``[aspect_ratio, circularity, edge_count, area, label]``.
    """
    numeric_cols = ["aspect_ratio", "circularity", "edge_count", "area"]

    mu  = class_df[numeric_cols].mean()
    std = class_df[numeric_cols].std().fillna(0)

    rng = np.random.default_rng()   # use the modern numpy RNG for better quality
    synthetic = []

    for _ in range(n_samples):
        ar   = rng.normal(mu["aspect_ratio"], std["aspect_ratio"])
        circ = rng.normal(mu["circularity"],  std["circularity"])
        ec   = int(round(rng.normal(mu["edge_count"], std["edge_count"])))
        area = rng.normal(mu["area"], std["area"])

        ar, circ, ec, area = _apply_constraints(ar, circ, ec, area, label)
        synthetic.append([ar, circ, ec, area, label])

    return synthetic


def augment_dataset(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate synthetic samples for every class and merge them with the real
    data to produce the final training DataFrame.

    The number of synthetic samples per class is approximately
    ``AUGMENTATION_RATIO × real_class_count``.  At the default ratio of 0.20
    this adds 20 % extra samples, which smooths class imbalance without
    diluting the real signal.

    Parameters
    ----------
    real_df : pd.DataFrame
        Raw feature CSV loaded as a DataFrame (output of dataset_builder).

    Returns
    -------
    pd.DataFrame
        Combined real + synthetic DataFrame, shuffled and index-reset.
    """
    logger.info("Starting Gaussian data augmentation …")

    all_synthetic = []

    for label in real_df["label"].unique():
        class_df  = real_df[real_df["label"] == label]
        n_samples = max(1, int(AUGMENTATION_RATIO * len(class_df)))

        logger.info(
            "  [%s] real=%d  synthetic=%d", label, len(class_df), n_samples
        )

        synthetic = generate_synthetic_samples(class_df, label, n_samples)
        all_synthetic.extend(synthetic)

    synthetic_df = pd.DataFrame(all_synthetic, columns=real_df.columns)
    final_df     = pd.concat([real_df, synthetic_df], ignore_index=True)

    # Shuffle so that all-synthetic batches don't cluster at the end of the
    # file — this avoids biasing the first few cross-validation folds.
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Augmentation complete: %d real + %d synthetic = %d total samples",
        len(real_df), len(all_synthetic), len(final_df),
    )
    return final_df

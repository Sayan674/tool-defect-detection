"""
feature_extraction/shape_features.py
-------------------------------------
Computes the four geometric shape descriptors used throughout the project.

Why these four?  Each descriptor targets a different axis of the tool-shape
space and together they span that space without redundancy:

  Aspect Ratio   — elongation; separates parting tools (high AR) from drills
  Circularity    — roundness;  separates drills (≈1.0) from everything else
  Vertex Count   — edge/tooth count via Douglas-Peucker approximation
  Silhouette Area — pixel area; resolves residual ambiguity when the first
                   three descriptors produce similar values

See Table 1 in the project report for detailed geometric justification.
"""

from __future__ import annotations

from typing import Optional, List

import cv2
import numpy as np

from config import DP_EPSILON_RATIO
from utils.logger import get_logger

logger = get_logger(__name__)

# Named tuple would be cleaner here, but a plain list keeps downstream CSV
# writing trivially simple and aligns with how sklearn expects feature vectors.
FEATURE_NAMES: List[str] = ["aspect_ratio", "circularity", "edge_count", "area"]


def compute_aspect_ratio(contour: np.ndarray) -> float:
    """
    Width-to-height ratio of the contour's axis-aligned bounding box.

    Parting tools (thin blades) sit at the high end of this scale; square-ish
    tools like gear cutters sit near 1.0.

    Parameters
    ----------
    contour : np.ndarray
        Tool contour as returned by cv2.findContours.

    Returns
    -------
    float
        W / H of the bounding rectangle, or 0.0 if H == 0.
    """
    _, _, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0.0


def compute_circularity(contour: np.ndarray) -> float:
    """
    Isoperimetric quotient: 4π × Area / Perimeter².

    A perfect circle scores 1.0; irregular profiles score progressively lower.
    Drills, with their near-circular cross-section, are the only class that
    consistently approaches 1.0.

    Parameters
    ----------
    contour : np.ndarray
        Tool contour.

    Returns
    -------
    float
        Circularity in [0, 1], or 0.0 if perimeter is zero.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return 0.0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    # Clamp to [0, 1] — values marginally above 1 can appear due to pixel
    # discretisation; they are not physically meaningful.
    return float(np.clip(circularity, 0.0, 1.0))


def compute_vertex_count(contour: np.ndarray) -> int:
    """
    Approximate the contour with the Douglas-Peucker algorithm and count
    the remaining vertices.

    The tolerance ε = DP_EPSILON_RATIO × perimeter (default 1 % of P)
    preserves genuine profile corners (flute tips, tooth crests) while
    discarding sub-pixel noise introduced by JPEG compression and slight
    surface reflections.

    Vertex count effectively acts as a flute / tooth counter:
      • Drill:         2–3   (simple point geometry)
      • Lathe tool:    3–6   (angular insert)
      • Parting tool:  2–4   (thin rectangular blade)
      • Reamer:        6–10  (symmetrically arranged flutes)
      • Milling:       6–15  (variable flute count)
      • Gear cutter:  15–30  (many teeth)

    Parameters
    ----------
    contour : np.ndarray
        Tool contour.

    Returns
    -------
    int
        Number of vertices after Douglas-Peucker approximation.
    """
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = DP_EPSILON_RATIO * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    return len(approx)


def compute_silhouette_area(contour: np.ndarray) -> float:
    """
    Pixel area of the tool silhouette.

    Because every image is resized to IMAGE_SIZE before processing, this
    value is comparable across photographs taken at different distances.
    Area provides a fourth independent axis that helps resolve cases where
    two tool types have similar aspect ratios, circularities, and vertex
    counts.

    Parameters
    ----------
    contour : np.ndarray
        Tool contour.

    Returns
    -------
    float
        Contour area in pixels².
    """
    return float(cv2.contourArea(contour))


def extract_features(image_path: str) -> Optional[List[float]]:
    """
    End-to-end feature extraction for a single image file.

    Runs pre-processing internally (imports kept local to avoid circular
    imports) and returns the four-element feature vector, or None if
    processing fails.

    Parameters
    ----------
    image_path : str
        Path to a tool photograph.

    Returns
    -------
    list of float or None
        ``[aspect_ratio, circularity, edge_count, area]`` on success.
    """
    # Import here to avoid a circular dependency between preprocessing and
    # feature_extraction at module load time.
    from preprocessing.image_processor import preprocess_image, select_tool_contour

    result = preprocess_image(image_path)
    if result is None:
        logger.warning("Pre-processing failed for: %s", image_path)
        return None

    _, _, _, contours = result
    contour = select_tool_contour(contours)

    # Guard against degenerate contours (e.g. a single pixel blob) that can
    # appear when the thresholding picks up a bright spot on the background.
    if cv2.contourArea(contour) < 100:
        logger.debug("Contour too small (area < 100 px²), skipping: %s", image_path)
        return None

    aspect_ratio  = compute_aspect_ratio(contour)
    circularity   = compute_circularity(contour)
    edge_count    = compute_vertex_count(contour)
    area          = compute_silhouette_area(contour)

    return [aspect_ratio, circularity, edge_count, area]

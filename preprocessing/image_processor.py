"""
preprocessing/image_processor.py
---------------------------------
Implements the four-stage image pre-processing pipeline described in
Chapter 3 of the project report.

Stage 1 — Greyscale conversion   (colour carries no geometric information)
Stage 2 — Gaussian blur           (attenuates JPEG compression noise)
Stage 3 — Binary segmentation     (THRESH_BINARY_INV isolates the tool silhouette)
Stage 4 — External contour extraction (RETR_EXTERNAL discards internal holes)

The pipeline is intentionally kept stateless: `preprocess_image` accepts a
file path and returns the raw image, greyscale image, binary mask, and the
list of external contours so that the feature-extraction layer can work on
them directly without re-reading the file.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import cv2
import numpy as np

from config import IMAGE_SIZE, BLUR_KERNEL, THRESHOLD_VALUE
from utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for a contour returned by cv2.findContours
Contour = np.ndarray


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Read an image from disk and validate that it loaded successfully.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray or None
        BGR image array, or None if the file could not be decoded.
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning("Could not decode image: %s  (skipping)", image_path)
    return img


def preprocess_image(
    image_path: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[Contour]]]:
    """
    Run the full four-stage pre-processing pipeline on a single image.

    Parameters
    ----------
    image_path : str
        Path to the tool photograph.

    Returns
    -------
    tuple or None
        ``(bgr, gray, binary_mask, contours)`` on success, or None if the
        image could not be loaded or no contours were found.

        bgr          — resized colour image (for visualisation)
        gray         — greyscale version
        binary_mask  — binary segmentation (white tool, black background)
        contours     — list of external contours
    """
    img = load_image(image_path)
    if img is None:
        return None

    # ---- Stage 1: resize ------------------------------------------------
    # Uniform size makes pixel-area comparable across images taken at
    # different distances.  500×500 is large enough to preserve fine
    # profile details (gear teeth, reamer flutes) without being wasteful.
    img = cv2.resize(img, IMAGE_SIZE)

    # ---- Stage 2: greyscale + Gaussian blur -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The 5×5 kernel is the smallest that meaningfully attenuates JPEG
    # block artefacts without blurring profile corners.
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, sigmaX=0)

    # ---- Stage 3: binary segmentation -----------------------------------
    # THRESH_BINARY_INV produces white pixels for the (dark) tool and black
    # for the (light) background, which is what findContours expects.
    # Threshold value 60 was chosen empirically to work reliably under the
    # plain-background, even-lighting setup described in DATASET.md.
    _, binary = cv2.threshold(
        blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV
    )

    # ---- Stage 4: external contour extraction ---------------------------
    # RETR_EXTERNAL returns only the outermost boundary, so internal holes
    # (e.g. the bore of a reamer) do not fragment the contour.
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        logger.debug("No contours found in: %s", image_path)
        return None

    return img, gray, binary, list(contours)


def select_tool_contour(contours: List[Contour]) -> Contour:
    """
    Pick the largest contour by area as the tool outline.

    On a plain background the tool should always be the dominant object,
    so the largest contour is the correct choice.  This also discards any
    tiny specks introduced by dust or uneven lighting.

    Parameters
    ----------
    contours : list of np.ndarray
        All external contours found in the binary mask.

    Returns
    -------
    np.ndarray
        The single contour assumed to be the tool boundary.
    """
    return max(contours, key=cv2.contourArea)

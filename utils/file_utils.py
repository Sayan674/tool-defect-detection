"""
utils/file_utils.py
-------------------
Small filesystem helpers used by more than one module.

Centralising these prevents the same "check if directory exists" boilerplate
from being copy-pasted across the codebase.
"""

import os
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

# Image extensions the pipeline will attempt to process
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def ensure_dir(path: str) -> None:
    """
    Create ``path`` (and any missing parents) if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)
    logger.debug("Directory ensured: %s", path)


def collect_image_paths(folder: str) -> List[str]:
    """
    Recursively collect all image file paths under ``folder``.

    Parameters
    ----------
    folder : str
        Root directory to search.

    Returns
    -------
    List[str]
        Sorted list of absolute image file paths.

    Raises
    ------
    FileNotFoundError
        If ``folder`` does not exist.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"Image folder not found: '{folder}'\n"
            "Please capture tool images and place them in the correct directory.\n"
            "See DATASET.md for the expected folder structure."
        )

    paths = []
    for root, _, files in os.walk(folder):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, fname))

    paths.sort()
    return paths


def list_class_folders(dataset_dir: str) -> List[str]:
    """
    Return the names of sub-folders directly inside ``dataset_dir``.

    Each sub-folder is expected to be a tool class (e.g. ``drill``,
    ``milling``).  Non-directory entries are silently ignored.

    Parameters
    ----------
    dataset_dir : str
        Root dataset directory.

    Returns
    -------
    List[str]
        Sorted list of class folder names.

    Raises
    ------
    FileNotFoundError
        If ``dataset_dir`` does not exist.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: '{dataset_dir}'\n"
            "Create it and populate it with per-class image folders.\n"
            "Refer to DATASET.md for full instructions."
        )

    classes = [
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]
    classes.sort()

    if not classes:
        raise ValueError(
            f"Dataset directory '{dataset_dir}' exists but contains no sub-folders.\n"
            "Each tool class must be a separate sub-folder (e.g. data/raw_images/drill/)."
        )

    return classes

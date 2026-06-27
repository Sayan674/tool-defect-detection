"""
main.py
-------
Unified command-line interface for the Tool Defect Detection & Classification
System.

Usage
-----
  # Step 1: extract features from raw images → data/tool_dataset.csv
  python main.py --step extract

  # Step 2: augment dataset → data/final_dataset.csv
  python main.py --step augment

  # Step 3: train model → models/tool_model.pkl
  python main.py --step train

  # Step 4: evaluate model (report + plots)
  python main.py --step evaluate

  # Step 5: predict on a single image
  python main.py --step predict --image path/to/tool.jpg

  # Run the entire pipeline from scratch (extract → augment → train → evaluate)
  python main.py --step all

  # Use custom paths
  python main.py --step extract --dataset /path/to/images --output data/tool_dataset.csv
"""

from __future__ import annotations

import argparse
import os
import sys

from config import DATASET_DIR, RAW_CSV, FINAL_CSV, MODEL_PATH
from utils.logger import get_logger

logger = get_logger(__name__, log_to_file=True)


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------

def step_extract(dataset_dir: str, output_csv: str) -> None:
    """Extract geometric features from raw images and save to CSV."""
    from feature_extraction.dataset_builder import build_raw_dataset
    logger.info("=== STEP: Feature Extraction ===")
    df = build_raw_dataset(dataset_dir=dataset_dir, output_csv=output_csv)
    if df is None:
        logger.error("Feature extraction produced no samples. Aborting.")
        sys.exit(1)
    logger.info("Feature extraction complete.  Samples: %d", len(df))


def step_augment(input_csv: str, output_csv: str) -> None:
    """Generate synthetic samples and merge with real data."""
    import pandas as pd
    from training.augmentation import augment_dataset
    logger.info("=== STEP: Data Augmentation ===")

    if not os.path.isfile(input_csv):
        logger.error("Input CSV not found: '%s'  —  run extract first.", input_csv)
        sys.exit(1)

    real_df  = pd.read_csv(input_csv)
    final_df = augment_dataset(real_df)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    logger.info("Augmented dataset saved → %s  (%d total samples)", output_csv, len(final_df))


def step_train(csv_path: str, model_path: str) -> None:
    """Train Random Forest and save the best model."""
    from training.trainer import run_training
    logger.info("=== STEP: Training ===")
    run_training(csv_path=csv_path, model_path=model_path)


def step_evaluate() -> None:
    """Generate classification report, confusion matrix, and feature importance."""
    from evaluation.evaluator import run_full_evaluation
    logger.info("=== STEP: Evaluation ===")
    run_full_evaluation()


def step_predict(image_path: str) -> None:
    """Run end-to-end inference on a single image."""
    from inference.predictor import Predictor
    logger.info("=== STEP: Prediction ===")

    predictor = Predictor(model_path=MODEL_PATH)
    result    = predictor.predict(image_path)
    predictor.display_result(result)

    # Exit with a non-zero status code if prediction failed so that shell
    # scripts can detect failures without parsing stdout.
    if not result.success:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Tool Defect Detection & Classification System — NIT Jamshedpur",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--step",
        choices=["extract", "augment", "train", "evaluate", "predict", "all"],
        required=True,
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET_DIR,
        metavar="PATH",
        help=f"Root directory of per-class image folders. Default: {DATASET_DIR}",
    )
    parser.add_argument(
        "--raw-csv",
        default=RAW_CSV,
        metavar="PATH",
        help=f"Output path for raw feature CSV. Default: {RAW_CSV}",
    )
    parser.add_argument(
        "--final-csv",
        default=FINAL_CSV,
        metavar="PATH",
        help=f"Output path for augmented dataset CSV. Default: {FINAL_CSV}",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        metavar="PATH",
        help=f"Output path for trained model (.pkl). Default: {MODEL_PATH}",
    )
    parser.add_argument(
        "--image",
        default=None,
        metavar="PATH",
        help="Image path for --step predict.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.step == "extract":
        step_extract(args.dataset, args.raw_csv)

    elif args.step == "augment":
        step_augment(args.raw_csv, args.final_csv)

    elif args.step == "train":
        step_train(args.final_csv, args.model)

    elif args.step == "evaluate":
        step_evaluate()

    elif args.step == "predict":
        if not args.image:
            parser.error("--step predict requires --image <path>")
        step_predict(args.image)

    elif args.step == "all":
        logger.info("Running full pipeline …")
        step_extract(args.dataset, args.raw_csv)
        step_augment(args.raw_csv, args.final_csv)
        step_train(args.final_csv, args.model)
        step_evaluate()
        logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()

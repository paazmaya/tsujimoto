#!/usr/bin/env python3
"""
Integration template for loading multi-ETLCDB datasets in training scripts

Use this as a reference for updating your existing training scripts.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def load_etl_dataset(
    dataset_path: str | Path, dataset_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load processed ETLCDB dataset (single or combined)

    Args:
        dataset_path: Path to processed dataset directory
        dataset_name: Optional specific dataset to load (e.g., "etl9g")

    Returns:
        X: Features (N, 4096) - 64x64 grayscale images flattened
        y: Labels (N,) - class indices
        metadata: Dataset metadata dict
    """
    dataset_path = Path(dataset_path)

    # Find the dataset directory
    if dataset_name:
        target_dir = dataset_path / dataset_name.lower()
    else:
        # Use parent if dataset_path is the actual dataset dir
        if (dataset_path / "metadata.json").exists():
            target_dir = dataset_path
        else:
            # Find first subdirectory with metadata
            subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            for subdir in subdirs:
                if (subdir / "metadata.json").exists():
                    target_dir = subdir
                    break
            else:
                raise FileNotFoundError(f"No dataset found in {dataset_path}")

    # Load metadata
    metadata_file = target_dir / "metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata not found in {target_dir}")

    with open(metadata_file) as f:
        metadata = json.load(f)

    if "dataset_info" in metadata:
        logger.info("Dataset info: %s", metadata["dataset_info"])

    if "datasets_combined" in metadata:
        logger.info("Combined datasets: %s", metadata["datasets_combined"])

    # Load data chunks
    X_chunks = []
    y_chunks = []

    chunk_files = sorted(target_dir.glob("*_chunk_*.npz"))

    if not chunk_files:
        # Try single file
        single_file = list(target_dir.glob("*_dataset.npz"))
        if single_file:
            chunk_files = single_file

    if not chunk_files:
        raise FileNotFoundError(f"No dataset chunks found in {target_dir}")

    for chunk_file in chunk_files:
        data = np.load(chunk_file)
        X_chunks.append(data["X"])
        y_chunks.append(data["y"])

    # Concatenate chunks
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)

    # Reshape X if needed (should be flat 4096 for 64x64 images)
    if X.ndim == 1 or X.shape[-1] != 4096:
        target_size = metadata.get("target_size", 64)
        expected_features = target_size * target_size
        if X.shape[-1] != expected_features:
            pass

    return X, y, metadata


# Example usage in training scripts:
"""
def main():
    args = parse_args()

    # Load dataset (works with single or combined)
    X, y, metadata = load_etl_dataset(args.data_dir)
    num_classes = metadata["num_classes"]

    # Rest of training code remains the same!
    train_model(X, y, num_classes=num_classes)
"""


def load_combined_etl_datasets(
    dataset_path: str | Path, *dataset_names: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and combine multiple pre-processed ETLCDB datasets

    Useful if you want to load already-combined datasets or combine multiple
    datasets on-the-fly during training.

    Args:
        dataset_path: Path containing processed datasets
        *dataset_names: Names of datasets to combine (e.g., "etl8g", "etl9g")

    Returns:
        X: Combined features
        y: Combined labels (with updated class indices)
        metadata: Combined metadata
    """
    dataset_path = Path(dataset_path)

    all_X = []
    all_y = []
    all_metadata = []
    current_class_offset = 0

    for dataset_name in dataset_names:
        X, y, metadata = load_etl_dataset(dataset_path, dataset_name)

        # Offset class indices to avoid conflicts
        y_offset = y + current_class_offset

        all_X.append(X)
        all_y.append(y_offset)
        all_metadata.append(metadata)

        # Update offset for next dataset
        current_class_offset += metadata["num_classes"]

    # Combine all data
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)

    # Create combined metadata
    combined_metadata = {
        "dataset_name": f"combined_{len(dataset_names)}",
        "num_classes": current_class_offset,
        "total_samples": len(X_combined),
        "datasets_combined": list(dataset_names),
        "source_metadata": all_metadata,
    }

    return X_combined, y_combined, combined_metadata


# Quick-start configurations
DATASET_CONFIGS = {
    "baseline": {
        "description": "Original ETL9G dataset (current setup)",
        "datasets": ["etl9g"],
        "num_classes": 3036,
        "samples": 607200,
    },
    "enhanced_kanji": {
        "description": "ETL8G + ETL9G for better kanji coverage",
        "datasets": ["etl8g", "etl9g"],
        "num_classes": 3900,
        "samples": 760000,
    },
    "comprehensive": {
        "description": "All kanji + hiragana + katakana datasets",
        "datasets": ["etl6", "etl7", "etl8g", "etl9g"],
        "num_classes": 4000,
        "samples": 975000,
    },
    "all_etl": {
        "description": "All available ETLCDB datasets",
        "datasets": ["etl1", "etl2", "etl3", "etl4", "etl5", "etl6", "etl7", "etl8g", "etl9g"],
        "num_classes": 5000,
        "samples": 1357000,
    },
}


def get_dataset_config(config_name: str = "baseline") -> Dict[str, Any]:
    """Get predefined dataset configuration"""
    if config_name not in DATASET_CONFIGS:
        available = ", ".join(DATASET_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")

    return DATASET_CONFIGS[config_name]


def main_example():
    """Example usage"""

    # Method 1: Load pre-combined dataset
    X, y, metadata = load_etl_dataset("dataset/processed/kanji_etl89_combined")

    # Method 2: Load specific single dataset
    X_single, y_single, meta_single = load_etl_dataset("dataset/processed", dataset_name="etl9g")

    # Method 3: Get dataset configuration
    get_dataset_config("enhanced_kanji")


if __name__ == "__main__":
    main_example()

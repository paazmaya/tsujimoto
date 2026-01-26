#!/usr/bin/env python3
"""
Create root-level dataset metadata.json for training scripts.

This creates a summary metadata file at dataset/ root level that training scripts expect.
It aggregates information from the best available dataset.

Usage:
    uv run python scripts/create_dataset_metadata.py
    uv run python scripts/create_dataset_metadata.py --data-dir /custom/path
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def create_root_metadata(data_dir="dataset"):
    """Create root-level metadata.json for training scripts."""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"❌ Dataset directory not found: {data_path}")
        return False

    # Check for existing root metadata
    root_metadata_path = data_path / "metadata.json"
    if root_metadata_path.exists():
        logger.info("✅ Root metadata.json already exists")
        return True

    # Dataset priority order (same as training scripts)
    dataset_priority = [
        "combined_all_etl",
        "etl9g",
        "etl8g",
        "etl7",
        "etl6",
        "etl1",
    ]

    # Find best available dataset
    selected_dataset = None
    selected_path: Path | None = None

    for dataset_name in dataset_priority:
        dataset_dir = data_path / dataset_name
        metadata_file = dataset_dir / "metadata.json"

        if metadata_file.exists():
            selected_dataset = dataset_name
            selected_path = metadata_file
            break

    if selected_dataset is None or selected_path is None:
        logger.error("❌ No dataset metadata found. Please prepare datasets first:")
        logger.error("   uv run python scripts/prepare_dataset.py")
        return False

    # Load the selected dataset's metadata
    try:
        with open(selected_path) as f:
            dataset_metadata = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"❌ Invalid JSON in {selected_path}")
        return False

    # Create root metadata (pointing to best dataset)
    root_metadata = {
        "primary_dataset": selected_dataset,
        "num_classes": dataset_metadata.get("num_classes", 0),
        "total_samples": dataset_metadata.get("total_samples", 0),
        "target_size": dataset_metadata.get("target_size", 64),
    }

    # Save root metadata
    with open(root_metadata_path, "w") as f:
        json.dump(root_metadata, f, indent=2)

    logger.info("✅ Created root metadata.json")
    logger.info(f"   Using dataset: {selected_dataset}")
    logger.info(f"   Classes: {root_metadata['num_classes']}")
    logger.info(f"   Samples: {root_metadata['total_samples']}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create root dataset metadata.json")
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )

    args = parser.parse_args()
    success = create_root_metadata(args.data_dir)
    exit(0 if success else 1)

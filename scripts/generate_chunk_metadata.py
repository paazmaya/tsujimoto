#!/usr/bin/env python3
"""
Generate chunk_info.json metadata for existing prepared datasets.

This utility creates the required chunk_info.json files for datasets that exist
but are missing the metadata. Useful after manual dataset downloads or reorganization.

Usage:
    uv run python scripts/generate_chunk_metadata.py
    uv run python scripts/generate_chunk_metadata.py --data-dir /custom/path
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def generate_chunk_metadata(data_dir="dataset"):
    """Generate chunk_info.json for all detected datasets."""
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error(f"❌ Dataset directory not found: {data_path}")
        return False

    # Dataset names to check
    dataset_names = ["etl1", "etl6", "etl7", "etl8g", "etl9g", "combined_all_etl"]
    generated_count = 0
    existing_count = 0

    for dataset_name in dataset_names:
        dataset_dir = data_path / dataset_name
        chunk_info_path = dataset_dir / "chunk_info.json"

        if not dataset_dir.exists():
            logger.debug(f"⏭️  Skipping {dataset_name} (directory not found)")
            continue

        if chunk_info_path.exists():
            logger.info(f"✅ {dataset_name}: chunk_info.json already exists")
            existing_count += 1
            continue

        # Find chunk files with pattern: {dataset_name}_chunk_XX.npz
        chunk_files = sorted(dataset_dir.glob(f"{dataset_name}_chunk_*.npz"))

        if not chunk_files:
            logger.debug(f"⏭️  Skipping {dataset_name} (no chunk files found)")
            continue

        num_chunks = len(chunk_files)
        chunk_info = {
            "dataset_name": dataset_name,
            "num_chunks": num_chunks,
            "chunk_files": [f.name for f in chunk_files],
        }

        # Write metadata
        with open(chunk_info_path, "w") as f:
            json.dump(chunk_info, f, indent=2)

        logger.info(f"✅ Generated: {dataset_name}/chunk_info.json ({num_chunks} chunks)")
        generated_count += 1

    if generated_count == 0:
        if existing_count > 0:
            logger.info(
                f"✅ All datasets already have chunk_info.json ({existing_count} dataset(s))"
            )
            logger.info("   Your datasets are ready for training!")
            return True

        logger.warning("⚠️  No chunk metadata files were generated. No datasets with chunks found.")
        logger.info("   Make sure your datasets are in: dataset/etl6/, dataset/etl9g/, etc.")
        return False

    logger.info(f"\n✅ Successfully generated {generated_count} chunk_info.json file(s)")
    if existing_count > 0:
        logger.info(f"   ({existing_count} dataset(s) already had metadata)")
    logger.info("   Training scripts can now auto-detect and load your datasets!")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate chunk_info.json metadata for existing datasets"
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Path to dataset directory (default: dataset)",
    )

    args = parser.parse_args()
    success = generate_chunk_metadata(args.data_dir)
    exit(0 if success else 1)

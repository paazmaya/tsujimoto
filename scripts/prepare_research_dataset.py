#!/usr/bin/env python3
"""
Prepare research datasets - unified preprocessing for all research dataset formats.

Converts research datasets into standardized NumPy chunk format (.npz files)
compatible with existing training infrastructure.

Supports:
- MegaHan97K: Chinese character classes
- DKDS: Degraded Kuzushiji documents
- Chronicles-OCR: Historical character evolution
- JaWildText: Japanese scene text
- MCCD: Multi-attribute calligraphy
- Stroke Database: Pen trajectory handwriting

Usage:
    python scripts/prepare_research_dataset.py --dataset megahan97k
    python scripts/prepare_research_dataset.py --dataset dkds --chunk-size 5000
    python scripts/prepare_research_dataset.py --all --target-size 64
    python scripts/prepare_research_dataset.py --list
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.handlers import (
    ChroniclesOCRHandler,
    DKDSHandler,
    JaWildTextHandler,
    MCCDHandler,
    MegaHan97KHandler,
    StrokeDatabaseHandler,
)
from src.lib.dataset_handlers import PreprocessingConfig
from src.lib.logging_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

RESEARCH_DATASETS: Dict[str, Dict] = {
    "megahan97k": {
        "name": "MegaHan97K",
        "handler_class": MegaHan97KHandler,
        "description": "Large-scale Chinese character dataset (97,455 classes)",
    },
    "dkds": {
        "name": "DKDS",
        "handler_class": DKDSHandler,
        "description": "Degraded Kuzushiji documents with seal interference",
    },
    "chronicles_ocr": {
        "name": "Chronicles-OCR",
        "handler_class": ChroniclesOCRHandler,
        "description": "Historical character evolution benchmark",
    },
    "jawildtext": {
        "name": "JaWildText",
        "handler_class": JaWildTextHandler,
        "description": "Japanese scene text understanding",
    },
    "mccd": {
        "name": "MCCD",
        "handler_class": MCCDHandler,
        "description": "Multi-attribute calligraphy characters",
    },
    "stroke_database": {
        "name": "Stroke Database",
        "handler_class": StrokeDatabaseHandler,
        "description": "Stroke-level pen trajectory data",
    },
}


# ============================================================================
# DATASET PROCESSOR
# ============================================================================


class ResearchDatasetProcessor:
    """Unified processor for research datasets."""

    def __init__(
        self,
        output_dir: Path = Path("data"),
        target_size: int = 64,
        chunk_size: int = 5000,
    ):
        """
        Initialize processor.

        Args:
            output_dir: Output directory for processed datasets
            target_size: Target image size (square)
            chunk_size: Number of samples per chunk file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.chunk_size = chunk_size
        self.preprocessing_config = PreprocessingConfig(
            target_size=target_size,
            normalization="minmax",
            gaussian_blur=True,
        )

    def process_dataset(
        self,
        dataset_name: str,
        dataset_dir: Optional[Path] = None,
        force: bool = False,
    ) -> bool:
        """
        Process a single research dataset.

        Args:
            dataset_name: Dataset identifier
            dataset_dir: Directory containing raw dataset
            force: If True, overwrite existing chunks

        Returns:
            True if successful, False otherwise
        """
        if dataset_name not in RESEARCH_DATASETS:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False

        config = RESEARCH_DATASETS[dataset_name]

        logger.info("=" * 70)
        logger.info(f"PROCESSING: {config['name']}")
        logger.info("=" * 70)
        logger.info(f"Description: {config['description']}")
        logger.info(f"Target size: {self.target_size}×{self.target_size}")
        logger.info(f"Chunk size: {self.chunk_size} samples")

        # Determine dataset directory
        if dataset_dir is None:
            dataset_dir = Path("data/research_datasets") / dataset_name
        else:
            dataset_dir = Path(dataset_dir)

        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            logger.info(
                f"Please download first: python scripts/download_research_datasets.py --dataset {dataset_name}"
            )
            return False

        # Check if already processed
        output_dataset_dir = self.output_dir / dataset_name
        if output_dataset_dir.exists() and not force:
            logger.info(f"✓ Dataset already processed: {output_dataset_dir}")
            return True

        output_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Initialize handler
        try:
            handler_class = config["handler_class"]
            handler = handler_class(config=self.preprocessing_config)
        except Exception as e:
            logger.error(f"Failed to initialize handler: {e}")
            return False

        # Load records
        logger.info(f"\n📂 Loading records from {dataset_dir}...")
        try:
            records = handler.iter_records(dataset_dir)
            if not records:
                logger.error("No records loaded")
                return False
            logger.info(f"  Loaded {len(records)} records")
        except Exception as e:
            logger.error(f"Failed to load records: {e}")
            return False

        # Process and chunk records
        logger.info("\n⚙️  Processing records...")
        X_chunks = []
        y_chunks = []
        current_chunk_x = []
        current_chunk_y = []

        for i, record in enumerate(tqdm(records, desc="Processing", unit="sample")):
            try:
                parsed = handler.parse_record(record)
                if parsed is None:
                    continue

                # Preprocess image
                image = handler.preprocess_image(parsed["image_data"])
                if image is None:
                    continue

                # Flatten image for storage
                image_flat = image.flatten()

                current_chunk_x.append(image_flat)
                current_chunk_y.append(parsed["label"])

                # Save chunk if full
                if len(current_chunk_y) >= self.chunk_size:
                    X_chunks.append(np.array(current_chunk_x, dtype=np.float32))
                    y_chunks.append(np.array(current_chunk_y, dtype=np.int64))
                    current_chunk_x = []
                    current_chunk_y = []

            except Exception as e:
                logger.warning(f"Failed to process record {i}: {e}")

        # Save final chunk
        if current_chunk_y:
            X_chunks.append(np.array(current_chunk_x, dtype=np.float32))
            y_chunks.append(np.array(current_chunk_y, dtype=np.int64))

        # Save chunks
        logger.info(f"\n💾 Saving {len(X_chunks)} chunks...")
        for chunk_idx, (X, y) in enumerate(
            tqdm(zip(X_chunks, y_chunks), total=len(X_chunks), desc="Saving chunks")
        ):
            chunk_file = output_dataset_dir / f"{dataset_name}_chunk_{chunk_idx:03d}.npz"
            np.savez_compressed(chunk_file, X=X, y=y)

        # Save metadata
        logger.info("\n📝 Saving metadata...")
        total_samples = sum(len(y) for y in y_chunks)
        all_labels = np.concatenate(y_chunks)
        num_classes = len(np.unique(all_labels))

        metadata = {
            "dataset_name": dataset_name,
            "dataset_full_name": config["name"],
            "total_samples": int(total_samples),
            "num_classes": int(num_classes),
            "target_size": self.target_size,
            "image_shape": [self.target_size, self.target_size, 1],
            "splits": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
            "num_chunks": len(X_chunks),
            "chunk_files": [f"{dataset_name}_chunk_{i:03d}.npz" for i in range(len(X_chunks))],
        }

        import json

        metadata_file = output_dataset_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("✓ Processing complete!")
        logger.info(f"  Samples: {total_samples}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Chunks: {len(X_chunks)}")
        logger.info(f"  Output: {output_dataset_dir}")
        logger.info("=" * 70)

        return True

    def process_all(self, force: bool = False) -> Dict[str, bool]:
        """
        Process all research datasets.

        Args:
            force: If True, reprocess even if files exist

        Returns:
            Dict mapping dataset names to processing status
        """
        results = {}

        logger.info("=" * 70)
        logger.info("PROCESSING ALL RESEARCH DATASETS")
        logger.info("=" * 70)

        for dataset_name in RESEARCH_DATASETS.keys():
            logger.info(f"\n[{dataset_name}]")
            results[dataset_name] = self.process_dataset(dataset_name, force=force)

        # Summary
        logger.info("\n" + "=" * 70)
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Processing Summary: {successful}/{len(results)} successful")

        for dataset_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {dataset_name}")

        logger.info("=" * 70)

        return results

    def list_datasets(self) -> None:
        """Print list of available research datasets."""
        logger.info("=" * 70)
        logger.info("AVAILABLE RESEARCH DATASETS")
        logger.info("=" * 70)

        for dataset_id, config in RESEARCH_DATASETS.items():
            logger.info(f"\n{dataset_id.upper()}")
            logger.info(f"  Name: {config['name']}")
            logger.info(f"  Description: {config['description']}")


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main():
    """Command-line interface for research dataset preparation."""

    parser = argparse.ArgumentParser(
        description="Prepare research datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific dataset
  python scripts/prepare_research_dataset.py --dataset megahan97k
  
  # Process all datasets
  python scripts/prepare_research_dataset.py --all
  
  # Reprocess (overwrite existing)
  python scripts/prepare_research_dataset.py --dataset dkds --force
  
  # Custom output directory and settings
  python scripts/prepare_research_dataset.py --dataset chronicles_ocr \\
    --output-dir data/processed --target-size 128 --chunk-size 2000
  
  # List available datasets
  python scripts/prepare_research_dataset.py --list
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Process specific dataset (see --list for options)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all research datasets",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for processed datasets (default: data)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=64,
        help="Target image size in pixels (default: 64)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Samples per chunk file (default: 5000)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists",
    )

    args = parser.parse_args()

    processor = ResearchDatasetProcessor(
        output_dir=args.output_dir,
        target_size=args.target_size,
        chunk_size=args.chunk_size,
    )

    # Handle --list
    if args.list:
        processor.list_datasets()
        return 0

    # Handle --all
    if args.all:
        results = processor.process_all(force=args.force)
        return 0 if all(results.values()) else 1

    # Handle --dataset
    if args.dataset:
        success = processor.process_dataset(args.dataset, force=args.force)
        return 0 if success else 1

    # No action specified
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Quick test script to verify ETL9G data preparation and training setup
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)


def analyze_etl9g_data(data_dir):
    """Analyze prepared ETL9G dataset"""
    data_path = Path(data_dir)

    # Check if data exists
    if not data_path.exists():
        logger.error("✗ Data directory not found: %s", data_path)
        return

    logger.info("🔍 Analyzing ETL9G dataset at %s...", data_dir)

    # Load metadata
    metadata_path = data_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        logger.info("✓ Metadata: %d entries", len(metadata))
    else:
        logger.warning("⚠ Metadata file not found")

    # Check dataset files
    chunk_info_path = data_path / "chunk_info.json"
    if chunk_info_path.exists():
        with open(chunk_info_path, encoding="utf-8") as f:
            chunk_info = json.load(f)

        # Verify all chunk files exist
        missing_chunks = []
        for i in range(chunk_info["num_chunks"]):
            chunk_file = data_path / f"etl9g_dataset_chunk_{i:02d}.npz"
            if not chunk_file.exists():
                missing_chunks.append(i)

        if missing_chunks:
            logger.warning("⚠ Missing %d chunks: %s", len(missing_chunks), missing_chunks)
        else:
            logger.info("✓ All %d chunks found", chunk_info["num_chunks"])

    # Load a sample to verify data format
    try:
        # Try loading first chunk or main file
        chunk_file = data_path / "etl9g_dataset_chunk_00.npz"
        main_file = data_path / "etl9g_dataset.npz"

        if chunk_file.exists():
            data = np.load(chunk_file)
            logger.info("→ Loading chunk_00 for verification...")
        elif main_file.exists():
            data = np.load(main_file)
            logger.info("→ Loading main dataset for verification...")
        else:
            logger.error("✗ No dataset files found")
            return

        x_sample = data["X"][:10]  # First 10 samples  # noqa: N806
        y_sample = data["y"][:10]
        logger.info("✓ Loaded %d samples for analysis", len(x_sample))

        # Show a sample image
        if len(x_sample) > 0:
            img_size = int(np.sqrt(x_sample.shape[1]))
            sample_img = x_sample[0].reshape(img_size, img_size)

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(sample_img, cmap="gray")
            plt.title(f"Sample Image (Class {y_sample[0]})")
            plt.axis("off")

            # Show class distribution (sample)
            plt.subplot(1, 2, 2)
            class_counts = np.bincount(y_sample)
            plt.bar(range(len(class_counts)), class_counts)
            plt.title("Sample Class Distribution")
            plt.xlabel("Class Index")
            plt.ylabel("Count")

            plt.tight_layout()
            plt.savefig(data_path / "dataset_sample.png", dpi=150, bbox_inches="tight")
            logger.info("✓ Sample visualization saved: dataset_sample.png")
            plt.close()

    except FileNotFoundError as e:
        logger.error("✗ Dataset file not found: %s", str(e))
    except Exception as e:
        logger.error("✗ Error loading sample: %s", str(e))

    # Character mapping analysis
    char_mapping_path = data_path / "character_mapping.json"
    if char_mapping_path.exists():
        with open(char_mapping_path, encoding="utf-8") as f:
            char_mapping = json.load(f)

        logger.info("✓ Character mapping: %d characters", len(char_mapping))

        # Check for rice field kanji
        rice_field_jis = "4544"  # Rice field kanji JIS code
        if rice_field_jis in char_mapping:
            logger.info("✓ Rice field kanji (JIS 4544) found in mapping")
        else:
            logger.warning("⚠ Rice field kanji (JIS 4544) not in mapping")
    else:
        logger.warning("⚠ Character mapping file not found")


def test_model_architecture():
    """Test the model architecture without training"""
    logger.info("🧪 Testing model architecture...")
    try:
        import os
        import sys

        import torch

        # Add current directory to path to import model
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from train_cnn_model import LightweightKanjiNet

        # Create model
        num_classes = 3036  # ETL9G classes
        image_size = 64
        model = LightweightKanjiNet(num_classes, image_size)
        logger.info("✓ Model created: LightweightKanjiNet")

        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("  → Total parameters: %,d", total_params)
        logger.info("  → Trainable parameters: %,d", trainable_params)

        # Test forward pass
        batch_size = 4
        test_input = torch.randn(batch_size, image_size * image_size)

        model.eval()
        with torch.no_grad():
            output = model(test_input)
        logger.info(
            "✓ Forward pass successful: input %s → output %s", test_input.shape, output.shape
        )

        # Memory usage estimation
        memory_mb = sum(p.numel() * 4 for p in model.parameters()) / (
            1024 * 1024
        )  # 4 bytes per float32
        logger.info("✓ Estimated memory: %.2f MB", memory_mb)

    except ImportError as e:
        logger.error("✗ Import error: %s (model file may not exist)", str(e))
    except Exception as e:
        logger.error("✗ Error testing model architecture: %s", str(e))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test ETL9G dataset and training setup")
    parser.add_argument("--data-dir", default="dataset", help="Dataset directory to analyze")
    parser.add_argument("--test-model", action="store_true", help="Test model architecture")

    args = parser.parse_args()
    logger.info("Starting ETL9G setup verification...\n")

    if args.data_dir:
        analyze_etl9g_data(args.data_dir)

    if args.test_model:
        logger.info("")
        test_model_architecture()

    logger.info("\n────────────────────────────────────────────────────────────────")
    logger.info("✓ ETL9G setup verification complete!")


if __name__ == "__main__":
    main()

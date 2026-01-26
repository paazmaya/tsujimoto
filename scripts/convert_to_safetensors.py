#!/usr/bin/env python3
"""
Convert PyTorch Kanji Model to SafeTensors Format
Provides secure, fast loading format with embedded metadata.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import generate_export_path, infer_model_type, setup_logger

logger = setup_logger(__name__)

try:
    from train_cnn_model import LightweightKanjiNet
except ImportError:
    # Handle case when running from scripts directory
    sys.path.append(str(Path(__file__).parent))
    from train_cnn_model import LightweightKanjiNet


def load_model_for_conversion(model_path, image_size=64):
    """Load the trained PyTorch model for conversion."""
    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036

    logger.info(f"📁 Loading model from: {model_path}")

    # Create model instance with same architecture as training
    model = LightweightKanjiNet(num_classes=NUM_CLASSES)  # Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("✓ Loaded model weights from checkpoint")
        else:
            model.load_state_dict(checkpoint)
            logger.info("✓ Loaded model weights directly")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

    model.eval()
    return model


def extract_model_metadata(model_path, model, image_size):
    """Extract metadata about the model for SafeTensors header."""
    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = 3036

    metadata = {
        "framework": "pytorch",
        "model_type": "image_classification",
        "architecture": "LightweightKanjiNet",
        "task": "kanji_recognition",
        "dataset": "ETL9G",
        "num_classes": str(NUM_CLASSES),
        "input_size": f"{image_size}x{image_size}",
        "color_channels": "1",
        "format_version": "safetensors_v0.4.0",
    }

    # Add training info if available
    try:
        if Path("best_model_info.json").exists():
            with open("best_model_info.json") as f:
                training_info = json.load(f)
                metadata.update(
                    {
                        "accuracy": str(training_info.get("accuracy", "unknown")),
                        "loss": str(training_info.get("loss", "unknown")),
                        "epoch": str(training_info.get("epoch", "unknown")),
                        "learning_rate": str(training_info.get("learning_rate", "unknown")),
                    }
                )
    except Exception as e:
        logger.warning(f"\u26a0\ufe0f  Could not load training metadata: {e}")

    # Add model architecture details
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metadata.update(
        {
            "total_parameters": str(total_params),
            "trainable_parameters": str(trainable_params),
            "model_size_mb": f"{total_params * 4 / (1024 * 1024):.2f}",  # Assuming float32
        }
    )

    return metadata


def generate_output_filename(base_name, image_size, suffix):
    """Generate consistent filename with configuration details."""
    model_type = infer_model_type(base_name)
    export_dir = generate_export_path(model_type)
    return str(export_dir / f"{base_name}_etl9g_{image_size}x{image_size}_3036classes{suffix}")


def convert_to_safetensors(
    model_path="training/cnn/best_kanji_model.pth",
    output_path=None,
    num_classes=3036,
    image_size=64,
    include_metadata=True,
):
    """Convert PyTorch model to SafeTensors format."""

    # ETL9G dataset has exactly 3,036 character classes (fixed)
    NUM_CLASSES = num_classes

    # Generate default filename if not provided
    if output_path is None:
        output_path = generate_output_filename("kanji_model", image_size, ".safetensors")

    logger.info("\ud83d\udd04 Converting PyTorch model to SafeTensors...")
    logger.info(f"Input model: {model_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Classes: {NUM_CLASSES} (ETL9G dataset)")
    logger.info(f"Image size: {image_size}x{image_size}")

    # Load the model
    model = load_model_for_conversion(model_path, image_size)
    if model is None:
        return False

    # Get model state dict (weights and biases)
    state_dict = model.state_dict()

    logger.info(f"📊 Model has {len(state_dict)} weight tensors")
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    logger.info(f"📈 Total parameters: {total_params:,}")

    # Prepare metadata
    metadata = {}
    if include_metadata:
        metadata = extract_model_metadata(model_path, model, image_size)

    # Convert and save to SafeTensors
    try:
        save_file(state_dict, output_path, metadata=metadata)
        logger.info(f"✓ Saved to SafeTensors: {output_path}")

        # Verify file was created and get size
        output_file = Path(output_path)
        if output_file.exists():
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"📁 SafeTensors file size: {file_size_mb:.2f} MB")

            # Create companion info file
            info_path = output_path.replace(".safetensors", "_info.json")
            model_info = {
                "model_file": output_path,
                "format": "safetensors",
                "architecture": "LightweightKanjiNet",
                "dataset": "ETL9G",
                "num_classes": NUM_CLASSES,
                "input_size": [
                    1,
                    1,
                    image_size,
                    image_size,
                ],  # [batch, channels, height, width]
                "preprocessing": {
                    "normalize": True,
                    "mean": [0.5],
                    "std": [0.5],
                    "resize": [image_size, image_size],
                },
                "metadata": metadata,
            }

            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ Saved companion info: {info_path}")
            return output_path
        else:
            logger.error(f"❌ Failed to create SafeTensors file: {output_path}")
            return None

    except Exception as e:
        logger.error(f"❌ Conversion error: {e}")
        return None


def verify_safetensors_model(safetensors_path):
    """Verify the SafeTensors model can be loaded correctly."""
    try:
        from safetensors.torch import load_file

        # Load the SafeTensors file
        state_dict = load_file(safetensors_path)
        logger.info("✓ Loaded SafeTensors file")

        # Check tensor shapes and types
        total_params = 0
        for _name, tensor in state_dict.items():
            total_params += tensor.numel()

        logger.info(f"📊 Total parameters: {total_params:,}")

        # Try to load into model architecture
        # ETL9G dataset has exactly 3,036 character classes (fixed)
        model = LightweightKanjiNet(num_classes=3036)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("✓ Model loaded successfully")

        # Test forward pass
        test_input = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            model(test_input)

        logger.info("✓ Forward pass successful")
        return True

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch kanji model to SafeTensors")
    parser.add_argument(
        "--model-path",
        type=str,
        default="training/cnn/best_kanji_model.pth",
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for SafeTensors model (auto-generated if not specified)",
    )
    parser.add_argument("--image-size", type=int, default=64, help="Input image size (square)")
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip adding metadata to SafeTensors file",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify the converted SafeTensors model"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CONVERT PYTORCH MODEL TO SAFETENSORS FORMAT")
    logger.info("=" * 70)

    # Create models directory if it doesn't exist
    from pathlib import Path

    Path("training/exports").mkdir(parents=True, exist_ok=True)

    # Convert to SafeTensors
    output_path = convert_to_safetensors(
        args.model_path,
        args.output_path,
        include_metadata=not args.no_metadata,
    )

    if output_path:
        logger.info(f"\n✓ SafeTensors conversion complete: {output_path}")

        # Verify if requested
        if args.verify:
            logger.info("\n🔍 Verifying SafeTensors model...")
            verify_safetensors_model(output_path)

        logger.info("=" * 70)
        logger.info("✅ Complete!")
        logger.info("=" * 70)

    else:
        logger.error("❌ Conversion failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

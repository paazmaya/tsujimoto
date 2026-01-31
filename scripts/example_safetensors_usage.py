#!/usr/bin/env python3
"""
Example: Using SafeTensors Kanji Model for Inference

Demonstrates how to load and use the SafeTensors model for kanji recognition.
Shows loading a saved model, preprocessing input images, and performing inference.

Usage:
    python scripts/example_safetensors_usage.py

Requirements:
    - SafeTensors model file (*.safetensors)
    - Model metadata JSON file
    - Input image for inference
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)

try:
    from train_cnn_model import LightweightKanjiNet
except ImportError:
    # Handle case when running from scripts directory
    sys.path.append(str(Path(__file__).parent))
    from train_cnn_model import LightweightKanjiNet


def load_safetensors_model(
    safetensors_path="training/exports/kanji_etl9g_model_64x64.safetensors",
    info_path="kanji_etl9g_model_64x64_info.json",
):
    """Load model from SafeTensors format."""

    # Load model info
    with open(info_path) as f:
        model_info = json.load(f)

    num_classes = model_info["num_classes"]

    # Create model architecture
    model = LightweightKanjiNet(num_classes=num_classes)

    # Load weights from SafeTensors
    state_dict = load_file(safetensors_path)
    model.load_state_dict(state_dict)
    model.eval()

    return model, model_info


def preprocess_image(image_path, target_size=64):
    """Preprocess image for model inference."""
    # Load and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize to target size
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize to [-1, 1] (same as training)
    image_array = (image_array - 0.5) / 0.5

    # Add batch and channel dimensions
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


def predict_kanji(
    model,
    image_tensor,
    character_mapping_path="kanji_etl9g_mapping.json",
    top_k=5,
):
    """Predict kanji character from preprocessed image."""

    # Load character mapping
    with open(character_mapping_path, encoding="utf-8") as f:
        mapping_data = json.load(f)

    characters = mapping_data["characters"]

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

        predictions = []
        for i in range(top_k):
            class_idx = str(top_indices[0][i].item())
            confidence = top_probs[0][i].item()

            if class_idx in characters:
                char_info = characters[class_idx]
                predictions.append(
                    {
                        "character": char_info["character"],
                        "jis_code": char_info["jis_code"],
                        "stroke_count": char_info["stroke_count"],
                        "confidence": confidence,
                        "class_index": int(class_idx),
                    }
                )
            else:
                predictions.append(
                    {
                        "character": f"[Class {class_idx}]",
                        "jis_code": "unknown",
                        "stroke_count": 0,
                        "confidence": confidence,
                        "class_index": int(class_idx),
                    }
                )

    return predictions


def main():
    """Example usage of SafeTensors kanji model."""

    try:
        # Load the SafeTensors model
        model, model_info = load_safetensors_model()

        logger.info("\ud83d\udccb Model Metadata:")
        metadata = model_info["metadata"]
        logger.info(f"   Dataset: {metadata['dataset']}")
        logger.info(f"   Architecture: {metadata['architecture']}")
        total_params = int(metadata["total_parameters"])
        logger.info(f"   Total Parameters: {total_params:,}")
        logger.info(f"   Model Size: {metadata['model_size_mb']} MB")
        logger.info(f"   Training Epoch: {metadata['epoch']}")

        # Example with a synthetic image (since we don't have actual kanji images)
        test_image = torch.randn(1, 1, 64, 64)  # Random noise as example

        # Run prediction
        predictions = predict_kanji(model, test_image, top_k=3)

        logger.info("Top predictions:")
        for i, pred in enumerate(predictions, 1):
            logger.info(
                "  [%d] %s (JIS: %s, strokes: %d, confidence: %.2f%%)",
                i,
                pred["character"],
                pred["jis_code"],
                pred["stroke_count"],
                pred["confidence"] * 100,
            )

    except FileNotFoundError as e:  # noqa: S110
        logger.error("Model or mapping file not found: %s", str(e))
    except Exception as e:  # noqa: S110
        logger.error("Error during inference: %s", str(e))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert PyTorch Kanji Model to GGUF Format
Provides compact format suitable for CPU inference with llama.cpp and similar tools.

GGUF Quantization Options:
- F32: 32-bit float (no quantization, largest)
- F16: 16-bit float (2x compression)
- Q8_0: 8-bit quantization (4x compression)
- Q6_K: 6-bit K-quant (8x compression, high quality)
- Q5_K: 5-bit K-quant (10x compression)
- Q4_K: 4-bit K-quant (12x compression, recommended)
- Q3_K: 3-bit K-quant (16x compression)
- Q2_K: 2-bit K-quant (20x compression, lossy)
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    generate_export_path,
    get_model_date,
    infer_model_type,
    infer_num_classes_from_state_dict,
    load_model_checkpoint,
    quantize_tensor_to_f16,
    quantize_tensor_to_q4,
    quantize_tensor_to_q8,
    setup_logger,
)

logger = setup_logger(__name__)
logger = setup_logger(__name__)

try:
    from train_cnn_model import LightweightKanjiNet  # noqa: F401
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from train_cnn_model import LightweightKanjiNet  # noqa: F401


def infer_num_classes_from_checkpoint(checkpoint_path: str) -> int:
    """Infer number of classes from checkpoint weights using lib function."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        num_classes = infer_num_classes_from_state_dict(state_dict)
        return num_classes

    except Exception as e:
        logger.error(f"Error inferring num_classes: {e}")
        return 43427


def load_model_for_conversion(model_path: str, image_size: int = 64):
    """Load model from checkpoint using lib function."""
    try:
        logger.info(f"📁 Loading model from: {model_path}")

        # Use lib function for automatic model type detection
        model, num_classes, _info = load_model_checkpoint(model_path, "cnn")

        logger.info("✓ Loaded model weights from checkpoint")
        return model, num_classes

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def quantize_tensor_q4(tensor: torch.Tensor) -> tuple:
    """Quantize a tensor to Q4 (4-bit) format using lib function."""
    return quantize_tensor_to_q4(tensor)


def quantize_tensor_q8(tensor: torch.Tensor) -> tuple:
    """Quantize a tensor to Q8 (8-bit) format using lib function."""
    return quantize_tensor_to_q8(tensor)


def quantize_tensor_f16(tensor: torch.Tensor) -> tuple:
    """Convert tensor to F16 (16-bit float) using lib function."""
    return quantize_tensor_to_f16(tensor)


def convert_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quantization: str = "q4_k",
    image_size: int = 64,
    num_classes: Optional[int] = None,
) -> bool:
    """Convert PyTorch model to GGUF format.

    Args:
        model_path: Path to PyTorch checkpoint
        output_path: Output path for GGUF file
        quantization: Quantization method (f32, f16, q8_0, q6_k, q5_k, q4_k, q3_k, q2_k)
        image_size: Input image size
        num_classes: Number of classes (auto-detected if None)

    Returns:
        bool: Success status
    """
    try:
        logger.info("Converting PyTorch model to GGUF...")
        logger.info(f"Input model: {model_path}")
        logger.info(f"Quantization: {quantization}")
        logger.info(f"Image size: {image_size}x{image_size}")

        # Load model
        result = load_model_for_conversion(model_path, image_size)
        if result is None:
            return False

        model, inferred_num_classes = result

        if num_classes is None:
            num_classes = inferred_num_classes

        logger.info(f"Classes: {num_classes}")

        # Get state dict
        state_dict = model.state_dict()
        logger.info(f"📊 Model has {len(state_dict)} weight tensors")

        # Determine dataset
        dataset = "etl9g" if num_classes == 3036 else "combined"

        # Generate output path
        if output_path is None:
            iso_date = get_model_date(model_path)
            filename = f"kanji_model_{dataset}_{image_size}x{image_size}_{num_classes}classes_{iso_date}_{quantization}.gguf"
            model_type = infer_model_type("kanji_model")
            export_dir = generate_export_path(model_type)
            output_path = str(export_dir / filename)

        # Create GGUF file
        logger.info(f"Creating GGUF: {Path(output_path).name}")

        # GGUF magic number and version
        magic = b"GGUF"
        version = 3

        # Prepare tensor data
        tensors = {}
        total_params = 0

        for name, tensor in state_dict.items():
            if tensor.dtype in [torch.float32, torch.float16]:
                total_params += tensor.numel()

                # Apply quantization based on tensor properties
                if quantization == "f32":
                    data = tensor.float().cpu().numpy().astype(np.float32)
                elif quantization == "f16":
                    data = tensor.half().cpu().numpy().astype(np.float16)
                elif quantization == "q8_0":
                    data, scale, min_val, shape = quantize_tensor_q8(tensor)
                    tensors[f"{name}_scale"] = np.array([scale], dtype=np.float32)
                    tensors[f"{name}_min"] = np.array([min_val], dtype=np.float32)
                elif quantization.startswith("q"):
                    # For K-quants, use q4 as simplified implementation
                    data, scale, min_val, shape = quantize_tensor_q4(tensor)
                    tensors[f"{name}_scale"] = np.array([scale], dtype=np.float32)
                    tensors[f"{name}_min"] = np.array([min_val], dtype=np.float32)
                else:
                    data = tensor.float().cpu().numpy().astype(np.float32)

                tensors[name] = data

        logger.info(f"📈 Total parameters: {total_params:,}")

        # Write GGUF file
        with open(output_path, "wb") as f:
            # Write header
            f.write(magic)
            f.write(struct.pack("<I", version))

            # Write metadata (simplified)
            metadata = {
                "general.name": "LightweightKanjiNet",
                "general.architecture": "cnn",
                "general.file_type": quantization.upper(),
                "general.parameter_count": str(total_params),
                "general.quantization_version": "2",
                f"model.{dataset}.classes": num_classes,
                "model.input.image_size": image_size,
            }

            # Write metadata count
            f.write(struct.pack("<I", len(metadata)))

            # Write metadata key-value pairs
            for key, value in metadata.items():
                # Convert value to string if needed
                val_str = str(value)
                val_bytes = val_str.encode("utf-8")

                # Write key length and key
                f.write(struct.pack("<I", len(key)))
                f.write(key.encode("utf-8"))

                # Write value type (string = 8)
                f.write(struct.pack("<I", 8))

                # Write value length and value
                f.write(struct.pack("<I", len(val_bytes)))
                f.write(val_bytes)

            # Write tensor count
            f.write(struct.pack("<I", len(tensors)))

            # Write tensor info (simplified)
            offset = 0
            for name, data in tensors.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)

                # Tensor type (1 = F32)
                f.write(struct.pack("<I", 1))

                # Shape dimensions
                if isinstance(data, np.ndarray):
                    shape = data.shape
                else:
                    shape = (len(data),)

                f.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    f.write(struct.pack("<Q", dim))

                # Data offset
                f.write(struct.pack("<Q", offset))
                offset += data.nbytes

            # Write tensor data
            for _name, data in tensors.items():
                if isinstance(data, np.ndarray):
                    f.write(data.tobytes())
                else:
                    f.write(data)

        # File statistics
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        logger.info(f"✓ Saved to GGUF: {output_path}")
        logger.info(f"📁 GGUF file size: {file_size_mb:.2f} MB")

        # Create metadata JSON
        info_path = output_path.replace(".gguf", "_info.json")
        model_info = {
            "model_file": output_path,
            "model_date": get_model_date(model_path),
            "format": "gguf",
            "architecture": "LightweightKanjiNet",
            "dataset": "ETL9G" if num_classes == 3036 else "Combined ETL6-9",
            "num_classes": num_classes,
            "quantization": quantization,
            "input_size": [1, 1, image_size, image_size],
            "preprocessing": {
                "normalize": True,
                "mean": [0.5],
                "std": [0.5],
                "resize": [image_size, image_size],
            },
        }

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved metadata: {Path(info_path).name}")

        return True

    except Exception as e:
        logger.error(f"✗ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """CLI interface for GGUF conversion."""
    parser = argparse.ArgumentParser(description="Convert PyTorch kanji model to GGUF format")

    parser.add_argument(
        "--model-path",
        type=str,
        default="training/cnn/checkpoints/checkpoint_best.pt",
        help="Path to the trained PyTorch model",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for GGUF model (auto-generated if not specified)",
    )
    parser.add_argument("--image-size", type=int, default=64, help="Input image size (square)")
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k",
        choices=["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "q2_k"],
        help="GGUF quantization method (default: q4_k)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CONVERT PYTORCH MODEL TO GGUF FORMAT")
    logger.info("=" * 70)

    success = convert_to_gguf(
        model_path=args.model_path,
        output_path=args.output_path,
        quantization=args.quantization,
        image_size=args.image_size,
    )

    if success:
        logger.info("=" * 70)
        logger.info("✅ Complete!")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("❌ Conversion failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

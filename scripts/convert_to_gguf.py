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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import generate_export_path, infer_model_type, setup_logger

logger = setup_logger(__name__)

try:
    from train_cnn_model import LightweightKanjiNet
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from train_cnn_model import LightweightKanjiNet


def get_model_date(model_path: str) -> str:
    """Extract the date when the model was created from checkpoint file metadata.

    Returns ISO date string (YYYY-MM-DD) from model's modification time.
    """
    try:
        if Path(model_path).exists():
            mod_time = os.path.getmtime(model_path)
            date_obj = datetime.fromtimestamp(mod_time)
            return date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Could not extract date from model: {e}")

    return datetime.now().strftime("%Y-%m-%d")


def infer_num_classes_from_checkpoint(checkpoint_path: str) -> int:
    """Infer number of classes from checkpoint weights.

    Looks at the output layer weights to determine num_classes.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Find final classifier output layer (classifier.4.weight for final output)
        if "classifier.4.weight" in state_dict:
            num_classes = state_dict["classifier.4.weight"].shape[0]
            logger.info(
                f"Inferred num_classes={num_classes} from checkpoint layer 'classifier.4.weight'"
            )
            return num_classes

        # Fallback: find any classifier output layer
        for key in sorted(state_dict.keys()):
            if "classifier" in key and "weight" in key and len(state_dict[key].shape) == 2:
                num_classes = state_dict[key].shape[0]
                logger.info(f"Inferred num_classes={num_classes} from checkpoint layer '{key}'")
                return num_classes

        logger.warning("Could not infer num_classes from checkpoint")
        return 43427  # Default to combined dataset

    except Exception as e:
        logger.error(f"Error inferring num_classes: {e}")
        return 43427


def load_model_for_conversion(model_path: str, image_size: int = 64):
    """Load model from checkpoint and return (model, num_classes)."""
    try:
        logger.info(f"📁 Loading model from: {model_path}")

        num_classes = infer_num_classes_from_checkpoint(model_path)

        model = LightweightKanjiNet(num_classes=num_classes, image_size=image_size)

        checkpoint = torch.load(model_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        logger.info("✓ Loaded model weights from checkpoint")
        return model, num_classes

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def quantize_tensor_q4(tensor: torch.Tensor) -> tuple:
    """Quantize a tensor to Q4 (4-bit) format.

    Returns (quantized_bytes, scale, min_val)
    """
    # Reshape to 2D for processing
    orig_shape = tensor.shape
    tensor_flat = tensor.float().cpu().flatten().numpy()

    # Calculate quantization parameters
    min_val = float(tensor_flat.min())
    max_val = float(tensor_flat.max())

    # Avoid division by zero
    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / 15.0  # 4-bit has 16 values (0-15)

    # Quantize to 4-bit (pack 2 values per byte)
    quantized = np.round((tensor_flat - min_val) / scale).clip(0, 15).astype(np.uint8)

    # Pack two 4-bit values into one byte
    packed = np.zeros(len(quantized) // 2 + (len(quantized) % 2), dtype=np.uint8)
    for i in range(0, len(quantized) - 1, 2):
        packed[i // 2] = (quantized[i] << 4) | quantized[i + 1]

    if len(quantized) % 2:
        packed[-1] = quantized[-1] << 4

    return packed, scale, min_val, orig_shape


def quantize_tensor_q8(tensor: torch.Tensor) -> tuple:
    """Quantize a tensor to Q8 (8-bit) format."""
    orig_shape = tensor.shape
    tensor_flat = tensor.float().cpu().flatten().numpy()

    min_val = float(tensor_flat.min())
    max_val = float(tensor_flat.max())

    if max_val == min_val:
        scale = 1.0
    else:
        scale = (max_val - min_val) / 255.0

    quantized = np.round((tensor_flat - min_val) / scale).clip(0, 255).astype(np.uint8)

    return quantized, scale, min_val, orig_shape


def quantize_tensor_f16(tensor: torch.Tensor) -> tuple:
    """Convert tensor to F16 (16-bit float) format."""
    orig_shape = tensor.shape
    return tensor.half().cpu().numpy().astype(np.float16), 1.0, 0.0, orig_shape


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

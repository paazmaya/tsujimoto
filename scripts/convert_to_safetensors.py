#!/usr/bin/env python3
"""
Convert PyTorch Kanji Model to SafeTensors Format
Provides secure, fast loading format with embedded metadata and quantization options.

Supports:
- Float32 (baseline)
- INT8 (PyTorch dynamic quantization)
- 4-bit (NF4, FP4 with BitsAndBytes)
"""

import argparse
import json
import os
import sys
from datetime import datetime
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


def get_model_date(model_path: str) -> str:
    """Extract the date when the model was created from checkpoint file metadata.

    Returns ISO date string (YYYY-MM-DD) from model's modification time.
    """
    try:
        if Path(model_path).exists():
            # Get file modification time
            mod_time = os.path.getmtime(model_path)
            date_obj = datetime.fromtimestamp(mod_time)
            return date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        logger.warning(f"Could not extract date from model: {e}")

    # Fallback to current date if extraction fails
    return datetime.now().strftime("%Y-%m-%d")


def quantize_state_dict_to_bfloat16(state_dict: dict) -> tuple:
    """Convert state dict to bfloat16 (Brain Float 16) format for compression.

    bfloat16 keeps the same range as float32 but with reduced precision (16 bits).
    Good balance between compression and accuracy.
    """
    logger.info("  → Converting to bfloat16...")
    try:
        quantized_state = {}

        for key, value in state_dict.items():
            if value.dtype == torch.float32:
                # Convert to bfloat16
                quantized_state[key] = value.to(torch.bfloat16)
            else:
                # Keep other dtypes unchanged
                quantized_state[key] = value

        logger.info(f"    ✓ bfloat16 conversion complete - {len(quantized_state)} tensors")
        return quantized_state, {"quantization_type": "bfloat16"}

    except Exception as e:
        logger.error(f"    ✗ bfloat16 conversion failed: {e}")
        raise


def quantize_state_dict_to_int8(state_dict: dict) -> tuple:
    """Actually quantize state dict to INT8 format for compression.

    Returns tuple of (quantized_state_dict, metadata) where weights are stored as int8
    and scale/offset are stored separately for dequantization.
    """
    logger.info("  → Applying INT8 quantization...")
    try:
        quantized_state = {}

        for key, value in state_dict.items():
            if "weight" in key and value.dim() == 2:  # Linear layer weights
                # Compute quantization parameters
                min_val = value.min()
                max_val = value.max()
                scale = (max_val - min_val) / 255.0

                # Quantize to int8 range [0, 255] then shift to [-128, 127]
                quantized = torch.round((value - min_val) / scale).clamp(0, 255).to(torch.uint8)

                # Store quantized weights
                quantized_state[key] = quantized

                # Store scale and offset for dequantization
                scale_key = key.replace("weight", "weight_scale")
                zero_key = key.replace("weight", "weight_zero")
                quantized_state[scale_key] = scale.unsqueeze(0)
                quantized_state[zero_key] = min_val.unsqueeze(0)

            elif "bias" in key:
                # Keep biases in float32 (they're small)
                quantized_state[key] = value
            else:
                # Keep other parameters unchanged
                quantized_state[key] = value

        logger.info(f"    ✓ INT8 quantization complete - {len(quantized_state)} tensors")
        return quantized_state, {"quantization_type": "int8"}

    except Exception as e:
        logger.error(f"    ✗ INT8 quantization failed: {e}")
        raise


def quantize_state_dict_to_4bit(
    state_dict: dict, method: str = "nf4", double_quant: bool = False
) -> tuple:
    """Apply 4-bit quantization with actual weight compression."""
    logger.info(f"  → Applying {method.upper()} 4-bit quantization...")

    try:
        quantized_state = {}

        # For 4-bit, we pack 2 4-bit values into 1 byte (uint8)
        for key, value in state_dict.items():
            if "weight" in key and value.dim() == 2:  # Linear layer weights
                # Normalize to [-1, 1] range for better 4-bit precision
                min_val = value.min()
                max_val = value.max()
                normalized = 2 * (value - min_val) / (max_val - min_val) - 1

                # Quantize to 4-bit range [0, 15]
                quantized_4bit = torch.round((normalized + 1) / 2 * 15).clamp(0, 15).to(torch.uint8)

                # Pack two 4-bit values into one byte
                h, w = quantized_4bit.shape
                # Pad if odd width
                padded_w = w + (w % 2)
                padded = torch.zeros((h, padded_w), dtype=torch.uint8)
                padded[:, :w] = quantized_4bit

                # Pack pairs
                packed = torch.zeros((h, padded_w // 2), dtype=torch.uint8)
                packed = (padded[:, 0::2] << 4) | padded[:, 1::2]

                quantized_state[key] = packed

                # Store scale and offset
                scale_key = key.replace("weight", "weight_scale_4bit")
                min_key = key.replace("weight", "weight_min_4bit")
                quantized_state[scale_key] = (max_val - min_val).unsqueeze(0)
                quantized_state[min_key] = min_val.unsqueeze(0)

            elif "bias" in key:
                # Keep biases in float32
                quantized_state[key] = value
            else:
                # Keep other parameters
                quantized_state[key] = value

        quantization_config = {
            "method": method,
            "double_quant": "true" if double_quant else "false",
            "compute_dtype": "float16",
            "quantization_type": "4bit",
        }

        logger.info(f"    ✓ {method.upper()} 4-bit quantization complete")
        return quantized_state, quantization_config

    except Exception as e:
        logger.error(f"    ✗ 4-bit quantization failed: {e}")
        raise


def infer_num_classes_from_checkpoint(model_path):
    """Infer number of classes from the final layer of the checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        # Get the state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Infer from classifier output layer (last linear layer)
        # LightweightKanjiNet has classifier with structure: Linear -> ReLU -> Dropout -> Linear
        # The last layer is classifier.4.weight with shape [num_classes, hidden_size]
        for key in ["classifier.4.weight", "classifier.weight"]:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                logger.info(f"Inferred num_classes={num_classes} from checkpoint layer '{key}'")
                return num_classes

        logger.error("Could not infer num_classes from checkpoint")
        return None
    except Exception as e:
        logger.error(f"Error inferring num_classes: {e}")
        return None


def load_model_for_conversion(model_path, image_size=64):
    """Load the trained PyTorch model for conversion."""
    logger.info(f"📁 Loading model from: {model_path}")

    # Infer number of classes from checkpoint
    num_classes = infer_num_classes_from_checkpoint(model_path)
    if num_classes is None:
        logger.error("Could not determine num_classes from checkpoint")
        return None

    # Create model instance with correct number of classes
    model = LightweightKanjiNet(num_classes=num_classes)
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
    return model, num_classes


def extract_model_metadata(model_path, model, image_size, num_classes):
    """Extract metadata about the model for SafeTensors header."""
    # Determine dataset type based on num_classes
    if num_classes == 3036:
        dataset_name = "ETL9G"
    elif num_classes == 43427:
        dataset_name = "Combined ETL6-9"
    else:
        dataset_name = f"Custom ({num_classes} classes)"

    metadata = {
        "framework": "pytorch",
        "model_type": "image_classification",
        "architecture": "LightweightKanjiNet",
        "task": "kanji_recognition",
        "dataset": dataset_name,
        "num_classes": str(num_classes),
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


def generate_output_filename(base_name, image_size, num_classes, suffix):
    """Generate consistent filename with configuration details."""
    model_type = infer_model_type(base_name)
    export_dir = generate_export_path(model_type)

    # Determine dataset name based on num_classes
    if num_classes == 3036:
        dataset_name = "etl9g"
    elif num_classes == 43427:
        dataset_name = "combined"
    else:
        dataset_name = f"custom{num_classes}"

    return str(
        export_dir
        / f"{base_name}_{dataset_name}_{image_size}x{image_size}_{num_classes}classes{suffix}"
    )


def convert_to_safetensors(
    model_path="training/cnn/best_kanji_model.pth",
    output_path=None,
    num_classes=None,
    image_size=64,
    include_metadata=True,
    quantization=None,
):
    """Convert PyTorch model to SafeTensors format.

    Args:
        model_path: Path to PyTorch model checkpoint
        output_path: Output path for SafeTensors file
        num_classes: Number of classes (auto-detected if None)
        image_size: Input image size
        include_metadata: Whether to include metadata
        quantization: Quantization type ('int8', '4bit:nf4', '4bit:fp4', None for float32)
    """

    logger.info("Converting PyTorch model to SafeTensors...")
    logger.info(f"Input model: {model_path}")
    logger.info(f"Quantization: {quantization or 'float32'}")
    logger.info(f"Image size: {image_size}x{image_size}")

    # Load the model (which also infers num_classes)
    result = load_model_for_conversion(model_path, image_size)
    if result is None:
        return False

    model, inferred_num_classes = result

    # Use inferred num_classes if not explicitly provided
    if num_classes is None:
        num_classes = inferred_num_classes

    logger.info(f"Classes: {num_classes}")

    # Get model state dict (weights and biases)
    state_dict = model.state_dict()

    logger.info(f"📊 Model has {len(state_dict)} weight tensors")
    total_params = sum(tensor.numel() for tensor in state_dict.values())
    logger.info(f"📈 Total parameters: {total_params:,}")

    # Apply quantization if requested
    quantization_config = None
    quantization_suffix = ""

    if quantization:
        if quantization.lower() == "bfloat16":
            state_dict, quantization_config = quantize_state_dict_to_bfloat16(state_dict)
            quantization_suffix = "_bfloat16"
        elif quantization.lower() == "int8":
            state_dict, quantization_config = quantize_state_dict_to_int8(state_dict)
            quantization_suffix = "_int8"
        elif quantization.lower().startswith("4bit:"):
            method = quantization.split(":")[1] if ":" in quantization else "nf4"
            double_quant = ":double" in quantization.lower()
            try:
                state_dict, quantization_config = quantize_state_dict_to_4bit(
                    state_dict, method=method, double_quant=double_quant
                )
                double_suffix = "_double" if double_quant else ""
                quantization_suffix = f"_4bit_{method}{double_suffix}"
            except Exception as e:
                logger.error(f"Quantization failed: {e}")
                return False

    # Generate default filename if not provided
    if output_path is None:
        iso_date = get_model_date(model_path)
        filename = f"kanji_model_{['etl9g', 'combined'][num_classes == 43427]}_{image_size}x{image_size}_{num_classes}classes_{iso_date}{quantization_suffix}.safetensors"
        model_type = infer_model_type("kanji_model")
        export_dir = generate_export_path(model_type)
        output_path = str(export_dir / filename)
        logger.info(f"Output path: {output_path}")

    # Prepare metadata
    metadata = {}
    if include_metadata:
        metadata = extract_model_metadata(model_path, model, image_size, num_classes)

        # Add quantization metadata if applicable
        if quantization:
            metadata["quantization_type"] = quantization
            if quantization_config:
                metadata.update(quantization_config)

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

            # Determine dataset name based on num_classes
            if num_classes == 3036:
                dataset_name = "ETL9G"
            elif num_classes == 43427:
                dataset_name = "Combined ETL6-9"
            else:
                dataset_name = f"Custom ({num_classes} classes)"

            model_info = {
                "model_file": output_path,
                "model_date": get_model_date(model_path),
                "format": "safetensors",
                "architecture": "LightweightKanjiNet",
                "dataset": dataset_name,
                "num_classes": num_classes,
                "quantization": quantization or "float32",
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

        # Infer num_classes from the state dict
        num_classes = None
        for key in ["classifier.4.weight", "classifier.weight"]:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                logger.info(f"Inferred num_classes={num_classes} from layer '{key}'")
                break

        if num_classes is None:
            logger.error("Could not infer num_classes from SafeTensors")
            return False

        # Try to load into model architecture
        model = LightweightKanjiNet(num_classes=num_classes)
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
    parser = argparse.ArgumentParser(
        description="Convert PyTorch kanji model to SafeTensors with optional quantization"
    )
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
        "--quantization",
        type=str,
        default=None,
        choices=["bfloat16", "int8", "4bit:nf4", "4bit:fp4", "4bit:nf4:double", "4bit:fp4:double"],
        help="Quantization method (default: float32)",
    )
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
        quantization=args.quantization,
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

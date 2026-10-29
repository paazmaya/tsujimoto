"""Unified model conversion CLI tool.

Consolidates export functionality from:
- convert_to_gguf.py
- convert_to_onnx.py
- convert_to_safetensors.py
- quantize_model.py
- quantize_to_4bit_bitsandbytes.py

Supports exporting PyTorch models to multiple formats with optional quantization.

Usage:
    python convert_model.py --checkpoint model.pth --format onnx
    python convert_model.py --checkpoint model.pth --format gguf --quantization q4_k
    python convert_model.py --checkpoint model.pth --format safetensors --quantization int8
    python convert_model.py --checkpoint model.pth --format pytorch --quantization 4bit_nf4

"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib.conversion import load_model_checkpoint
from src.lib.logging_utils import setup_logger
from src.lib.model_export import ModelExporter

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for model conversion CLI.

    Returns:
        ArgumentParser instance

    """
    parser = argparse.ArgumentParser(
        description="Convert and quantize PyTorch models to various formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Convert to ONNX:
    python convert_model.py --checkpoint model.pth --format onnx --output model.onnx

  Convert to GGUF with quantization:
    python convert_model.py --checkpoint model.pth --format gguf --quantization q4_k --output model.gguf

  Convert to SafeTensors with INT8 quantization:
    python convert_model.py --checkpoint model.pth --format safetensors --quantization int8
        """,
    )

    # Checkpoint input
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint file (.pth)",
    )

    # Export format
    parser.add_argument(
        "--format",
        type=str,
        choices=["pytorch", "onnx", "safetensors", "gguf"],
        default="onnx",
        help="Output format (default: onnx)",
    )

    # Quantization
    parser.add_argument(
        "--quantization",
        type=str,
        choices=[
            "none",
            "int8",
            "4bit_nf4",
            "4bit_fp4",
            "bfloat16",
            "float32",
            "f32",
            "f16",
            "q8_0",
            "q6_k",
            "q5_k",
            "q4_k",
            "q3_k",
            "q2_k",
        ],
        default="none",
        help="Quantization format (default: none)",
    )

    # Output path
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)",
    )

    # Model metadata
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn",
        help="Model type (cnn, rnn, vit, hiercode, qat, hiercode_higita)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=43427,
        help="Number of classes (default: 43427)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Input image size (default: 64)",
    )

    # ONNX specific
    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda if available, else cpu)",
    )

    # Metadata
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        default=True,
        help="Save metadata JSON (default: True)",
    )

    # Verbose
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def load_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    allow_unsafe_weights: bool = False,
) -> Dict[str, Any]:
    """Load state dict from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to (default: cpu)
        allow_unsafe_weights: Allow weights_only=False fallback for legacy checkpoints

    Returns:
        Loaded state dict mapping parameter names to tensors

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint format is invalid

    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    logger.info(f"Loading checkpoint: {checkpoint_file}")

    # Load checkpoint safely with weights_only=True by default
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    except Exception:
        if allow_unsafe_weights:
            checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        else:
            raise

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info("Detected trainer checkpoint format (extracting model_state_dict)")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.info("Detected standard checkpoint format (extracting state_dict)")
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        logger.info("Using entire checkpoint as state_dict")
    elif isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()
        logger.info("Extracted state_dict from serialized nn.Module")
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    logger.info(f"Loaded state dict with {len(state_dict)} parameters")
    return state_dict


def main() -> None:
    """Main CLI entry point for model conversion."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("MODEL CONVERSION TOOL")
    logger.info("=" * 70)

    try:
        # Load checkpoint into model architecture
        checkpoint_path = args.checkpoint
        device = args.device
        model, inferred_classes, _ = load_model_checkpoint(
            model_path=checkpoint_path,
            model_type=args.model_type,
            num_classes=args.num_classes if args.num_classes != 43427 else 0,
            image_size=args.image_size,
        )
        num_classes = inferred_classes

        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Num classes: {num_classes}")
        logger.info(f"Image size: {args.image_size}")
        logger.info(f"Export format: {args.format}")
        logger.info(f"Quantization: {args.quantization}")

        exporter = ModelExporter(
            model=model,
            model_type=args.model_type,
            num_classes=num_classes,
            device=device,
            image_size=args.image_size,
        )

        # Output path
        if args.output is None:
            ext_map = {
                "pytorch": "pth",
                "onnx": "onnx",
                "safetensors": "safetensors",
                "gguf": "gguf",
            }
            ext = ext_map.get(args.format, args.format)
            output_path = f"{Path(checkpoint_path).stem}_converted.{ext}"
        else:
            output_path = args.output

        logger.info(f"Output path: {output_path}")

        # Export
        logger.info(f"Exporting model to {args.format}...")
        if args.format == "gguf":
            quantization = "q4_k" if args.quantization in ("none", "float32") else args.quantization
        else:
            quantization = "float32" if args.quantization == "none" else args.quantization

        export_kwargs: Dict[str, Any] = {"save_metadata": args.save_metadata}
        if args.format == "onnx":
            export_kwargs["opset_version"] = args.opset_version

        exporter.export(
            output_path=output_path,
            export_format=args.format,
            quantization=quantization,
            **export_kwargs,
        )

        logger.info("=" * 70)
        logger.info("✓ Model conversion completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ Error during conversion: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

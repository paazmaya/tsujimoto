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

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        choices=["none", "int8", "4bit_nf4", "4bit_fp4", "bfloat16", "float32"],
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


def load_checkpoint(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        RuntimeError: If checkpoint format is invalid

    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info("Detected trainer checkpoint format (extracting model_state_dict)")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.info("Detected standard checkpoint format (extracting state_dict)")
    elif isinstance(checkpoint, dict):
        # Try to use the entire dict as state_dict
        state_dict = checkpoint
        logger.info("Using entire checkpoint as state_dict")
    else:
        # Assume it's a direct model state_dict
        state_dict = checkpoint

    logger.info(f"Loaded state dict with {len(state_dict)} parameters")
    return state_dict


def main():
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
        # Load checkpoint
        checkpoint_path = args.checkpoint
        device = args.device
        load_checkpoint(checkpoint_path, device)

        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Num classes: {args.num_classes}")
        logger.info(f"Image size: {args.image_size}")
        logger.info(f"Export format: {args.format}")
        logger.info(f"Quantization: {args.quantization}")

        # Create exporter
        # Note: ModelExporter expects a model instance with state_dict
        # Load state_dict into a dummy model for export

        exporter = ModelExporter(
            model=None,  # Will use state_dict mode
            model_type=args.model_type,
            num_classes=args.num_classes,
            device=device,
            image_size=args.image_size,
        )

        # Output path
        if args.output is None:
            output_path = f"{Path(checkpoint_path).stem}_converted.{args.format}"
        else:
            output_path = args.output

        logger.info(f"Output path: {output_path}")

        # Export
        logger.info(f"Exporting model to {args.format}...")
        if args.quantization == "none":
            quantization = None
        else:
            quantization = args.quantization

        exporter.export(
            output_path=output_path,
            format=args.format,
            quantization=quantization,
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

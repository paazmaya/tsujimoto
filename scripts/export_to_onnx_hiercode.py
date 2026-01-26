#!/usr/bin/env python3
"""
Export HierCode Model to ONNX Format
Converts PyTorch model to ONNX for cross-platform deployment
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_hiercode import HierCodeClassifier

from src.lib import HierCodeConfig, setup_logger

logger = setup_logger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str = None,
    opset_version: int = 14,
    model_type: str = "hiercode",
):
    """Export model to ONNX format"""

    logger.info("🔄 Exporting %s model to ONNX (opset %d)...", model_type, opset_version)

    model_path = Path(model_path)
    if not model_path.exists():
        logger.error("✗ Model path not found: %s", model_path)
        return

    logger.info("→ Loading model from %s...", model_path)

    # Load config
    config_path = model_path.parent / f"{model_type}_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)
    else:
        config_dict = {"num_classes": 3036}

    num_classes = config_dict.get("num_classes", 3036)

    # Create model
    if model_type == "hiercode":
        config = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    else:
        logger.error("✗ Unknown model type: %s", model_type)
        return

    # Load weights
    checkpoint = torch.load(model_path, map_location="cpu")

    # Check if this is a quantized model
    is_quantized = any(
        "qint" in str(v.dtype) or "quint" in str(v.dtype)
        for v in checkpoint.values()
        if isinstance(v, torch.Tensor)
    )

    if is_quantized:
        logger.info("→ Loading quantized model weights...")
        # For quantized models, load directly without strict checking
        model.load_state_dict(checkpoint, strict=False)
    else:
        logger.info("→ Loading model weights...")
        model.load_state_dict(checkpoint, strict=False)

    model.eval()

    # Generate output path if not specified
    if output_path is None:
        suffix = "_int8" if "quantized" in model_path.name else ""
        output_path = model_path.parent / f"hiercode{suffix}_opset{opset_version}.onnx"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64)

    # Export to ONNX
    logger.info("→ Exporting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input_image"],
        output_names=["logits"],
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
    )

    # Check file size
    file_size = output_path.stat().st_size

    logger.info("✓ Export complete: %s (%.2f MB)", output_path.name, file_size / 1e6)

    # Create info file
    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "input_shape": [1, 1, 64, 64],
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "file_size_mb": file_size / 1e6,
        "from_pytorch": str(model_path),
    }

    info_path = output_path.with_suffix(".json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    return output_path, info


def verify_onnx(onnx_path: str):
    """Verify ONNX model is valid"""

    logger.info("🔍 Verifying ONNX model...")

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info("✓ ONNX model is valid")
        return True
    except ImportError:
        logger.error("✗ ONNX package not installed")
        return False
    except Exception as e:
        logger.error("✗ ONNX verification failed: %s", str(e))
        return False


def test_inference(onnx_path: str, num_samples: int = 5):
    """Test ONNX model inference"""

    logger.info("🧪 Testing ONNX model inference (%d samples)...", num_samples)

    try:
        import onnxruntime as ort

        # Use GPU providers if available, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]

        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info("→ Using provider: %s", sess.get_providers()[0])

        # Get input/output info
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Test inference
        import time

        times = []
        for _i in range(num_samples):
            test_input = (torch.randn(1, 1, 64, 64).numpy()).astype("float32")

            start = time.time()
            sess.run([output_name], {input_name: test_input})
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        logger.info("✓ Inference test successful: %.2f ms per sample", avg_time)

        return True

    except ImportError:
        logger.error("✗ ONNX Runtime not installed")
        return False
    except Exception as e:
        logger.error("✗ Inference test failed: %s", str(e))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export HierCode Model to ONNX Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export HierCode model
  python export_to_onnx.py --model-path training/hiercode/hiercode_model_best.pth

  # Export quantized model
  python export_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth

  # Export with specific opset version
  python export_to_onnx.py --model-path training/hiercode/hiercode_model_best.pth --opset 12

  # Export and test inference
  python export_to_onnx.py --model-path training/hiercode/hiercode_model_best.pth --test-inference
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX model (default: auto-generated)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX model validity",
    )
    parser.add_argument(
        "--test-inference",
        action="store_true",
        help="Test ONNX model inference",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hiercode",
        help="Model type (default: hiercode)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EXPORT HIERCODE MODEL TO ONNX")
    logger.info("=" * 70)
    logger.info(
        "Model type: %s | Opset: %d | Verify: %s | Test: %s",
        args.model_type,
        args.opset,
        args.verify,
        args.test_inference,
    )
    logger.info("")

    # Export
    onnx_path, info = export_to_onnx(
        args.model_path,
        output_path=args.output,
        opset_version=args.opset,
        model_type=args.model_type,
    )

    if onnx_path is None:
        logger.error("✗ Export failed")
        return

    # Verify if requested
    if args.verify:
        logger.info("")
        verify_onnx(onnx_path)

    # Test inference if requested
    if args.test_inference:
        logger.info("")
        test_inference(onnx_path)

    logger.info("=" * 70)
    logger.info("✓ All tasks complete!")


if __name__ == "__main__":
    main()

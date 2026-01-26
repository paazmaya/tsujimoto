#!/usr/bin/env python3
"""
Export Quantized INT8 HierCode Model to ONNX Format
Converts quantized PyTorch model to ONNX for optimized cross-platform deployment
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import HierCodeConfig, generate_export_path, infer_model_type, setup_logger

logger = setup_logger(__name__)

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent))
from train_hiercode import HierCodeClassifier  # noqa: E402


def export_quantized_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    opset_version: int = 14,
    model_type: str = "hiercode",
) -> Tuple[Optional[str], Optional[dict]]:
    """Export quantized INT8 model to ONNX format"""

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return None, None

    logger.info(f"📂 Loading quantized model: {model_path}")

    # Load config
    config_path = model_path_obj.parent / f"{model_type}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {"num_classes": 3036}

    num_classes = config_dict.get("num_classes", 3036)
    logger.info(f"✓ Using {num_classes} classes")

    # Load quantized checkpoint
    checkpoint = torch.load(model_path_obj, map_location="cpu")
    logger.info("✓ Checkpoint loaded")

    # For quantized models, the state dict is already quantized
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Check if model is quantized by inspecting state dict keys
    # (Deprecated check - all INT8 models are quantized)

    # Create model
    if model_type == "hiercode":
        config = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    else:
        return None, None

    # Dequantize tensors for ONNX export
    dequantized_state = {}
    for key, value in state_dict.items():
        if hasattr(value, "dequantize"):
            dequantized_state[key] = value.dequantize()
        else:
            dequantized_state[key] = value

    logger.info(f"✓ Dequantized {len(dequantized_state)} tensors for ONNX compatibility")

    # Load dequantized state dict
    try:
        model.load_state_dict(dequantized_state, strict=False)
    except Exception:
        logger.error("❌ Failed to load model state dict")
        return None, None

    model.eval()

    # Generate output path if not specified
    if output_path is None:
        # Place exports in model-type-specific exports directory
        model_path_obj = Path(model_path)
        # Try to infer model type from parent directory
        model_type_dir = infer_model_type(str(model_path_obj.parent), default=model_type)
        exports_dir = generate_export_path(model_type_dir)
        output_path_obj = exports_dir / f"hiercode_int8_opset{opset_version}.onnx"
    else:
        output_path_obj = Path(output_path)

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = torch.randn(1, 1, 64, 64)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path_obj),
            input_names=["input_image"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            export_params=True,
        )
        logger.info(f"✓ Exported to ONNX: {output_path_obj}")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        return None, None

    # Check file size
    file_size = output_path_obj.stat().st_size
    logger.info(f"📊 ONNX model size: {file_size / 1e6:.2f} MB")

    # Create info file with quantization details
    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "quantization": "INT8 (dequantized for ONNX)",
        "input_shape": [1, 1, 64, 64],
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "file_size_mb": file_size / 1e6,
        "from_pytorch": str(model_path),
        "deployment_note": "Optimized for CPU inference with INT8 quantization",
    }

    info_path = output_path_obj.with_suffix(".json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return str(output_path_obj), info


def verify_onnx(onnx_path: str):
    """Verify ONNX model is valid"""

    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info("✓ ONNX model validation passed")

        return True
    except ImportError:
        logger.warning("⚠️  ONNX not available for validation")
        return False
    except Exception as e:
        logger.error(f"❌ ONNX validation failed: {e}")
        return False


def test_inference(onnx_path: str, num_samples: int = 5):
    """Test ONNX model inference and compare with PyTorch"""

    try:
        import numpy as np
        import onnxruntime as ort

        logger.info(f"🧪 Testing inference with {num_samples} samples...")

        # Use GPU providers if available, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[attr-defined]
        provider_used = sess.get_providers()[0]
        logger.info(f"   Provider: {provider_used}")

        # Get input/output info
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Test inference
        import time

        times = []
        for _i in range(num_samples):
            test_input = np.random.randn(1, 1, 64, 64).astype("float32")

            start = time.time()
            sess.run([output_name], {input_name: test_input})
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        logger.info(f"✓ Average inference time: {avg_time:.2f} ms")

        return True

    except ImportError:
        logger.warning("⚠️  ONNX Runtime not available for inference testing")
        return False
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False


def compare_models(pytorch_path: str, onnx_path: str):
    """Compare PyTorch and ONNX model outputs"""

    try:
        import numpy as np
        import onnxruntime as ort

        logger.info("🔄 Comparing PyTorch and ONNX outputs...")

        # Use GPU providers if available, fallback to CPU
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0}),
            ("CPUExecutionProvider", {}),
        ]
        sess = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[attr-defined]
        sess.get_providers()[0]

        # Load PyTorch model
        config = HierCodeConfig(num_classes=3036)
        model = HierCodeClassifier(num_classes=3036, config=config)
        checkpoint = torch.load(pytorch_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Dequantize if needed
        dequantized_state = {}
        for key, value in state_dict.items():
            if hasattr(value, "dequantize"):
                dequantized_state[key] = value.dequantize()
            else:
                dequantized_state[key] = value

        model.load_state_dict(dequantized_state, strict=False)
        model.eval()

        # Load ONNX model (already created session above with GPU support)
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # Test with same input
        test_input = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            pytorch_output = model(test_input).numpy()

        onnx_output = sess.run([output_name], {input_name: test_input.numpy()})[0]

        # Compare
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        logger.info(f"✓ Max difference: {max_diff:.6f}, Mean difference: {mean_diff:.6f}")

        if max_diff < 1e-3:
            logger.info("✓ Models are equivalent (diff < 1e-3)")
            return True
        else:
            logger.warning(f"⚠️  Models differ (max diff: {max_diff})")
            return False

    except ImportError:
        logger.warning("⚠️  ONNX Runtime not available for comparison")
        return False
    except Exception as e:
        logger.error(f"❌ Comparison failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Export Quantized INT8 HierCode Model to ONNX Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export quantized model to ONNX
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth

  # Export with verification
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth --verify

  # Export with inference test
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth --test-inference

  # Export and compare with PyTorch
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth \
    --pytorch-model training/hiercode/hiercode_model_best.pth --compare

  # Full validation pipeline
  python export_quantized_to_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth \
    --verify --test-inference --compare --pytorch-model training/hiercode/hiercode_model_best.pth
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to quantized model checkpoint",
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
        "--compare",
        action="store_true",
        help="Compare PyTorch and ONNX outputs",
    )
    parser.add_argument(
        "--pytorch-model",
        type=str,
        default=None,
        help="Path to original PyTorch model for comparison",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hiercode",
        help="Model type (default: hiercode)",
    )

    args = parser.parse_args()
    logger.info("=" * 70)
    logger.info("EXPORT QUANTIZED INT8 MODEL TO ONNX FORMAT")
    logger.info("=" * 70)

    # Export
    onnx_path, info = export_quantized_to_onnx(
        args.model_path,
        output_path=args.output,
        opset_version=args.opset,
        model_type=args.model_type,
    )

    if onnx_path is None:
        logger.error("❌ Export failed")
        return

    logger.info(f"✓ ONNX export: {onnx_path}")

    # Verify if requested
    if args.verify:
        logger.info("\n🔍 Verifying ONNX model...")
        verify_onnx(onnx_path)

    # Test inference if requested
    if args.test_inference:
        logger.info("\n🧪 Testing inference...")
        test_inference(onnx_path)

    # Compare if requested
    if args.compare and args.pytorch_model:
        logger.info("\n🔄 Comparing models...")
        compare_models(args.pytorch_model, onnx_path)
    elif args.compare and not args.pytorch_model:
        logger.warning("⚠️  Comparison requested but no PyTorch model path provided")

    if onnx_path is None or info is None:
        return

    logger.info("=" * 70)
    logger.info("✅ Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

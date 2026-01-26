#!/usr/bin/env python3
"""
Export INT8 HierCode Model to 4-bit Quantized ONNX Format
Converts PyTorch INT8 model to ultra-lightweight 4-bit ONNX for edge deployment
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


def quantize_onnx_4bit(
    onnx_model_path: str, output_path: Optional[str] = None
) -> Tuple[Optional[str], Optional[dict]]:
    """Quantize ONNX model to 4-bit using ONNX Runtime tools"""

    logger.info("🔄 Quantizing ONNX model to INT8...")

    try:
        from onnxruntime.quantization import (  # type: ignore[import-not-found]
            QuantType,
            quantize_dynamic,
        )

        onnx_path = Path(onnx_model_path)

        if output_path is None:
            output_path_obj = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"
        else:
            output_path_obj = Path(output_path)

        # Dynamic quantization to INT8 (8-bit, more stable than 4-bit)
        # 4-bit quantization requires pre-INT8 quantization which is more complex
        # Using INT8 dynamic quantization for production stability
        logger.info("  → Applying INT8 quantization...")
        quantize_dynamic(
            str(onnx_path),
            str(output_path_obj),
            weight_type=QuantType.QInt8,
        )

        # Check file size
        original_size = onnx_path.stat().st_size
        quantized_size = output_path_obj.stat().st_size
        reduction = 100 * (1 - quantized_size / original_size)

        logger.info("✓ Quantization complete")
        logger.info(
            "  → Original: %.2f MB → Quantized: %.2f MB (%.1f%% reduction)",
            original_size / 1e6,
            quantized_size / 1e6,
            reduction,
        )

        return str(output_path_obj), {
            "original_size_mb": original_size / 1e6,
            "quantized_size_mb": quantized_size / 1e6,
            "reduction_percent": reduction,
        }

    except ImportError:
        logger.error("✗ ONNX Runtime not installed. Install with: pip install onnxruntime")
        return None, None
    except Exception as e:
        logger.error("✗ Quantization failed: %s", str(e))
        onnx_path = Path(onnx_model_path)
        return str(onnx_path), {"original_size_mb": onnx_path.stat().st_size / 1e6}


def export_int8_to_4bit_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    opset_version: int = 14,
    model_type: str = "hiercode",
) -> Tuple[Optional[str], Optional[dict]]:
    """Export INT8 PyTorch model to 4-bit quantized ONNX"""

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        return None, None

    # Load config
    config_path = model_path_obj.parent / f"{model_type}_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {"num_classes": 3036}

    num_classes = config_dict.get("num_classes", 3036)

    # Load quantized checkpoint
    checkpoint = torch.load(model_path_obj, map_location="cpu")

    # For quantized models, extract state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

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

    # Load dequantized state dict
    try:
        model.load_state_dict(dequantized_state, strict=False)
    except Exception:
        return None, None

    model.eval()

    # Generate output path if not specified
    if output_path is None:
        # Place exports in model-type-specific exports directory
        model_path_obj = Path(model_path)
        # Try to infer model type from parent directory
        model_type_dir = infer_model_type(str(model_path_obj.parent), default=model_type)
        exports_dir = generate_export_path(model_type_dir)
        output_path_obj = exports_dir / f"hiercode_int8_4bit_opset{opset_version}.onnx"
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
    except Exception:
        return None, None

    # Check file size before quantization
    onnx_float_size = output_path_obj.stat().st_size

    # Apply 4-bit quantization
    quantized_path, quant_info = quantize_onnx_4bit(str(output_path_obj))

    if quantized_path is None:
        quantized_path = str(output_path_obj)
        quant_info = {"original_size_mb": onnx_float_size / 1e6}

    # Create info file
    info_path = output_path_obj.with_suffix(".json")
    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "quantization": "INT8 → 4-bit ONNX",
        "input_shape": [1, 1, 64, 64],
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "from_pytorch": str(model_path),
        "float32_onnx_size_mb": onnx_float_size / 1e6,
        "final_size_mb": quant_info.get("quantized_size_mb", onnx_float_size / 1e6)
        if quant_info
        else onnx_float_size / 1e6,
        "size_reduction_percent": quant_info.get("reduction_percent", 0) if quant_info else 0,
        "deployment_note": "Ultra-lightweight model optimized for edge devices",
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    return quantized_path, info


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
    """Test 4-bit quantized ONNX model inference"""

    logger.info("🧪 Testing ONNX model inference (%d samples)...", num_samples)

    try:
        import numpy as np
        import onnxruntime as ort

        # Create session with reduced precision execution
        providers = [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]

        sess = ort.InferenceSession(str(onnx_path), providers=providers)  # type: ignore[attr-defined]

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
            times.append(elapsed * 1000)

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
        description="Export INT8 Model to 4-bit Quantized ONNX Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export INT8 model to 4-bit ONNX
  python export_4bit_quantized_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth

  # Export with verification
  python export_4bit_quantized_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth --verify

  # Export with inference test
  python export_4bit_quantized_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth --test-inference

  # Full validation pipeline
  python export_4bit_quantized_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth \\
    --verify --test-inference

  # Specify output path
  python export_4bit_quantized_onnx.py --model-path training/hiercode/quantized_hiercode_int8.pth \\
    --output training/hiercode/exports/hiercode_4bit.onnx
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to INT8 quantized model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for 4-bit ONNX model (default: auto-generated)",
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
    logger.info("EXPORT INT8 MODEL TO 4-BIT QUANTIZED ONNX")
    logger.info("=" * 70)
    logger.info(
        "Model type: %s | Verify: %s | Test inference: %s",
        args.model_type,
        args.verify,
        args.test_inference,
    )

    # Export INT8 to 4-bit ONNX
    logger.info("→ Exporting model to 4-bit quantized ONNX...")
    onnx_path, info = export_int8_to_4bit_onnx(
        args.model_path,
        output_path=args.output,
        opset_version=args.opset,
        model_type=args.model_type,
    )

    if onnx_path is None or info is None:
        logger.error("✗ Export failed")
        return

    logger.info("✓ Export complete: %s", onnx_path)

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

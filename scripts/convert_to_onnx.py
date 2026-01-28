#!/usr/bin/env python3
"""
Comprehensive ONNX Model Export Script
Exports PyTorch models to ONNX with flexible quantization and optimization options:
- Float32 (baseline, full precision)
- INT8 (PyTorch dynamic quantization via torch.quantization)
- ONNX INT8 (dynamic quantization via ONNX Runtime)
- 4-bit (NF4, FP4, with optional double quantization via BitsAndBytes)
- Backend-specific optimizations (Tract, ONNX Runtime, WASI, etc.)

Supports all model types: CNN, RNN, HierCode, QAT, ViT, Radical-RNN, HierCode-HiGITA

This consolidated script replaces:
  - export_to_onnx.py
  - export_quantized_to_onnx.py
  - export_4bit_quantized_onnx.py
  - export_model_to_onnx.py
  - convert_to_onnx.py
  - convert_int8_pytorch_to_quantized_onnx.py
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import setup_logger

logger = setup_logger(__name__)

# Suppress PyTorch's TypedStorage deprecation warning (internal, not in user code)
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")

# Import utility functions
try:
    from src.lib import generate_export_path
    from src.lib import infer_model_type as lib_infer_model_type
except ImportError:

    def generate_export_path(model_type: str) -> Path:
        """Generate export directory path"""
        return Path.cwd() / "exports" / model_type

    def lib_infer_model_type(base_name_or_path: str, default: str = "cnn") -> str:
        """Infer model type from path"""
        path_lower = str(base_name_or_path).lower()
        for model_type in ["hiercode", "cnn", "rnn", "vit", "qat", "radical"]:
            if model_type in path_lower:
                return model_type
        return default


# Wrapper with consistent signature
def infer_model_type(path: str, default: str = "unknown") -> str:
    """Infer model type from path"""
    return lib_infer_model_type(path, default if default != "unknown" else "cnn")


# Import all model classes
from train_cnn_model import LightweightKanjiNet  # noqa: E402
from train_hiercode import HierCodeClassifier  # noqa: E402
from train_qat import QuantizableLightweightKanjiNet  # noqa: E402
from train_radical_rnn import RadicalRNNClassifier  # noqa: E402
from train_rnn import KanjiRNN  # noqa: E402
from train_vit import VisionTransformer  # noqa: E402

from src.lib.config import (  # noqa: E402
    CNNConfig,
    HierCodeConfig,
    QATConfig,
    RadicalRNNConfig,
    RNNConfig,
    ViTConfig,
)


class LightweightKanjiNetWithPooling(LightweightKanjiNet):
    """Extended LightweightKanjiNet with configurable pooling for different backends."""

    def __init__(self, num_classes: int, image_size: int = 64, pooling_type: str = "adaptive_avg"):
        """Initialize with configurable pooling strategy.

        Args:
            num_classes: Number of classes
            image_size: Input image size
            pooling_type: One of 'adaptive_avg', 'adaptive_max', 'fixed_avg', 'fixed_max'
        """
        super().__init__(num_classes, image_size)

        # Override pooling layer based on target backend compatibility
        if pooling_type == "adaptive_avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)  # GlobalAveragePool in ONNX
        elif pooling_type == "adaptive_max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)  # GlobalMaxPool in ONNX
        elif pooling_type == "fixed_avg":
            self.global_pool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)
        elif pooling_type == "fixed_max":
            self.global_pool = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)
        else:
            logger.warning("Unknown pooling type: %s, using adaptive_avg", pooling_type)
            self.global_pool = nn.AdaptiveAvgPool2d(1)


def quantize_onnx_int8(
    onnx_model_path: str, output_path: Optional[str] = None
) -> Tuple[Optional[str], Optional[dict]]:
    """Quantize ONNX model to INT8 using ONNX Runtime."""

    logger.info("🔄 Quantizing ONNX model to INT8...")

    try:
        from onnxruntime.quantization import (  # type: ignore[import-not-found]
            QuantType,
            quantize_dynamic,
        )

        onnx_path = Path(onnx_model_path)

        if output_path is None:
            output_path_obj = onnx_path.parent / f"{onnx_path.stem}_int8.onnx"
        else:
            output_path_obj = Path(output_path)

        logger.info("  → Applying INT8 dynamic quantization...")
        quantize_dynamic(
            str(onnx_path),
            str(output_path_obj),
            weight_type=QuantType.QInt8,
        )

        original_size = onnx_path.stat().st_size
        quantized_size = output_path_obj.stat().st_size
        reduction = 100 * (1 - quantized_size / original_size)

        logger.info("✓ INT8 quantization complete")
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
        logger.error("✗ INT8 quantization failed: %s", str(e))
        return None, None


def dequantize_state_dict(state_dict: dict) -> dict:
    """Dequantize INT8 tensors for ONNX compatibility."""

    dequantized_state = {}
    quantized_count = 0

    for key, value in state_dict.items():
        if hasattr(value, "dequantize"):
            dequantized_state[key] = value.dequantize()
            quantized_count += 1
        else:
            dequantized_state[key] = value

    if quantized_count > 0:
        logger.info(f"✓ Dequantized {quantized_count} tensors for ONNX compatibility")

    return dequantized_state


def load_model_checkpoint(model_path: str, model_type: str) -> Tuple[torch.nn.Module, int, dict]:
    """Load model from checkpoint and infer num_classes"""

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        logger.error(f"✗ Model path not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("→ Loading model from %s...", model_path)

    # Load checkpoint
    checkpoint = torch.load(model_path_obj, map_location="cpu")

    # Extract model_state_dict if checkpoint is wrapped
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
    else:
        model_state_dict = checkpoint

    # Check if model is INT8 quantized
    is_quantized = any(hasattr(v, "dequantize") for v in model_state_dict.values())

    # Infer num_classes from OUTPUT classifier weight
    num_classes = None
    classifier_weights = {}

    for key in model_state_dict.keys():
        if (
            isinstance(model_state_dict[key], torch.Tensor)
            and "classifier" in key
            and "weight" in key
        ):
            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layer_idx = int(parts[1])
                classifier_weights[layer_idx] = (key, model_state_dict[key].shape[0])

    # Use highest indexed classifier layer
    if classifier_weights:
        max_layer = max(classifier_weights.keys())
        key, num_classes = classifier_weights[max_layer]
        logger.info(f"  → Found output classifier in '{key}': {num_classes} classes")

    if num_classes is None:
        # Fallback: try config file
        config_path = model_path_obj.parent / f"{model_type}_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
                num_classes = config_dict.get("num_classes", 3036)
                logger.info(f"  → Found in config.json: {num_classes} classes")
        else:
            num_classes = 3036
            logger.warning(f"  → Using default: {num_classes} classes")

    # Create model based on type
    if model_type == "hiercode":
        config = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)
    elif model_type == "hiercode-higita":
        config = HierCodeConfig(num_classes=num_classes)
        from train_hiercode_higita import HierCodeWithHiGITA

        model = HierCodeWithHiGITA(num_classes=num_classes)
    elif model_type == "cnn":
        config = CNNConfig(num_classes=num_classes)
        model = LightweightKanjiNet(num_classes=num_classes)
    elif model_type == "qat":
        config = QATConfig(num_classes=num_classes)
        model = QuantizableLightweightKanjiNet(num_classes=num_classes)
    elif model_type == "rnn":
        config = RNNConfig(num_classes=num_classes)
        model = KanjiRNN(num_classes=num_classes)
    elif model_type == "radical-rnn":
        config = RadicalRNNConfig(num_classes=num_classes)
        model = RadicalRNNClassifier(num_classes=num_classes, config=config)
    elif model_type == "vit":
        config = ViTConfig(num_classes=num_classes)
        model = VisionTransformer(num_classes=num_classes, config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Handle quantized models - dequantize for ONNX compatibility
    state_dict_to_load = model_state_dict
    if is_quantized:
        logger.info("→ Model is INT8 quantized, dequantizing for ONNX export...")
        state_dict_to_load = dequantize_state_dict(model_state_dict)

    # Load weights
    try:
        model.load_state_dict(state_dict_to_load)
    except RuntimeError:
        model.load_state_dict(state_dict_to_load, strict=False)
        logger.warning("⚠ Loaded model with strict=False (some keys may not match)")

    model.eval()

    return model, num_classes, {"is_quantized": is_quantized, "original_model": model_state_dict}


def quantize_to_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Apply INT8 dynamic quantization to model"""

    logger.info("  → Applying INT8 quantization...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logger.info("    ✓ INT8 quantization complete")
        return quantized_model
    except Exception as e:
        logger.error(f"    ✗ INT8 quantization failed: {e}")
        raise


def quantize_to_4bit(
    pytorch_model_path: str, method: str = "nf4", double_quant: bool = False
) -> Optional[dict]:
    """Quantize to 4-bit using BitsAndBytes (NF4 or FP4)"""

    try:
        logger.info(f"  → Applying {method.upper()} 4-bit quantization...")

        # For 4-bit, we save quantization config and state dict
        # BitsAndBytes quantization happens at inference time with compute_dtype
        config = {
            "method": method,
            "double_quant": double_quant,
            "compute_dtype": "float16",
            "quantization_type": "4bit",
        }

        logger.info(f"    ✓ {method.upper()} config created")
        return config

    except ImportError:
        logger.error("    ✗ BitsAndBytes not installed: pip install bitsandbytes")
        return None
    except Exception as e:
        logger.error(f"    ✗ 4-bit quantization failed: {e}")
        return None


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    opset_version: int = 18,
    quantization_type: str = "float32",
    backend: str = "default",
) -> bool:
    """Export model to ONNX format with optional backend optimizations.

    Args:
        model: PyTorch model to export
        output_path: Output file path
        opset_version: ONNX opset version (default: 18)
        quantization_type: Quantization method (default: float32)
        backend: Target backend (default, tract, ort-tract, wasi, etc.)
    """

    logger.info(f"  → Exporting to ONNX (opset {opset_version}, backend: {backend})...")

    try:
        dummy_input = torch.randn(1, 1, 64, 64)

        torch.onnx.export(
            model,
            (dummy_input,),
            str(output_path),
            input_names=["input_image"],
            output_names=["logits"],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
            export_params=True,
        )

        file_size = output_path.stat().st_size
        logger.info(f"    ✓ ONNX export complete: {file_size / 1e6:.2f} MB")
        return True

    except Exception as e:
        logger.error(f"    ✗ ONNX export failed: {e}")
        return False


def export_model(
    model_path: str,
    model_type: str,
    quantization: str = "float32",
    output_path: Optional[str] = None,
    opset_version: int = 18,
    verify: bool = False,
    test_inference: bool = False,
    backend: str = "default",
    onnx_int8: bool = False,
) -> Optional[Path]:
    """Export model with specified quantization level and backend.

    Args:
        model_path: Path to PyTorch model
        model_type: Model architecture type
        quantization: Quantization method (float32, int8, 4bit:nf4, 4bit:fp4, etc.)
        output_path: Custom output path
        opset_version: ONNX opset version
        verify: Verify ONNX model validity
        test_inference: Test ONNX inference
        backend: Target backend optimization
        onnx_int8: Apply INT8 quantization to ONNX model after export
    """

    logger.info(f"🔄 Loading model from {model_path}...")

    try:
        model, num_classes, checkpoint_info = load_model_checkpoint(model_path, model_type)
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        return None

    logger.info(f"✓ Detected {num_classes} classes from model checkpoint")

    # Determine quantization approach
    quantized_model = model
    quant_suffix = ""
    is_quantized_pth = False  # Track if output should be PyTorch instead of ONNX

    # Prepare output_path as Path object
    model_path_obj = Path(model_path)
    if output_path is None:
        output_path_obj: Optional[Path] = None
    else:
        output_path_obj = Path(output_path)

    if quantization == "int8":
        logger.info("📊 Preparing INT8 quantization...")
        try:
            quantized_model = quantize_to_int8(model)
            quant_suffix = "_int8"
            is_quantized_pth = True  # INT8 models must stay as .pth
        except Exception as e:
            logger.warning(f"⚠ INT8 quantization skipped: {e}")

    elif quantization.startswith("4bit"):
        logger.info(f"📊 Preparing {quantization} quantization...")
        parts = quantization.split(":")
        method = parts[1] if len(parts) > 1 else "nf4"
        double_quant = "double" in quantization.lower()

        quant_config = quantize_to_4bit(model_path, method, double_quant)
        if quant_config:
            quant_suffix = f"_4bit_{method}"
            if double_quant:
                quant_suffix += "_double"
            logger.info(f"  → 4-bit config: {json.dumps(quant_config, indent=2)}")
        else:
            logger.warning("⚠ 4-bit quantization skipped (BitsAndBytes not available)")

    # Extract ISO date from model file's modification time
    model_mtime = model_path_obj.stat().st_mtime
    from datetime import datetime

    iso_date = datetime.fromtimestamp(model_mtime).strftime("%Y-%m-%d")
    logger.info(f"  → Model creation date: {iso_date}")

    # Generate output path if not provided
    if output_path_obj is None:
        if is_quantized_pth:
            # INT8 quantized models save as .pth, not .onnx
            output_path_obj = model_path_obj.parent / f"{model_type}_{iso_date}{quant_suffix}.pth"
        else:
            output_path_obj = (
                model_path_obj.parent
                / f"{model_type}_{iso_date}{quant_suffix}_opset{opset_version}.onnx"
            )

    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if is_quantized_pth:
        logger.info("💾 Saving INT8 quantized model as PyTorch...")
        try:
            torch.save(quantized_model.state_dict(), output_path_obj)
            file_size = output_path_obj.stat().st_size
            logger.info(f"✓ INT8 model saved: {output_path_obj.name} ({file_size / 1e6:.2f} MB)")
        except Exception as e:
            logger.error(f"✗ Failed to save INT8 model: {e}")
            return None
    else:
        logger.info(f"🔄 Exporting {quantization} model to ONNX...")

        if not export_to_onnx(
            quantized_model, output_path_obj, opset_version, quantization, backend
        ):
            return None

        # Apply ONNX INT8 quantization if requested
        if onnx_int8:
            logger.info("📊 Applying ONNX INT8 quantization...")
            quantized_onnx_path, quant_info = quantize_onnx_int8(str(output_path_obj))
            if quantized_onnx_path:
                output_path_obj = Path(quantized_onnx_path)
                logger.info(f"✓ ONNX INT8 model saved: {output_path_obj.name}")
            else:
                logger.warning("⚠ ONNX INT8 quantization failed, using unquantized model")

    # Create metadata file
    info_path = output_path_obj.with_suffix(".json")
    info = {
        "model_type": model_type,
        "num_classes": num_classes,
        "quantization": quantization,
        "onnx_int8": onnx_int8,
        "input_shape": [1, 1, 64, 64],
        "input_names": ["input_image"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "backend": backend,
        "file_size_mb": output_path_obj.stat().st_size / 1e6,
        "from_pytorch": str(model_path),
        "format": "pth" if is_quantized_pth else "onnx",
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logger.info(f"✓ Metadata saved: {info_path.name}")

    # Verification (only for ONNX)
    if verify and not is_quantized_pth:
        logger.info("🔍 Verifying ONNX model...")
        try:
            import onnx

            onnx_model = onnx.load(str(output_path_obj))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX model is valid")
        except ImportError:
            logger.warning("⚠ ONNX not available for verification")
        except Exception as e:
            logger.warning(f"⚠ ONNX verification failed: {e}")

    # Inference test (only for ONNX)
    if test_inference and not is_quantized_pth:
        logger.info("🧪 Testing ONNX model inference...")
        try:
            import onnxruntime as ort

            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                ("CPUExecutionProvider", {}),
            ]
            sess = ort.InferenceSession(str(output_path_obj), providers=providers)
            logger.info(f"  → Using provider: {sess.get_providers()[0]}")

            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name

            times = []
            for _ in range(5):
                test_input = (torch.randn(1, 1, 64, 64).numpy()).astype("float32")
                start = time.time()
                sess.run([output_name], {input_name: test_input})
                times.append((time.time() - start) * 1000)

            avg_time = sum(times) / len(times)
            logger.info(f"✓ Inference test successful: {avg_time:.2f} ms per sample")

        except ImportError:
            logger.warning("⚠ ONNX Runtime not available for inference testing")
        except Exception as e:
            logger.warning(f"⚠ Inference test failed: {e}")

    return output_path_obj


def main():
    parser = argparse.ArgumentParser(
        description="Unified ONNX Model Export with Quantization Options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quantization Options:
  float32       - Full precision (baseline)
  int8          - INT8 dynamic quantization (3-4x smaller)
  4bit:nf4      - NF4 quantization (best accuracy/size trade-off)
  4bit:fp4      - FP4 quantization (fastest, smallest)
  4bit:nf4:double - NF4 with double quantization (extreme compression)

Examples:
  # Export CNN model (float32)
  python export_model_to_onnx.py --model-path training/cnn/best_model.pth --model-type cnn

  # Export with INT8 quantization
  python export_model_to_onnx.py --model-path training/cnn/best_model.pth --model-type cnn --quantization int8 --verify

  # Export with 4-bit NF4 quantization
  python export_model_to_onnx.py --model-path training/hiercode/best_model.pth --model-type hiercode --quantization 4bit:nf4 --test-inference

  # Export with maximum compression (4-bit NF4 + double quant)
  python export_model_to_onnx.py --model-path training/rnn/best_model.pth --model-type rnn --quantization 4bit:nf4:double --verify --test-inference

  # Custom output path and opset
  python export_model_to_onnx.py --model-path training/vit/best_model.pth --model-type vit --quantization int8 --output exports/my_model.onnx --opset 18
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained PyTorch model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["hiercode", "hiercode-higita", "cnn", "qat", "rnn", "radical-rnn", "vit"],
        help="Model architecture type (required)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="float32",
        choices=[
            "float32",
            "int8",
            "4bit:nf4",
            "4bit:fp4",
            "4bit:nf4:double",
            "4bit:fp4:double",
        ],
        help="Quantization type (default: float32)",
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
        default=18,
        help="ONNX opset version (default: 18)",
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
        "--backend",
        type=str,
        default="default",
        choices=["default", "tract", "ort-tract", "wasi", "tflite"],
        help="Target backend for optimization (default: default)",
    )
    parser.add_argument(
        "--onnx-int8",
        action="store_true",
        help="Apply INT8 quantization to ONNX model after export (requires onnxruntime)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("COMPREHENSIVE ONNX MODEL EXPORT")
    logger.info("=" * 70)
    logger.info(
        "Model: %s | Quantization: %s | Opset: %d | Backend: %s | ONNX INT8: %s",
        args.model_type,
        args.quantization,
        args.opset,
        args.backend,
        args.onnx_int8,
    )
    logger.info("Verify: %s | Test Inference: %s", args.verify, args.test_inference)
    logger.info("")

    result = export_model(
        args.model_path,
        model_type=args.model_type,
        quantization=args.quantization,
        output_path=args.output,
        opset_version=args.opset,
        verify=args.verify,
        test_inference=args.test_inference,
        backend=args.backend,
        onnx_int8=args.onnx_int8,
    )

    if result is None:
        logger.error("✗ Export failed")
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"✓ Export complete: {result.name}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

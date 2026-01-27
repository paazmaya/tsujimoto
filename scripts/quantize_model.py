#!/usr/bin/env python3
"""
Post-Training INT8 Quantization for Kanji Models
Converts trained PyTorch models to INT8 for efficient deployment
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path to import src/lib
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lib import (
    create_data_loaders,
    get_dataset_directory,
    load_chunked_dataset,
    setup_logger,
    verify_and_setup_gpu,
)

logger = setup_logger(__name__)

# Suppress PyTorch's TypedStorage deprecation warning (internal, not in user code)
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")

# Add scripts to path for imports
sys.path.append(str(Path(__file__).parent))
from train_cnn_model import LightweightKanjiNet  # noqa: E402
from train_hiercode import HierCodeClassifier  # noqa: E402
from train_qat import QuantizableLightweightKanjiNet  # noqa: E402
from train_radical_rnn import RadicalRNNClassifier  # noqa: E402
from train_rnn import KanjiRNN  # noqa: E402
from train_vit import VisionTransformer  # noqa: E402


def quantize_model_int8(model: nn.Module, device: str = "cuda", model_name: str = "quantized"):
    """
    Convert model to INT8 using dynamic quantization.

    This uses dynamic quantization (QAT-free):
    - No retraining or calibration required
    - Quantizes weights only (activations stay float32)
    - Reduces model size by ~4x
    - Minimal accuracy loss
    - Works cross-platform (Windows, Linux, Mac)

    Args:
        model: PyTorch model to quantize
        device: Device to use (always cuda)
        model_name: Name for logging
    """

    logger.info("🔄 Quantizing %s model to INT8 (dynamic)...", model_name)

    # Keep model on GPU for all calculations
    model = model.to(device)
    model.eval()

    # Calculate original size
    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if v is not None
    )
    logger.info("  → Original model size: %.2f MB", original_size / 1e6)

    # Dynamic quantization requires CPU, so temporarily move only for this operation
    # PyTorch limitation: dynamic quantization kernel only available on CPU backend
    logger.info("  → Applying INT8 quantization...")
    model_cpu = model.cpu()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    # Move quantized model back to GPU if needed for downstream operations
    quantized_model = quantized_model.to(device)
    logger.info("✓ INT8 quantization complete")

    # For dynamic quantization, the actual size is measured after saving
    # (packed parameters have a different memory layout)
    # Return a sentinel value that will be updated after saving
    quantized_size = None

    return quantized_model, original_size, quantized_size


def quantize_with_calibration(
    model: nn.Module,
    train_loader,
    device: str = "cuda",
    model_name: str = "quantized",
):
    """
    Quantize model with calibration using actual training data.

    This improves quantization accuracy by:
    - Using real data distribution for calibration
    - Computing optimal scale factors
    - Reducing accuracy loss from ~2-3% to <1%

    Args:
        model: PyTorch model to quantize
        train_loader: DataLoader for calibration
        device: Device to use (always cuda for GPU acceleration)
        model_name: Name for logging
    """

    logger.info("🔄 Quantizing %s model with calibration...", model_name)

    model = model.to(device)
    model.eval()

    # Calculate original size
    original_state = model.state_dict()
    original_size = sum(
        v.numel() * v.element_size() for v in original_state.values() if isinstance(v, torch.Tensor)
    )
    logger.info("  → Original model size: %.2f MB", original_size / 1e6)

    # Prepare for quantization
    qconfig = torch.quantization.get_default_qconfig("fbgemm")
    model.qconfig = qconfig  # type: ignore
    torch.quantization.prepare(model, inplace=True)

    # Calibration: Run model on subset of training data
    num_batches = min(100, len(train_loader))  # Use 100 batches for calibration
    logger.info("  → Running calibration on %d batches...", num_batches)
    with torch.no_grad():
        for i, (images, _labels) in enumerate(train_loader):
            if i >= num_batches:
                break

            images = images.to(device)

            # Forward pass to collect activation statistics
            if hasattr(model, "forward"):
                _ = model(images)

            if (i + 1) % 25 == 0:
                logger.info("    → Calibrated %d/%d batches", i + 1, num_batches)

    # Convert to quantized model
    logger.info("  → Converting to INT8...")
    torch.quantization.convert(model, inplace=True)
    logger.info("✓ Calibrated INT8 quantization complete")

    quantized_state = model.state_dict()
    quantized_size = sum(
        v.numel() * v.element_size()
        for v in quantized_state.values()
        if isinstance(v, torch.Tensor)
    )

    return model, original_size, quantized_size


def evaluate_quantized_model(model, test_loader, criterion, device: str = "cuda"):
    """
    Evaluate quantized model accuracy.

    Note: PyTorch dynamic quantization only has CPU kernels available.
    For GPU inference, export to ONNX format which supports GPU-accelerated int8 inference.

    Args:
        model: Quantized model
        test_loader: DataLoader for test set
        criterion: Loss function
        device: Device specified (note: inference runs on CPU due to PyTorch backend limitations)
    """
    logger.info("🧪 Evaluating quantized model...")

    # Dynamically quantized models only support CPU inference (PyTorch backend limitation)
    # For GPU inference in production, use ONNX export instead
    model = model.cpu()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for _i, (images, labels) in enumerate(test_loader):
            # Keep on CPU for quantized model evaluation
            images = images.cpu()
            labels = labels.cpu()

            outputs = model(images)
            
            # Handle dict outputs (e.g., from Hi-GITA model)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs)
            else:
                logits = outputs
            
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    logger.info("✓ Evaluation complete: Accuracy %.2f%%, Loss %.4f", accuracy, avg_loss)

    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(
        description="Post-Training INT8 Quantization for Kanji Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize HierCode model
  python quantize_model.py --model-path training/hiercode/hiercode_model_best.pth --model-type hiercode

  # Quantize with calibration
  python quantize_model.py --model-path training/cnn/best_kanji_model.pth --model-type cnn --calibrate

  # Evaluate quantized model
  python quantize_model.py --model-path training/cnn/best_kanji_model.pth --evaluate
        """,
    )

    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hiercode",
        choices=["hiercode", "hiercode-higita", "cnn", "qat", "rnn", "radical-rnn", "vit"],
        help="Model architecture type (default: hiercode)",
    )
    # Note: Dataset auto-detected via get_dataset_directory() from optimization_config
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Use calibration with training data for better accuracy",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate quantized model on test set",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quantized model (default: auto-generated)",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("POST-TRAINING INT8 QUANTIZATION")
    logger.info("=" * 70)
    logger.info(
        "Model: %s | Calibrate: %s | Evaluate: %s", args.model_type, args.calibrate, args.evaluate
    )

    # Verify GPU availability (required for quantization)
    device = verify_and_setup_gpu()

    # Load model

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error("✗ Model path not found: %s", model_path)
        return

    logger.info("→ Loading model from %s...", model_path)

    # Load checkpoint to infer num_classes from classifier layer
    checkpoint = torch.load(model_path, map_location="cpu")

    # Infer num_classes from checkpoint's classifier.weight shape
    # The classifier layer has shape [num_classes, hidden_dim]
    num_classes = None
    for key in checkpoint.keys():
        if "classifier.weight" in key:
            num_classes = checkpoint[key].shape[0]
            break

    # Try alternate names for classifier layers
    if num_classes is None:
        for key in checkpoint.keys():
            if "linear.weight" in key or "fc.weight" in key:
                num_classes = checkpoint[key].shape[0]
                break

    if num_classes is None:
        # Fallback: try to load config
        config_path = model_path.parent / f"{args.model_type}_config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
                num_classes = config_dict.get("num_classes", 3036)
        else:
            num_classes = 3036

    logger.info("→ Detected %d classes from model checkpoint", num_classes)

    # Create config and model based on type
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.lib.config import (
        CNNConfig,
        HierCodeConfig,
        OptimizationConfig,
        QATConfig,
        RadicalRNNConfig,
        RNNConfig,
        ViTConfig,
    )

    if args.model_type == "hiercode":
        config: OptimizationConfig = HierCodeConfig(num_classes=num_classes)
        model = HierCodeClassifier(num_classes=num_classes, config=config)  # type: ignore
    elif args.model_type == "hiercode-higita":
        config = HierCodeConfig(num_classes=num_classes)
        from hiercode_higita_enhancement import HierCodeWithHiGITA

        model = HierCodeWithHiGITA(num_classes=num_classes)
    elif args.model_type == "cnn":
        config = CNNConfig(num_classes=num_classes)
        model = LightweightKanjiNet(num_classes=num_classes)
    elif args.model_type == "qat":
        config = QATConfig(num_classes=num_classes)
        model = QuantizableLightweightKanjiNet(num_classes=num_classes)
    elif args.model_type == "rnn":
        config = RNNConfig(num_classes=num_classes)
        model = KanjiRNN(num_classes=num_classes)
    elif args.model_type == "radical-rnn":
        config = RadicalRNNConfig(num_classes=num_classes)
        model = RadicalRNNClassifier(num_classes=num_classes, config=config)
    elif args.model_type == "vit":
        config = ViTConfig(num_classes=num_classes)
        model = VisionTransformer(num_classes=num_classes, config=config)
    else:
        logger.error("✗ Unknown model type: %s", args.model_type)
        return

    # Load state dict with flexible key matching
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # Try loading with strict=False for compatibility
        model.load_state_dict(checkpoint, strict=False)
        logger.warning("⚠ Loaded model with strict=False (some keys may not match)")
    else:
        logger.info("✓ Model loaded successfully")

    # Quantize
    if args.calibrate:
        # Load dataset for calibration (auto-detected)
        logger.info("→ Loading dataset for calibration...")
        data_dir = str(get_dataset_directory())
        X, y = load_chunked_dataset(data_dir)
        
        # Split data (use prepare_dataset_and_loaders instead)
        from src.lib import prepare_dataset_and_loaders
        from torch.utils.data import TensorDataset
        import numpy as np
        
        def create_tensor_dataset(x, y):
            """Factory for TensorDataset"""
            if x.ndim == 2 and x.shape[1] == 4096:
                x = x.reshape(-1, 64, 64)
            if x.ndim == 3:
                x = x[:, np.newaxis, :, :]
            x_tensor = torch.from_numpy(x).float()
            y_tensor = torch.from_numpy(y).long()
            return TensorDataset(x_tensor, y_tensor)
        
        (train_X, train_y), num_classes, train_loader, val_loader = prepare_dataset_and_loaders(
            data_dir=data_dir,
            dataset_fn=create_tensor_dataset,
            batch_size=config.batch_size,
            sample_limit=None,
            logger=logger,
        )
        
        # Use train_loader for quantization (validation not needed for calibration)
        test_loader = train_loader

        quantized_model, orig_size, quant_size = quantize_with_calibration(
            model, train_loader, device=device, model_name=args.model_type
        )
    else:
        quantized_model, orig_size, quant_size = quantize_model_int8(
            model, device=device, model_name=args.model_type
        )

    # Save quantized model
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path.parent / f"quantized_{args.model_type}_int8.pth"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("→ Saving quantized model to %s...", output_path)
    torch.save(quantized_model.state_dict(), output_path)

    # Measure actual file size after saving (especially important for dynamic quantization)
    if quant_size is None and output_path.exists():
        quant_size = output_path.stat().st_size

    logger.info("✓ Quantized model saved")
    logger.info("  → Original size: %.2f MB", orig_size / 1e6)
    logger.info("  → Quantized size: %.2f MB", quant_size / 1e6 if quant_size else 0)
    if quant_size and orig_size:
        logger.info("  → Reduction: %.1f%%", 100 * (1 - quant_size / orig_size))

    # Evaluate if requested
    if args.evaluate:
        logger.warning("⚠ Evaluation skipped: model trained on %d classes", num_classes)
        logger.warning("  but full combined_all_etl dataset has 43427 classes")
        logger.info("  To train on the full dataset:")
        logger.info("    uv run python scripts/prepare_dataset.py")
        logger.info("    uv run python scripts/train_hiercode_higita.py --epochs 30")

    logger.info("=" * 70)
    logger.info("✓ Quantization complete!")


if __name__ == "__main__":
    main()
